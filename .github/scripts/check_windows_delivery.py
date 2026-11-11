#!/usr/bin/env python3
"""Validate the delivered notebook and offline player on native Windows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import locale
import math
import os
from pathlib import Path
import platform
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


sys.dont_write_bytecode = True
KERNEL_NAME = "iemocap-demo-windows"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# 전달 목록의 모든 입력 파일과 목록 자체를 검사 전후 같은 바이트로 비교함.
# 실행 결과와 가상환경은 전달 목록의 입력 파일에 포함되지 않음.
def snapshot_inputs(package):
    manifest_path = package / "PACKAGE_MANIFEST.json"
    names = set(read_json(manifest_path)["files"]) | {"PACKAGE_MANIFEST.json"}
    snapshot = {}
    for name in sorted(names):
        path = (package / name).resolve()
        require(path.is_relative_to(package) and path.is_file(), f"Missing package input: {name}")
        snapshot[name] = sha(path)
    return snapshot


def snapshot_digest(snapshot):
    content = json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class QuietHandler(SimpleHTTPRequestHandler):
    # HTTP 요청은 현재 PC의 임의 포트로만 받으며 별도 접근 로그를 남기지 않음.
    def log_message(self, format, *args):
        pass


# 전용 설정으로 실제 서버를 시작하고 런타임 기록의 토큰으로 API를 확인함.
# 토큰은 메모리와 서버 임시 기록에서만 사용하며 표준 출력·보고서에 넣지 않음.
def check_jupyter_server(package, diagnostics):
    config = package / "jupyter_lab_config.py"
    require(config.is_file(), "The dedicated JupyterLab configuration is missing")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [sys.executable, "-m", "jupyterlab", "--config", str(config),
               "--no-browser", "--ip=127.0.0.1", "--port", str(port), "--ServerApp.port_retries=0"]
    result, base_url, token = None, None, None
    started = time.perf_counter()
    diagnostics.update(passed=False, expected_port=port, isolated_runtime=True,
                       runtime_candidate_count=0, candidates=[], parse_errors=0)
    with tempfile.TemporaryDirectory(prefix="iemocap-jupyter-runtime-") as runtime:
        environment = os.environ.copy()
        environment["JUPYTER_RUNTIME_DIR"] = runtime
        # Jupyter 기본 시작 메시지에는 인증 URL이 포함될 수 있으므로 저장하지 않음.
        process = subprocess.Popen(command, cwd=package, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        diagnostics["launcher_pid"] = process.pid
        server_info = None

        def api(endpoint, method="GET"):
            request = Request(base_url + endpoint.lstrip("/"),
                              data=b"{}" if method == "POST" else None, method=method,
                              headers={"Authorization": "token " + token, "Content-Type": "application/json"})
            diagnostics["last_api_endpoint"] = endpoint
            try:
                with urlopen(request, timeout=5) as response:
                    body = response.read()
                    return response.status, json.loads(body.decode("utf-8")) if body else {}
            except HTTPError as error:
                diagnostics["last_http_error_code"] = error.code
                raise
            except (URLError, TimeoutError) as error:
                diagnostics["last_connection_error_type"] = type(error).__name__
                raise

        def port_open():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return True
            except OSError:
                return False

        def stop_tree(pid):
            # Windows 가상환경 실행기와 실제 서버의 PID가 다를 수 있어 자식까지 종료함.
            try:
                stopped = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                diagnostics.setdefault("tree_termination", []).append({"pid": pid, "returncode": stopped.returncode})
            except subprocess.TimeoutExpired:
                diagnostics.setdefault("tree_termination", []).append({"pid": pid, "timed_out": True})

        try:
            deadline = time.monotonic() + 60
            status = None
            while time.monotonic() < deadline:
                diagnostics["launcher_returncode"] = process.poll()
                paths = list(Path(runtime).glob("jpserver-*.json"))
                diagnostics["runtime_candidate_count"] = len(paths)
                diagnostics["candidates"] = []
                for path in paths:
                    try:
                        candidate = read_json(path)
                        parsed = urlsplit(candidate["url"])
                        root_matches = Path(candidate["root_dir"]).resolve() == package
                        port_matches = candidate.get("port") == port and parsed.port == port
                        loopback = parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")
                    except (OSError, ValueError, KeyError, TypeError):
                        diagnostics["parse_errors"] += 1
                        continue
                    diagnostics["candidates"].append(dict(file=path.name, pid=candidate.get("pid"),
                        port=candidate.get("port"), root_matches_package=root_matches,
                        requested_port_matches=port_matches, loopback=loopback))
                    # 다른 서버가 쓰지 않는 임시 폴더와 요청 포트·패키지 경로로 식별함.
                    # 가상환경 실행기의 부모 PID와 같다는 조건은 사용하지 않음.
                    if root_matches and port_matches and loopback:
                        server_info = candidate
                        diagnostics["server_pid"] = candidate.get("pid")
                        diagnostics["server_pid_matches_launcher"] = candidate.get("pid") == process.pid
                        break
                require(process.poll() is None or server_info is not None,
                        f"JupyterLab launcher exited without a matching server (exit {process.returncode})")
                if server_info is not None:
                    parsed = urlsplit(server_info["url"])
                    require(parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")
                            and parsed.port == port, "The Jupyter server is not using the selected loopback address")
                    base_url = f"http://127.0.0.1:{port}" + parsed.path.rstrip("/") + "/"
                    token = server_info.get("token")
                    require(isinstance(token, str) and bool(token), "The Jupyter server did not enable token authentication")
                    try:
                        status_code, status = api("api/status")
                    except (HTTPError, URLError, TimeoutError):
                        time.sleep(0.25)
                        continue
                    require(status_code == 200, "Jupyter /api/status did not return HTTP 200")
                    diagnostics["status_http_code"] = status_code
                    break
                time.sleep(0.25)
            require(status is not None, "JupyterLab did not become ready within 60 seconds")
            require(Path(server_info["root_dir"]).resolve() == package, "The dedicated Jupyter root directory was not applied")
            kernels_code, kernels = api("api/kernelspecs")
            diagnostics.update(kernelspecs_http_code=kernels_code, observed_default_kernel=kernels.get("default"),
                               observed_kernels=sorted(kernels.get("kernelspecs", {})))
            require(kernels_code == 200 and kernels.get("default") == KERNEL_NAME,
                    "The Jupyter API reports a different default kernel")
            require(set(kernels.get("kernelspecs", {})) == {KERNEL_NAME},
                    "The Jupyter API exposes kernels outside the dedicated configuration")
            result = dict(passed=True, configuration=str(config), configuration_sha256=sha(config),
                          loopback_only=True, status_http_code=status_code, kernelspecs_http_code=kernels_code,
                          root_dir_matches_package=True, default_kernel=kernels["default"],
                          available_kernels=sorted(kernels["kernelspecs"]),
                          api_status={key: status.get(key) for key in ("connections", "kernels", "started", "last_activity")},
                          startup_seconds=time.perf_counter() - started)
        finally:
            if base_url is not None and token is not None:
                try:
                    api("api/shutdown", method="POST")
                except (HTTPError, URLError, TimeoutError, ValueError):
                    pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stop_tree(process.pid)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if port_open() and server_info is not None:
                server_pid = server_info.get("pid")
                if isinstance(server_pid, int) and server_pid > 0:
                    stop_tree(server_pid)
            diagnostics.update(launcher_returncode=process.poll(), elapsed_seconds=time.perf_counter() - started,
                               server_port_closed=not port_open())
            require(process.poll() is not None, "The temporary JupyterLab server could not be stopped")
            require(diagnostics["server_port_closed"], "The temporary JupyterLab server port is still open")
            # 런타임 폴더는 이 블록 종료 시 삭제되며 인증 기록이 산출물에 남지 않음.
    result.update(server_stopped=True, temporary_runtime_removed=True)
    diagnostics.update(result)
    result = diagnostics
    return result


# 실제 Microsoft Edge에서 재생 버튼을 누르고 시간 진행과 종료를 확인함.
# 영상 DOM의 play() 호출 대신 브라우저의 기본 재생 컨트롤을 클릭함.
def check_edge_playback(package, output, utterance_id, diagnostics):
    from playwright.sync_api import sync_playwright
    from video_playback import load_video_playback

    dialogue = utterance_id.rsplit("_", 1)[0]
    media_root = package / "data/media"
    mp4 = media_root / dialogue / f"{utterance_id}.mp4"
    _, webm, standalone = load_video_playback(mp4)
    preparation = read_json(mp4.with_suffix(".playback.json"))
    handler = partial(QuietHandler, directory=str(media_root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/" + quote(standalone.relative_to(media_root).as_posix())
    state_script = """v => ({
        current_time: v.currentTime, duration: v.duration, paused: v.paused,
        ended: v.ended, ready_state: v.readyState, controls: v.controls,
        width: v.videoWidth, height: v.videoHeight,
        direct_webm_source: v.getAttribute('src').startsWith('data:video/webm;base64,'),
        error: v.error ? {code: v.error.code, message: v.error.message} : null
    })"""
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            try:
                diagnostics.update(passed=False, browser="Microsoft Edge", version=browser.version, headless=True)
                page = browser.new_page(viewport={"width": 1100, "height": 900})
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                video = page.locator("video")
                require(video.count() == 1, "The offline player must contain one video")
                page.wait_for_function("""() => {
                    const v = document.querySelector('video');
                    return v && (v.error || (v.readyState >= 2 && Number.isFinite(v.duration)));
                }""", timeout=30000)
                before = video.evaluate(state_script)
                diagnostics["before"] = before
                require(before["error"] is None, f"Edge failed to load WebM: {before}")
                require(before["controls"] and before["direct_webm_source"], "Unexpected video player markup")
                require((before["width"], before["height"]) == (720, 480), "Unexpected video dimensions")
                duration = before["duration"]
                require(math.isfinite(duration) and duration > 0, "Invalid Edge media duration")
                validation = preparation["validation"]
                require(abs(duration - validation["expected_seconds"]) <= validation["duration_tolerance_seconds"],
                        "Edge media duration exceeds the existing preparation tolerance")
                page.screenshot(path=str(output / "edge_before_playback.png"), full_page=True)
                video.scroll_into_view_if_needed()
                bounds = video.bounding_box()
                require(bounds is not None, "The video element is not visible")
                # Windows Edge 화면에서 확인한 재생 버튼 위치임. 맨 아래 탐색 막대의 위쪽임.
                play_position = {"x": 24, "y": bounds["height"] - 48}
                diagnostics["native_play_control_position"] = play_position
                video.click(position=play_position)
                try:
                    page.wait_for_function("""() => {
                        const v = document.querySelector('video');
                        return v && v.currentTime > 0.05 && !v.paused && !v.error;
                    }""", timeout=10000)
                except Exception:
                    diagnostics["after_play_control"] = video.evaluate(state_script)
                    page.screenshot(path=str(output / "edge_playback_failed.png"), full_page=True)
                    raise
                started = video.evaluate(state_script)
                diagnostics["started"] = started
                page.wait_for_function("""() => {
                    const v = document.querySelector('video');
                    return v && (v.ended || v.error);
                }""", timeout=max(30000, int((duration + 20) * 1000)))
                ended = video.evaluate(state_script)
                require(ended["ended"] and ended["error"] is None, f"Edge playback did not finish cleanly: {ended}")
                page.screenshot(path=str(output / "edge_playback_complete.png"), full_page=True)
                return dict(passed=True, browser="Microsoft Edge", channel="msedge", version=browser.version,
                            headless=True, input_action="native video play control click", loopback_only=True,
                            media=str(webm), standalone_html=str(standalone), webm_sha256=sha(webm),
                            before=before, started=started, ended=ended,
                            screenshots=["edge_before_playback.png", "edge_playback_complete.png"])
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def validate(package, output, report):
    require(platform.system() == "Windows", "This validation must run on native Windows")
    require(sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 10), "CPython 3.10 is required")
    require(struct.calcsize("P") == 8, "64-bit Python is required")
    venv = package / ".venv"
    expected_python = venv / "Scripts/python.exe"
    require(Path(sys.executable).resolve() == expected_python.resolve(), "Run with the package .venv/Scripts/python.exe")
    require(Path(sys.prefix).resolve() == venv.resolve(), "The active Python prefix is not the package virtual environment")
    require(" " in str(package) and any(ord(c) > 127 for c in str(package)),
            "Use a package path containing both non-ASCII characters and a space")
    sys.path.insert(0, str(package))

    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client.kernelspec import KernelSpecManager
    import numpy as np
    import torch

    require(torch.__version__.split("+")[0] == "2.7.1" and "+cpu" in torch.__version__, "PyTorch 2.7.1 CPU wheel is required")
    require(torch.version.cuda is None, "A CPU-only PyTorch build is required")
    spec = KernelSpecManager().get_kernel_spec(KERNEL_NAME)
    require(Path(spec.argv[0]).resolve() == expected_python.resolve(), "The registered notebook kernel points to another Python")
    report["kernel"] = dict(name=KERNEL_NAME, argv=spec.argv, resource_dir=spec.resource_dir,
                            executable_matches_package_venv=True)
    report["runtime"] = dict(torch=torch.__version__, torch_cuda=torch.version.cuda, numpy=np.__version__)
    packages = ("torch", "numpy", "scikit-learn", "ipython", "matplotlib", "ipykernel", "nbformat",
                "nbclient", "nbconvert", "librosa", "soundfile", "tokenizers", "jupyterlab", "playwright")
    report["installed_packages"] = {name: importlib.metadata.version(name) for name in packages}

    before_inputs = snapshot_inputs(package)
    report["input_preservation"] = dict(files=len(before_inputs), before_sha256=snapshot_digest(before_inputs))
    report["jupyterlab_server_api"] = {"passed": False}
    try:
        check_jupyter_server(package, report["jupyterlab_server_api"])
    except Exception:
        # 서버 설정 실패도 기록하되 독립적인 노트북·Edge 검증은 계속 진행함.
        # 예외에는 토큰이나 인증 URL이 들어가지 않고 안전한 진단 필드만 남음.
        report["jupyterlab_server_api"]["error"] = traceback.format_exc()
    source_path = package / "현장시연.ipynb"
    with source_path.open(encoding="utf-8") as stream:
        notebook = nbformat.read(stream, as_version=4)
    code_count = sum(cell.cell_type == "code" for cell in notebook.cells)
    require(len(notebook.cells) == 64 and code_count == 31, "Unexpected delivered notebook cell count")
    before_results = {path.resolve() for path in (package / "results").glob("notebook_*") if path.is_dir()}
    started = time.perf_counter()
    client = NotebookClient(notebook, timeout=900, kernel_name=KERNEL_NAME,
                            resources={"metadata": {"path": str(package)}})
    # 실패한 셀 출력도 파일로 남겨 실제 Windows 환경의 오류를 확인할 수 있게 함.
    try:
        client.execute()
    finally:
        with (output / "현장시연_실행완료.ipynb").open("w", encoding="utf-8", newline="\n") as stream:
            nbformat.write(notebook, stream)
        after_inputs = snapshot_inputs(package)
        changed = sorted(name for name in before_inputs.keys() | after_inputs.keys()
                         if before_inputs.get(name) != after_inputs.get(name))
        report["input_preservation"].update(after_sha256=snapshot_digest(after_inputs), changed_files=changed,
                                             passed=not changed)
        require(not changed, f"Delivered input files were modified: {changed}")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [item for cell in code_cells for item in cell.get("outputs", []) if item.output_type == "error"]
    require(not errors, "Notebook execution produced error outputs")
    require([cell.execution_count for cell in code_cells] == list(range(1, 32)), "Notebook cells were not executed in order")
    visual_records = [item.get("data", {}).get("application/json", {})
                      for cell in code_cells for item in cell.get("outputs", [])]
    visual_records = [item["visual_transform_audit"] for item in visual_records
                      if isinstance(item, dict) and "visual_transform_audit" in item]
    require(len(visual_records) == 1 and len(visual_records[0]) == 3, "Missing RGB transformation audit")
    require(all(row["passed"] and row["max_rgb_difference"] <= 1
                and row["max_boundary_distance"] <= 1e-4 and row["normalization_atol"] == 1e-6
                and len(row["rounding_differences"]) == row["different_rgb_elements"]
                for row in visual_records[0]), "RGB rounding boundary verification failed")
    report["visual_transform_audit"] = visual_records[0]
    first_output = "".join(item.get("text", "") for item in code_cells[0].get("outputs", []) if item.output_type == "stream")
    python_lines = [line for line in first_output.splitlines() if line.startswith("Python:")]
    require(len(python_lines) == 1 and Path(python_lines[0].split(":", 1)[1].strip()).resolve() == expected_python.resolve(),
            "The executed notebook reported a different Python interpreter")
    after_results = {path.resolve() for path in (package / "results").glob("notebook_*") if path.is_dir()}
    created = after_results - before_results
    require(len(created) == 1, f"Expected one fresh notebook result directory, found {len(created)}")
    result = created.pop()
    status = read_json(result / "status.json")
    audit = read_json(result / "independent_audit.json")
    preprocessing = read_json(result / "preprocessing_example_audit.json")
    example = read_json(result / "notebook_example_audit.json")
    summary = read_json(result / "summary.json")
    require(status["status"] == "complete" and status["training"] is False, "Inference did not complete")
    require(audit["passed"] and audit["comparison_rows"] == 10039 and len(audit["folds"]) == 10,
            "The delivered independent prediction audit did not pass")
    require(audit["prediction_atol"] == 1e-4 and audit["metric_recompute_atol"] == 1e-10 and audit["new_training"] is False,
            "The original independent audit conditions changed")
    require(preprocessing["passed"] and preprocessing["pretrained_forward"] == "prepared_intermediate_outputs",
            "Preprocessing verification did not pass")
    comparisons = preprocessing["feature_comparison"]
    require(len(comparisons) == 4 and {row["입력"] for row in comparisons} == {"audio", "prosody", "text", "visual"}
            and all(row["검사"] == "통과" for row in comparisons), "Not all four preprocessing comparisons passed")
    require(all(value["passed"] for value in example.values()), "Example predictions differ from the full evaluation")
    require(summary["device"] == "cpu" and summary["threads"] == 4 and summary["training"] is False,
            "Unexpected inference device, thread count, or training state")
    report["notebook"] = dict(passed=True, cells=64, code_cells_executed=31, error_outputs=0,
                              kernel_executable_confirmed=True, seconds=time.perf_counter() - started,
                              fresh_result=str(result), source_sha256=sha(source_path))
    report["independent_audit"] = audit
    report["preprocessing"] = preprocessing
    report["summary"] = summary
    report["example_audit"] = example
    body, _ = HTMLExporter().from_notebook_node(notebook)
    (output / "현장시연_설명과결과.html").write_text(body, encoding="utf-8", newline="\n")
    report["edge_playback"] = {"passed": False}
    report["edge_playback"].update(check_edge_playback(package, output, preprocessing["utterance_id"],
                                                     report["edge_playback"]))
    final_inputs = snapshot_inputs(package)
    require(final_inputs == before_inputs, "Package input files changed during browser verification")
    report["input_preservation"]["after_browser_sha256"] = snapshot_digest(final_inputs)
    report["scope"] = dict(native_windows=True, notebook_kernel_execution=True,
                           offline_edge_video_playback=True,
                           jupyterlab_server_api=report["jupyterlab_server_api"]["passed"],
                           jupyterlab_full_browser_ui=False,
                           new_training=False, original_model_audit_tolerances_preserved=True,
                           rgb_rounding_boundary_checks=True,
                           linux_byte_or_dtype_equality_required=False)
    require(report["jupyterlab_server_api"]["passed"],
            "JupyterLab server API validation failed; see jupyterlab_server_api diagnostics")


def main():
    parser = argparse.ArgumentParser(description="실제 Windows에서 노트북 전체 실행과 Edge 영상 재생을 검증함")
    parser.add_argument("--package", type=Path, required=True, help="한글과 공백을 포함한 경로의 전달 패키지")
    parser.add_argument("--output", type=Path, required=True, help="검증 보고서와 실행 출력 저장 폴더")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    package, output = args.package.resolve(), args.output.resolve()
    require(package.is_dir(), "The package directory does not exist")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "windows_validation.json"
    require(not destination.exists(), "The validation report already exists")
    report = dict(schema=1, passed=False, started_at=datetime.now(timezone.utc).isoformat(),
                  package=str(package), output=str(output),
                  host=dict(system=platform.system(), release=platform.release(), version=platform.version(),
                            platform=platform.platform(), machine=platform.machine(),
                            processor=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
                            logical_cpu_count=os.cpu_count()),
                  python=dict(version=sys.version, executable=sys.executable, prefix=sys.prefix,
                              pointer_bits=struct.calcsize("P") * 8, implementation=sys.implementation.name,
                              preferred_encoding=locale.getpreferredencoding(False),
                              filesystem_encoding=sys.getfilesystemencoding(), utf8_mode=sys.flags.utf8_mode))
    try:
        validate(package, output, report)
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8", newline="\n")
        print(json.dumps({"passed": report["passed"], "report": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
