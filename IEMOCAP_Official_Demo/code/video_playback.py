#!/usr/bin/env python3
"""Prepare and verify an offline WebM player for an existing IEMOCAP clip."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MP4 = ROOT / "data/media/Ses01F_impro01/Ses01F_impro01_F011.mp4"


# 파일을 작은 조각으로 읽어 원본과 재생 자료의 SHA-256을 계산함.
def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# 기록의 파일명·크기·해시를 모두 검사함. 심볼릭 링크로 다른 파일을 읽지 않음.
def _check_file(path: Path, record: dict) -> None:
    if not isinstance(record, dict) or record.get("path") != path.name:
        raise ValueError(f"재생 기록의 파일명이 다름: {path.name}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"일반 파일을 확인할 수 없음: {path}")
    if path.stat().st_size != record.get("bytes") or _sha(path) != record.get("sha256"):
        raise ValueError(f"파일 크기 또는 해시가 다름: {path.name}")


def _file_record(path: Path) -> dict:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}


# 기존 미디어 목록을 기준으로 MP4와 같은 발화의 원본 WAV 복사본을 연결함.
# 반환값은 MP4 경로, WAV 경로, 기존 목록 경로, 발화 메타데이터임.
def _source_files(mp4_path: str | Path) -> tuple[Path, Path, Path, dict]:
    requested = Path(mp4_path)
    if requested.is_symlink():
        raise ValueError("MP4 경로는 일반 파일이어야 함")
    mp4 = requested.resolve(strict=True)
    media_root = (ROOT / "data/media").resolve()
    if not mp4.is_relative_to(media_root) or mp4.suffix != ".mp4":
        raise ValueError("현재 패키지의 data/media 안에 있는 MP4가 필요함")
    if re.fullmatch(r"[A-Za-z0-9_]+", mp4.stem) is None:
        raise ValueError("발화 파일명 형식이 잘못됨")
    manifest_path = mp4.parent / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("미디어 목록은 일반 파일이어야 함")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("dialogue_id") != mp4.parent.name:
        raise ValueError("기존 미디어 준비 상태 또는 대화 ID가 다름")
    rows = manifest.get("utterances", [])
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == mp4.stem]
    if len(matches) != 1:
        raise ValueError("미디어 목록에서 해당 발화를 하나로 확인할 수 없음")
    utterance = matches[0]
    wav = mp4.with_suffix(".wav")
    _check_file(mp4, utterance.get("video"))
    _check_file(wav, utterance.get("audio"))
    if utterance["audio"].get("source_sha256") != utterance["audio"]["sha256"]:
        raise ValueError("원본 WAV와 준비된 WAV의 해시가 다름")
    return mp4, wav, manifest_path, utterance


def _output_paths(mp4: Path) -> tuple[Path, Path, Path]:
    return mp4.with_suffix(".webm"), mp4.with_suffix(".playback.html"), mp4.with_suffix(".playback.json")


# source 하위 태그 대신 video의 src에 데이터를 직접 넣음.
# 영상 한 개만 표시하며 JavaScript나 외부 파일 요청 없이 재생할 수 있음.
def _player_html(webm: Path, utterance: dict) -> str:
    encoded = base64.b64encode(webm.read_bytes()).decode("ascii")
    uid = html.escape(utterance["id"])
    text = html.escape(utterance["text"])
    return ("<section><h3>영상 재생</h3>"
            f"<p>발화 ID: {uid}</p><p>{text}</p>"
            f'<video src="data:video/webm;base64,{encoded}" controls playsinline '
            'preload="metadata" width="720" style="max-width:100%;height:auto">'
            "이 환경에서는 영상 재생 화면을 표시할 수 없음.</video>"
            "<p>재생 버튼을 눌러 같은 발화의 영상과 음성을 확인함.</p></section>")


def _standalone_html(player: str) -> str:
    return ("<!doctype html>\n<html lang=\"ko\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>영상 재생</title><style>body{font-family:sans-serif;max-width:900px;"
            "margin:32px auto;padding:0 16px;line-height:1.6}</style></head><body>"
            + player + "</body></html>\n")


# 현장 호출 함수임. 변환을 실행하지 않고 준비된 자료의 연결·무결성만 검사함.
# 입력은 기존 MP4 경로이며 반환값은 표시 HTML, WebM 경로, 독립 HTML 경로임.
def load_video_playback(mp4_path: str | Path) -> tuple[str, Path, Path]:
    mp4, wav, manifest_path, utterance = _source_files(mp4_path)
    webm, standalone, sidecar = _output_paths(mp4)
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"WebM 재생 자료를 먼저 준비해야 함: {sidecar}")
    record = _read_json(sidecar)
    if record.get("schema") != 1 or record.get("status") != "complete":
        raise ValueError("재생 자료 준비 완료 기록을 확인할 수 없음")
    if record.get("utterance_id") != utterance["id"]:
        raise ValueError("재생 자료의 발화 ID가 다름")
    metadata = {key: utterance[key] for key in ("id", "text", "start", "end", "speaker", "held_out_fold")}
    if record.get("utterance") != metadata:
        raise ValueError("재생 자료의 원문·시각·화자·시험 폴드가 다름")
    for key, path in (("mp4", mp4), ("wav", wav), ("media_manifest", manifest_path)):
        _check_file(path, record.get("sources", {}).get(key))
    for key, path in (("webm", webm), ("html", standalone)):
        _check_file(path, record.get("outputs", {}).get(key))
    if record.get("validation", {}).get("full_decode_passed") is not True:
        raise ValueError("준비 단계의 전체 영상·음성 디코딩 검사 기록이 없음")
    player = _player_html(webm, utterance)
    if standalone.read_text(encoding="utf-8") != _standalone_html(player):
        raise ValueError("독립 재생 HTML과 현재 WebM·발화 정보가 다름")
    return player, webm, standalone


# subprocess에는 인자 목록을 전달함. 셸 확장 없이 최대 두 코덱 스레드를 사용함.
def _run(command: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"ffmpeg 실행 실패: {result.stderr[-3000:]}")
    return result


# 완료 표시인 sidecar는 마지막에 교체함. 실패 시 이번에 옮긴 파일을 되돌림.
# 기존 파일 교체는 prepare의 overwrite=True 또는 CLI의 --overwrite가 필요함.
def _publish(staged: list[Path], targets: list[Path], overwrite: bool, work: Path) -> None:
    existing = [path for path in targets if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError("기존 재생 파일이 있음. 교체하려면 --overwrite를 명시해야 함")
    backups, published = [], []
    try:
        for target in existing:
            backup = work / (target.name + ".previous")
            target.replace(backup)
            backups.append((target, backup))
        for source, target in zip(staged, targets):
            source.replace(target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        for target, backup in reversed(backups):
            backup.replace(target)
        raise


# 준비 단계에서만 실행함. 기존 MP4의 영상과 원본 WAV 복사본으로 WebM을 생성함.
# 임시 파일을 전체 디코딩한 뒤 독립 HTML과 해시 기록을 함께 게시함.
def prepare(mp4_path: str | Path, ffmpeg_path: str | Path | None = None, *, overwrite: bool = False) -> dict:
    mp4, wav, manifest_path, utterance = _source_files(mp4_path)
    targets = list(_output_paths(mp4))
    if any(path.exists() or path.is_symlink() for path in targets) and not overwrite:
        raise FileExistsError("기존 재생 파일이 있음. 교체하려면 --overwrite를 명시해야 함")
    executable = str(ffmpeg_path) if ffmpeg_path is not None else shutil.which("ffmpeg")
    if not executable or not Path(executable).is_file():
        raise ValueError("ffmpeg 실행 파일 경로를 --ffmpeg로 지정해야 함")
    executable = str(Path(executable).resolve())
    sources = {key: _file_record(path) for key, path in (("mp4", mp4), ("wav", wav), ("media_manifest", manifest_path))}
    expected_seconds = float(utterance["audio"]["duration_seconds"])
    fps = float(utterance["video"]["inspection"]["fps"])
    if not math.isfinite(expected_seconds) or expected_seconds <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("원본 재생 시간 또는 FPS가 잘못됨")
    # 컨테이너 표시의 0.01초 반올림과 한 프레임, Vorbis 블록 길이를 허용함.
    duration_tolerance = 1 / fps + 2048 / utterance["audio"]["sample_rate"] + 0.01
    with tempfile.TemporaryDirectory(prefix=f".{mp4.stem}.playback-", dir=mp4.parent) as directory:
        work = Path(directory)
        webm, standalone, sidecar = [work / path.name for path in targets]
        command = [executable, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                   "-threads", "2", "-i", str(mp4), "-threads", "2", "-i", str(wav),
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libvpx", "-threads:v", "2",
                   "-filter_threads", "1", "-pix_fmt", "yuv420p", "-deadline", "good", "-cpu-used", "2",
                   "-b:v", "1M", "-crf", "10", "-c:a", "libvorbis", "-threads:a", "1",
                   "-q:a", "4", "-shortest", str(webm)]
        _run(command)
        inspection = subprocess.run([executable, "-hide_banner", "-i", str(webm)], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace")
        info = inspection.stderr
        duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
        if "Video: vp8" not in info or "Audio: vorbis" not in info or duration is None:
            raise ValueError("WebM의 VP8 영상·Vorbis 음성·재생 시간을 확인할 수 없음")
        hours, minutes, seconds = map(float, duration.groups())
        actual_seconds = hours * 3600 + minutes * 60 + seconds
        if abs(actual_seconds - expected_seconds) > duration_tolerance:
            raise ValueError("WebM 재생 시간이 원본 WAV와 다름")
        _run([executable, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-threads", "2",
              "-i", str(webm), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
        standalone.write_text(_standalone_html(_player_html(webm, utterance)), encoding="utf-8")
        record = {"schema": 1, "status": "complete", "prepared_at": datetime.now(timezone.utc).isoformat(),
                  "utterance_id": utterance["id"],
                  "utterance": {key: utterance[key] for key in ("id", "text", "start", "end", "speaker", "held_out_fold")},
                  "sources": sources, "outputs": {"webm": _file_record(webm), "html": _file_record(standalone)},
                  "conversion_args": command, "ffmpeg_version": _run([executable, "-version"]).stdout.splitlines()[0],
                  "validation": {"full_decode_passed": True, "video_codec": "vp8", "audio_codec": "vorbis",
                                 "expected_seconds": expected_seconds, "reported_seconds": actual_seconds,
                                 "duration_tolerance_seconds": duration_tolerance}}
        # 변환 도중 원본이 바뀌지 않았는지 다시 검사하고 새 파일 세 개만 게시함.
        for key, path in (("mp4", mp4), ("wav", wav), ("media_manifest", manifest_path)):
            _check_file(path, sources[key])
        sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _publish([webm, standalone, sidecar], targets, overwrite, work)
    load_video_playback(mp4)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="기존 발화 영상의 WebM 재생 자료와 오프라인 HTML을 준비함")
    parser.add_argument("--mp4", type=Path, default=DEFAULT_MP4, help="기존 MP4 경로. 기본값은 긴 문장 예시 F011")
    parser.add_argument("--ffmpeg", type=Path, help="ffmpeg 실행 파일 경로. 생략하면 PATH에서 검색함")
    parser.add_argument("--overwrite", action="store_true", help="검증 완료 후 기존 재생 자료 세 개를 교체함")
    args = parser.parse_args()
    result = prepare(args.mp4, args.ffmpeg, overwrite=args.overwrite)
    print(json.dumps({"상태": result["status"], "발화 ID": result["utterance_id"],
                      "재생 자료": [str(path) for path in _output_paths(args.mp4.resolve())]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
