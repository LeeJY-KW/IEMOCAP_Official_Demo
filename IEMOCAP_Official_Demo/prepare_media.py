#!/usr/bin/env python3
"""IEMOCAP 시험 대화의 원본 음성과 재생용 영상을 미리 준비함."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
import wave


ROOT = Path(__file__).resolve().parent
FEATURE_MANIFEST = ROOT / "data/features/manifest.json"
PROTOCOL_PATHS = {
    task: ROOT / "evidence" / task / "protocol.json"
    for task in ("classification", "va")
}


# 조건이 맞지 않으면 구체적인 원인을 표시하고 준비를 중단함.
def require(condition, message):
    if not condition:
        raise ValueError(message)


# 큰 원본 영상도 일정한 메모리 사용량으로 읽어 SHA256을 계산함.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ID가 폴더 이름이나 파일 이름에 들어가므로 경로 구분자와 상위 경로 이동을 차단함.
def safe_id(value, kind):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_]+", value),
            f"잘못된 {kind}: {value!r}")
    return value


# 지정한 대화를 고정된 시험 폴드와 연결함. 기본값은 1번 폴드의 첫 시험 대화임.
# 반환값은 대화 정보, 해당 시험 폴드 번호, 두 과제의 평가 설정 해시임.
def select_dialogue(requested):
    feature_manifest = read_json(FEATURE_MANIFEST)
    protocols = {task: read_json(path) for task, path in PROTOCOL_PATHS.items()}
    folds = protocols["classification"]["folds"]
    require(folds == protocols["va"]["folds"], "분류와 회귀의 폴드 구성이 다름")
    require([fold["fold"] for fold in folds] == [1, 2, 3, 4, 5], "폴드 번호·순서가 잘못됨")
    dialogue_id = safe_id(folds[0]["test"][0] if requested is None else requested, "대화 ID")
    dialogues = feature_manifest["dialogues"]
    indexed = {item["id"]: item for item in dialogues}
    require(len(indexed) == len(dialogues), "특징 목록의 대화 ID가 중복됨")
    require(dialogue_id in indexed, f"특징 목록에 없는 대화임: {dialogue_id}")
    dialogue = indexed[dialogue_id]
    session = dialogue["session"]
    require(type(session) is int and 1 <= session <= 5, "세션 번호가 잘못됨")
    require(dialogue_id.startswith(f"Ses{session:02d}"), "대화 ID와 세션 번호가 다름")
    matches = [fold for fold in folds if dialogue_id in fold["test"]]
    require(len(matches) == 1, "대화가 정확히 한 개 시험 폴드에 속해야 함")
    fold = matches[0]
    require(fold["fold"] == session, "시험 세션 번호와 폴드 번호가 다름")
    require(dialogue_id not in fold["train"] + fold["dev"], "선택 폴드의 학습·개발 자료와 겹침")
    groups = {}
    for role in ("train", "dev", "test"):
        ids = fold[role]
        require(len(ids) == len(set(ids)) and all(did in indexed for did in ids),
                f"{role} 대화 목록에 중복 또는 알 수 없는 ID가 있음")
        groups[role] = {indexed[did]["session"] for did in ids}
    require(groups["test"] == {session} and groups["dev"] == {session % 5 + 1}
            and groups["train"] == set(range(1, 6)) - groups["test"] - groups["dev"],
            "학습·개발·시험 세션 구성이 고정 평가 설정과 다름")

    # 특징 목록의 원래 발화 순서와 텍스트·시각·화자를 그대로 보존함.
    utterances = dialogue["utterances"]
    require(utterances, "대화에 발화가 없음")
    ids = []
    for utterance in utterances:
        uid = safe_id(utterance["id"], "발화 ID")
        require(re.fullmatch(re.escape(dialogue_id) + r"_[FM]\d+", uid),
                f"대화에 속하지 않는 발화 ID임: {uid}")
        start, end = utterance["start"], utterance["end"]
        require(all(type(value) in (int, float) and math.isfinite(value) for value in (start, end))
                and 0 <= start < end, f"발화 시각이 잘못됨: {uid}")
        require(isinstance(utterance["text"], str), f"텍스트 형식이 잘못됨: {uid}")
        require(type(utterance["speaker"]) is int and utterance["speaker"] in (0, 1),
                f"화자 값이 잘못됨: {uid}")
        ids.append(uid)
    require(len(ids) == len(set(ids)), "발화 ID가 중복됨")
    return dialogue, fold["fold"], {task: sha256(path) for task, path in PROTOCOL_PATHS.items()}


# 사용자가 지정한 원본 루트 아래의 고정 상대 경로만 사용함.
def source_file(source_root, relative):
    path = (source_root / relative).resolve()
    require(path.is_relative_to(source_root), f"원본 루트를 벗어난 경로임: {relative}")
    require(path.is_file(), f"원본 파일이 없음: {path}")
    return path


def find_ffmpeg(requested):
    executable = shutil.which(str(requested)) if requested else shutil.which("ffmpeg")
    require(executable is not None, "ffmpeg를 찾을 수 없음. --ffmpeg로 실행 파일 경로를 지정해야 함")
    executable = str(Path(executable).resolve())
    require(os.access(executable, os.X_OK), f"ffmpeg 실행 권한이 없음: {executable}")
    result = subprocess.run([executable, "-version"], capture_output=True, text=True, check=True)
    require(bool(result.stdout.strip()), "ffmpeg 버전 정보를 읽을 수 없음")
    return executable, result.stdout.splitlines()[0]


# ffmpeg의 입력 정보에서 길이와 스트림 종류를 읽음. 출력 파일을 지정하지 않아 반환 코드는 1일 수 있음.
def inspect_media(ffmpeg, path):
    result = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)],
                            capture_output=True, text=True)
    text = result.stderr
    duration = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", text)
    require(duration is not None, f"미디어 길이를 읽을 수 없음: {path}\n{text[-1000:]}")
    seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    streams = [line.strip() for line in text.splitlines() if re.search(r"Stream #.*(?:Video|Audio):", line)]
    video = next((line for line in streams if "Video:" in line), None)
    fps = re.search(r"([\d.]+) fps", video or "")
    require(math.isfinite(seconds) and seconds > 0, f"미디어 길이가 잘못됨: {path}")
    return {"duration_seconds": seconds, "streams": streams,
            "has_video": video is not None, "has_audio": any("Audio:" in line for line in streams),
            "fps": float(fps[1]) if fps else None}


# 원본 WAV의 샘플 수와 재생 시간을 기록함. 복사할 음성의 신호 값은 변경하지 않음.
def inspect_wave(path):
    with wave.open(str(path), "rb") as audio:
        frames, rate = audio.getnframes(), audio.getframerate()
        require(frames > 0 and rate > 0, f"비어 있거나 잘못된 WAV임: {path}")
        return {"frames": frames, "sample_rate": rate, "channels": audio.getnchannels(),
                "sample_width_bytes": audio.getsampwidth(), "duration_seconds": frames / rate}


def artifact_info(path):
    require(path.is_file() and path.stat().st_size > 0, f"생성 파일이 비어 있음: {path}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


# 영상은 원본 전체 프레임을 유지하고 발화 시간 구간만 잘라 H.264로 저장함.
# 같은 발화의 원본 WAV를 AAC 오디오 트랙으로 넣어 영상과 음성을 함께 재생하게 함.
# 셸을 사용하지 않고 인자 목록으로 실행하며 디코딩·인코딩 스레드를 최대 2개로 제한함.
def clip_command(ffmpeg, video, audio, start, end, output):
    return [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-threads", "2", "-ss", f"{start:.6f}", "-i", str(video),
            "-threads", "2", "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
            "-t", f"{end - start:.6f}", "-c:v", "libx264", "-threads:v", "2",
            "-filter_threads", "1", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-threads:a", "1", "-b:a", "128k", "-movflags", "+faststart",
            "-shortest", str(output)]


# 모든 발화와 목록이 준비된 임시 폴더만 완료 폴더로 이동함.
# 덮어쓰기를 요청한 경우에도 교체 직전까지 기존 폴더를 보존하며 이동 실패 시 복원함.
def publish_bundle(temporary, target, overwrite):
    require(not target.is_symlink(), f"출력 경로가 심볼릭 링크임: {target}")
    require(not target.exists() or overwrite, f"기존 자료가 있음. 교체하려면 --overwrite 필요: {target}")
    backup = None
    if target.exists():
        require(target.is_dir(), f"출력 경로가 폴더가 아님: {target}")
        backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
        target.rename(backup)
    try:
        temporary.rename(target)
    except BaseException:
        if backup is not None and not target.exists():
            backup.rename(target)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def prepare(iemocap_root, dialogue_id=None, ffmpeg_path=None, overwrite=False):
    source_root = Path(iemocap_root).expanduser().resolve()
    require(source_root.is_dir(), f"IEMOCAP 원본 폴더가 없음: {source_root}")
    feature_hash = sha256(FEATURE_MANIFEST)
    dialogue, fold, protocol_hashes = select_dialogue(dialogue_id)
    did, session = dialogue["id"], dialogue["session"]
    output_root = ROOT / "data/media"
    target = output_root / did
    require(not target.is_symlink(), f"출력 경로가 심볼릭 링크임: {target}")
    require(not target.exists() or overwrite, f"기존 자료가 있음. 교체하려면 --overwrite 필요: {target}")
    ffmpeg, version = find_ffmpeg(ffmpeg_path)
    video_relative = Path(f"Session{session}/dialog/avi/DivX/{did}.avi")
    video = source_file(source_root, video_relative)
    video_info = inspect_media(ffmpeg, video)
    require(video_info["has_video"], f"원본에 영상 스트림이 없음: {video}")
    frame_tolerance = 1 / video_info["fps"] if video_info["fps"] else 0.05
    require(max(u["end"] for u in dialogue["utterances"]) <= video_info["duration_seconds"] + frame_tolerance,
            "원본 영상 길이보다 발화 종료 시각이 큼. 원본 영상의 시간축을 먼저 확인해야 함")
    video_hash = sha256(video)

    # 변환을 시작하기 전에 대화 전체의 WAV 존재 여부와 형식을 확인함.
    sources = []
    for utterance in dialogue["utterances"]:
        relative = Path(f"Session{session}/sentences/wav/{did}/{utterance['id']}.wav")
        audio = source_file(source_root, relative)
        sources.append((utterance, relative, audio, inspect_wave(audio), sha256(audio)))
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{did}.preparing-", dir=output_root))
    try:
        records = []
        for index, (utterance, relative, audio, audio_info, audio_hash) in enumerate(sources, 1):
            uid = utterance["id"]
            local_audio, local_video = temporary / f"{uid}.wav", temporary / f"{uid}.mp4"
            shutil.copy2(audio, local_audio)
            audio_record = artifact_info(local_audio)
            require(audio_record["sha256"] == audio_hash, f"WAV 복사본이 원본과 다름: {uid}")
            command = clip_command(ffmpeg, video, local_audio, utterance["start"], utterance["end"], local_video)
            process = subprocess.run(command, capture_output=True, text=True)
            require(process.returncode == 0, f"영상 변환 실패: {uid}\n{process.stderr[-2000:]}")
            clip_info = inspect_media(ffmpeg, local_video)
            require(clip_info["has_video"] and clip_info["has_audio"], f"생성 클립의 영상·음성 스트림이 누락됨: {uid}")
            audio_record.update(source_path=relative.as_posix(), source_sha256=audio_hash, **audio_info)
            video_record = artifact_info(local_video)
            video_record.update(source_path=video_relative.as_posix(), source_sha256=video_hash,
                                conversion_args=command, inspection=clip_info)
            records.append({key: utterance[key] for key in ("id", "text", "start", "end", "speaker")}
                           | {"held_out_fold": fold, "audio": audio_record, "video": video_record})
            print(f"미디어 준비 {index}/{len(sources)} | {uid}", flush=True)

        # 준비에 사용한 원본 영상과 평가 정의가 작업 도중 바뀌지 않았는지 다시 확인함.
        require(sha256(video) == video_hash, "준비 도중 원본 영상이 변경됨")
        require(sha256(FEATURE_MANIFEST) == feature_hash, "준비 도중 특징 목록이 변경됨")
        require(all(sha256(PROTOCOL_PATHS[task]) == value for task, value in protocol_hashes.items()),
                "준비 도중 평가 설정이 변경됨")
        manifest = {"schema": 1, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
                    "dialogue_id": did, "session": session, "held_out_fold": fold,
                    "utterance_count": len(records), "source_root": str(source_root),
                    "feature_manifest_sha256": feature_hash, "protocol_sha256": protocol_hashes,
                    "source_timestamp_offset_seconds": 0.0,
                    "ffmpeg": {"executable": ffmpeg, "version": version, "max_codec_threads": 2},
                    "source_video": {"path": video_relative.as_posix(), "bytes": video.stat().st_size,
                                     "sha256": video_hash, "inspection": video_info}, "utterances": records}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        publish_bundle(temporary, target, overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"준비 완료: {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iemocap-root", type=Path, required=True, help="Session1~5를 포함한 IEMOCAP 원본 폴더")
    parser.add_argument("--dialogue", help="준비할 시험 대화 ID. 기본값은 1번 폴드의 첫 시험 대화")
    parser.add_argument("--ffmpeg", type=Path, help="ffmpeg 실행 파일 경로. 생략 시 PATH에서 검색함")
    parser.add_argument("--overwrite", action="store_true", help="준비가 모두 끝난 뒤 기존 대화 미디어 폴더를 교체함")
    args = parser.parse_args()
    try:
        prepare(args.iemocap_root, args.dialogue, args.ffmpeg, args.overwrite)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, wave.Error) as error:
        parser.exit(1, f"미디어 준비 실패: {error}\n")


if __name__ == "__main__":
    main()
