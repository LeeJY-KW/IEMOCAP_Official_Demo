"""IEMOCAP manifest and resumable, utterance-local frozen feature extraction.

Run ``python -m emotionheart.inductive.data --help`` from the workspace root.
All private inputs stay local. No labels enter the feature extractors.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time


LABELS = ["ang", "hap", "exc", "sad", "fru", "neu"]
DIMS = {"audio": 1536, "prosody": 16, "text": 768, "visual": 512}
ANNOTATION = re.compile(r"\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+(\w+)")
TRANSCRIPT = re.compile(r"(Ses\S+)\s+\[([\d.]+)\s*-\s*([\d.]+)\]:\s*(.*)")
PROSODY_NAMES = [
    "log1p_duration_seconds", "log_rms_mean", "log_rms_std",
    "log_rms_mean_absolute_delta", "voiced_log_f0_mean", "voiced_log_f0_std",
    "voiced_log_f0_iqr", "voiced_log_f0_mean_absolute_delta", "voiced_fraction",
    "zcr_mean", "zcr_std", "centroid_hz_mean", "centroid_hz_std",
    "rolloff_85_hz_mean", "spectral_flatness_mean", "quiet_fraction",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def source_stat(path):
    stat = Path(path).stat()
    if stat.st_size == 0:
        raise ValueError(f"Empty source file: {path}")
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def split_sessions(test_session):
    """Five disjoint folds: one test, next cyclic dev, remaining three train."""
    if test_session not in range(1, 6):
        raise ValueError("test_session must be 1..5")
    dev = test_session % 5 + 1
    return {"train": [s for s in range(1, 6) if s not in (test_session, dev)],
            "dev": [dev], "test": [test_session]}


def build_manifest(dataset, cache):
    root, cache = Path(dataset).resolve(), Path(cache)
    dialogues, all_ids, sources = [], set(), {}
    for session in range(1, 6):
        folder = root / f"Session{session}"
        annotations = sorted(p for p in (folder / "dialog/EmoEvaluation").glob("*.txt")
                             if not p.name.startswith("._"))
        if not annotations:
            raise FileNotFoundError(f"No annotations in {folder}")
        for annotation in annotations:
            name = annotation.stem
            transcript = folder / "dialog/transcriptions" / f"{name}.txt"
            visual = folder / "dialog/avi/DivX" / f"{name}.avi"
            sources[str(visual)] = source_stat(visual)
            texts = {}
            for path in (annotation, transcript):
                sources[str(path)] = {**source_stat(path),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            annotation_lines = annotation.read_text(encoding="utf-8-sig").splitlines()
            annotated_ids = {m[3] for line in annotation_lines if (m := ANNOTATION.match(line))}
            extra_transcripts = 0
            for line in transcript.read_text(encoding="utf-8-sig").splitlines():
                match = TRANSCRIPT.match(line)
                if match:
                    if match[1] not in annotated_ids:
                        extra_transcripts += 1
                        continue
                    if match[1] in texts:
                        raise ValueError(f"Duplicate transcript: {match[1]}")
                    texts[match[1]] = match[4]
            utterances = []
            for line in annotation_lines:
                match = ANNOTATION.match(line)
                if not match:
                    continue
                start, end, uid, emotion = match.groups()
                start, end = float(start), float(end)
                if uid in all_ids or not uid.startswith(name + "_"):
                    raise ValueError(f"Duplicate or wrong-dialogue annotation: {uid}")
                if uid not in texts or not texts[uid].strip():
                    raise ValueError(f"Missing transcript: {uid}")
                if not 0 <= start < end:
                    raise ValueError(f"Invalid annotation time: {uid}")
                speaker_code = uid.rsplit("_", 1)[-1][0]
                if speaker_code not in "FM":
                    raise ValueError(f"Unrecognized speaker: {uid}")
                audio = folder / "sentences/wav" / name / f"{uid}.wav"
                sources[str(audio)] = source_stat(audio)
                utterances.append({"id": uid, "start": start, "end": end,
                                   "speaker": "FM".index(speaker_code),
                                   "audio_path": str(audio), "text": texts[uid],
                                   "visual_path": str(visual),
                                   "label": LABELS.index(emotion) if emotion in LABELS else -100,
                                   "original_emotion": emotion})
                all_ids.add(uid)
            if not utterances:
                raise ValueError(f"No utterances in {annotation}")
            utterances.sort(key=lambda u: (u["start"], u["end"], u["id"]))
            dialogues.append({"id": name, "session": session, "utterances": utterances,
                              "unsegmented_transcript_rows_ignored": extra_transcripts})
    audio_ids = {p.stem for p in root.glob("Session*/sentences/wav/*/*.wav")
                 if not p.name.startswith("._")}
    if audio_ids != all_ids:
        raise ValueError(f"Annotation/audio mismatch: unannotated={sorted(audio_ids-all_ids)[:10]}, "
                         f"missing_audio={sorted(all_ids-audio_ids)[:10]}")
    manifest = {"schema_version": 1, "labels": LABELS, "dataset_root": str(root),
                "extractors": {"audio": "microsoft/wavlm-base-plus final-layer mean+std",
                               "text": "FacebookAI/roberta-base token mean, excluding specials",
                               "visual": "torchvision ResNet18 IMAGENET1K_V1, whole-frame mean of 3",
                               "prosody": PROSODY_NAMES},
                "dialogues": dialogues, "sources": sources,
                "folds": {str(i): split_sessions(i) for i in range(1, 6)}}
    manifest["fingerprint"] = digest(manifest)
    target = cache / "manifest.json"
    if target.exists() and json.loads(target.read_text())["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Manifest changed; use a separate cache directory to avoid stale features")
    write_json(target, manifest)
    return manifest


def load_manifest(cache):
    manifest = json.loads((Path(cache) / "manifest.json").read_text())
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if digest(payload) != manifest["fingerprint"]:
        raise ValueError("Manifest fingerprint mismatch")
    return manifest


def load_audio(path):
    import librosa
    import numpy as np
    import soundfile as sf
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[0] == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Empty/nonfinite audio: {path}")
    waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    return waveform


def extract_prosody(waveform):
    """16 raw statistics, no corpus/speaker normalization or label information.

    Natural-log RMS has floor 1e-7. YIN uses 50..500Hz, 64ms frames/20ms hop.
    Voicing is an explicit energy/flatness proxy, not calibrated pitch confidence:
    RMS > max(-45dBFS, peak RMS-35dB) and spectral flatness < .25. Quiet is
    the complement of the energy gate. F0 deltas require adjacent voiced frames.
    Empty voiced sets map F0 statistics to zero; voiced_fraction exposes this.
    """
    import librosa
    import numpy as np
    y = np.pad(waveform, (0, max(0, 1024 - len(waveform))))
    spectrum = abs(librosa.stft(y, n_fft=1024, hop_length=320))
    rms = librosa.feature.rms(S=spectrum, frame_length=1024)[0]
    flatness = librosa.feature.spectral_flatness(S=spectrum)[0]
    energy_gate = rms > max(10 ** (-45 / 20), float(rms.max()) * 10 ** (-35 / 20))
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=16000, frame_length=1024, hop_length=320)
    voiced = energy_gate & (flatness < .25) & np.isfinite(f0)
    log_f0 = np.log(np.maximum(f0, 1))
    selected = log_f0[voiced]
    adjacent = voiced[:-1] & voiced[1:]
    f0_delta = np.abs(np.diff(log_f0))[adjacent]
    log_rms = np.log(np.maximum(rms, 1e-7))
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=320)[0]
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=16000)[0]
    rolloff = librosa.feature.spectral_rolloff(S=spectrum, sr=16000, roll_percent=.85)[0]
    values = [np.log1p(len(waveform) / 16000), log_rms.mean(), log_rms.std(),
              np.abs(np.diff(log_rms)).mean() if len(log_rms) > 1 else 0,
              selected.mean() if len(selected) else 0, selected.std() if len(selected) else 0,
              np.diff(np.percentile(selected, [25, 75]))[0] if len(selected) else 0,
              f0_delta.mean() if len(f0_delta) else 0, voiced.mean(), zcr.mean(), zcr.std(),
              centroid.mean(), centroid.std(), rolloff.mean(), flatness.mean(), (~energy_gate).mean()]
    return np.asarray(values, dtype=np.float32)


def visual_sampling(start, end, fps, frame_count):
    """Sample only the observed annotation/video intersection, without time shifts.

    Some source AVIs end during the final annotated utterance. Preserve this
    coverage loss explicitly; never replace missing frames with other times.
    """
    import numpy as np
    if not np.isfinite([start, end, fps]).all() or end <= start or fps <= 0 or frame_count <= 0:
        raise ValueError("Invalid video sampling interval or metadata")
    observed_start, observed_end = max(0.0, start), min(end, frame_count / fps)
    if observed_end <= observed_start:
        raise ValueError("Annotation has no observed video overlap")
    times = observed_start + (np.arange(3) + .5) * (observed_end - observed_start) / 3
    indices = np.floor(times * fps).astype(int)
    if np.any(indices < 0) or np.any(indices >= frame_count):
        raise ValueError("Sampling did not resolve to available video frames")
    return {"original_interval": [start, end],
            "observed_interval": [observed_start, observed_end],
            "visual_coverage_fraction": (observed_end - observed_start) / (end - start),
            "partial_clip": observed_start != start or observed_end != end,
            "sample_times": times.tolist(), "frame_indices": indices.tolist(),
            "frame_times": (indices / fps).tolist(),
            "fps": fps, "frame_count": frame_count, "speaker_crop": False}


def make_extractor(modality, device, chunk_seconds):
    """Return a local callable and the exact public-model/library provenance."""
    import numpy as np
    packages = ["numpy", "librosa", "soundfile"] if modality == "prosody" else ["numpy", "torch"]
    if modality in ("audio", "text"):
        packages += ["transformers", "huggingface-hub"]
    if modality == "visual":
        packages += ["torchvision", "opencv-python-headless"]
    info = {"modality": modality, "implementation": "iemocap-local-features-v1",
            "packages": {p: importlib.metadata.version(p) for p in packages}}
    if modality == "prosody":
        info.update({"names": PROSODY_NAMES, "sample_rate": 16000,
                     "frame_samples": 1024, "hop_samples": 320,
                     "voicing": "YIN 50..500Hz; energy>-45dBFS and peak-35dB; flatness<0.25"})
        def prosody(utterance):
            y = load_audio(utterance["audio_path"])
            return extract_prosody(y), {"duration_seconds": len(y) / 16000, "truncated": False}
        return prosody, info
    import torch
    if modality == "audio":
        from transformers import AutoFeatureExtractor, AutoModel
        name = "microsoft/wavlm-base-plus"
        processor = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval().to(device)
        info.update({"model": name, "revision": model.config._commit_hash,
                     "sample_rate": 16000, "chunk_seconds": chunk_seconds,
                     "pooling": "all final hidden frames mean+population std; no overlap",
                     "waveform_normalization": "whole utterance before chunking"})
        def audio(utterance):
            y = load_audio(utterance["audio_path"])
            inputs = processor(y, sampling_rate=16000, return_tensors="pt").input_values
            length = max(400, round(chunk_seconds * 16000))
            frames = []
            with torch.inference_mode():
                for start in range(0, inputs.shape[1], length):
                    piece = inputs[:, start:start + length]
                    if piece.shape[1] < 400:
                        piece = torch.nn.functional.pad(piece, (0, 400 - piece.shape[1]))
                    frames.append(model(piece.to(device)).last_hidden_state[0].float().cpu())
            states = torch.cat(frames)
            feature = torch.cat([states.mean(0), states.std(0, unbiased=False)]).numpy()
            return feature, {"duration_seconds": len(y) / 16000, "chunks": len(frames),
                             "hidden_frames": len(states), "truncated": False}
        return audio, info
    if modality == "text":
        from transformers import AutoModel, AutoTokenizer
        name = "FacebookAI/roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval().to(device)
        info.update({"model": name, "revision": model.config._commit_hash,
                     "pooling": "mean of nonspecial tokens across complete 510-token chunks"})
        def text(utterance):
            tokens = tokenizer.encode(utterance["text"], add_special_tokens=False)
            if not tokens:
                raise ValueError(f"No text tokens: {utterance['id']}")
            states = []
            with torch.inference_mode():
                for start in range(0, len(tokens), 510):
                    ids = tokenizer.build_inputs_with_special_tokens(tokens[start:start + 510])
                    hidden = model(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0]
                    states.append(hidden[1:-1].float().cpu())
            return torch.cat(states).mean(0).numpy(), {"tokens": len(tokens), "truncated": False}
        return text, info
    import cv2
    from torchvision.models import ResNet18_Weights, resnet18
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)
    transform = weights.transforms()
    info.update({"implementation": "iemocap-local-visual-features-v2",
                 "model": "torchvision/resnet18", "weights": "IMAGENET1K_V1",
                 "weights_url": weights.url, "whole_frame": True,
                 "sampling": "3 equal interval centers within annotation/video intersection",
                 "partial_clips": "record observed interval and coverage; no time shift or imputation"})
    capture, current_path = None, None
    def visual(utterance):
        nonlocal capture, current_path
        if current_path != utterance["visual_path"]:
            if capture is not None:
                capture.release()
            current_path = utterance["visual_path"]
            capture = cv2.VideoCapture(current_path)
            if not capture.isOpened():
                raise ValueError(f"Cannot decode AVI: {current_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise ValueError(f"Invalid AVI metadata: {current_path}")
        try:
            details = visual_sampling(utterance["start"], utterance["end"], fps, count)
        except ValueError as error:
            raise ValueError(f"{utterance['id']}: {error}") from error
        frames = []
        for index in details["frame_indices"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}: {utterance['id']}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(torch.from_numpy(rgb).permute(2, 0, 1)))
        with torch.inference_mode():
            feature = model(torch.stack(frames).to(device)).mean(0).float().cpu().numpy()
        return feature, details
    return visual, info


def feature_path(cache, uid, modality):
    return Path(cache) / "utterances" / modality / f"{uid}.npz"


def save_npz(path, **values):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **values)
    tmp.replace(path)


def read_feature(path, dimension, fingerprint):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        feature = data["feature"]
        if str(data["fingerprint"]) != fingerprint:
            raise ValueError(f"Stale feature: {path}")
        if feature.shape != (dimension,) or feature.dtype != np.float32 or not np.isfinite(feature).all():
            raise ValueError(f"Invalid feature: {path}")
        return feature


def extraction_fingerprint(manifest, provenance):
    return digest({"manifest": manifest["fingerprint"], "extractor": provenance})


def assemble(cache, manifest):
    import numpy as np
    provenance = {}
    for modality in DIMS:
        path = Path(cache) / f"extractor_{modality}.json"
        if not path.exists():
            return 0
        provenance[modality] = json.loads(path.read_text())
    fingerprints = {m: extraction_fingerprint(manifest, p) for m, p in provenance.items()}
    completed = 0
    for dialogue in manifest["dialogues"]:
        utterances = dialogue["utterances"]
        if not all(feature_path(cache, u["id"], m).exists() for u in utterances for m in DIMS):
            continue
        arrays = {m: np.stack([read_feature(feature_path(cache, u["id"], m), dim, fingerprints[m])
                               for u in utterances]) for m, dim in DIMS.items()}
        arrays.update({"speaker": np.array([u["speaker"] for u in utterances], dtype=np.int64),
                       "labels": np.array([u["label"] for u in utterances], dtype=np.int64),
                       "order": np.arange(len(utterances), dtype=np.int64),
                       "times": np.array([[u["start"], u["end"]] for u in utterances], dtype=np.float32),
                       "ids": np.array([u["id"] for u in utterances]),
                       "fingerprint": np.array(digest(fingerprints))})
        save_npz(Path(cache) / "dialogues" / f"{dialogue['id']}.npz", **arrays)
        completed += 1
    return completed


def extract(cache, modalities, device="cuda", limit=0, chunk_seconds=20, threads=4):
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    manifest = load_manifest(cache)
    utterances = [u for d in manifest["dialogues"] for u in d["utterances"]]
    if limit:
        utterances = utterances[:limit]
    for modality in modalities:
        print(f"Loading {modality} extractor", flush=True)
        extractor, provenance = make_extractor(modality, device, chunk_seconds)
        path = Path(cache) / f"extractor_{modality}.json"
        if path.exists() and json.loads(path.read_text()) != provenance:
            raise ValueError(f"Extractor changed: {path}; use a separate cache")
        write_json(path, provenance)
        fingerprint = extraction_fingerprint(manifest, provenance)
        start, done, reused = time.monotonic(), 0, 0
        for i, utterance in enumerate(utterances, 1):
            target = feature_path(cache, utterance["id"], modality)
            if target.exists():
                read_feature(target, DIMS[modality], fingerprint)
                reused += 1
            else:
                feature, details = extractor(utterance)
                feature = np.asarray(feature, dtype=np.float32)
                if feature.shape != (DIMS[modality],) or not np.isfinite(feature).all():
                    raise ValueError(f"Invalid extracted {modality}: {utterance['id']}")
                save_npz(target, feature=feature, fingerprint=np.array(fingerprint),
                         metadata=np.array(json.dumps(details, sort_keys=True)))
                done += 1
            if i % 25 == 0 or i == len(utterances):
                print(json.dumps({"stage": modality, "processed": i, "total": len(utterances),
                                  "new": done, "reused": reused,
                                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    completed = assemble(cache, manifest)
    print(json.dumps({"assembled_dialogues": completed, "total_dialogues": len(manifest["dialogues"])}))


def audit(cache, require_complete=True):
    import numpy as np
    manifest = load_manifest(cache)
    for path, expected in manifest["sources"].items():
        current = source_stat(path)
        if any(current[k] != expected[k] for k in current):
            raise ValueError(f"Source changed since manifest: {path}")
        if "sha256" in expected and hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Source checksum changed: {path}")
    counts, complete, expected_fingerprints = Counter(), 0, {}
    for modality in DIMS:
        path = Path(cache) / f"extractor_{modality}.json"
        if path.exists():
            expected_fingerprints[modality] = extraction_fingerprint(manifest, json.loads(path.read_text()))
    for dialogue in manifest["dialogues"]:
        utterances = dialogue["utterances"]
        counts.update(LABELS[u["label"]] if u["label"] >= 0 else "ignored" for u in utterances)
        path = Path(cache) / "dialogues" / f"{dialogue['id']}.npz"
        if not path.exists():
            continue
        if len(expected_fingerprints) != len(DIMS):
            raise ValueError("Dialogue cache exists without all extractor provenance")
        with np.load(path, allow_pickle=False) as arrays:
            for modality, dimension in DIMS.items():
                feature = arrays[modality]
                if feature.shape != (len(utterances), dimension) or feature.dtype != np.float32:
                    raise ValueError(f"Invalid shape/dtype: {path}:{modality}")
                if not np.isfinite(feature).all():
                    raise ValueError(f"Nonfinite feature: {path}:{modality}")
            for key, expected in (("ids", [u["id"] for u in utterances]),
                                  ("labels", [u["label"] for u in utterances]),
                                  ("speaker", [u["speaker"] for u in utterances]),
                                  ("order", list(range(len(utterances))))):
                if arrays[key].tolist() != expected:
                    raise ValueError(f"Manifest/cache mismatch: {path}:{key}")
            if str(arrays["fingerprint"]) != digest(expected_fingerprints):
                raise ValueError(f"Stale dialogue cache: {path}")
            expected_times = np.array([[u["start"], u["end"]] for u in utterances], dtype=np.float32)
            if not np.array_equal(arrays["times"], expected_times):
                raise ValueError(f"Manifest/cache mismatch: {path}:times")
        complete += 1
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_fingerprint": manifest["fingerprint"], "dialogues": len(manifest["dialogues"]),
              "utterances": sum(counts.values()), "label_counts": dict(counts),
              "complete_dialogue_caches": complete, "complete": complete == len(manifest["dialogues"]),
              "folds": manifest["folds"], "dimensions": DIMS}
    write_json(Path(cache) / "audit.json", report)
    if require_complete and not report["complete"]:
        raise ValueError(f"Incomplete extraction: {complete}/{report['dialogues']} dialogues; see audit.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest_parser = commands.add_parser("manifest")
    manifest_parser.add_argument("--dataset", type=Path, required=True)
    manifest_parser.add_argument("--cache", type=Path, required=True)
    extract_parser = commands.add_parser("extract")
    extract_parser.add_argument("--cache", type=Path, required=True)
    extract_parser.add_argument("--modalities", default="audio,prosody,text,visual")
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--limit-utterances", type=int, default=0)
    extract_parser.add_argument("--chunk-seconds", type=float, default=20)
    extract_parser.add_argument("--threads", type=int, default=4)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--cache", type=Path, required=True)
    audit_parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.command == "manifest":
        result = build_manifest(args.dataset, args.cache)
        print(json.dumps({"dialogues": len(result["dialogues"]),
                          "utterances": sum(len(d["utterances"]) for d in result["dialogues"]),
                          "fingerprint": result["fingerprint"]}))
    elif args.command == "extract":
        modalities = args.modalities.split(",")
        if not modalities or any(m not in DIMS for m in modalities) or args.chunk_seconds <= 0:
            parser.error("Expected modalities audio,prosody,text,visual and positive chunk seconds")
        if args.limit_utterances < 0 or args.threads < 1:
            parser.error("limit must be nonnegative and threads positive")
        extract(args.cache, modalities, args.device, args.limit_utterances, args.chunk_seconds, args.threads)
    else:
        print(json.dumps(audit(args.cache, not args.allow_incomplete), indent=2))


if __name__ == "__main__":
    main()
