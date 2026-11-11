#!/usr/bin/env python3
"""원본 발화에서 특징과 중간 계산값을 준비함. 학습 완료 추출기를 오프라인으로 사용함."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'evidence/classification/source_snapshot/data.py'
ATOL = 1e-4
RTOL = 1e-4
DIMS = {'audio': 1536, 'prosody': 16, 'text': 768, 'visual': 512}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def source_module():
    # 보존된 학습 당시 전처리 코드에서 원본 읽기·운율·프레임 선택 함수를 가져옴.
    spec = importlib.util.spec_from_file_location('frozen_iemocap_preprocessing', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_versions():
    names = ['numpy', 'torch', 'transformers', 'huggingface-hub', 'librosa',
             'soundfile', 'torchvision', 'opencv-python-headless', 'Pillow']
    return {name: importlib.metadata.version(name) for name in names}


def check_versions(provenance, actual):
    for modality, info in provenance.items():
        for name, version in info['packages'].items():
            if actual[name] != version:
                raise ValueError(f'{modality}: {name} version {actual[name]} != recorded {version}')


def compare_feature(actual, expected, modality):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise ValueError(f'Invalid {modality} feature: {actual.shape}, {actual.dtype}')
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(passed=bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
                shape=list(actual.shape), max_abs_difference=float(difference.max()),
                mean_abs_difference=float(difference.mean()), atol=ATOL, rtol=RTOL)


def prosody_intermediates(waveform, frozen):
    # 원래 16차원 운율 계산에 사용하는 프레임별 수치와 스펙트럼을 함께 보존함.
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
    f0_delta = np.abs(np.diff(log_f0))[voiced[:-1] & voiced[1:]]
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
    feature = np.asarray(values, dtype=np.float32)
    if not np.array_equal(feature, frozen.extract_prosody(waveform)):
        raise ValueError('Expanded prosody implementation differs from frozen source')
    return dict(prosody_feature=feature, prosody_stft_magnitude=spectrum,
                prosody_rms=rms, prosody_log_rms=log_rms, prosody_flatness=flatness,
                prosody_energy_gate=energy_gate, prosody_f0=f0, prosody_log_f0=log_f0,
                prosody_voiced=voiced, prosody_zcr=zcr, prosody_centroid_hz=centroid,
                prosody_rolloff_85_hz=rolloff)


def prepare(args):
    import cv2
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
    from torchvision.models import ResNet18_Weights, resnet18

    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    frozen = source_module()
    feature_manifest = read_json(ROOT / 'data/features/manifest.json')
    dialogue = next((d for d in feature_manifest['dialogues'] if d['id'] == args.dialogue), None)
    if dialogue is None:
        raise ValueError(f'Unknown dialogue: {args.dialogue}')
    utterances = dialogue['utterances'][:args.limit_utterances or None]
    provenance = {m: read_json(ROOT / f'data/features/extractor_{m}.json') for m in DIMS}
    versions = package_versions()
    check_versions(provenance, versions)
    snapshots = {m: args.hf_cache / ('models--' + provenance[m]['model'].replace('/', '--')) /
                 'snapshots' / provenance[m]['revision'] for m in ('audio', 'text')}
    weight_paths = {'audio': snapshots['audio'] / 'pytorch_model.bin',
                    'text': snapshots['text'] / 'model.safetensors',
                    'visual': args.resnet_weights}
    for path in [*weight_paths.values(), snapshots['audio'] / 'preprocessor_config.json',
                 snapshots['text'] / 'tokenizer.json']:
        if not path.is_file():
            raise FileNotFoundError(f'Offline extractor file missing: {path}')
    video_path = args.iemocap_root / f'Session{dialogue["session"]}/dialog/avi/DivX/{args.dialogue}.avi'
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output = args.output or ROOT / 'data/preprocessing' / args.dialogue
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    # 현장 노트북에서도 동일한 토큰화 규칙을 사용할 수 있도록 원본 설정을 복사함.
    shutil.copyfile(snapshots['text'] / 'tokenizer.json', output / 'tokenizer.json')
    start = time.perf_counter()
    record = dict(schema=1, status='running', dialogue=args.dialogue,
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  training=False, device=args.device, threads=args.threads,
                  backend=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                               cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                               cudnn_benchmark=torch.backends.cudnn.benchmark,
                               cudnn_deterministic=torch.backends.cudnn.deterministic,
                               cuda_version=torch.version.cuda,
                               device_name=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU'),
                  python=sys.version, executable=sys.executable, packages=versions,
                  original_feature_provenance=provenance,
                  source_code={'path': str(SOURCE.relative_to(ROOT)), 'sha256': sha256(SOURCE)},
                  preparation_code={'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
                  reference_features={'path': f'data/features/dialogues/{args.dialogue}.npz',
                                      'sha256': sha256(ROOT / f'data/features/dialogues/{args.dialogue}.npz')},
                  raw_video={'path': str(video_path), 'sha256': sha256(video_path), 'bytes': video_path.stat().st_size},
                  pretrained_weights={m: {'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size}
                                      for m, p in weight_paths.items()},
                  extractor_configs={m: {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size}
                                         for p in snapshots[m].iterdir() if p.is_file() and p.suffix in ('.json', '.txt')}
                                     for m in ('audio', 'text')},
                  portable_tokenizer=dict(path='tokenizer.json', sha256=sha256(output / 'tokenizer.json')),
                  comparison_tolerance={'atol': ATOL, 'rtol': RTOL},
                  requested_utterances=len(utterances), dialogue_utterances=len(dialogue['utterances']), utterances=[])
    write_json(output / 'manifest.json', record)
    print(f'Loading offline extractors for {len(utterances)} utterances', flush=True)
    # 기록된 revision의 로컬 파일만 사용하여 추출기를 불러옴.
    processor = AutoFeatureExtractor.from_pretrained(snapshots['audio'], local_files_only=True)
    audio_model = AutoModel.from_pretrained(snapshots['audio'], local_files_only=True).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(snapshots['text'], local_files_only=True)
    text_model = AutoModel.from_pretrained(snapshots['text'], local_files_only=True).eval().to(args.device)
    visual_model = resnet18(weights=None)
    visual_model.load_state_dict(torch.load(weight_paths['visual'], map_location='cpu', weights_only=True), strict=True)
    visual_model.fc = torch.nn.Identity()
    visual_model = visual_model.eval().to(args.device)
    for model in (audio_model, text_model, visual_model):
        model.requires_grad_(False)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    # 실제 저장 설정은 do_normalize=False임. 이 값을 그대로 적용하여 입력을 보존함.
    record['effective_audio_processor'] = processor.to_dict()
    record['effective_text_tokenizer'] = dict(type=type(tokenizer).__name__, model_max_length=tokenizer.model_max_length,
                                              special_tokens=tokenizer.special_tokens_map)
    record['effective_visual_transform'] = dict(resize_size=transform.resize_size, crop_size=transform.crop_size,
                                               mean=transform.mean, std=transform.std,
                                               interpolation=str(transform.interpolation), antialias=transform.antialias,
                                               source_color='BGR', model_color='RGB', tensor_range_before_normalize=[0, 1])
    record['effective_prosody'] = dict(sample_rate=16000, frame_samples=1024, hop_samples=320,
                                     stft_window='hann', center=True, stft_pad_mode='constant',
                                     yin_fmin=50, yin_fmax=500, yin_trough_threshold=0.1,
                                     energy_dbfs_floor=-45, energy_peak_offset_db=-35, flatness_max=0.25,
                                     log_rms_floor=1e-7, feature_names=frozen.PROSODY_NAMES)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f'Cannot open {video_path}')
    fps, frame_count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    with np.load(ROOT / f'data/features/dialogues/{args.dialogue}.npz', allow_pickle=False) as z:
        reference = {k: z[k].copy() for k in ('ids', *DIMS)}
    lookup = {str(uid): i for i, uid in enumerate(reference['ids'])}
    chunk_samples = max(400, round(float(provenance['audio']['chunk_seconds']) * 16000))
    try:
        for count, utterance in enumerate(utterances, 1):
            uid = utterance['id']
            audio_path = args.iemocap_root / f'Session{dialogue["session"]}/sentences/wav/{args.dialogue}/{uid}.wav'
            waveform = frozen.load_audio(audio_path)
            audio_info = sf.info(audio_path)
            arrays = dict(audio_waveform_16k=waveform)
            arrays.update(prosody_intermediates(waveform, frozen))
            # 모노 16 kHz 파형을 processor에 전달한 뒤 최대20초 단위로 순서대로 처리함.
            processor_input = processor(waveform, sampling_rate=16000, return_tensors='pt').input_values
            arrays['audio_processor_input'] = processor_input.numpy()
            hidden_chunks, spans, input_lengths, hidden_counts = [], [], [], []
            with torch.inference_mode():
                for offset in range(0, processor_input.shape[1], chunk_samples):
                    piece = processor_input[:, offset:offset + chunk_samples]
                    spans.append([offset, offset + piece.shape[1]])
                    if piece.shape[1] < 400:
                        piece = torch.nn.functional.pad(piece, (0, 400 - piece.shape[1]))
                    states = audio_model(piece.to(args.device)).last_hidden_state[0].float().cpu()
                    hidden_chunks.append(states)
                    input_lengths.append(piece.shape[1])
                    hidden_counts.append(states.shape[0])
            states = torch.cat(hidden_chunks)
            arrays.update(audio_hidden_states=states.numpy(), audio_chunk_spans=np.asarray(spans),
                          audio_chunk_input_lengths=np.asarray(input_lengths), audio_chunk_hidden_counts=np.asarray(hidden_counts),
                          audio_feature=torch.cat([states.mean(0), states.std(0, unbiased=False)]).numpy())
            # 발화 토큰을510개씩 처리하고 특수토큰 양끝을 제외한 마지막층 벡터를 모음.
            tokens = tokenizer.encode(utterance['text'], add_special_tokens=False)
            if not tokens:
                raise ValueError(f'Empty text tokens: {uid}')
            text_states, input_ids, special_masks, offsets = [], [], [], [0]
            with torch.inference_mode():
                for offset in range(0, len(tokens), 510):
                    piece = tokens[offset:offset + 510]
                    ids = tokenizer.build_inputs_with_special_tokens(piece)
                    hidden = text_model(input_ids=torch.tensor([ids], device=args.device)).last_hidden_state[0]
                    text_states.append(hidden[1:-1].float().cpu())
                    input_ids.extend(ids)
                    special_masks.extend([1] + [0] * len(piece) + [1])
                    offsets.append(len(input_ids))
            states = torch.cat(text_states)
            arrays.update(text_token_ids=np.asarray(tokens, dtype=np.int64), text_input_ids=np.asarray(input_ids, dtype=np.int64),
                          text_special_tokens_mask=np.asarray(special_masks, dtype=np.bool_),
                          text_chunk_offsets=np.asarray(offsets, dtype=np.int64), text_hidden_states=states.numpy(),
                          text_feature=states.mean(0).numpy())
            # 원본 주석 구간을3등분한 각 중앙시각의 RGB 프레임을 PNG로 저장함.
            visual_info = frozen.visual_sampling(utterance['start'], utterance['end'], fps, frame_count)
            visual_inputs, frame_records = [], []
            for frame_number, index in enumerate(visual_info['frame_indices']):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Cannot decode frame {index}: {uid}')
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                png = output / 'frames' / f'{uid}_{frame_number}.png'
                Image.fromarray(rgb).save(png, format='PNG')
                frame_records.append(dict(path=str(png.relative_to(output)), sha256=sha256(png),
                                          shape=list(rgb.shape), frame_index=int(index)))
                visual_inputs.append(transform(torch.from_numpy(rgb).permute(2, 0, 1)))
            inputs = torch.stack(visual_inputs)
            with torch.inference_mode():
                frame_features = visual_model(inputs.to(args.device))
                visual_feature = frame_features.mean(0).float().cpu().numpy()
            arrays.update(visual_inputs_224=inputs.numpy(), visual_frame_features=frame_features.float().cpu().numpy(),
                          visual_feature=visual_feature)
            comparisons = {m: compare_feature(arrays[f'{m}_feature'], reference[m][lookup[uid]], m) for m in DIMS}
            target = output / f'{uid}.npz'
            np.savez_compressed(target, **arrays)
            if target.stat().st_size >= 100 * 1024 * 1024:
                raise ValueError(f'Intermediate archive exceeds100MiB: {target}')
            detail = dict(id=uid, order=lookup[uid], speaker=utterance['speaker'], start=utterance['start'], end=utterance['end'],
                          text=utterance['text'], file=target.name, sha256=sha256(target), bytes=target.stat().st_size,
                          raw_audio=dict(path=str(audio_path), sha256=sha256(audio_path), bytes=audio_path.stat().st_size,
                                         sample_rate=audio_info.samplerate, channels=audio_info.channels, frames=audio_info.frames,
                                         subtype=audio_info.subtype),
                          audio=dict(sample_rate=16000, samples=len(waveform), chunks=len(hidden_chunks),
                                     hidden_frames=len(arrays['audio_hidden_states']), chunk_seconds=provenance['audio']['chunk_seconds']),
                          text_processing=dict(token_count=len(tokens), tokens=tokenizer.convert_ids_to_tokens(tokens),
                                               decoded_tokens=[tokenizer.decode([tid]) for tid in tokens], chunks=len(text_states)),
                          visual={**visual_info, 'frames': frame_records},
                          arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()}, comparisons=comparisons)
            record['utterances'].append(detail)
            write_json(output / 'manifest.json', record)
            print(json.dumps(dict(utterance=uid, completed=count, total=len(utterances),
                                  differences={m: v['max_abs_difference'] for m, v in comparisons.items()},
                                  passed=all(v['passed'] for v in comparisons.values())), ensure_ascii=False), flush=True)
            if not all(v['passed'] for v in comparisons.values()):
                raise ValueError(f'Frozen feature comparison failed: {uid}')
        record.update(status='complete', all_feature_comparisons_passed=True,
                      seconds=time.perf_counter() - start, completed_utterances=len(record['utterances']))
        write_json(output / 'manifest.json', record)
        print('Completed: ' + str(output), flush=True)
    except Exception as error:
        record.update(status='failed', error=str(error), seconds=time.perf_counter() - start)
        write_json(output / 'manifest.json', record)
        raise
    finally:
        capture.release()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iemocap-root', type=Path, required=True)
    parser.add_argument('--dialogue', default='Ses01F_impro01')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit-utterances', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--hf-cache', type=Path, default=Path.home() / '.cache/huggingface/hub')
    parser.add_argument('--resnet-weights', type=Path, default=Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or args.limit_utterances < 0:
        parser.error('threads must be1..8; limit-utterances must be nonnegative')
    prepare(args)


if __name__ == '__main__':
    main()
