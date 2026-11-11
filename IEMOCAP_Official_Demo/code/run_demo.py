#!/usr/bin/env python3
"""Offline, inference-only IEMOCAP evaluation with ten frozen Base checkpoints."""
# 이 파일은 학습이 끝난 분류 5개·회귀 5개 모델을 불러와 실제 추론하고 채점함.
# 실행 순서: 파일·정답·분리 검증 → 폴드별 추론 → 지표 집계 → 비교표·보고서 저장.
# 현장에서는 준비된 특징과 학습 완료 가중치로 추론·채점함.
# 별도 구현에 의한 저장 예측의 독립 재채점은 audit_demo.py에서 수행함.
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

# 패키지를 읽을 때 __pycache__를 생성하지 않도록 하여 전달 폴더의 불필요한 변경을 줄임.
sys.dont_write_bytecode = True
import numpy as np
import torch

from model import InductiveGraph

# code 상위 패키지를 기준으로 자료를 찾음. PyCharm 작업 폴더가 달라도 같은 자료를 사용함.
# 아래 특징 차원과 클래스 순서는 학습 완료 가중치·정답 정의와 동일하게 유지해야 함.
ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9), name='Asia/Seoul')
DIMS = dict(audio=1536, prosody=16, text=768, visual=512)
CLASSES = ['hap', 'sad', 'ang', 'fea', 'sur', 'dis', 'neu']
KOREAN = ['행복', '슬픔', '분노', '공포', '놀람', '혐오', '중립']
# classification은 7개 감정 분류, va는 정서가·각성도 회귀임.
# 분류와 회귀용 모델을 각각 학습했으며 두 과제의 지표를 별도로 산출함.
TASKS = ('classification', 'va')
KEYS = dict(classification=('accuracy', 'weighted_f1'), va=('ccc_valence', 'ccc_arousal', 'mean_ccc'))


# JSON 읽기: 파일 경로를 받아 사전 또는 목록 형태의 내용을 반환함.
# 원본 평가 설정과 결과 파일 모두 UTF-8로 읽음.
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# JSON 저장: 경로와 직렬화할 값을 받아 들여쓰기한 UTF-8 파일로 씀.
# 한글은 그대로 보존하며 NaN·무한대처럼 유효하지 않은 JSON 수치는 허용하지 않음.
def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


# 파일 무결성 검사에 사용할 SHA256 문자열을 반환함.
# 큰 가중치·특징 파일을 한꺼번에 메모리에 읽지 않고 4 MiB씩 나누어 계산함.
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


# 사전·목록 등 JSON으로 표현 가능한 값의 SHA256을 계산함.
# 키 순서를 고정하므로 같은 평가 설정은 사전 삽입 순서와 관계없이 같은 해시를 가짐.
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# 검증 조건이 거짓이면 원인을 담은 오류로 즉시 중단함.
# 데이터나 가중치가 기대한 조건과 다를 때 다음 추론 단계로 진행하지 않게 함.
def require(condition, message):
    if not condition:
        raise ValueError(message)


# 검사·실행 시작과 완료 시각을 한국 표준시가 포함된 문자열로 반환함.
def timestamp():
    return datetime.now(KST).isoformat()


# 지정한 과제와 폴드의 best.pt를 CPU 메모리로 읽어 가중치·정규화 통계·선택 기록을 반환함.
# 번호는 1~5이며 지정한 과제·폴드의 학습 완료 가중치를 사용함.
def load_checkpoint(task, fold):
    # 호출 전 verify에서 패키지 해시와 원래 모델 선택 확정 기록을 대조함.
    return torch.load(ROOT / 'weights' / task / f'fold{fold}' / 'best.pt', map_location='cpu', weights_only=False)


# 추론 전 사전 검사. 기본 입력은 추가 정규화 재검산 여부(False)임.
# 반환값은 검사 보고서, 과제별 정답 정의, 평가 설정, 원본 대화 정보의 네 묶음임.
# 가중치·정답·특징·학습 기록을 읽기 전용으로 검사함.
def verify(audit_normalizers=False):
    started = time.perf_counter()
    # 1단계: 전달 파일 목록의 존재 여부와 해시를 검사함.
    # 실행 코드·평가 자료는 고정하고, 안내 문서·설명 셀·실행 출력 편집은 허용함.
    manifest = read(ROOT / 'PACKAGE_MANIFEST.json')
    for name, item in manifest['files'].items():
        p = ROOT / name
        require(p.is_file(), f'Missing package file: {name}')
        if item.get('verification') == 'documentation':
            continue
        if 'notebook_code_sha256' in item:
            notebook = read(p)
            code = [''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
            require(digest(code) == item['notebook_code_sha256'], f'Notebook source changed: {name}')
        elif 'notebook_source_sha256' in item:
            # Jupyter가 자동 저장하는 출력·실행 번호·편집기 메타데이터는 비교에서 제외함.
            # 셀 종류와 코드·설명 원문의 내용 및 순서는 해시로 검사함.
            notebook = read(p)
            source = [(c['cell_type'], ''.join(c['source'])) for c in notebook['cells']]
            require(digest(source) == item['notebook_source_sha256'], f'Notebook source changed: {name}')
        else:
            require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], f'Package hash mismatch: {name}')
    # 2단계: 특징 목록과 IEMOCAP의 원본 사람 주석을 직접 읽음.
    # 발화 ID 중복, 주석 형식, 전체 10,039개 발화의 누락 여부를 확인함.
    feature_manifest = read(ROOT / 'data/features/manifest.json')
    original = {d['id']: d for d in feature_manifest['dialogues']}
    require(len(original) == len(feature_manifest['dialogues']) == 151, 'Invalid feature manifest')
    # 원본 주석에서 시작·종료 시각, 발화 ID, 감정, 정서가, 활성도 값을 읽음.
    # 연속 감정의 정답은 원본 정서가와 활성도(각성도) 주석임.
    pattern = re.compile(r'^\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+(\w+)\s+\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]')
    annotations = {}
    for p in sorted((ROOT / 'data/annotations').glob('Session*/*.txt')):
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            if not line.startswith('['):
                continue
            m = pattern.match(line)
            require(m is not None, f'Malformed annotation: {p.name}')
            require(m[3] not in annotations, f'Duplicate source ID: {m[3]}')
            annotations[m[3]] = dict(emotion=m[4], va=[float(m[5]), float(m[6])], times=[float(m[1]), float(m[2])])
    require(len(annotations) == 10039, 'Annotation coverage mismatch')
    # 3단계: 분류와 회귀 각각의 평가 설정, 정답 정의, 모델 선택 확정 기록을 검증함.
    # 과거 절대 경로는 출처 기록으로만 쓰고 실제 파일은 현재 패키지 안에서 찾음.
    protocols, datasets, splits, normalizer_checks = {}, {}, [], []
    for task in TASKS:
        base = ROOT / 'evidence' / task
        protocol, gate = read(base / 'protocol.json'), read(base / 'evaluation_gate.json')
        dataset = read(ROOT / 'data' / f'{task}_dataset.json')
        protocols[task], datasets[task] = protocol, dataset
        require(protocol['task'] == dataset['task'] == task, 'Task mismatch')
        require(protocol['protocol'] == 'certification-fivefold-v1', 'Unexpected model family')
        require(dataset['labels'] == (CLASSES if task == 'classification' else ['valence', 'arousal']), 'Label order mismatch')
        require(sha(ROOT / 'data' / f'{task}_dataset.json') == protocol['dataset_sha256'], 'Frozen descriptor mismatch')
        require(sha(ROOT / 'data/features/manifest.json') == dataset['original_manifest_sha256'], 'Original manifest mismatch')
        # 시험 평가에 앞서 모든 폴드의 모델 선택이 확정되었는지 기록을 대조함.
        # 현재 모델 코드와 보존한 학습 당시 코드가 원래 기록의 해시와 같은지도 확인함.
        require(gate['protocol_hash'] == digest(protocol) and gate['all_folds_locked'], 'Invalid test gate')
        require(sha(ROOT / 'code/model.py') == protocol['sources']['model.py'], 'Model code differs from training snapshot')
        for name, value in protocol['sources'].items():
            require(sha(base / 'source_snapshot' / name) == value, f'Source hash mismatch: {name}')
        for historic_path, expected in dataset['source_hashes'].items():
            old = Path(historic_path)
            local = ROOT / 'data/features/manifest.json' if old.name == 'manifest.json' else ROOT / 'data/annotations' / f'Session{int(old.name[3:5])}' / old.name
            require(sha(local) == expected, f'Annotation source changed: {old.name}')
        # 원본 주석, 특징 목록, 과제별 정답 정의의 대화·발화 ID 및 순서를 맞춤.
        # 분류는 원본 7개 감정만 정답 번호로 사용하고 나머지는 -100으로 채점에서 제외함.
        # 회귀 정답에는 원본 정서가·각성도 주석 수치를 그대로 사용함.
        indexed = {d['id']: d for d in dataset['dialogues']}
        require(len(indexed) == len(dataset['dialogues']) == 151 and set(indexed) == set(original), 'Descriptor coverage mismatch')
        for did, d in indexed.items():
            require(d['group'] == original[did]['session'] == int(did[3:5]), 'Session identity mismatch')
            us = original[did]['utterances']
            require([u['id'] for u in d['utterances']] == [u['id'] for u in us], 'Descriptor ID order mismatch')
            for u, source in zip(d['utterances'], us):
                a = annotations[u['id']]
                require(a['emotion'] == source['original_emotion'] and a['times'] == [source['start'], source['end']], 'Original source alignment mismatch')
                expected = (CLASSES.index(a['emotion']) if a['emotion'] in CLASSES else -100) if task == 'classification' else a['va']
                require(u['target'] == expected, f'Ground truth differs from source: {u["id"]}')
        # 4단계: 학습 당시 확정해 저장한 5겹 교차검증 구성을 검증함.
        # k번 세션은 k번 폴드의 시험 자료, 다음 순환 세션은 개발 자료, 나머지 3개는 학습 자료임.
        # 그 발화를 추론하는 모델의 학습·개발에 시험 대화나 화자가 들어 있으면 중단함.
        require(protocol['folds'] == dataset['folds'] and len(protocol['folds']) == 5, 'Fold declarations differ')
        coverage = []
        for number, fold in enumerate(protocol['folds'], 1):
            require(fold['fold'] == number, 'Invalid fold order')
            flat = sum([fold[role] for role in ('train', 'dev', 'test')], [])
            require(len(flat) == len(set(flat)) == 151 and set(flat) == set(indexed), 'Split overlap or missing dialogue')
            groups = {role: {indexed[d]['group'] for d in fold[role]} for role in ('train', 'dev', 'test')}
            require(groups['test'] == {number} and groups['dev'] == {number % 5 + 1} and groups['train'] == set(range(1, 6)) - groups['test'] - groups['dev'], 'Session leakage')
            # 세션과 화자 성별을 조합한 화자 ID로 학습·개발·시험 간 화자 중복도 확인함.
            speakers = {role: {u['id'][:5] + '_' + u['id'].rsplit('_', 1)[1][0] for d in fold[role] for u in indexed[d]['utterances']} for role in groups}
            require(all(not speakers[a] & speakers[b] for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]), 'Speaker leakage')
            coverage.extend(fold['test'])
            # 5단계: 해당 폴드의 선택 기록과 best.pt가 동일한 학습·개발·시험 구성에 속하는지 확인함.
            # 50에포크 기록에서 개발 점수가 처음 최댓값이 된 모델을 선택했는지 대조함.
            folder = base / f'fold{number}'
            lock = read(folder / 'selection.json')
            require(sha(folder / 'selection.json') == gate['selection_hashes'][str(number)], 'Selection lock changed')
            require(sha(ROOT / 'weights' / task / f'fold{number}/best.pt') == lock['checkpoint_sha256'], 'Checkpoint hash mismatch')
            require(lock['identity'] == digest(dict(protocol=protocol, fold=fold)) and lock['fold'] == number and not lock['test_loaded_during_training'], 'Invalid checkpoint identity')
            history = [json.loads(line) for line in (folder / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
            require([x['epoch'] for x in history] == list(range(1, 51)) and lock['epochs_completed'] == 50, 'Incomplete training evidence')
            # history가 에포크 오름차순이므로 동점이면 max가 먼저 등장한 에포크를 선택함.
            best = max(history, key=lambda x: x['dev_score'])
            require((best['epoch'], best['dev_score']) == (lock['best_epoch'], lock['dev_score']), 'Selection did not use earliest dev maximum')
            saved = load_checkpoint(task, number)
            require(saved['identity'] == lock['identity'] and saved['task'] == task and saved['protocol_hash'] == digest(protocol), 'Checkpoint belongs to another split')
            require((saved['epoch'], saved['dev_score']) == (lock['best_epoch'], lock['dev_score']), 'Checkpoint selection mismatch')
            # 6단계: 가중치와 함께 저장된 학습 자료의 정규화 평균·표준편차를 검사함.
            # 기본 검사는 저장 통계의 해시·차원·유한성·양의 표준편차를 확인함.
            stats = saved['normalizer']
            require(digest({k: {n: v.tolist() for n, v in fields.items()} for k, fields in stats.items()}) == lock['normalizer_hash'], 'Normalizer hash mismatch')
            if audit_normalizers:
                # 선택 검사: 해당 폴드의 학습 특징만으로 평균·표준편차를 다시 계산해 대조함.
                # 재검산한 값은 출처 확인에 사용하며 저장된 정규화 통계와 모델은 유지함.
                raw = {name: [] for name in DIMS}
                for did in fold['train']:
                    with np.load(ROOT / 'data/features/dialogues' / f'{did}.npz', allow_pickle=False) as z:
                        for name in DIMS:
                            raw[name].append(z[name])
                for name in DIMS:
                    # 학습 당시와 같은 정밀도·최소 표준편차 규칙으로 계산하여 정확히 비교함.
                    # 통계 계산에 사용되는 특징은 해당 폴드의 학습 자료로 한정함.
                    x = np.concatenate(raw[name]).astype(np.float64)
                    require(np.array_equal(x.mean(0).astype(np.float32), stats[name]['mean']), 'Train-only mean mismatch')
                    require(np.array_equal(np.maximum(x.std(0), 1e-5).astype(np.float32), stats[name]['std']), 'Train-only std mismatch')
                normalizer_checks.append(f'{task}/fold{number}')
            for name, dim in DIMS.items():
                require(stats[name]['mean'].shape == stats[name]['std'].shape == (dim,) and np.isfinite(stats[name]['mean']).all() and np.isfinite(stats[name]['std']).all() and (stats[name]['std'] > 0).all(), 'Invalid normalization statistics')
            splits.append(dict(task=task, fold=number, train_sessions=sorted(groups['train']), dev_session=next(iter(groups['dev'])), test_session=number, train_test_speaker_overlap=0, selected_epoch=lock['best_epoch'], checkpoint_sha256=lock['checkpoint_sha256']))
        # 각 대화가 과제별로 한 번만 시험 대상이 되었는지 확인함.
        # OOF는 각 표본을 그 표본의 학습·개발에 쓰지 않은 모델로 예측한 결과를 뜻함.
        require(len(coverage) == len(set(coverage)) == 151, 'Incomplete OOF coverage')
        # 7단계: 특징 캐시의 해시, 발화 순서, 배열 크기, 화자와 시각 정보를 확인함.
        # 캐시와 정답의 행 순서가 다르면 같은 발화끼리 비교할 수 없으므로 오류로 처리함.
        for did, expected_hash in protocol['cache_hashes'].items():
            p = ROOT / 'data/features/dialogues' / f'{did}.npz'
            require(sha(p) == expected_hash, f'Feature cache changed: {did}')
            with np.load(p, allow_pickle=False) as z:
                require(z['ids'].tolist() == [u['id'] for u in indexed[did]['utterances']], 'Feature/target order mismatch')
                for name, dim in DIMS.items():
                    require(z[name].shape == (len(z['ids']), dim) and np.isfinite(z[name]).all(), f'Invalid features: {name}')
                require(np.array_equal(z['speaker'], [u['speaker'] for u in original[did]['utterances']]), 'Speaker cache differs')
                require(np.array_equal(z['times'], np.asarray([[u['start'], u['end']] for u in original[did]['utterances']], dtype=np.float32)), 'Time cache differs')
    # 두 과제가 같은 세션 분리를 사용해야 발화별 분류·회귀 결과를 올바르게 연결할 수 있음.
    # 통과 보고서는 이후 preflight.json에 저장하여 현장 실행의 검사 근거로 보존함.
    require(protocols['classification']['folds'] == protocols['va']['folds'], 'Tasks use different folds')
    report = dict(passed=True, checked_at=timestamp(), package_manifest_sha256=sha(ROOT / 'PACKAGE_MANIFEST.json'), files_checked=len(manifest['files']), folds_checked=10, annotations_checked=len(annotations), normalizers_recomputed_from_train_only=normalizer_checks, splits=splits, seconds=time.perf_counter() - started)
    return report, datasets, protocols, original


# 과제, 정답 배열 y, 새 예측 배열 p를 받아 채점 수와 지표를 사전으로 반환함.
# 분류 p는 [발화 수, 7] 확률, 회귀 p는 [발화 수, 2] 정서가·각성도 예측임.
# 이 함수의 계산을 별도 구현으로 다시 확인하는 독립 검증은 audit_demo.py가 담당함.
def metrics(task, y, p):
    if task == 'classification':
        # 분류에서 -100은 원본 7개 클래스 밖의 감정임. 문맥에는 남기고 채점에서만 제외함.
        # 가장 높은 예측 확률의 클래스를 선택한 뒤 정답 행·예측 열의 혼동행렬을 만듦.
        valid = y != -100
        true, pred = y[valid], p[valid].argmax(1)
        cm = np.zeros((7, 7), dtype=np.int64)
        np.add.at(cm, (true, pred), 1)
        # 클래스별 F1의 분모는 정답 표본 수와 예측 표본 수의 합임.
        # 정답·예측이 모두 없는 클래스의 F1은 0으로 두고, 정답 표본 비율로 가중평균함.
        support, predicted = cm.sum(1), cm.sum(0)
        f1 = np.divide(2.0 * cm.diagonal(), support + predicted, out=np.zeros(7), where=support + predicted > 0)
        return dict(n=len(true), accuracy=float(np.trace(cm) / len(true)), weighted_f1=float(f1 @ support / len(true)), support=support.tolist(), per_class_f1=f1.tolist(), confusion=cm.tolist())
    # 회귀는 정서가와 각성도 정답이 모두 1~5인 발화만 두 차원에 공통으로 사용함.
    # 범위 밖 원본 값은 그대로 보존하고 채점 제외 여부를 기록함.
    keep = ((y >= 1) & (y <= 5)).all(1)
    a, b = y[keep].astype(np.float64), p[keep].astype(np.float64)
    # CCC는 상관뿐 아니라 예측·정답의 평균과 분산 차이도 반영함.
    # 64비트 부동소수점과 모집단 분산·공분산으로 두 차원을 각각 계산한 뒤 산술평균함.
    ccc = 2 * ((a - a.mean(0)) * (b - b.mean(0))).mean(0) / (a.var(0) + b.var(0) + (a.mean(0) - b.mean(0)) ** 2)
    return dict(n=len(a), ccc_valence=float(ccc[0]), ccc_arousal=float(ccc[1]), mean_ccc=float(ccc.mean()))


# 동일한 열을 가진 행 사전 목록을 CSV로 저장함. 첫 행의 키 순서를 열 순서로 씀.
# UTF-8 BOM을 넣어 한글 표를 일반적인 스프레드시트 프로그램에서도 읽기 쉽게 함.
def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 요약 사전에서 5개 대표 지표의 폴드 평균·표본표준편차를 짧은 문자열로 만듦.
# 터미널 출력과 summary.txt가 같은 값을 보여주도록 공통으로 사용함.
def compact(summary):
    lines = ['IEMOCAP | frozen Base | fresh inference | 5-fold mean +/- sample SD']
    for task in TASKS:
        for key in KEYS[task]:
            v = summary['tasks'][task]['fold_mean'][key]
            lines.append(f'{key:16s} {v["mean"]:.6f} +/- {v["sample_sd"]:.6f}')
    return '\n'.join(lines)


# 실행 요약과 발화별 비교 행을 받아 독립 실행 가능한 HTML 보고서를 저장함.
# 전체 폴드 점수와 원본 감정·V/A 및 예측을 표시하며 폴드·발화 ID·텍스트 검색을 제공함.
# 보고서에 필요한 데이터가 파일 안에 들어 있으므로 별도 서버나 인터넷 연결이 필요 없음.
def make_html(path, summary, rows):
    # JSON 안의 < 문자를 이스케이프하여 발화 텍스트가 HTML 태그로 해석되는 것을 방지함.
    payload = json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>IEMOCAP 시험 결과</title>
<style>body{font:15px system-ui,sans-serif;margin:28px;color:#172b4d;background:#f5f7fa}h1{font-size:25px}.panel{background:white;padding:18px;border-radius:10px;margin:16px 0}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dde3eb}th{position:sticky;top:0;background:#eaf0f8}input,select,button{font:inherit;padding:7px;margin:4px}td:last-child{min-width:260px}.muted{color:#526178}</style>
<h1>IEMOCAP 시험 결과</h1>
<div class="panel"><pre id="summary"></pre><table><thead><tr><th>Fold</th><th>분류 표본</th><th>Accuracy</th><th>Weighted F1</th><th>V/A 표본</th><th>CCC-V</th><th>CCC-A</th><th>평균 CCC</th></tr></thead><tbody id="folds"></tbody></table></div>
<div class="panel"><label>Fold <select id="fold"><option value="">전체</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label><input id="search" placeholder="발화 ID 또는 텍스트 검색" size="38"><button id="prev">이전</button><button id="next">다음</button><span id="count"></span><p class="muted">전체 발화를 ID 순서로 표시함. 채점에서 제외된 발화도 문맥과 원본 정답을 보존함. V/A 열은 정답 → 예측 순서임.</p></div>
<table><thead><tr><th>Fold</th><th>발화 ID</th><th>원본 감정</th><th>예측 감정</th><th>분류 채점</th><th>Valence</th><th>Arousal</th><th>V/A 채점</th><th>텍스트</th></tr></thead><tbody id="rows"></tbody></table>
<script id="payload" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);let page=0;const size=40;
const f=x=>Number(x).toFixed(4);const s=data.summary;document.getElementById('summary').textContent='실행: '+s.completed_at+' | 장치: '+s.device+' | 추론: '+s.inference_seconds.toFixed(2)+'초\\n'+Object.entries(s.tasks).flatMap(([task,v])=>Object.entries(v.fold_mean).map(([k,m])=>k+': '+f(m.mean)+' ± '+f(m.sample_sd))).join('\\n');
function row(parent, values){const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});parent.append(tr)}
s.tasks.classification.folds.forEach((c,i)=>{const v=s.tasks.va.folds[i];row(document.getElementById('folds'),[i+1,c.n,f(c.accuracy),f(c.weighted_f1),v.n,f(v.ccc_valence),f(v.ccc_arousal),f(v.mean_ccc)])});
function render(){const query=document.getElementById('search').value.toLowerCase(),fold=document.getElementById('fold').value;const filtered=data.rows.filter(r=>(!fold||r.fold==fold)&&(!query||(r.utterance_id+' '+r.text).toLowerCase().includes(query)));page=Math.max(0,Math.min(page,Math.ceil(filtered.length/size)-1));const el=document.getElementById('rows');el.replaceChildren();filtered.slice(page*size,(page+1)*size).forEach(r=>row(el,[r.fold,r.utterance_id,r.original_emotion,r.predicted_emotion_ko,r.classification_valid?'포함':'제외: '+r.classification_exclusion,f(r.gt_valence)+' → '+f(r.pred_valence),f(r.gt_arousal)+' → '+f(r.pred_arousal),r.va_valid?'포함':'제외: 범위 밖',r.text]));document.getElementById('count').textContent=filtered.length+'개 / '+(page+1)+' 페이지'}
document.getElementById('search').oninput=()=>{page=0;render()};document.getElementById('fold').onchange=()=>{page=0;render()};document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};render();</script></html>'''
    Path(path).write_text(html.replace('PAYLOAD', payload), encoding='utf-8')


# 현장 추론의 본체. 새 결과 폴더, 연산 장치(cpu 또는 cuda), CPU 스레드 수를 받음.
# 사전 검사 후 모델 10개를 순서대로 추론·채점하고 결과 폴더 경로를 반환함.
# 저장된 가중치를 불러온 모델의 순전파를 실제 실행하여 새 예측을 생성함.
def run(output, device, threads):
    # 먼저 실행 장치를 확인하고 전체 사전 검사를 통과한 경우에만 추론을 시작함.
    torch.set_num_threads(threads)
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable; use --device cpu')
    report, datasets, protocols, original = verify()
    print(f'PASS: {report["files_checked"]} files, 10 selected checkpoints, 5 disjoint folds, source ground truth', flush=True)
    # 결과 폴더가 이미 있으면 중단하여 이전 결과를 덮어쓰지 않음.
    # 진행 상태와 사전 검사 기록을 먼저 남기고, 모든 저장이 끝나야 complete로 갱신함.
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'preflight.json', report)
    write(output / 'status.json', dict(status='running', started_at=timestamp(), training=False))
    # 재현에 필요한 Python·PyTorch·NumPy 버전, 장치, 스레드 수, 패키지 해시를 결과에 함께 남김.
    all_predictions, summary = {}, dict(schema=1, model='Base InductiveGraph multi_graph observed', mode='fresh_checkpoint_inference', training=False, modalities=['audio', 'text', 'video'], device=device, threads=threads, environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, executable=sys.executable), package_manifest_sha256=report['package_manifest_sha256'], tasks={})
    inference_start = time.perf_counter()
    for task in TASKS:
        indexed = {d['id']: d for d in datasets[task]['dialogues']}
        fold_results, joined = [], []
        # 분류와 회귀는 별도 모델이며 각 과제의 5개 폴드를 순서대로 평가함.
        # 각 과제의 k번 폴드 모델로 해당 k번 시험 세션을 추론함.
        for fold in protocols[task]['folds']:
            number = fold['fold']
            saved = load_checkpoint(task, number)
            # 학습 때와 같은 구조를 만들고 저장된 전체 가중치를 엄격히 불러옴.
            # 평가 모드와 기울기 계산 해제로 학습 완료 가중치를 고정한 채 추론함.
            model = InductiveGraph(DIMS, 7 if task == 'classification' else 2, variant='multi_graph', **protocols[task]['graph']).to(device)
            model.load_state_dict(saved['model'], strict=True)
            model.eval().requires_grad_(False)
            ids, dialogue_ids, ys, ps = [], [], [], []
            # 추론 전용 모드에서 시험 대화를 하나씩 처리함.
            # 한 대화의 모든 발화를 문맥으로 함께 사용하며 대화별로 독립 처리함.
            with torch.inference_mode():
                for did in fold['test']:
                    with np.load(ROOT / 'data/features/dialogues' / f'{did}.npz', allow_pickle=False) as z:
                        # 가중치에 저장된 학습 자료의 평균·표준편차로만 특징을 정규화함.
                        # 시험 특징에도 해당 폴드의 저장된 학습 통계를 그대로 적용함.
                        # 입력은 음성·운율·텍스트·영상 특징과 화자·순서·시각 정보뿐임.
                        # 평가 정답은 추론을 끝낸 뒤 채점과 결과 비교에만 사용함.
                        batch = {name: torch.as_tensor((z[name] - saved['normalizer'][name]['mean']) / saved['normalizer'][name]['std'], dtype=torch.float32, device=device) for name in DIMS}
                        batch.update({name: torch.as_tensor(z[name], device=device) for name in ('speaker', 'order', 'times')})
                        raw = model(batch)
                        # 분류 출력은 소프트맥스로 7개 감정의 확률로 바꿈.
                        # 회귀 출력은 학습 때와 같은 변환으로 정서가·각성도 각각의 1~5 범위 값을 만듦.
                        pred = raw.softmax(-1) if task == 'classification' else 3 + 2 * torch.tanh(raw)
                        ps.append(pred.cpu().numpy())
                        ids.extend(z['ids'].tolist())
                        dialogue_ids.extend([did] * len(z['ids']))
                    # 정답은 해당 대화의 예측을 끝낸 다음 채점·비교표 작성용으로만 모음.
                    ys.extend(u['target'] for u in indexed[did]['utterances'])
            y = np.asarray(ys, dtype=np.int64 if task == 'classification' else np.float32)
            p = np.concatenate(ps)
            require(np.isfinite(p).all(), 'Nonfinite output')
            valid = y != -100 if task == 'classification' else ((y >= 1) & (y <= 5)).all(1)
            # 채점에서 제외되는 발화도 예측·원본 정답·ID와 함께 보존함.
            # 유효 표본 표시(target_valid)가 실제 지표 계산에 포함된 대상을 구분함.
            arrays = dict(ids=np.asarray(ids), dialogues=np.asarray(dialogue_ids), targets=y, predictions=p, target_valid=valid, folds=np.full(len(y), number))
            np.savez_compressed(output / f'{task}_fold{number}.npz', **arrays)
            # 먼저 각 폴드에서 개별 점수를 계산하여 대표 집계의 근거를 남김.
            score = dict(fold=number, selected_epoch=saved['epoch'], **metrics(task, y, p))
            fold_results.append(score)
            joined.append(arrays)
            print(f'{task:14s} fold {number}/5 | test Session {number} | n={score["n"]} | ' + ' | '.join(f'{k}={score[k]:.6f}' for k in KEYS[task]), flush=True)
            del model, saved
        # 5개 폴드의 시험 예측을 합쳐 과제별 OOF 파일을 만듦.
        # 전체 10,039개 발화가 중복·누락 없이 한 번씩 있고 채점 표본 수도 맞는지 검사함.
        combined = {k: np.concatenate([a[k] for a in joined]) for k in joined[0]}
        expected_ids = {u['id'] for d in datasets[task]['dialogues'] for u in d['utterances']}
        require(len(combined['ids']) == len(set(combined['ids'])) == 10039 and set(combined['ids']) == expected_ids, 'OOF coverage failure')
        require(int(combined['target_valid'].sum()) == (4639 if task == 'classification' else 10035), 'Scored population mismatch')
        np.savez_compressed(output / f'{task}_oof.npz', **combined)
        all_predictions[task] = combined
        # 대표 결과는 폴드별 점수 5개의 산술평균과 표본표준편차(ddof=1)임.
        # 전체 예측을 합쳐 한 번에 채점한 pooled OOF 점수는 별도의 보조 결과로 보존함.
        # 폴드 평균과 OOF 통합 점수는 표본 가중 및 계산 방식이 다르므로 구분해 기록함.
        summary['tasks'][task] = dict(folds=fold_results, fold_mean={key: dict(mean=float(np.mean([r[key] for r in fold_results])), sample_sd=float(np.std([r[key] for r in fold_results], ddof=1))) for key in KEYS[task]}, pooled_oof_secondary=metrics(task, combined['targets'], combined['predictions']), context_utterances=len(combined['ids']), scored_utterances=int(combined['target_valid'].sum()))
    summary['inference_seconds'] = time.perf_counter() - inference_start
    summary['completed_at'] = timestamp()
    # 발화 ID를 기준으로 분류·회귀 결과를 연결하여 같은 발화끼리 비교함.
    # 같은 발화의 분류·회귀 모델이 동일한 시험 폴드를 사용했는지도 확인함.
    # 대응한 발화의 분류·회귀 정답과 예측을 나란히 표시함.
    c, v = all_predictions['classification'], all_predictions['va']
    vi = {uid: i for i, uid in enumerate(v['ids'])}
    utterances = {u['id']: u for d in original.values() for u in d['utterances']}
    rows = []
    for i, uid in enumerate(c['ids']):
        j, source = vi[uid], utterances[uid]
        require(c['folds'][i] == v['folds'][j], 'Combined annotation uses wrong fold')
        pred = int(c['predictions'][i].argmax())
        rows.append(dict(fold=int(c['folds'][i]), utterance_id=str(uid), dialogue_id=str(c['dialogues'][i]), start=source['start'], end=source['end'], original_emotion=source['original_emotion'], predicted_emotion=CLASSES[pred], predicted_emotion_ko=KOREAN[pred], classification_valid=bool(c['target_valid'][i]), classification_correct=bool(c['targets'][i] == pred) if c['target_valid'][i] else '', classification_exclusion='' if c['target_valid'][i] else 'original label outside seven classes', gt_valence=float(v['targets'][j, 0]), pred_valence=float(v['predictions'][j, 0]), gt_arousal=float(v['targets'][j, 1]), pred_arousal=float(v['predictions'][j, 1]), va_valid=bool(v['target_valid'][j]), text=source['text']))
    # 전체 발화를 ID 순으로 정렬하여 정답 여부와 관계없이 모두 표시함.
    # 발화별 비교 CSV, 폴드별 지표 CSV, 구조화 요약 JSON, 짧은 요약 TXT, HTML 화면을 저장함.
    rows.sort(key=lambda r: r['utterance_id'])
    save_csv(output / 'utterance_comparison.csv', rows)
    fold_rows = [dict(fold=i + 1, classification_n=summary['tasks']['classification']['folds'][i]['n'], va_n=summary['tasks']['va']['folds'][i]['n'], **{k: summary['tasks'][t]['folds'][i][k] for t in TASKS for k in KEYS[t]}) for i in range(5)]
    save_csv(output / 'fold_metrics.csv', fold_rows)
    write(output / 'summary.json', summary)
    (output / 'summary.txt').write_text(compact(summary) + '\n', encoding='utf-8')
    make_html(output / 'report.html', summary, rows)
    # 모든 결과 파일을 저장한 뒤 완료 상태를 기록함. 이후 별도 독립 검증을 수행할 수 있음.
    write(output / 'status.json', dict(status='complete', completed_at=summary['completed_at'], training=False))
    print('\n' + compact(summary) + f'\nResults: {output}', flush=True)
    return output


# 명령줄 입력 처리: verify는 사전 검사만, run은 검사 후 실제 추론·채점을 수행함.
# --audit-normalizers는 verify에서만 사용하는 읽기 전용 추가 검사임.
# run의 기본값은 CPU 4스레드이며 결과 폴더를 지정하지 않으면 실행 시각으로 새 폴더를 만듦.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    v = sub.add_parser('verify', help='Check hashes, folds, checkpoints, original ground truth')
    v.add_argument('--audit-normalizers', action='store_true', help='Read-only check against original train data; does not change weights or statistics')
    v.add_argument('--output', type=Path, help='Save the provenance audit outside immutable package inputs')
    r = sub.add_parser('run', help='Fresh inference only; never reads reference predictions')
    r.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    r.add_argument('--threads', type=int, default=4)
    r.add_argument('--output', type=Path, default=ROOT / 'results' / datetime.now(KST).strftime('run_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    # 상대 출력 경로는 작업 폴더와 관계없이 패키지 루트를 기준으로 해석함.
    if args.output is not None and not args.output.is_absolute():
        args.output = ROOT / args.output
    # 검사 보고서는 화면에 출력하며 --output을 주면 새 JSON 파일에도 저장함.
    if args.command == 'verify':
        report, *_ = verify(args.audit_normalizers)
        if args.output:
            require(not args.output.exists(), 'Audit output already exists')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        # 실행 자원 범위와 새 출력 경로를 확인한 다음 실제 추론으로 들어감.
        require(1 <= args.threads <= 32, 'threads must be 1..32')
        require(not args.output.exists(), 'Output exists; select a new directory')
        run(args.output, args.device, args.threads)


# 명령줄에서 직접 실행할 때만 main을 호출함. 노트북에서는 함수를 불러와 사용함.
if __name__ == '__main__':
    main()
