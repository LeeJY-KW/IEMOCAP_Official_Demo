#!/usr/bin/env python3
"""Offline, inference-only IEMOCAP evaluation with ten frozen Base checkpoints."""
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

sys.dont_write_bytecode = True
import numpy as np
import torch

from model import InductiveGraph

ROOT = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9), name='Asia/Seoul')
DIMS = dict(audio=1536, prosody=16, text=768, visual=512)
CLASSES = ['hap', 'sad', 'ang', 'fea', 'sur', 'dis', 'neu']
KOREAN = ['행복', '슬픔', '분노', '공포', '놀람', '혐오', '중립']
TASKS = ('classification', 'va')
KEYS = dict(classification=('accuracy', 'weighted_f1'), va=('ccc_valence', 'ccc_arousal', 'mean_ccc'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp():
    return datetime.now(KST).isoformat()


def load_checkpoint(task, fold):
    # The package hashes and original selection locks are checked before this.
    return torch.load(ROOT / 'weights' / task / f'fold{fold}' / 'best.pt', map_location='cpu', weights_only=False)


def verify(audit_normalizers=False):
    started = time.perf_counter()
    manifest = read(ROOT / 'PACKAGE_MANIFEST.json')
    for name, item in manifest['files'].items():
        p = ROOT / name
        require(p.is_file(), f'Missing package file: {name}')
        if 'notebook_source_sha256' in item:
            # Jupyter saves outputs, execution counts and UI metadata automatically.
            # Keep all code/Markdown cells fixed while allowing those saved outputs.
            notebook = read(p)
            source = [(c['cell_type'], ''.join(c['source'])) for c in notebook['cells']]
            require(digest(source) == item['notebook_source_sha256'], f'Notebook source changed: {name}')
        else:
            require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], f'Package hash mismatch: {name}')
    feature_manifest = read(ROOT / 'data/features/manifest.json')
    original = {d['id']: d for d in feature_manifest['dialogues']}
    require(len(original) == len(feature_manifest['dialogues']) == 151, 'Invalid feature manifest')
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
        require(gate['protocol_hash'] == digest(protocol) and gate['all_folds_locked'], 'Invalid test gate')
        require(sha(ROOT / 'model.py') == protocol['sources']['model.py'], 'Model code differs from training snapshot')
        for name, value in protocol['sources'].items():
            require(sha(base / 'source_snapshot' / name) == value, f'Source hash mismatch: {name}')
        for historic_path, expected in dataset['source_hashes'].items():
            old = Path(historic_path)
            local = ROOT / 'data/features/manifest.json' if old.name == 'manifest.json' else ROOT / 'data/annotations' / f'Session{int(old.name[3:5])}' / old.name
            require(sha(local) == expected, f'Annotation source changed: {old.name}')
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
        require(protocol['folds'] == dataset['folds'] and len(protocol['folds']) == 5, 'Fold declarations differ')
        coverage = []
        for number, fold in enumerate(protocol['folds'], 1):
            require(fold['fold'] == number, 'Invalid fold order')
            flat = sum([fold[role] for role in ('train', 'dev', 'test')], [])
            require(len(flat) == len(set(flat)) == 151 and set(flat) == set(indexed), 'Split overlap or missing dialogue')
            groups = {role: {indexed[d]['group'] for d in fold[role]} for role in ('train', 'dev', 'test')}
            require(groups['test'] == {number} and groups['dev'] == {number % 5 + 1} and groups['train'] == set(range(1, 6)) - groups['test'] - groups['dev'], 'Session leakage')
            speakers = {role: {u['id'][:5] + '_' + u['id'].rsplit('_', 1)[1][0] for d in fold[role] for u in indexed[d]['utterances']} for role in groups}
            require(all(not speakers[a] & speakers[b] for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]), 'Speaker leakage')
            coverage.extend(fold['test'])
            folder = base / f'fold{number}'
            lock = read(folder / 'selection.json')
            require(sha(folder / 'selection.json') == gate['selection_hashes'][str(number)], 'Selection lock changed')
            require(sha(ROOT / 'weights' / task / f'fold{number}/best.pt') == lock['checkpoint_sha256'], 'Checkpoint hash mismatch')
            require(lock['identity'] == digest(dict(protocol=protocol, fold=fold)) and lock['fold'] == number and not lock['test_loaded_during_training'], 'Invalid checkpoint identity')
            history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
            require([x['epoch'] for x in history] == list(range(1, 51)) and lock['epochs_completed'] == 50, 'Incomplete training evidence')
            best = max(history, key=lambda x: x['dev_score'])
            require((best['epoch'], best['dev_score']) == (lock['best_epoch'], lock['dev_score']), 'Selection did not use earliest dev maximum')
            saved = load_checkpoint(task, number)
            require(saved['identity'] == lock['identity'] and saved['task'] == task and saved['protocol_hash'] == digest(protocol), 'Checkpoint belongs to another split')
            require((saved['epoch'], saved['dev_score']) == (lock['best_epoch'], lock['dev_score']), 'Checkpoint selection mismatch')
            stats = saved['normalizer']
            require(digest({k: {n: v.tolist() for n, v in fields.items()} for k, fields in stats.items()}) == lock['normalizer_hash'], 'Normalizer hash mismatch')
            if audit_normalizers:
                # Read-only provenance audit; these values never replace saved statistics.
                raw = {name: [] for name in DIMS}
                for did in fold['train']:
                    with np.load(ROOT / 'data/features/dialogues' / f'{did}.npz', allow_pickle=False) as z:
                        for name in DIMS:
                            raw[name].append(z[name])
                for name in DIMS:
                    x = np.concatenate(raw[name]).astype(np.float64)
                    require(np.array_equal(x.mean(0).astype(np.float32), stats[name]['mean']), 'Train-only mean mismatch')
                    require(np.array_equal(np.maximum(x.std(0), 1e-5).astype(np.float32), stats[name]['std']), 'Train-only std mismatch')
                normalizer_checks.append(f'{task}/fold{number}')
            for name, dim in DIMS.items():
                require(stats[name]['mean'].shape == stats[name]['std'].shape == (dim,) and np.isfinite(stats[name]['mean']).all() and np.isfinite(stats[name]['std']).all() and (stats[name]['std'] > 0).all(), 'Invalid normalization statistics')
            splits.append(dict(task=task, fold=number, train_sessions=sorted(groups['train']), dev_session=next(iter(groups['dev'])), test_session=number, train_test_speaker_overlap=0, selected_epoch=lock['best_epoch'], checkpoint_sha256=lock['checkpoint_sha256']))
        require(len(coverage) == len(set(coverage)) == 151, 'Incomplete OOF coverage')
        for did, expected_hash in protocol['cache_hashes'].items():
            p = ROOT / 'data/features/dialogues' / f'{did}.npz'
            require(sha(p) == expected_hash, f'Feature cache changed: {did}')
            with np.load(p, allow_pickle=False) as z:
                require(z['ids'].tolist() == [u['id'] for u in indexed[did]['utterances']], 'Feature/target order mismatch')
                for name, dim in DIMS.items():
                    require(z[name].shape == (len(z['ids']), dim) and np.isfinite(z[name]).all(), f'Invalid features: {name}')
                require(np.array_equal(z['speaker'], [u['speaker'] for u in original[did]['utterances']]), 'Speaker cache differs')
                require(np.array_equal(z['times'], np.asarray([[u['start'], u['end']] for u in original[did]['utterances']], dtype=np.float32)), 'Time cache differs')
    require(protocols['classification']['folds'] == protocols['va']['folds'], 'Tasks use different folds')
    report = dict(passed=True, checked_at=timestamp(), package_manifest_sha256=sha(ROOT / 'PACKAGE_MANIFEST.json'), files_checked=len(manifest['files']), folds_checked=10, annotations_checked=len(annotations), normalizers_recomputed_from_train_only=normalizer_checks, splits=splits, seconds=time.perf_counter() - started)
    return report, datasets, protocols, original


def metrics(task, y, p):
    if task == 'classification':
        valid = y != -100
        true, pred = y[valid], p[valid].argmax(1)
        cm = np.zeros((7, 7), dtype=np.int64)
        np.add.at(cm, (true, pred), 1)
        support, predicted = cm.sum(1), cm.sum(0)
        f1 = np.divide(2.0 * cm.diagonal(), support + predicted, out=np.zeros(7), where=support + predicted > 0)
        return dict(n=len(true), accuracy=float(np.trace(cm) / len(true)), weighted_f1=float(f1 @ support / len(true)), support=support.tolist(), per_class_f1=f1.tolist(), confusion=cm.tolist())
    keep = ((y >= 1) & (y <= 5)).all(1)
    a, b = y[keep].astype(np.float64), p[keep].astype(np.float64)
    ccc = 2 * ((a - a.mean(0)) * (b - b.mean(0))).mean(0) / (a.var(0) + b.var(0) + (a.mean(0) - b.mean(0)) ** 2)
    return dict(n=len(a), ccc_valence=float(ccc[0]), ccc_arousal=float(ccc[1]), mean_ccc=float(ccc.mean()))


def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact(summary):
    lines = ['IEMOCAP | frozen Base | fresh inference | 5-fold mean +/- sample SD']
    for task in TASKS:
        for key in KEYS[task]:
            v = summary['tasks'][task]['fold_mean'][key]
            lines.append(f'{key:16s} {v["mean"]:.6f} +/- {v["sample_sd"]:.6f}')
    return '\n'.join(lines)


def make_html(path, summary, rows):
    payload = json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>IEMOCAP 평가 시연</title>
<style>body{font:15px system-ui,sans-serif;margin:28px;color:#172b4d;background:#f5f7fa}h1{font-size:25px}.panel{background:white;padding:18px;border-radius:10px;margin:16px 0}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dde3eb}th{position:sticky;top:0;background:#eaf0f8}input,select,button{font:inherit;padding:7px;margin:4px}td:last-child{min-width:260px}.muted{color:#526178}</style>
<h1>IEMOCAP 이산·연속 감정 평가 시연</h1><div class="panel">학습 완료 Base의 분류 5개·V/A 회귀 5개 가중치를 사용한 새 추론 결과입니다. 각 Session k는 Fold k 모델로만 평가했습니다. 정답은 IEMOCAP 원본 주석이며 모델 입력에는 제공하지 않았습니다.<br><span class="muted">고정 음성·텍스트·영상 특징 입력 · 발화 단위 · 두 과제 별도 학습 · 통제 환경 예비 검증 · 혐오 정답 2개 · 기관 인증 완료 아님</span></div>
<div class="panel"><pre id="summary"></pre><table><thead><tr><th>Fold</th><th>분류 표본</th><th>Accuracy</th><th>Weighted F1</th><th>V/A 표본</th><th>CCC-V</th><th>CCC-A</th><th>평균 CCC</th></tr></thead><tbody id="folds"></tbody></table></div>
<div class="panel"><label>Fold <select id="fold"><option value="">전체</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label><input id="search" placeholder="발화 ID 또는 텍스트 검색" size="38"><button id="prev">이전</button><button id="next">다음</button><span id="count"></span><p class="muted">ID 순서로 표시합니다. 성공 사례를 선별하지 않았습니다. 점수 제외 발화도 문맥과 원래 정답을 보존합니다. V/A 열은 정답 → 예측입니다.</p></div>
<table><thead><tr><th>Fold</th><th>발화 ID</th><th>원본 감정</th><th>예측 감정</th><th>분류 채점</th><th>Valence</th><th>Arousal</th><th>V/A 채점</th><th>텍스트</th></tr></thead><tbody id="rows"></tbody></table>
<script id="payload" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);let page=0;const size=40;
const f=x=>Number(x).toFixed(4);const s=data.summary;document.getElementById('summary').textContent='실행: '+s.completed_at+' | 장치: '+s.device+' | 추론: '+s.inference_seconds.toFixed(2)+'초\\n'+Object.entries(s.tasks).flatMap(([task,v])=>Object.entries(v.fold_mean).map(([k,m])=>k+': '+f(m.mean)+' ± '+f(m.sample_sd))).join('\\n');
function row(parent, values){const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});parent.append(tr)}
s.tasks.classification.folds.forEach((c,i)=>{const v=s.tasks.va.folds[i];row(document.getElementById('folds'),[i+1,c.n,f(c.accuracy),f(c.weighted_f1),v.n,f(v.ccc_valence),f(v.ccc_arousal),f(v.mean_ccc)])});
function render(){const query=document.getElementById('search').value.toLowerCase(),fold=document.getElementById('fold').value;const filtered=data.rows.filter(r=>(!fold||r.fold==fold)&&(!query||(r.utterance_id+' '+r.text).toLowerCase().includes(query)));page=Math.max(0,Math.min(page,Math.ceil(filtered.length/size)-1));const el=document.getElementById('rows');el.replaceChildren();filtered.slice(page*size,(page+1)*size).forEach(r=>row(el,[r.fold,r.utterance_id,r.original_emotion,r.predicted_emotion_ko,r.classification_valid?'포함':'제외: '+r.classification_exclusion,f(r.gt_valence)+' → '+f(r.pred_valence),f(r.gt_arousal)+' → '+f(r.pred_arousal),r.va_valid?'포함':'제외: 범위 밖',r.text]));document.getElementById('count').textContent=filtered.length+'개 / '+(page+1)+' 페이지'}
document.getElementById('search').oninput=()=>{page=0;render()};document.getElementById('fold').onchange=()=>{page=0;render()};document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};render();</script></html>'''
    Path(path).write_text(html.replace('PAYLOAD', payload), encoding='utf-8')


def run(output, device, threads):
    torch.set_num_threads(threads)
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable; use --device cpu')
    report, datasets, protocols, original = verify()
    print(f'PASS: {report["files_checked"]} files, 10 selected checkpoints, 5 disjoint folds, source ground truth', flush=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'preflight.json', report)
    write(output / 'status.json', dict(status='running', started_at=timestamp(), training=False))
    all_predictions, summary = {}, dict(schema=1, model='Base InductiveGraph multi_graph observed', mode='fresh_checkpoint_inference', training=False, modalities=['audio', 'text', 'video'], device=device, threads=threads, environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, executable=sys.executable), package_manifest_sha256=report['package_manifest_sha256'], tasks={})
    inference_start = time.perf_counter()
    for task in TASKS:
        indexed = {d['id']: d for d in datasets[task]['dialogues']}
        fold_results, joined = [], []
        for fold in protocols[task]['folds']:
            number = fold['fold']
            saved = load_checkpoint(task, number)
            model = InductiveGraph(DIMS, 7 if task == 'classification' else 2, variant='multi_graph', **protocols[task]['graph']).to(device)
            model.load_state_dict(saved['model'], strict=True)
            model.eval().requires_grad_(False)
            ids, dialogue_ids, ys, ps = [], [], [], []
            with torch.inference_mode():
                for did in fold['test']:
                    with np.load(ROOT / 'data/features/dialogues' / f'{did}.npz', allow_pickle=False) as z:
                        # Only features and structural metadata go into forward().
                        # Cached labels and descriptor targets are never model inputs.
                        batch = {name: torch.as_tensor((z[name] - saved['normalizer'][name]['mean']) / saved['normalizer'][name]['std'], dtype=torch.float32, device=device) for name in DIMS}
                        batch.update({name: torch.as_tensor(z[name], device=device) for name in ('speaker', 'order', 'times')})
                        raw = model(batch)
                        pred = raw.softmax(-1) if task == 'classification' else 3 + 2 * torch.tanh(raw)
                        ps.append(pred.cpu().numpy())
                        ids.extend(z['ids'].tolist())
                        dialogue_ids.extend([did] * len(z['ids']))
                    ys.extend(u['target'] for u in indexed[did]['utterances'])
            y = np.asarray(ys, dtype=np.int64 if task == 'classification' else np.float32)
            p = np.concatenate(ps)
            require(np.isfinite(p).all(), 'Nonfinite output')
            valid = y != -100 if task == 'classification' else ((y >= 1) & (y <= 5)).all(1)
            arrays = dict(ids=np.asarray(ids), dialogues=np.asarray(dialogue_ids), targets=y, predictions=p, target_valid=valid, folds=np.full(len(y), number))
            np.savez_compressed(output / f'{task}_fold{number}.npz', **arrays)
            score = dict(fold=number, selected_epoch=saved['epoch'], **metrics(task, y, p))
            fold_results.append(score)
            joined.append(arrays)
            print(f'{task:14s} fold {number}/5 | test Session {number} | n={score["n"]} | ' + ' | '.join(f'{k}={score[k]:.6f}' for k in KEYS[task]), flush=True)
            del model, saved
        combined = {k: np.concatenate([a[k] for a in joined]) for k in joined[0]}
        expected_ids = {u['id'] for d in datasets[task]['dialogues'] for u in d['utterances']}
        require(len(combined['ids']) == len(set(combined['ids'])) == 10039 and set(combined['ids']) == expected_ids, 'OOF coverage failure')
        require(int(combined['target_valid'].sum()) == (4639 if task == 'classification' else 10035), 'Scored population mismatch')
        np.savez_compressed(output / f'{task}_oof.npz', **combined)
        all_predictions[task] = combined
        summary['tasks'][task] = dict(folds=fold_results, fold_mean={key: dict(mean=float(np.mean([r[key] for r in fold_results])), sample_sd=float(np.std([r[key] for r in fold_results], ddof=1))) for key in KEYS[task]}, pooled_oof_secondary=metrics(task, combined['targets'], combined['predictions']), context_utterances=len(combined['ids']), scored_utterances=int(combined['target_valid'].sum()))
    summary['inference_seconds'] = time.perf_counter() - inference_start
    summary['completed_at'] = timestamp()
    c, v = all_predictions['classification'], all_predictions['va']
    vi = {uid: i for i, uid in enumerate(v['ids'])}
    utterances = {u['id']: u for d in original.values() for u in d['utterances']}
    rows = []
    for i, uid in enumerate(c['ids']):
        j, source = vi[uid], utterances[uid]
        require(c['folds'][i] == v['folds'][j], 'Combined annotation uses wrong fold')
        pred = int(c['predictions'][i].argmax())
        rows.append(dict(fold=int(c['folds'][i]), utterance_id=str(uid), dialogue_id=str(c['dialogues'][i]), start=source['start'], end=source['end'], original_emotion=source['original_emotion'], predicted_emotion=CLASSES[pred], predicted_emotion_ko=KOREAN[pred], classification_valid=bool(c['target_valid'][i]), classification_correct=bool(c['targets'][i] == pred) if c['target_valid'][i] else '', classification_exclusion='' if c['target_valid'][i] else 'original label outside seven classes', gt_valence=float(v['targets'][j, 0]), pred_valence=float(v['predictions'][j, 0]), gt_arousal=float(v['targets'][j, 1]), pred_arousal=float(v['predictions'][j, 1]), va_valid=bool(v['target_valid'][j]), text=source['text']))
    rows.sort(key=lambda r: r['utterance_id'])
    save_csv(output / 'utterance_comparison.csv', rows)
    fold_rows = [dict(fold=i + 1, classification_n=summary['tasks']['classification']['folds'][i]['n'], va_n=summary['tasks']['va']['folds'][i]['n'], **{k: summary['tasks'][t]['folds'][i][k] for t in TASKS for k in KEYS[t]}) for i in range(5)]
    save_csv(output / 'fold_metrics.csv', fold_rows)
    write(output / 'summary.json', summary)
    (output / 'summary.txt').write_text(compact(summary) + '\n', encoding='utf-8')
    make_html(output / 'report.html', summary, rows)
    write(output / 'status.json', dict(status='complete', completed_at=summary['completed_at'], training=False))
    print('\n' + compact(summary) + f'\nResults: {output}', flush=True)
    return output


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
    if args.command == 'verify':
        report, *_ = verify(args.audit_normalizers)
        if args.output:
            require(not args.output.exists(), 'Audit output already exists')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        require(1 <= args.threads <= 32, 'threads must be 1..32')
        require(not args.output.exists(), 'Output exists; select a new directory')
        run(args.output, args.device, args.threads)


if __name__ == '__main__':
    main()
