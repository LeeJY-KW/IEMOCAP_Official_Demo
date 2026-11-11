#!/usr/bin/env python3
"""Independently score fresh predictions and compare the frozen reference run."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def independent_score(task, y, p):
    if task == 'classification':
        mask = y != -100
        truth, prediction = y[mask], p[mask].argmax(1)
        return dict(accuracy=float(accuracy_score(truth, prediction)), weighted_f1=float(f1_score(truth, prediction, average='weighted', labels=list(range(7)), zero_division=0)), confusion=confusion_matrix(truth, prediction, labels=list(range(7))).tolist())
    mask = np.logical_and(y >= 1, y <= 5).all(axis=1)
    scores = []
    for dimension in range(2):
        truth, prediction = y[mask, dimension].astype('float64'), p[mask, dimension].astype('float64')
        covariance = np.cov(truth, prediction, ddof=0)
        scores.append(float(2 * covariance[0, 1] / (covariance[0, 0] + covariance[1, 1] + (np.mean(truth) - np.mean(prediction)) ** 2)))
    return dict(ccc_valence=scores[0], ccc_arousal=scores[1], mean_ccc=sum(scores) / 2)


def audit(output):
    output = Path(output).resolve()
    check(read(output / 'status.json')['status'] == 'complete', 'Inference did not complete')
    summary = read(output / 'summary.json')
    check(summary['mode'] == 'fresh_checkpoint_inference' and summary['training'] is False, 'Unexpected run mode')
    records, max_difference, merged = [], 0.0, {}
    for task in ('classification', 'va'):
        descriptor = read(ROOT / 'data' / f'{task}_dataset.json')
        index = {d['id']: d for d in descriptor['dialogues']}
        old_summary = read(ROOT / 'evidence' / task / 'test_summary.json')
        parts, scores = [], []
        for fold in descriptor['folds']:
            number = fold['fold']
            with np.load(output / f'{task}_fold{number}.npz', allow_pickle=False) as z:
                fresh = {k: z[k].copy() for k in z.files}
            with np.load(ROOT / 'evidence' / task / f'fold{number}/test_predictions.npz', allow_pickle=False) as z:
                old = {k: z[k].copy() for k in z.files}
            expected = [u for did in fold['test'] for u in index[did]['utterances']]
            check(fresh['ids'].tolist() == [u['id'] for u in expected], 'Fresh prediction IDs are not the frozen test set')
            check(fresh['dialogues'].tolist() == [did for did in fold['test'] for u in index[did]['utterances']], 'Dialogue assignment mismatch')
            check(np.array_equal(fresh['targets'], np.asarray([u['target'] for u in expected], dtype=fresh['targets'].dtype)), 'Ground truth mismatch')
            check(np.all(fresh['folds'] == number), 'Wrong checkpoint fold assignment')
            for key in ('ids', 'dialogues', 'targets', 'target_valid'):
                check(np.array_equal(fresh[key], old[key]), f'Reference alignment mismatch: {task}/{number}/{key}')
            valid = fresh['targets'] != -100 if task == 'classification' else ((fresh['targets'] >= 1) & (fresh['targets'] <= 5)).all(1)
            check(np.array_equal(valid, fresh['target_valid']), 'Invalid score mask')
            check(np.isfinite(fresh['predictions']).all(), 'Nonfinite predictions')
            if task == 'classification':
                check(np.allclose(fresh['predictions'].sum(1), 1, atol=1e-6, rtol=0), 'Invalid probability sum')
                check(np.array_equal(fresh['predictions'].argmax(1), old['predictions'].argmax(1)), 'Class decisions differ from archived inference')
            else:
                check(((fresh['predictions'] >= 1) & (fresh['predictions'] <= 5)).all(), 'V/A output outside 1..5')
            difference = float(np.abs(fresh['predictions'].astype('float64') - old['predictions']).max())
            max_difference = max(max_difference, difference)
            check(difference <= 1e-4, 'Fresh inference differs from reference beyond float32 CPU/GPU tolerance')
            score = independent_score(task, fresh['targets'], fresh['predictions'])
            reported = summary['tasks'][task]['folds'][number - 1]
            for key in summary['tasks'][task]['fold_mean']:
                check(abs(score[key] - reported[key]) < 1e-10, f'Independent metric mismatch: {key}')
            if task == 'classification':
                check(score['confusion'] == reported['confusion'], 'Confusion matrix mismatch')
            check(reported['n'] == int(valid.sum()), 'Reported denominator mismatch')
            records.append(dict(task=task, fold=number, context=len(valid), scored=int(valid.sum()), max_abs_prediction_difference=difference))
            parts.append(fresh)
            scores.append(score)
        combined = {key: np.concatenate([x[key] for x in parts]) for key in parts[0]}
        with np.load(output / f'{task}_oof.npz', allow_pickle=False) as z:
            for key in combined:
                check(np.array_equal(z[key], combined[key]), 'Saved OOF differs from per-fold predictions')
        check(len(combined['ids']) == len(set(combined['ids'])) == 10039, 'OOF duplicate or missing ID')
        check(int(combined['target_valid'].sum()) == (4639 if task == 'classification' else 10035), 'OOF denominator mismatch')
        pooled = independent_score(task, combined['targets'], combined['predictions'])
        for key, value in summary['tasks'][task]['fold_mean'].items():
            check(abs(np.mean([x[key] for x in scores]) - value['mean']) < 1e-10, 'Fold mean differs')
            check(abs(np.std([x[key] for x in scores], ddof=1) - value['sample_sd']) < 1e-10, 'Sample SD differs')
            check(abs(value['mean'] - old_summary[key]['mean']) < 1e-5, 'Mean differs from archived evaluation')
            check(abs(pooled[key] - summary['tasks'][task]['pooled_oof_secondary'][key]) < 1e-10, 'Pooled metric differs')
        merged[task] = {str(uid): {key: combined[key][i] for key in ('targets', 'predictions', 'target_valid', 'folds')} for i, uid in enumerate(combined['ids'])}
    with (output / 'utterance_comparison.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == len({r['utterance_id'] for r in rows}) == 10039, 'Comparison CSV coverage mismatch')
    labels = ['hap', 'sad', 'ang', 'fea', 'sur', 'dis', 'neu']
    for row in rows:
        c, v = [merged[task][row['utterance_id']] for task in ('classification', 'va')]
        check(int(row['fold']) == c['folds'] == v['folds'], 'CSV fold mismatch')
        check(row['predicted_emotion'] == labels[int(c['predictions'].argmax())], 'CSV classification mismatch')
        check(row['classification_valid'] == str(bool(c['target_valid'])) and row['va_valid'] == str(bool(v['target_valid'])), 'CSV scoring mask mismatch')
        for key, value in [('gt_valence', v['targets'][0]), ('gt_arousal', v['targets'][1]), ('pred_valence', v['predictions'][0]), ('pred_arousal', v['predictions'][1])]:
            check(float(row[key]) == float(value), f'CSV V/A mismatch: {key}')
    report = dict(passed=True, source='independent sklearn metrics / numpy covariance / frozen saved predictions', folds=records, comparison_rows=len(rows), max_abs_prediction_difference=max_difference, prediction_atol=1e-4, metric_recompute_atol=1e-10, historical_mean_atol=1e-5, new_training=False)
    dest = output / 'independent_audit.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS | 10 folds | 10039 comparison rows | max prediction difference {max_difference:.9g}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    audit(parser.parse_args().output)
