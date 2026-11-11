#!/usr/bin/env python3
"""Independently score fresh predictions and compare the frozen reference run."""
import argparse
import csv
import json
from pathlib import Path
import sys

# 검증하면서 원본 패키지에 파이썬 바이트코드 파일이 생기지 않도록 함.
sys.dont_write_bytecode = True
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]


# 기능: UTF-8 JSON 파일을 읽음.
# 입력: 파일 경로 path. 반환: JSON을 해석한 딕셔너리 또는 목록.
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# 기능: 검증 조건이 거짓이면 해당 원인을 담은 오류로 즉시 중단함.
# 입력: 참·거짓 조건 condition과 오류 설명 message. 반환값은 없음.
def check(condition, message):
    if not condition:
        raise ValueError(message)


# 기능: 별도 구현으로 분류·회귀 지표를 계산하여 추론 결과를 독립 검증함.
# 입력: 과제 task, 정답 배열 y, 예측 배열 p. 반환: 해당 과제의 지표 딕셔너리.
def independent_score(task, y, p):
    if task == 'classification':
        # -100은 채점 제외 표시임. 나머지는 가장 큰 예측 확률의 클래스로 비교함.
        # 클래스 순서를 0~6으로 고정하고 정답 표본 수로 가중한 F1을 계산함.
        mask = y != -100
        truth, prediction = y[mask], p[mask].argmax(1)
        return dict(accuracy=float(accuracy_score(truth, prediction)), weighted_f1=float(f1_score(truth, prediction, average='weighted', labels=list(range(7)), zero_division=0)), confusion=confusion_matrix(truth, prediction, labels=list(range(7))).tolist())
    # 정서가와 각성도가 모두 1~5인 발화만 두 차원의 공통 채점 대상으로 삼음.
    mask = np.logical_and(y >= 1, y <= 5).all(axis=1)
    scores = []
    for dimension in range(2):
        # 재계산은 float64로 수행하며, 분산·공분산은 모집단 정의(ddof=0)를 씀.
        truth, prediction = y[mask, dimension].astype('float64'), p[mask, dimension].astype('float64')
        covariance = np.cov(truth, prediction, ddof=0)
        scores.append(float(2 * covariance[0, 1] / (covariance[0, 0] + covariance[1, 1] + (np.mean(truth) - np.mean(prediction)) ** 2)))
    # 두 차원의 CCC를 각각 유지하고 산술평균을 연속 감정의 대표 지표로 반환함.
    return dict(ccc_valence=scores[0], ccc_arousal=scores[1], mean_ccc=sum(scores) / 2)


# 기능: 새 추론의 저장 예측과 보고서를 독립 채점하고 기존 확정 결과와 대조함.
# 입력: run_demo.py가 생성한 결과 폴더 output. 반환: 전체 검증 결과 딕셔너리.
# 저장된 예측을 읽어 재채점하고 결과 폴더에 independent_audit.json을 저장함.
def audit(output):
    output = Path(output).resolve()
    # 정상 완료된 추론 전용 실행인지 먼저 확인함. 미완료 결과는 채점하지 않음.
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
            # 현재 예측과 원래 보존된 예측을 각각 읽되, pickle 객체 로딩은 허용하지 않음.
            with np.load(output / f'{task}_fold{number}.npz', allow_pickle=False) as z:
                fresh = {k: z[k].copy() for k in z.files}
            with np.load(ROOT / 'evidence' / task / f'fold{number}/test_predictions.npz', allow_pickle=False) as z:
                old = {k: z[k].copy() for k in z.files}
            # 고정 평가 정의에서 시험 발화의 순서를 만듦. ID만 같고 순서가 다른
            # 결과나 다른 폴드의 예측·정답이 섞인 결과도 아래 검사에서 거부함.
            expected = [u for did in fold['test'] for u in index[did]['utterances']]
            check(fresh['ids'].tolist() == [u['id'] for u in expected], 'Fresh prediction IDs are not the frozen test set')
            check(fresh['dialogues'].tolist() == [did for did in fold['test'] for u in index[did]['utterances']], 'Dialogue assignment mismatch')
            check(np.array_equal(fresh['targets'], np.asarray([u['target'] for u in expected], dtype=fresh['targets'].dtype)), 'Ground truth mismatch')
            check(np.all(fresh['folds'] == number), 'Wrong checkpoint fold assignment')
            for key in ('ids', 'dialogues', 'targets', 'target_valid'):
                check(np.array_equal(fresh[key], old[key]), f'Reference alignment mismatch: {task}/{number}/{key}')
            # 원본 정답으로 유효 표본 표시를 다시 계산하여 저장된 표시와 대조함.
            valid = fresh['targets'] != -100 if task == 'classification' else ((fresh['targets'] >= 1) & (fresh['targets'] <= 5)).all(1)
            check(np.array_equal(valid, fresh['target_valid']), 'Invalid score mask')
            check(np.isfinite(fresh['predictions']).all(), 'Nonfinite predictions')
            if task == 'classification':
                # 확률의 합은 1이어야 하며, 최종 클래스 결정은 과거 결과와 완전히 같아야 함.
                check(np.allclose(fresh['predictions'].sum(1), 1, atol=1e-6, rtol=0), 'Invalid probability sum')
                check(np.array_equal(fresh['predictions'].argmax(1), old['predictions'].argmax(1)), 'Class decisions differ from archived inference')
            else:
                check(((fresh['predictions'] >= 1) & (fresh['predictions'] <= 5)).all(), 'V/A output outside 1..5')
            # CPU와 GPU의 float32 연산 차이에는 최대 절대 오차 1e-4만 허용함.
            # 이 허용 오차와 아래의 동일 예측 재채점 오차 1e-10은 비교 목적이 다름.
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
        # 폴드별 예측을 연결한 값과 저장된 OOF를 대조함. OOF는 각 발화를
        # 그 발화의 학습·개발에 사용하지 않은 폴드 모델로 한 번씩 예측한 결과임.
        combined = {key: np.concatenate([x[key] for x in parts]) for key in parts[0]}
        with np.load(output / f'{task}_oof.npz', allow_pickle=False) as z:
            for key in combined:
                check(np.array_equal(z[key], combined[key]), 'Saved OOF differs from per-fold predictions')
        check(len(combined['ids']) == len(set(combined['ids'])) == 10039, 'OOF duplicate or missing ID')
        check(int(combined['target_valid'].sum()) == (4639 if task == 'classification' else 10035), 'OOF denominator mismatch')
        # 전체 문맥 발화는 10,039개지만 채점 대상은 분류 4,639개, 회귀 10,035쌍임.
        # 폴드 평균±표본표준편차(ddof=1)와 전체 OOF 통합 점수를 따로 검증함.
        pooled = independent_score(task, combined['targets'], combined['predictions'])
        for key, value in summary['tasks'][task]['fold_mean'].items():
            check(abs(np.mean([x[key] for x in scores]) - value['mean']) < 1e-10, 'Fold mean differs')
            check(abs(np.std([x[key] for x in scores], ddof=1) - value['sample_sd']) < 1e-10, 'Sample SD differs')
            # 과거 실행의 평균과 비교할 때에는 연산 환경 차이를 고려한 1e-5를 적용함.
            check(abs(value['mean'] - old_summary[key]['mean']) < 1e-5, 'Mean differs from archived evaluation')
            check(abs(pooled[key] - summary['tasks'][task]['pooled_oof_secondary'][key]) < 1e-10, 'Pooled metric differs')
        merged[task] = {str(uid): {key: combined[key][i] for key in ('targets', 'predictions', 'target_valid', 'folds')} for i, uid in enumerate(combined['ids'])}
    # 사람이 확인하는 CSV도 같은 발화 ID에 연결된 예측·정답·채점 여부인지 검사함.
    # 동일한 발화의 분류·회귀 정답과 예측을 한 표에서 함께 확인함.
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
    # 모든 검사가 끝난 경우에만 통과 보고서를 저장함. 원본 예측 파일은 수정하지 않음.
    report = dict(passed=True, source='independent sklearn metrics / numpy covariance / frozen saved predictions', folds=records, comparison_rows=len(rows), max_abs_prediction_difference=max_difference, prediction_atol=1e-4, metric_recompute_atol=1e-10, historical_mean_atol=1e-5, new_training=False)
    dest = output / 'independent_audit.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS | 10 folds | 10039 comparison rows | max prediction difference {max_difference:.9g}')
    return report


if __name__ == '__main__':
    # 터미널 사용 예: python audit_demo.py results/site_run_01
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output
    audit(output if output.is_absolute() else ROOT / output)
