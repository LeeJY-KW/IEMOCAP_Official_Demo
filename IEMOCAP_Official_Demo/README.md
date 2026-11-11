# IEMOCAP 공인시험 평가 시연 준비본

**현장에서는 학습 완료 가중치를 불러와 추론과 채점을 실행함.** 공인시험 항목 3번 이산 감정 7분류와 4번 정서가·각성도 회귀를 평가함. 분류용 5개·V/A 회귀용 5개, 총 10개 가중치와 대응 시험 데이터가 포함되어 있음. 모델은 이 프로젝트의 Base `InductiveGraph(variant='multi_graph', context_gate='observed')`이며 두 과제는 별도로 학습한 모델임.

IEMOCAP 통제 환경의 발화 단위 사람 주석을 정답으로 사용함. 이산 감정은 원본 감정 클래스와 비교해 가중 F1을 계산하고, 연속 감정은 원본 정서가·각성도와 비교해 각 차원의 CCC 및 평균 CCC를 계산함. 두 과제의 정답과 예측은 발화 ID로 연결해 한 표에서 확인함.

## 바로 실행

현재 서버에서 검증된 인터프리터는 `/home/neuroai-server05/PycharmProjects/ICASSP_Hetero/.venv-emotionheart/bin/python`임. 아래 명령으로 해당 환경에서 바로 실행할 수 있음.

```bash
cd /home/neuroai-server05/PycharmProjects/감성_감정/공인인증/IEMOCAP_Official_Demo
/home/neuroai-server05/PycharmProjects/ICASSP_Hetero/.venv-emotionheart/bin/python run_demo.py run
```

기본 실행 환경은 CPU 4스레드임. 패키지 내부 가중치와 특징으로 오프라인 추론을 수행함. 매번 새 `results/run_날짜_시간/`에 결과를 저장하며 기존 결과를 보존함. 10개 모델의 `forward()`를 실제 실행함.

다른 PC에서는 **이 폴더 전체**를 복사하고 Python 3.10 환경에서 설치 후 실행함. 가상환경과 패키지 설치는 현장 방문 전에 완료함. 아래 패키지 설치에는 인터넷 연결이 필요함.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_demo.py verify
python run_demo.py run --device cpu --output results/site_run_01
python audit_demo.py results/site_run_01
```

`site_run_01`이 이미 있으면 새 이름을 사용함. `verify`는 파일 SHA256, 모델 저장 파일 소속 폴드, 개발 자료를 기준으로 한 선택 기록, 원본 정답, 데이터 분리를 검사함. `audit_demo.py`는 이번에 계산한 예측을 독립적으로 채점하고 과거 확정된 결과와 비교함. 과거 예측은 재현성 검증의 비교 자료로 사용함.

노트북 사용 시 `python -m pip install -r requirements.txt`로 라이브러리를 준비하고, Jupyter 또는 IDE의 노트북 편집기에서 해당 Python 환경을 커널로 선택함. `현장시연.ipynb`는 46개 셀(실행 코드 22개)로 구성했으며, 각 단계에 목적·사용 함수·인자·반환값·확인할 결과를 작성함. 위에서부터 `Shift+Enter`로 실행함.

노트북 진행 순서는 **실행 준비 → 데이터와 원본 정답 → 5겹 교차검증 분리와 가중치 → 한 대화의 모델 로딩·정규화·추론 → 전체 5겹 교차검증 평가 → 지표와 그래프 → 독립 검증·결과 보존**임. 한 대화의 추론 과정은 노트북 안에서 실제 연산 코드로 보여주며, 전체 평가는 같은 `run_demo.run()`을 호출함. 정답 비교표와 그래프가 코드 셀 아래에 표시됨.

노트북의 자동 저장으로 바뀌는 출력·실행 번호·편집기 메타데이터는 허용함. 코드와 설명 셀의 순서·내용은 `notebook_source_sha256`으로 검사하고 가중치·특징·주석 등 나머지 전달 파일은 기존 SHA256으로 검사함. 소스 수정이 필요한 경우 전달 준비 단계에서 해시 목록을 다시 생성함. HTML 보고서는 브라우저에서 파일을 직접 열어 확인할 수 있음.

## 현장 시연 순서

1. `python run_demo.py verify`의 통과와 폴드 분리표를 확인함.
2. `python run_demo.py run --output results/site_run_01`을 실행함. 확정된 가중치와 정규화 통계로 추론·채점을 수행함.
3. `summary.txt`와 `fold_metrics.csv`에서 폴드별 결과 및 5겹 교차검증 평균±표본표준편차를 확인함.
4. `report.html`에서 폴드 또는 발화 ID를 선택해 원본 감정과 예측 감정, 실제 V/A와 예측 V/A를 비교함. 전체 10,039개 발화를 ID 순으로 제공함.
5. `python audit_demo.py results/site_run_01`을 실행하고 `independent_audit.json`을 보존함.

## 5겹 교차검증과 시험 자료 구성

| 폴드 / 가중치 번호 | 학습 세션 | 개발 세션 | 현장 시험 세션 |
|---|---|---|---|
| 1 | 3, 4, 5 | 2 | 1 |
| 2 | 1, 4, 5 | 3 | 2 |
| 3 | 1, 2, 5 | 4 | 3 |
| 4 | 1, 2, 3 | 5 | 4 |
| 5 | 2, 3, 4 | 1 | 5 |

5겹 교차검증의 학습과 모델 선택은 완료된 상태임. 현장에서는 각 폴드의 고정된 시험 자료를 해당 폴드 모델로 평가함. `k번 세션 → k번 폴드` 연결을 유지하며, **각 발화는 해당 세션을 학습·개발에서 제외한 모델로 예측함.** 각 발화는 과제별 OOF(전체 폴드의 시험 예측) 결과에 정확히 한 번 포함됨.

가중치와 특징 정규화 통계는 2026-09-30 완료한 `certification_classification_v1` / `certification_va_v1`의 원본 바이트를 보존함. 각 폴드의 50에포크 기록 중 분류는 개발 자료의 가중 F1, 회귀는 개발 자료의 평균 CCC가 처음으로 최댓값을 기록한 모델 저장 파일을 사용함. 5개 모델 선택 확정 기록이 모두 생성된 후 시험 자료를 평가한 원래 기록도 함께 제공함. `verify --audit-normalizers`는 저장 통계가 학습 자료에서만 계산되었는지 읽기 전용으로 재검산함.

## 데이터와 점수

| 항목 | 사용 내용 |
|---|---|
| 입력 | 음성 WavLM 1536차원·운율 16차원, RoBERTa 텍스트 768차원, ResNet18 영상 512차원의 고정 특징 |
| 문맥 | 151개 대화, 10,039개 발화; 대화 내부 양방향 문맥 |
| 이산 정답 | 원본 hap/sad/ang/fea/sur/dis/neu: 행복/슬픔/분노/공포/놀람/혐오/중립, 4,639개 |
| 이산 채점 | 정확도와 가중 F1; 주 지표는 가중 F1 |
| 분류 제외 | exc/fru/xxx/oth는 원본 라벨을 보존한 채 채점에서 제외하고 문맥으로 유지 |
| 연속 정답 | 원본 발화별 정서가·활성도 평균 주석; 활성도를 각성도로 표기 |
| 연속 채점 | CCC-V, CCC-A 및 두 값의 평균; 10,035개 V/A 쌍 |
| V/A 제외 | 원본 정서가 5.5인 4개 발화는 원값 보존, 두 차원 모두 채점 제외 |
| 집계 | 폴드별 지표 산술평균±표본표준편차(ddof=1); 전체 폴드의 시험 예측을 합친 OOF 점수는 별도 보조 지표 |

가중 F1은 각 클래스 F1에 정답 표본 비율을 곱해 합산함. CCC는 `2*cov(y,p)/(var(y)+var(p)+(mean(y)-mean(p))^2)`이며 전체 발화 쌍의 모집단 분산·공분산을 사용함. 각 폴드에서 V/A 두 차원의 CCC를 각각 구한 뒤 평균함. 혐오 정답은 전체 2개임.

`data/features`의 특징 캐시를 입력으로 사용하고, 원본 사람 주석과 대조한 평가 정의를 정답으로 사용함. 모델 순전파에는 특징·화자·발화 순서·시간을 전달함.

이 패키지는 **사전 추출한 멀티모달 특징을 입력하는 평가**임. 원본 출처 경로는 이력으로 보존하며 실행 경로는 패키지 내부 상대 경로를 사용함.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `run_demo.py`, `model.py` | 추론 전용 진입점과 원본 모델 코드 |
| `현장시연.ipynb` | 준비 → 무결성 검사 → 추론 → 정답 비교 → 독립 채점 |
| `weights/{classification,va}/fold1..5/best.pt` | 확정한 10개 가중치와 학습 자료 정규화 통계 |
| `data/features`, `data/annotations`, `data/*_dataset.json` | 고정 특징, 원본 주석, 원본 평가 정의 |
| `evidence/` | 평가 설정, 모델 선택 기록, 학습 이력, 원본 코드 사본, 과거 예측 및 복사 출처 |
| `PACKAGE_MANIFEST.json` | 전달 파일별 SHA256·크기. 결과 폴더는 대상 제외 |
| `results/*/preflight.json` | 실제 실행 당시 무결성과 분리 검사 |
| `results/*/*_fold*.npz`, `*_oof.npz` | 이번 실행에서 계산한 예측·정답·ID·채점 마스크 |
| `results/*/utterance_comparison.csv`, `report.html` | 이산·연속 정답과 예측을 발화별로 연결한 결과 |
| `results/*/independent_audit.json` | 별도 지표 구현과 확정된 결과의 대조 |

`evidence/source_snapshot`에는 학습에 사용한 원본 코드 사본을 출처 기록으로 보존함.

## 시험 설명에 사용할 문구

> 학습이 완료된 이산 감정 분류 모델 및 연속 감정 회귀 모델의 폴드별 가중치를 불러옴. IEMOCAP의 5개 세션을 기준으로 구성한 5겹 교차검증에서, 각 폴드의 학습·개발에서 제외한 시험 세션의 멀티모달 특징을 입력함. 예측 감정 클래스와 원본 감정 라벨을 비교하여 가중 F1을 산출하고, 예측 정서가·각성도와 원본 발화별 주석을 비교하여 각 차원의 CCC 및 평균 CCC를 산출함. 현장에서는 확정된 모델의 추론과 채점을 수행함.
