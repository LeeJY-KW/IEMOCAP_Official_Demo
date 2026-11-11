# IEMOCAP 공인시험

공인시험 항목 **3번 이산 감정 인식**과 **4번 연속 감정 인식**의 시연 자료임.
학습 완료된 분류 5개·정서가 및 각성도 회귀 5개 가중치를 불러와 추론하고 IEMOCAP 원본 사람 주석과 비교함.

## PyCharm 실행

1. `IEMOCAP_Official_Demo` 폴더를 프로젝트로 열고 Python 3.10 64비트 인터프리터를 선택함.
2. 같은 인터프리터를 사용하는 PyCharm 터미널에서 아래 명령을 실행함. 작업 폴더는 `requirements.txt`가 있는 패키지 루트임.

```bash
python -m pip install --upgrade pip==25.3
python -m pip install -r requirements.txt
```

3. PyCharm 2026.2에서 [Model_verification.ipynb](Model_verification.ipynb)를 열어 같은 프로젝트 인터프리터를 선택함. 커널 설치 안내가 나오면 PyCharm에서 설치함.
4. 노트북 상단 `Configure Jupyter Server → Managed Server`에서 `Autodetected execution mode`를 해제하고 `Execution mode`를 `IPyKernel`로 선택함. 위에서부터 실행하며, 이전 위치에서 실행한 노트북은 커널을 재시작함.

`requirements.txt`에는 모델 추론·전처리·미디어 표시용 패키지 8개만 포함함. 노트북 실행 커널은 PyCharm에서 준비함. 실행 방식은 [PyCharm 2026.2 공식 안내](https://www.jetbrains.com/help/pycharm/2026.2/configuring-jupyter-notebook.html)를 따름.

설치 후에는 준비된 가중치·특징·원본 주석·미디어로 오프라인 실행함. 기본 연산은 CPU 4스레드임. 결과는 프로젝트 작업 폴더와 관계없이 패키지 루트의 `results/notebook_날짜_시간`에 저장됨.

## 시험 순서

1. 실행 준비: 라이브러리·경로·평가 파일 확인
2. 평가 데이터와 정답 준비: 원본 정답·시험 발화·원문·음성·영상 확인
3. 전처리 및 특징 추출: 음성·운율·텍스트·영상 전처리와 Feature pooling 확인
4. 평가 조건 확인: Fold별 가중치와 학습·개발·시험 Session 확인
5. 예시 대화 추론: 정규화·모델 추론·정답 비교
6. 전체 5-fold cross-validation 평가: Fold별 점수와 평균·표본표준편차 산출
7. 결과 및 시각화: 모델 최종 성능 표와 예시 대화의 정서가·각성도 정답·예측 확인

미디어·전처리는 약 9.81초의 `Ses01F_impro01_F011` 한 발화로 설명함. 음성·영상은 재생 버튼으로 확인함. 모델 최종 성능은 전체 5개 시험 Session의 평가 결과로 산출함.

WavLM·RoBERTa·ResNet18의 고정 가중치로 미리 계산한 중간 출력을 포함함. 현장에서는 원본 음성 처리·운율 계산·토큰화·RGB 변환·특징 집계·학습 통계 정규화를 실행함. 영상 입력은 정규화 일치와 RGB 반올림 경계를 나누어 검사하며, 계산값과 차이 수를 표시함.

## 평가 기준

각 Session은 해당 Session을 학습·개발에 사용하지 않은 Fold의 모델로 평가함. 분류와 회귀는 별도로 학습한 모델이며 현장에서는 새 학습을 수행하지 않음.

| 항목 | 평가 내용 |
|---|---|
| 이산 감정 | 행복·슬픔·분노·공포·놀람·혐오·중립 7분류, 유효 정답 4,639개 |
| 연속 감정 | 정서가·각성도, 유효 정답 10,035쌍 |
| 전체 문맥 | 151개 대화, 10,039개 발화 |
| 주 지표 | Weighted F1 및 정서가·각성도 CCC의 평균 |
| 집계 | 5개 Fold 지표의 산술평균 ± 표본표준편차 |
| 기준 | Weighted F1 70% 이상, 평균 CCC 0.60 이상 |

이번 시험에서 측정한 5개 Fold 지표의 산술평균으로 판정함. 반올림 전 Weighted F1이 0.70 이상, 평균 CCC가 0.60 이상이면 각각 `PASS`이며, 두 항목 모두 충족하면 종합 `PASS`임.
Accuracy·Weighted F1의 평균은 %, 표본표준편차는 %p로 표시함. CCC와 기계 판독용 JSON·CSV의 지표는 원래 척도로 저장함.

분류 채점에서는 원본 7종 이외의 라벨을 문맥으로 보존하고, 연속 채점에서는 범위를 벗어난 정서가 4개를 두 차원의 공통 마스크로 제외함. 노트북에 정답 정의와 유효 표본 수를 표시함.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `Model_verification.ipynb` | 현장에서 순서대로 실행하는 시험 노트북 |
| `code/` | 모델·추론·독립 채점·영상 재생 Python 코드 |
| `requirements.txt` | 현장 실행에 사용하는 단일 설치 목록 |
| `data/` | 특징·원본 주석·음성·영상·전처리 중간 자료 |
| `weights/` | 학습 완료 가중치 10개 |
| `evidence/` | Fold 분리·가중치 선택·학습 당시 원본 코드 등 평가 근거 |
| `results/` | 시험 실행 결과 |
| `PACKAGE_MANIFEST.json` | 전달 자료의 무결성 검사 목록 |

`summary.json`, `summary.txt`, `report.html`에서 이번 시험의 측정값·기준·성능 판정을 확인함. `fold_metrics.csv`에는 Fold별 지표, `utterance_comparison.csv`에는 전체 발화의 정답·예측, `independent_audit.json`에는 계산 검증 기록을 저장함.


시험을 실행하면 해당 실행의 결과가 `results/notebook_날짜_시간`에 생성됨. 결과는 해당 폴더의 예측·Ground Truth·성능 지표와 검정 기록으로 확인함.
