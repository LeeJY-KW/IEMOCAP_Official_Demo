# IEMOCAP 감정 인식 평가 시연

학습이 완료된 멀티모달 감정 인식 모델을 이용한 **이산·연속 감정 5겹 교차검증 평가 시연** 자료입니다. 연구·시험 준비용 비공개 저장소입니다.

현장에서는 준비된 가중치를 불러와 추론하고, IEMOCAP 원본 사람 주석(정답)과 비교합니다. 분류 5개와 V/A 회귀 5개, 총 10개 가중치와 필요한 특징·주석을 포함합니다.

## 먼저 볼 자료

- **[설명·코드·실행 결과가 포함된 노트북](IEMOCAP_Official_Demo/results/notebook_review_20261111/현장시연_실행완료.ipynb)** — GitHub에서 셀별 설명과 실제 출력 확인
- [새로 실행할 현장 노트북](IEMOCAP_Official_Demo/현장시연.ipynb)
- [상세 실행 안내](IEMOCAP_Official_Demo/README.md)
- [5겹 교차검증 결과 요약](IEMOCAP_Official_Demo/results/notebook_review_20261111/summary.txt)
- [실제 Jupyter 커널 실행 검증](IEMOCAP_Official_Demo/results/notebook_review_20261111/notebook_execution_audit.json)
- [독립 채점 검증](IEMOCAP_Official_Demo/results/notebook_review_20261111/independent_audit.json)

노트북은 **목적 → 함수·인자·반환값 → 실행 코드 → 결과 확인** 순서의 46개 셀로 구성했습니다. 실제 Jupyter 커널에서 코드 셀 22개를 모두 실행했고, 정답 비교표와 그래프 2개를 포함합니다.

## 실행 순서

1. 실행 환경과 파일 무결성 확인
2. 멀티모달 특징 및 원본 감정·V/A 정답 확인
3. 학습·개발·시험 세션과 가중치 선택 기록 확인
4. 한 대화의 가중치 로딩 → 정규화 → 실제 추론 → 정답 비교
5. 전체 5겹 교차검증 추론 및 정확도·가중 F1·CCC 계산
6. 폴드별 성능과 발화별 V/A 그래프 확인
7. 독립 검증 및 결과 보존

## 설치와 실행

Python 3.10 환경을 사용합니다. 저장소 전체를 내려받아야 가중치·특징을 포함한 평가가 가능합니다. Git LFS는 사용하지 않습니다.

```bash
git clone https://github.com/LeeJY-KW/IEMOCAP_Official_Demo.git
cd IEMOCAP_Official_Demo/IEMOCAP_Official_Demo
python -m venv .venv
```

Linux에서는 `source .venv/bin/activate`, Windows PowerShell에서는 `.venv\Scripts\Activate.ps1`로 환경을 활성화합니다.

```bash
python -m pip install -r requirements.txt
python run_demo.py verify
```

Jupyter 또는 IDE에서 이 Python 환경을 커널로 선택하고 `현장시연.ipynb`를 위에서부터 실행합니다. 실행 완료본의 임시 경로는 검증 당시 기록이며, 다시 실행하면 현재 패키지의 `results/notebook_...`에 새 결과가 생성됩니다. 학습·개발·시험 분리와 모델 설정은 고정되어 있습니다.

터미널에서 전체 평가만 실행할 수도 있습니다.

```bash
python run_demo.py run --device cpu --output results/site_run_01
python audit_demo.py results/site_run_01
```

같은 결과 폴더를 덮어쓰지 않으므로 재실행에는 새 이름을 사용합니다. 노트북 출력·실행 번호의 자동 저장은 허용하며 코드·설명과 가중치·데이터의 무결성은 검사합니다. 설치 후 추론에는 인터넷이나 GPU가 필요하지 않습니다.

## 평가 정의

| 폴드 | 학습 세션 | 개발 세션 | 시험 세션 |
|---|---|---|---|
| 1 | 3, 4, 5 | 2 | 1 |
| 2 | 1, 4, 5 | 3 | 2 |
| 3 | 1, 2, 5 | 4 | 3 |
| 4 | 1, 2, 3 | 5 | 4 |
| 5 | 2, 3, 4 | 1 | 5 |

k번 세션은 그 세션을 학습·개발에 쓰지 않은 k번 폴드 가중치로만 평가합니다. 분류와 회귀는 같은 Base 구조를 각각 학습한 별도 모델이며 5개 모델의 앙상블을 사용하지 않습니다.

- 입력: 사전 추출한 음성·운율·텍스트·영상 특징
- 문맥: 151개 대화, 10,039개 발화
- 이산 정답: 원본 `hap/sad/ang/fea/sur/dis/neu` 4,639개. 나머지 감정은 채점에서 제외하고 문맥 유지
- 연속 정답: 원본 발화별 정서가·각성도 10,035쌍. 정서가 5.5인 4개 발화는 원값을 보존하고 두 차원 모두 채점 제외
- 대표 집계: 5개 폴드 지표의 산술평균 ± 표본표준편차. 전체 폴드의 시험 예측을 합친 OOF 점수는 별도 보조 지표

## 재현 결과

| 지표 | 5겹 교차검증 평균 ± 표본표준편차 |
|---|---:|
| 정확도 | 0.776622 ± 0.031902 |
| 가중 F1 | 0.773211 ± 0.030954 |
| 정서가 CCC | 0.750615 ± 0.037333 |
| 각성도 CCC | 0.684792 ± 0.017355 |
| 평균 CCC | 0.717703 ± 0.022541 |

한 대화의 설명용 추론과 전체 평가 예측의 일치를 확인했고, 전체 저장 예측을 별도로 재채점했습니다. 실행 시의 검사와 근거는 `results/notebook_review_20261111`에 있습니다.

## 결과 확인

`results/notebook_review_20261111/현장시연_설명과결과.html`은 설명과 코드·출력을 함께 읽는 미리보기입니다. `report.html`은 전체 발화의 정답·예측을 검색하는 화면입니다. HTML 파일은 저장소를 내려받은 뒤 브라우저에서 열면 됩니다. GitHub Pages로 공개하지 않았습니다.

IEMOCAP 통제 환경의 발화 단위 예비 검증 자료입니다. 혐오 정답은 전체 2개입니다. 자체 수집 자료의 어노테이션 성능, 실환경·프레임 단위 평가 또는 기관 인증 완료를 의미하지 않습니다. 원음·영상과 특징 추출기 가중치는 포함하지 않았으며 현재 입력은 사전 추출 특징입니다.
