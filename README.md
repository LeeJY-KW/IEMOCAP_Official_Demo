# IEMOCAP 공인시험

3번 이산 감정 인식과 4번 정서가·각성도 인식을 평가하는 비공개 시험 자료임.
학습 완료 가중치로 각 Fold의 학습·개발에 사용하지 않은 IEMOCAP Session을 추론하고 원본 라벨과 비교함.

1. 내부 `IEMOCAP_Official_Demo` 폴더를 PyCharm 프로젝트로 엶.
2. Python 3.10 64비트 프로젝트 인터프리터에서 `requirements.txt` 하나로 설치함.
3. PyCharm 2026.2에서 [Model_verification.ipynb](IEMOCAP_Official_Demo/Model_verification.ipynb)를 열고 커널 설치 안내가 나오면 PyCharm에서 설치함. 실행 방식을 `IPyKernel`로 선택한 뒤 위에서부터 실행함.


[설치·시험 안내](IEMOCAP_Official_Demo/README.md)

현장 진행은 **원문·음성·영상 → 전처리 → Fold·가중치 확인 → 모델 추론 → 전체 5겹 평가 → 성능 기준 판정** 순서임.
현장 노트북은 프로젝트 최상위의 `Model_verification.ipynb`, Python 코드는 `code/`, 평가 자료는 `data/`, 가중치는 `weights/`, 평가 근거는 `evidence/`, 결과는 `results/`에 있음.

시험을 실행하면 해당 실행의 결과가 `results/notebook_날짜_시간`에 생성됨. 결과는 해당 폴더의 예측·Ground Truth·성능 지표와 검정 기록으로 확인함.

시험 기준은 **5개 Fold의 평균 Weighted F1 70% 이상**, **평균 CCC 0.60 이상**임. 이번 시험 측정값·기준·항목별 판정을 표시하며, 두 항목 모두 충족하면 종합 `PASS`임.
