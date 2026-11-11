# IEMOCAP 감정 인식 평가 시연

학습이 완료된 멀티모달 감정 인식 모델을 이용한 **이산·연속 감정 5겹 교차검증 평가 시연** 자료임. 공인시험 항목 3번 이산 감정 7분류와 4번 정서가·각성도 회귀를 준비하는 비공개 저장소임.

현장에서는 준비된 가중치를 불러와 추론하고, IEMOCAP 통제 환경의 원본 발화별 사람 주석(정답)과 비교함. 분류 5개와 V/A 회귀 5개, 총 10개 가중치와 필요한 특징·주석, 예시 원본 음성·영상 및 전처리 자료를 포함함.

## 먼저 볼 자료

- **[설명·코드·실행 결과가 포함된 노트북](IEMOCAP_Official_Demo/results/notebook_playback_utf8_20261111/현장시연_실행완료.ipynb)** — GitHub에서 셀별 설명과 실제 출력 확인
- [새로 실행할 현장 노트북](IEMOCAP_Official_Demo/현장시연.ipynb)
- [상세 실행 안내](IEMOCAP_Official_Demo/README.md)
- [5겹 교차검증 결과 요약](IEMOCAP_Official_Demo/results/notebook_playback_utf8_20261111/summary.txt)
- [실제 Jupyter 커널 실행 검증](IEMOCAP_Official_Demo/results/notebook_playback_utf8_20261111/notebook_execution_audit.json)
- [독립 채점 검증](IEMOCAP_Official_Demo/results/notebook_playback_utf8_20261111/independent_audit.json)

노트북은 **목적 → 함수·인자·반환값 → 실행 코드 → 결과 확인** 순서의 64개 셀(실행 코드 31개)로 구성함. 원문·음성·영상, 전처리 과정, 정답 비교표와 성능 그래프를 같은 흐름에서 확인함.

## 실행 순서

1. 실행 환경과 파일 무결성 확인
2. 멀티모달 특징과 원본 정답, 선택 발화의 음성·원문·영상 및 전처리 확인
3. 학습·개발·시험 세션과 가중치 선택 기록 확인
4. 한 대화의 가중치 로딩 → 정규화 → 실제 추론 → 정답 비교
5. 전체 5겹 교차검증 추론 및 정확도·가중 F1·CCC 계산
6. 폴드별 성능과 대화 전체의 V/A 시계열 그래프 확인
7. 독립 검증 및 결과 보존

## 설치와 실행

Python 3.10 환경을 사용함. 저장소 전체를 내려받으면 가중치·특징을 포함한 평가를 실행할 수 있음.

```bash
git clone https://github.com/LeeJY-KW/IEMOCAP_Official_Demo.git
cd IEMOCAP_Official_Demo/IEMOCAP_Official_Demo
python -m venv .venv
```

Linux에서는 `source .venv/bin/activate`, Windows PowerShell에서는 `.venv\Scripts\Activate.ps1`로 환경을 활성화함.

```bash
python -m pip install -r requirements.txt
python run_demo.py verify
```

Jupyter 또는 IDE에서 이 Python 환경을 커널로 선택하고 `현장시연.ipynb`를 위에서부터 실행함. 실행 완료본의 임시 경로는 검증 당시 기록이며, 다시 실행하면 현재 패키지의 `results/notebook_...`에 새 결과가 생성됨. 학습·개발·시험 분리와 모델 설정은 고정되어 있음.

Windows에서도 JSON과 원본 코드 텍스트를 UTF-8로 읽도록 인코딩을 명시함. 기존 버전에서 `UnicodeDecodeError`가 발생했다면 먼저 패키지에 최신 `현장시연.ipynb`와 `video_playback.py`가 반영됐는지 확인함. 이후 Jupyter 커널을 재시작하고 맨 위 셀부터 전체 실행함.

터미널에서 전체 평가를 바로 실행할 수도 있음.

```bash
python run_demo.py run --device cpu --output results/site_run_01
python audit_demo.py results/site_run_01
```

재실행에는 새 결과 폴더 이름을 사용해 이전 결과를 보존함. 노트북 출력·실행 번호의 자동 저장은 허용하며 코드·설명과 가중치·데이터의 무결성을 검사함. 설치를 완료한 뒤 CPU로 오프라인 추론을 수행할 수 있음.

## 원본 미디어와 전처리 확인

2-4~2-12절의 미디어·전처리 예시는 약 9.81초 길이의 `Ses01F_impro01_F011`을 사용함. 같은 발화의 원문·음성·영상과 전처리 과정을 순서대로 확인함. 노트북을 위에서부터 실행하면 됨.

2-6절은 `<video src>`에 WebM 데이터를 직접 포함한 HTML 플레이어로 재생함. 기존 MP4와 원본 WAV는 그대로 보존함. [별도 재생 HTML](IEMOCAP_Official_Demo/data/media/Ses01F_impro01/Ses01F_impro01_F011.playback.html)을 내려받은 패키지에서 브라우저로 열어 재생할 수도 있음. `video_playback.py`가 WebM·HTML의 사전 준비와 실행 시 원본·재생 파일의 해시 검사를 담당함. 현장에서는 기존 `requirements.txt` 환경과 준비된 파일을 사용함.

| 포함 폴더 | 내용 | 용량 |
|---|---|---:|
| `IEMOCAP_Official_Demo/data/media/Ses01F_impro01` | 원본 WAV 30개, H.264/AAC MP4 30개, 예시 WebM·재생 HTML 및 해시 목록 | WAV·MP4 묶음 약 17.1 MiB |
| `IEMOCAP_Official_Demo/data/preprocessing/Ses01F_impro01` | 중간 출력 NPZ 30개, 원본 RGB PNG 90개, 토크나이저·추출기 설정·검증 기록 | 약 64.0 MiB |

현장에서는 원본 음성 읽기·모노화·16 kHz 처리, librosa 운율 계산, tokenizers 토큰화, 원본 RGB 프레임의 크기 조절·중앙 자르기·정규화, 특징 집계와 학습 통계 정규화를 실제 실행함. WavLM·RoBERTa·ResNet18 순전파의 중간 출력은 준비 단계에서 정확한 고정 가중치와 설정으로 CUDA에서 계산해 저장한 자료를 사용함. 2-12절에는 원본 `make_extractor` 전체 코드를 표시함.

현장 커널은 Python 3.10·CPU 4스레드를 유지함. `requirements.txt` 하나로 현장 라이브러리를 설치하며 `librosa`, `soundfile`, `tokenizers`가 포함됨. 준비된 미디어·프레임·중간 출력·토크나이저가 함께 있어 현장에서 필요한 자료를 오프라인으로 읽을 수 있음. 패키지에는 예시 대화 30발화의 자료가 준비되어 있으며, 미디어·전처리 시연에는 선택한 긴 문장 자료를 사용함. 전체 10,039개 발화의 감정 모델 입력은 검증된 기존 특징 캐시를 그대로 사용함.

준비용 PC의 패키지 폴더에서 기본 예시 대화의 자료를 처음 생성하는 명령은 아래와 같음.

```bash
python prepare_media.py --iemocap-root <root> --ffmpeg <path>
python prepare_preprocessing.py --iemocap-root <root> --device cuda
```

기존 재생 자료를 교체할 때는 `--overwrite`를 추가하며, 전처리 재생성 결과는 `--output <새폴더>`로 새 위치에 저장함. 다른 시험 대화는 두 명령에 같은 `--dialogue <대화ID>`를 지정해 준비함.

`<root>`는 IEMOCAP 원본 루트, `<path>`는 FFmpeg 실행 파일 경로임. 전처리 재생성에는 `requirements-preprocessing.txt`에 맞춘 별도 환경과 원래 Hugging Face 로컬 캐시·ResNet18 가중치가 필요함. 현장에서는 저장소에 준비된 파일을 그대로 사용함.

## 평가 정의

| 폴드 | 학습 세션 | 개발 세션 | 시험 세션 |
|---|---|---|---|
| 1 | 3, 4, 5 | 2 | 1 |
| 2 | 1, 4, 5 | 3 | 2 |
| 3 | 1, 2, 5 | 4 | 3 |
| 4 | 1, 2, 3 | 5 | 4 |
| 5 | 2, 3, 4 | 1 | 5 |

k번 세션은 그 세션을 학습·개발에서 제외한 k번 폴드 가중치로 평가함. 분류와 회귀는 같은 Base 구조를 각각 학습한 별도 모델임.

- 입력: 사전 추출한 음성·운율·텍스트·영상 특징
- 문맥: 151개 대화, 10,039개 발화
- 이산 정답: 원본 `hap/sad/ang/fea/sur/dis/neu` 4,639개. 나머지 감정은 채점에서 제외하고 문맥으로 유지함. 혐오 정답은 2개임.
- 연속 정답: 원본 발화별 정서가·각성도 10,035쌍. 정서가 5.5인 4개 발화는 원값을 보존하고 두 차원 모두 채점에서 제외함.
- 대표 집계: 5개 폴드 지표의 산술평균 ± 표본표준편차. 전체 폴드의 시험 예측을 합친 OOF 점수는 별도 보조 지표임.

## 재현 결과

| 지표 | 5겹 교차검증 평균 ± 표본표준편차 |
|---|---:|
| 정확도 | 0.776622 ± 0.031902 |
| 가중 F1 | 0.773211 ± 0.030954 |
| 정서가 CCC | 0.750615 ± 0.037333 |
| 각성도 CCC | 0.684792 ± 0.017355 |
| 평균 CCC | 0.717703 ± 0.022541 |

한 대화의 설명용 추론과 전체 평가 예측의 일치를 확인했으며, 전체 저장 예측을 별도로 재채점함. 실행 시의 검사와 근거는 `results/notebook_playback_utf8_20261111`에 있음.

## 결과 확인

`results/notebook_playback_utf8_20261111/현장시연_설명과결과.html`은 설명과 코드·출력을 함께 읽는 미리보기임. 노트북에는 정답·예측 비교표의 처음 12행, 대화 전체의 V/A 시계열 그래프, 전체 CSV의 처음 10행을 표시함. `report.html`은 전체 발화의 정답·예측을 검색하는 화면임. HTML 파일은 저장소를 내려받은 뒤 브라우저에서 열어 확인함.
