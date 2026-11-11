# IEMOCAP 공인시험 평가 시연 준비본

**현장에서는 학습 완료 가중치를 불러와 추론과 채점을 실행함.** 공인시험 항목 3번 이산 감정 7분류와 4번 정서가·각성도 회귀를 평가함. 분류용 5개·V/A 회귀용 5개, 총 10개 가중치와 대응 시험 데이터가 포함되어 있음. 예시 대화의 음성·원문·영상과 전처리 중간 자료도 함께 준비함. 미디어·전처리 예시는 길이가 약 9.81초인 `Ses01F_impro01_F011` 발화를 사용함. 모델은 이 프로젝트의 Base `InductiveGraph(variant='multi_graph', context_gate='observed')`이며 두 과제는 별도로 학습한 모델임.

IEMOCAP 통제 환경의 발화 단위 사람 주석을 정답으로 사용함. 이산 감정은 원본 감정 클래스와 비교해 가중 F1을 계산하고, 연속 감정은 원본 정서가·각성도와 비교해 각 차원의 CCC 및 평균 CCC를 계산함. 두 과제의 정답과 예측은 발화 ID로 연결해 한 표에서 확인함.

## Windows 설치와 실행

Windows 64비트와 Python 3.10 64비트를 사용함. 저장소 전체를 내려받아 압축을 풀고, `setup_windows.cmd`와 `start_windows.cmd`가 있는 `IEMOCAP_Official_Demo` 폴더를 엶.

1. 최초 한 번 `setup_windows.cmd`를 실행함. 이 폴더의 `.venv`에 CPU용 PyTorch와 `requirements.txt` 패키지를 설치하고 전용 Jupyter 커널을 등록함. 설치에는 인터넷 연결이 필요함.
2. 설치가 끝나면 `start_windows.cmd`를 실행함. 파일 검사 후 브라우저에 JupyterLab과 `현장시연.ipynb`가 열림.
3. 커널 이름이 **IEMOCAP Windows (Python 3.10)**인지 확인하고 위에서부터 셀을 실행함. 기존 오류로 중단된 경우에는 커널을 재시작한 뒤 전체 실행함.

두 실행 파일은 자체 폴더를 작업 경로로 사용하고 UTF-8 모드를 설정함. Jupyter 서버와 커널 모두 `.venv\Scripts\python.exe`를 사용하며, 커널 목록에는 이 프로젝트의 전용 커널만 표시함. 전역 Python이나 다른 프로젝트의 커널이 섞이면 초기 검사에서 중단함.

기본 추론 장치는 CPU 4스레드임. 설치가 끝난 현장에서는 준비된 특징·가중치·미디어로 오프라인 실행함. 원본 특징 추출기를 다시 설치하거나 학습할 필요 없이, 준비된 중간 출력으로 전처리 과정을 확인함. 새 결과는 `results/notebook_날짜_시간`에 저장됨.

| 실행 파일 | 수행 내용 |
|---|---|
| `setup_windows.cmd` | Python 3.10 64비트 확인, `.venv` 설치, CPU PyTorch, 패키지 및 전용 커널 등록, 파일 검사 |
| `start_windows.cmd` | 전용 Python·커널·가중치·자료 확인 후 JupyterLab에서 노트북 열기 |
| `start_windows.cmd --check-only` | 브라우저를 열지 않고 설치 상태와 실행 경로 검사 |

터미널에서 전체 평가만 실행할 때도 같은 가상환경을 사용함.

```bat
.venv\Scripts\python.exe run_demo.py run --device cpu --output results\site_run_01
.venv\Scripts\python.exe audit_demo.py results\site_run_01
```

`site_run_01`이 이미 있으면 새 이름을 사용함. 평가용 가중치는 학습 완료 상태이며, 이 명령은 추론과 독립 채점만 수행함.

노트북은 64개 셀(실행 코드 31개)로 구성했으며, 각 단계에 목적·사용 함수·인자·반환값·확인할 결과를 작성함. 진행 순서는 **실행 준비 → 원본 정답·미디어 → 전처리 → 폴드·가중치 → 한 대화 추론 → 전체 5겹 평가 → 그래프 → 독립 검증·결과 보존**임. 정답 비교표의 처음 12행과 대화 전체 V/A 그래프, 결과 CSV의 처음 10행을 표시함.

텍스트 자료는 UTF-8로 읽고 저장함. 전달된 노트북의 출력·실행 번호·편집기 메타데이터는 저장할 수 있으며, 코드·설명과 가중치·특징·주석은 무결성 검사로 확인함. 영상은 WebM으로 재생하며 별도 HTML도 패키지 안에 포함됨.

## 원문·미디어·전처리 확인

2-4~2-12절의 미디어·전처리 예시는 약 9.81초 길이의 `Ses01F_impro01_F011` 발화로 고정함. 같은 발화의 원문·원본 음성·동일 구간 영상과 전처리 과정을 순서대로 확인함. 노트북을 위에서부터 실행하고, 음성과 영상은 재생 버튼을 눌러 확인함.

2-6절은 `<video src>`에 WebM 데이터를 직접 포함한 HTML 플레이어로 재생함. 기존 MP4와 원본 WAV는 그대로 보존함. [별도 재생 HTML](data/media/Ses01F_impro01/Ses01F_impro01_F011.playback.html)을 브라우저에서 열어 같은 영상을 재생할 수도 있음. `video_playback.py`는 방문 전에 WebM·HTML을 준비하고 현장 실행 시 원본과 재생 파일의 해시를 검사함. 현장에서는 기존 `requirements.txt` 환경과 준비된 파일을 사용함.

| 노트북 절 | 현장에서 실행·확인하는 내용 |
|---|---|
| 2-4~2-6 | 예시 발화의 원문, 원본 WAV 재생, 같은 발화 구간의 WebM 임베드 재생 |
| 2-7~2-8 | 전체 전처리 흐름, 원본 음성 읽기·모노화·16 kHz 처리, WavLM 입력과 파형 |
| 2-9 | 원본 음성에서 librosa로 운율 16차원 통계를 실제 계산 |
| 2-10 | 저장된 원래 토크나이저로 실제 토큰화하고 토큰 ID·특수 토큰·입력 순서를 표시 |
| 2-11 | 원본 RGB 3프레임의 크기 조절·중앙 자르기·정규화를 실제 실행하고 입력 텐서 대조 |
| 2-12 | 원본 `make_extractor` 전체 코드 표시, 준비된 중간 출력으로 특징 평균·표준편차 계산 및 기존 특징 대조 |
| 4-3~4-5 | 학습 자료의 저장 통계로 정규화하고 대화 전체를 추론하여 정답·예측 비교표의 처음 12행 확인 |

WavLM·RoBERTa·ResNet18의 순전파는 준비 단계에서 정확한 고정 가중치와 설정으로 CUDA에서 실행해 중간 출력을 저장함. 현장에서는 그 출력을 불러와 WavLM 숨은 상태의 평균·모집단 표준편차, RoBERTa 일반 토큰 평균, ResNet18 3프레임 평균을 직접 계산함. 음성 처리·운율 계산·토큰화·영상 변환과 특징 집계의 실행 결과를 준비 기록 및 기존 특징과 대조함.

패키지에는 예시 대화 `Ses01F_impro01`의 30발화 자료가 준비되어 있으며, 미디어·전처리 시연에는 위 예시 발화의 자료를 사용함. 전체 10,039개 발화의 감정 모델 입력은 검증된 기존 `data/features`를 그대로 사용함. 현장 커널은 Python 3.10·CPU 4스레드이며, 필요한 미디어와 중간 출력·토크나이저가 패키지에 포함되어 있어 준비된 파일로 오프라인 진행이 가능함. 현장 라이브러리는 `requirements.txt` 하나로 설치하며 전처리 확인용 `librosa`, `soundfile`, `tokenizers`도 포함함.

| 준비 폴더 | 포함 자료 | 용량 |
|---|---|---:|
| `data/media/Ses01F_impro01` | 원본 WAV 30개, H.264/AAC MP4 30개, 예시 WebM·재생 HTML 및 해시 목록 | WAV·MP4 묶음 약 17.1 MiB |
| `data/preprocessing/Ses01F_impro01` | 중간 출력 NPZ 30개, 원본 RGB PNG 90개, 토크나이저·추출기 설정·검증 기록 | 약 64.0 MiB |

이 두 폴더를 포함한 전체 패키지와 현장 Python 환경을 준비하면, 현장에서 재생·전처리 확인·감정 모델 추론을 이어서 실행할 수 있음.

## 현장 시연 순서

1. 노트북을 위에서부터 실행하여 파일 검사, 원문·음성·영상, 전처리 과정과 폴드 분리표를 확인함. 터미널 사전 검사는 `.venv\Scripts\python.exe run_demo.py verify`로 실행함.
2. 노트북의 전체 평가 셀을 실행함. 터미널에서는 `.venv\Scripts\python.exe run_demo.py run --output results\site_run_01`을 실행함. 확정된 가중치와 정규화 통계로 추론·채점을 수행함.
3. `summary.txt`와 `fold_metrics.csv`에서 폴드별 결과 및 5겹 교차검증 평균±표본표준편차를 확인함.
4. `report.html`에서 폴드 또는 발화 ID를 선택해 원본 감정과 예측 감정, 실제 V/A와 예측 V/A를 비교함. 전체 10,039개 발화를 ID 순으로 제공함.
5. `.venv\Scripts\python.exe audit_demo.py results\site_run_01`을 실행하고 `independent_audit.json`을 보존함.

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

이 패키지는 **검증된 멀티모달 특징으로 감정 모델을 평가하고, 같은 원본의 미디어와 전처리 과정을 함께 확인하는 구성**임. 원본 출처 경로는 이력으로 보존하며 실행 경로는 패키지 내부 상대 경로를 사용함.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `setup_windows.cmd`, `start_windows.cmd`, `jupyter_lab_config.py` | Windows 설치와 전용 커널 실행 설정 |
| `run_demo.py`, `model.py` | 추론 전용 진입점과 원본 모델 코드 |
| `현장시연.ipynb` | 원문·미디어 → 전처리 → 모델 추론 → 정답 비교 → 독립 채점 |
| `weights/{classification,va}/fold1..5/best.pt` | 확정한 10개 가중치와 학습 자료 정규화 통계 |
| `data/features`, `data/annotations`, `data/*_dataset.json` | 고정 특징, 원본 주석, 원본 평가 정의 |
| `data/media`, `data/preprocessing` | 예시 원본 음성·재생 영상, 전처리 중간 출력·프레임·설정 |
| `prepare_media.py`, `prepare_preprocessing.py` | 방문 전 재생 자료와 전처리 자료를 생성하는 준비 코드 |
| `video_playback.py` | WebM·별도 재생 HTML 사전 준비와 실행 시 파일 해시 검사 |
| `requirements.txt`, `requirements-preprocessing.txt` | 현장 실행 환경과 자료 재생성용 별도 환경 |
| `evidence/` | 평가 설정, 모델 선택 기록, 학습 이력, 원본 코드 사본, 과거 예측 및 복사 출처 |
| `PACKAGE_MANIFEST.json` | 전달 파일별 SHA256·크기. 결과 폴더는 대상 제외 |
| `results/*/preflight.json` | 실제 실행 당시 무결성과 분리 검사 |
| `results/*/*_fold*.npz`, `*_oof.npz` | 이번 실행에서 계산한 예측·정답·ID·채점 마스크 |
| `results/*/utterance_comparison.csv`, `report.html` | 이산·연속 정답과 예측을 발화별로 연결한 결과 |
| `results/*/independent_audit.json` | 별도 지표 구현과 확정된 결과의 대조 |

`evidence/source_snapshot`에는 학습에 사용한 원본 코드 사본을 출처 기록으로 보존함.

## 자료를 다시 준비하는 경우

제공된 `data/media`와 `data/preprocessing`은 현장에서 그대로 사용함. 준비용 PC에서 기본 예시 대화의 자료를 처음 생성하는 명령은 아래와 같음.

```bash
python prepare_media.py --iemocap-root <root> --ffmpeg <path>
python prepare_preprocessing.py --iemocap-root <root> --device cuda
```

기존 재생 자료를 교체할 때는 `--overwrite`를 추가하며, 전처리 재생성 결과는 `--output <새폴더>`로 새 위치에 저장함. 다른 시험 대화는 두 명령에 같은 `--dialogue <대화ID>`를 지정해 준비함.

`<root>`는 IEMOCAP의 `Session1`~`Session5`가 있는 폴더, `<path>`는 FFmpeg 실행 파일 경로임. 재생 자료 생성은 FFmpeg를 사용함. 전처리 자료 생성은 `requirements-preprocessing.txt`에 맞춘 별도 환경과 원래 Hugging Face 로컬 캐시, ResNet18 가중치가 필요함. 원본 특징 추출 당시의 라이브러리·모델 설정·가중치 해시를 맞춰 준비하며, 실제 사용 환경과 비교 결과를 준비 폴더에 기록함. 이 단계는 고정된 추출기의 순전파를 실행하는 자료 준비 과정임.

## 최신 실행 결과

- [설명과 실행 결과가 포함된 노트북](results/notebook_playback_utf8_20261111/현장시연_실행완료.ipynb)
- [원문·미디어·전처리·평가 결과 HTML](results/notebook_playback_utf8_20261111/현장시연_설명과결과.html)
- [실제 Jupyter 커널 실행 검증](results/notebook_playback_utf8_20261111/notebook_execution_audit.json)
- [독립 채점 검증](results/notebook_playback_utf8_20261111/independent_audit.json)

## 시험 설명에 사용할 문구

> 학습이 완료된 이산 감정 분류 모델 및 연속 감정 회귀 모델의 폴드별 가중치를 불러옴. IEMOCAP의 5개 세션을 기준으로 구성한 5겹 교차검증에서, 각 폴드의 학습·개발에서 제외한 시험 세션의 멀티모달 특징을 입력함. 예측 감정 클래스와 원본 감정 라벨을 비교하여 가중 F1을 산출하고, 예측 정서가·각성도와 원본 발화별 주석을 비교하여 각 차원의 CCC 및 평균 CCC를 산출함. 현장에서는 확정된 모델의 추론과 채점을 수행함.
