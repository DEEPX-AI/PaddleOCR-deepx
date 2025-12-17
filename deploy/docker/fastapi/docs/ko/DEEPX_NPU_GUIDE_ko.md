# DEEPX NPU 지원 - 완전 가이드

PaddleOCR FastAPI 서비스에 DEEPX NPU 하드웨어 가속을 추가하는 완전한 가이드

## 📚 목차

1. [빠른 시작](#빠른-시작)
2. [자동 설치](#자동-설치)
3. [구성 요소](#구성-요소)
4. [환경 변수](#환경-변수)
5. [모델 구조](#모델-구조)
6. [사용법](#사용법)
7. [테스트](#테스트)
8. [문제 해결](#문제-해결)
9. [참고 자료](#참고-자료)

---

## 빠른 시작

### 1단계: 자동 설치

DEEPX NPU 전용 설정 스크립트를 사용하여 모든 의존성을 자동으로 설치:

```bash
# 전체 NPU 환경 자동 설정
./local_deepx_setup.sh --dx_rt /path/to/dx_rt
```

**설정 스크립트가 자동으로 수행하는 작업:**
- ✅ Python 3.11+ venv 생성 및 활성화
- ✅ DX_RT 빌드 (dx-engine 자동 설치)
- ✅ PaddlePaddle 3.0.0 설치
- ✅ PyTorch 2.3.0 + torchvision + torchaudio 설치
- ✅ ONNX Runtime 1.18.0 설치
- ✅ 추가 의존성 (scikit-image, imgaug, shapely, pyclipper, jiwer)
- ✅ PaddleOCR v5 모델 다운로드 (server/mobile)
- ✅ DEEPX NPU 모델 확인
- ✅ RT 최적화 환경 변수 설정 파일 생성 (deepx_env.sh)
- ✅ deepx/engine 경로 검증
- ✅ 의존성 설치 검증

### 2단계: 서비스 시작

```bash
# venv 활성화 및 NPU 환경 설정
source venv/bin/activate
source deepx_env.sh  # RT 최적화 환경 변수 적용

# 또는 run.sh 사용 (자동으로 deepx_env.sh 적용)
./run.sh
```

**run.sh가 자동으로 수행하는 작업:**
- ✅ venv 활성화
- ✅ `deepx_env.sh` 자동 감지 및 적용
- ✅ RT 최적화 환경 변수 설정
- ✅ OCR 서비스 실행

### 3단계: CPU OCR 테스트

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1
  }'
```

### 4단계: DEEPX NPU OCR 테스트

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1,
    "deepx": true
  }'
```

---

## 자동 설치

### 개요

`local_deepx_setup.sh`는 deepx 프로젝트의 설정 방식을 FastAPI 서비스에 포팅한 스크립트입니다.

### 포팅된 주요 기능

| 기능 | deepx | FastAPI 포팅 |
|------|-------|--------------|
| **환경 설정** | startup.sh | local_deepx_setup.sh |
| **venv 생성** | ✅ | ✅ |
| **DX_RT 빌드** | build.sh | ✅ (자동 호출) |
| **PyTorch 설치** | requirements.txt (2.3.0) | ✅ (2.3.0) |
| **dx-engine 설치** | ✅ (1.1.2) | ✅ (dx_rt build를 통해) |
| **ONNX Runtime** | requirements.txt (1.18.0) | ✅ (1.18.0) |
| **모델 확인** | - | ✅ (deepx/engine/model_files) |
| **RT 최적화** | set_env.sh | deepx_env.sh |
| **환경 변수 적용** | source set_env.sh | source deepx_env.sh |

### 기본 사용법

```bash
# 필수: dx_rt 경로 지정
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# Server 모델 사용 (기본값)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# Mobile 모델 사용
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --use-mobile
```

### 고급 옵션

```bash
# NPU 설정 건너뛰기 (CPU만)
./local_deepx_setup.sh --no-npu

# PaddleOCR 모델만 다운로드 (DEEPX 모델 제외)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --no-deepx-models

# RT 최적화 스레드 수 커스터마이즈
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --inter-threads 2 --intra-threads 4

# GPU 버전 PaddlePaddle 사용
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --gpu

# Python 버전 지정
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --python python3.11
```

### 전체 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dx_rt PATH` | dx_rt 디렉토리 경로 (NPU 필수) | - |
| `--gpu` | GPU 버전 PaddlePaddle 설치 | cpu |
| `--use-mobile` | Mobile 모델 사용 | server |
| `--no-models` | PaddleOCR 모델 다운로드 건너뛰기 | false |
| `--no-deepx-models` | DEEPX 모델 다운로드 건너뛰기 | false |
| `--no-npu` | NPU 설정 건너뛰기 (CPU만) | false |
| `--python VERSION` | Python 버전 지정 | python3.11 |
| `--version VERSION` | PaddleOCR 버전 지정 | 3.3.2 |
| `--inter-threads N` | CUSTOM_INTER_OP_THREADS_COUNT | 1 |
| `--intra-threads N` | CUSTOM_INTRA_OP_THREADS_COUNT | 3 |

---

## 구성 요소

### 1. 필수 Python 패키지

```txt
# Core OCR dependencies
opencv-python-headless==4.7.0.72
numpy==1.26.4
torch==2.3.0                    ← NPU 필수
torchvision==0.18.0             ← NPU 필수
torchaudio==2.3.0               ← NPU 필수
onnxruntime==1.18.0             ← NPU 필수

# Image processing dependencies
opencv-contrib-python==4.7.0.72
scikit-image                    ← NPU 필수
imgaug                          ← NPU 필수
shapely                         ← NPU 필수
pyclipper                       ← NPU 필수

# Utility dependencies
jiwer                           ← NPU 필수
```

### 2. DX_RT 빌드

dx_rt의 `build.sh`를 호출하여 dx-engine을 자동으로 설치합니다:

```bash
# 자동 실행됨:
cd /path/to/dx_rt
./build.sh
```

빌드 결과:
- dx-engine이 venv에 자동 설치됨
- 필요한 라이브러리가 컴파일됨

### 3. RT 최적화 환경 설정

deepx의 환경 설정 로직을 `deepx_env.sh`로 포팅:

**생성되는 파일 (deepx_env.sh):**
```bash
#!/bin/bash
# DEEPX NPU Environment Configuration

# RT Optimization Settings
export CUSTOM_INTER_OP_THREADS_COUNT=1
export CUSTOM_INTRA_OP_THREADS_COUNT=3

# Optional settings (uncomment if needed)
# export DXRT_DYNAMIC_CPU_THREAD=3
# export DXRT_TASK_MAX_LOAD=4
# export NFH_INPUT_WORKER_THREADS=5
# export NFH_OUTPUT_WORKER_THREADS=6

# Library paths
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"

echo "✓ DEEPX NPU environment configured"
echo "  CUSTOM_INTER_OP_THREADS_COUNT=${CUSTOM_INTER_OP_THREADS_COUNT}"
echo "  CUSTOM_INTRA_OP_THREADS_COUNT=${CUSTOM_INTRA_OP_THREADS_COUNT}"
```

### 4. 자동 환경 적용 (run.sh)

`run.sh`가 시작 시 `deepx_env.sh`를 자동으로 적용:

```bash
# run.sh 내부:
DEEPX_ENV_FILE="$SCRIPT_DIR/deepx_env.sh"
if [ -f "$DEEPX_ENV_FILE" ]; then
    echo "🔧 Applying DEEPX NPU environment settings..."
    source "$DEEPX_ENV_FILE"
    echo ""
fi
```

---

## 환경 변수

### RT 최적화 변수

| 변수 | 기본값 | 설명 | 출처 |
|------|--------|------|------|
| `CUSTOM_INTER_OP_THREADS_COUNT` | 1 | Inter-op 스레드 수 | deepx/set_env.sh |
| `CUSTOM_INTRA_OP_THREADS_COUNT` | 3 | Intra-op 스레드 수 | deepx/set_env.sh |
| `DXRT_DYNAMIC_CPU_THREAD` | - | 동적 CPU 스레드 | deepx/set_env.sh |
| `DXRT_TASK_MAX_LOAD` | - | 최대 작업 부하 | deepx/set_env.sh |
| `NFH_INPUT_WORKER_THREADS` | - | 입력 워커 스레드 | deepx/set_env.sh |
| `NFH_OUTPUT_WORKER_THREADS` | - | 출력 워커 스레드 | deepx/set_env.sh |
| `LD_LIBRARY_PATH` | ${VIRTUAL_ENV}/lib:... | 라이브러리 경로 | deepx/startup.sh |

### 커스터마이징

설치 시 옵션으로 지정하거나 `deepx_env.sh`를 직접 편집:

```bash
# 방법 1: 설치 시 지정
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --inter-threads 2 --intra-threads 4

# 방법 2: deepx_env.sh 편집 후 적용
vi deepx_env.sh
source deepx_env.sh
```

---

## 모델 구조

### PaddleOCR 모델 ($HOME/.paddlex/official_models)

```
~/.paddlex/official_models/
├── PP-OCRv5_server_det/          # Server 검출 모델
├── PP-OCRv5_server_rec/          # Server 인식 모델
├── PP-OCRv5_mobile_det/          # Mobile 검출 모델
├── PP-OCRv5_mobile_rec/          # Mobile 인식 모델
├── PP-LCNet_x1_0_doc_ori/        # 문서 방향 분류
├── UVDoc/                        # 문서 왜곡 보정
└── PP-LCNet_x1_0_textline_ori/   # 텍스트라인 방향
```

### DEEPX 모델 (deepx/engine/model_files)

```
deepx/engine/model_files/
├── server/                       # DEEPX Server 모델
│   ├── det/
│   ├── rec/
│   └── cls/
├── mobile/                       # DEEPX Mobile 모델
│   ├── det/
│   ├── rec/
│   └── cls/
└── *.txt                         # 딕셔너리 파일
```

---

## 사용법

### 서비스 시작

```bash
# 방법 1: 수동 활성화
source venv/bin/activate
source deepx_env.sh
python ocr_service.py

# 방법 2: run.sh 사용 (권장)
./run.sh
```

### API 사용

#### CPU 모드 (기본)

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1
  }'
```

#### NPU 모드 (Async)

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1,
    "deepx": true
  }'
```

#### NPU 모드 (Sync)

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1,
    "deepx": true,
    "sync": true
  }'
```

### Python 클라이언트 예제

```python
import requests
import base64

# 이미지 읽기
with open("test.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# CPU OCR
response = requests.post(
    "http://localhost:8080/api/v1/ocr",
    json={
        "file": img_base64,
        "fileType": 1
    }
)

# NPU OCR (Async)
response = requests.post(
    "http://localhost:8080/api/v1/ocr",
    json={
        "file": img_base64,
        "fileType": 1,
        "deepx": True
    }
)

# NPU OCR (Sync)
response = requests.post(
    "http://localhost:8080/api/v1/ocr",
    json={
        "file": img_base64,
        "fileType": 1,
        "deepx": True,
        "sync": True
    }
)

print(response.json())
```

---

## 테스트

### 테스트 스크립트 실행

```bash
# 서비스 시작
./run.sh

# 테스트 실행
./run_tests.sh --all

# NPU로 테스트
./run_tests.sh --all --deepx true

# CPU로 테스트
./run_tests.sh --all --deepx false

# Sync 모드로 테스트
./run_tests.sh --all --deepx true --sync
```

### 로그 확인

#### CPU 모드
```
🚀 Using CPU for inference
✅ CPU OCR completed: 10 results
```

#### NPU 모드 (Async)
```
🚀 Using DEEPX NPU for inference (async mode)
🔧 Initializing DEEPX NPU OCR engine...
✅ DEEPX NPU OCR initialized successfully
✅ NPU OCR completed: 10 results
```

#### NPU 모드 (Sync)
```
🚀 Using DEEPX NPU for inference (sync mode)
🔧 Initializing DEEPX NPU OCR engine...
✅ DEEPX NPU OCR initialized successfully
✅ NPU OCR completed: 10 results
```

---

## 문제 해결

### 1. Python 버전 오류

**증상:**
```
Error: Python 3.10 is too old for DEEPX NPU support
Python 3.11+ is REQUIRED for DEEPX NPU
```

**해결:**
```bash
# Python 3.11 설치
sudo apt install python3.11

# 또는 Python 3.11로 재설치
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --python python3.11
```

### 2. dx_rt 경로 오류

**증상:**
```
Error: --dx_rt option is required for NPU setup
```

**해결:**
```bash
# dx_rt 경로를 반드시 지정
./local_deepx_setup.sh --dx_rt /path/to/dx_rt
```

### 3. dx_rt 빌드 실패

**증상:**
```
Error: DX_RT build failed with exit code 1
```

**해결:**
```bash
# 수동으로 dx_rt 빌드 확인
cd /path/to/dx_rt
./build.sh

# 빌드 로그 확인
cat build.log
```

### 4. PyTorch 미설치

**증상:**
```
❌ PyTorch not found
```

**해결:**
```bash
source venv/bin/activate
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
```

### 5. dx-engine 미설치

**증상:**
```
❌ dx-engine not found
```

**해결:**
```bash
# dx_rt 빌드를 통해 자동 설치됨
cd /path/to/dx_rt
./build.sh

# 또는 수동 설치
source venv/bin/activate
pip install dx-engine==1.1.2
```

### 6. deepx 경로 오류

**증상:**
```
❌ deepx not found at /data/home/dhyang/git/github/PaddleOCR/deepx
```

**해결:**
```bash
# 경로 확인
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx/engine/

# 심볼릭 링크 확인
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx
```

### 7. DEEPX 모델 없음

**증상:**
```
⚠ DEEPX models not found or incomplete
```

**해결:**
```bash
# 모델 확인
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx/engine/model_files/server/
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx/engine/model_files/mobile/

# 모델이 없으면 deepx/engine/model_files에 배치 필요
```

### 8. RT 최적화 미적용

**증상:**
- NPU 성능이 예상보다 낮음

**해결:**
```bash
# 환경 변수 확인
echo $CUSTOM_INTER_OP_THREADS_COUNT  # 1
echo $CUSTOM_INTRA_OP_THREADS_COUNT  # 3

# 수동 적용
source deepx_env.sh

# 서비스 재시작
./run.sh
```

### 9. Import 에러

**증상:**
```
ModuleNotFoundError: No module named 'dx_engine'
```

**해결:**
```bash
# venv 활성화 확인
source venv/bin/activate

# dx_rt 빌드 재실행
cd /path/to/dx_rt
./build.sh

# 패키지 확인
pip list | grep dx-engine
```

---

## 참고 자료

### 디렉토리 구조

```
deploy/docker/fastapi/
├── ocr_service.py           # NPU 지원 추가
├── local_deepx_setup.sh     # NPU 자동 설치 스크립트
├── deepx_env.sh             # RT 최적화 환경 변수 (자동 생성)
├── run.sh                   # 서비스 시작 (deepx_env.sh 자동 적용)
└── docs/ko/
    └── DEEPX_NPU_GUIDE_ko.md    # 이 파일
```

### 코드 변경 요약

#### 1. Request Model

```python
class BaiduOCRRequest(BaseModel):
    # ... 기존 필드들 ...
    deepx: Optional[bool] = Field(False, description="Use DEEPX NPU")
    sync: Optional[bool] = Field(False, description="Use sync mode (PaddleOcr)")
```

#### 2. NPU 초기화 (Async)

```python
def get_npu_ocr_instance():
    """DEEPX NPU OCR 엔진 초기화 (Async)"""
    import sys
    sys.path.insert(0, str(DEEPX_ENGINE_PATH))
    from engine.async_paddleocr import AsyncPipelineOCR
    return AsyncPipelineOCR(use_doc_orientation=False, use_doc_unwarping=False)
```

#### 3. NPU 초기화 (Sync)

```python
def get_npu_ocr_instance_sync():
    """DEEPX NPU OCR 엔진 초기화 (Sync)"""
    import sys
    sys.path.insert(0, str(DEEPX_ENGINE_PATH))
    from engine.paddleocr import PaddleOcr
    return PaddleOcr(use_doc_orientation=False, use_doc_unwarping=False)
```

#### 4. 분기 로직

```python
if request.deepx:
    if request.sync:
        # NPU Sync 모드
        result = process_image_with_npu_sync(img_np, request)
    else:
        # NPU Async 모드
        result = process_image_with_npu(img_np, request)
else:
    # CPU 모드
    result = process_image_with_cpu(img_np, request)
```

### 포팅 요약

| 항목 | deepx | FastAPI 포팅 | 파일 |
|------|-------|--------------|------|
| 환경 설정 | startup.sh | local_deepx_setup.sh | ✅ |
| DX_RT 빌드 | build.sh | 자동 호출 | ✅ |
| 의존성 | requirements.txt | 스크립트 내장 | ✅ |
| 모델 확인 | - | 경로 검증 | ✅ |
| RT 최적화 | set_env.sh | deepx_env.sh | ✅ |
| 자동 적용 | 수동 source | run.sh 자동 | ✅ |
| NPU 초기화 | Python 코드 | ocr_service.py | ✅ |
| Async/Sync | AsyncPipelineOCR/PaddleOcr | 둘 다 지원 | ✅ |
| 버전 관리 | requirements.txt | 명시적 버전 | ✅ |

### 주요 기능

✅ **구현 완료**
1. `/api/v1/ocr` 엔드포인트에 `deepx` 파라미터 추가
2. `deepx: true` → DEEPX NPU 사용
3. `deepx: false` 또는 생략 → CPU 사용
4. `sync: true` → Sync 모드 (PaddleOcr)
5. `sync: false` 또는 생략 → Async 모드 (AsyncPipelineOCR)
6. CPU/NPU 분기 로직 구현
7. NPU Async/Sync 초기화 및 추론 함수 구현
8. 자동 환경 설정 (deepx_env.sh)
9. RT 최적화 환경 변수 적용

모든 deepx의 NPU 관련 설정이 FastAPI 서비스에 완전히 포팅되었습니다! 🎉
