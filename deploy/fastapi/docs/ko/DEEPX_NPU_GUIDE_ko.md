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
- ✅ python 3.10+ venv 생성 및 활성화
- ✅ DX_RT 빌드 (dx-engine 자동 설치)
- ✅ PaddlePaddle 3.0.0 설치
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
./run.sh --ocr-version v6 --model-size medium
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
    "device": "npu"
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
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --python python3.10
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
| `--python VERSION` | Python 버전 지정 | python3.10 |
| `--version VERSION` | PaddleOCR 버전 지정 | 3.3.2 |
| `--inter-threads N` | CUSTOM_INTER_OP_THREADS_COUNT | 1 |
| `--intra-threads N` | CUSTOM_INTRA_OP_THREADS_COUNT | 3 |

---

## 구성 요소

### 1. 필수 Python 패키지

- [cpu requirement](../requirements.txt)
- [gpu requirement](../requirements-gpu.txt)

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

**설정 파일:**
- `.env.deepx`: 기본 RT 최적화 값을 저장하는 마스터 설정 파일
- `deepx_env.sh`: `.env.deepx`에서 기본값을 읽어오는 자동 생성 스크립트

**생성되는 파일 (.env.deepx):**
```bash
# DEEPX NPU Environment Variables
# Auto-generated from deepx_env.sh
# Used by VS Code debugger launch configurations
# Default values: 1 2 1 3 2 4

# RT Optimization Settings (Default values from deepx_env.sh)
CUSTOM_INTER_OP_THREADS_COUNT=1
CUSTOM_INTRA_OP_THREADS_COUNT=2
DXRT_DYNAMIC_CPU_THREAD=1
DXRT_TASK_MAX_LOAD=3
NFH_INPUT_WORKER_THREADS=2
NFH_OUTPUT_WORKER_THREADS=4
```

**생성되는 파일 (deepx_env.sh):**
```bash
#!/bin/bash
# DEEPX NPU Environment Configuration
# Source this file before running the FastAPI service with NPU support
# Usage: source ./deepx_env.sh [CUSTOM_INTER_OP_THREADS_COUNT] [CUSTOM_INTRA_OP_THREADS_COUNT] [DXRT_DYNAMIC_CPU_THREAD] [DXRT_TASK_MAX_LOAD] [NFH_INPUT_WORKER_THREADS] [NFH_OUTPUT_WORKER_THREADS]
# Example: source ./deepx_env.sh 1 2 1 3 2 4
# Default values (from .env.deepx): 1 2 1 3 2 4

# Set default values (loaded from .env.deepx)
DEFAULT_CUSTOM_INTER_OP_THREADS_COUNT=1
DEFAULT_CUSTOM_INTRA_OP_THREADS_COUNT=2
DEFAULT_DXRT_DYNAMIC_CPU_THREAD=1
DEFAULT_DXRT_TASK_MAX_LOAD=3
DEFAULT_NFH_INPUT_WORKER_THREADS=2
DEFAULT_NFH_OUTPUT_WORKER_THREADS=4

if [ "$1" = "-1" ]; then
    unset CUSTOM_INTER_OP_THREADS_COUNT
elif [ -n "$1" ]; then
    export CUSTOM_INTER_OP_THREADS_COUNT=$1
else
    export CUSTOM_INTER_OP_THREADS_COUNT=$DEFAULT_CUSTOM_INTER_OP_THREADS_COUNT
fi
# ... (다른 변수들도 동일한 로직)

# Library paths
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
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
| `CUSTOM_INTER_OP_THREADS_COUNT` | 1 | Inter-op 스레드 수 | .env.deepx |
| `CUSTOM_INTRA_OP_THREADS_COUNT` | 2 | Intra-op 스레드 수 | .env.deepx |
| `DXRT_DYNAMIC_CPU_THREAD` | 1 | 동적 CPU 스레드 | .env.deepx |
| `DXRT_TASK_MAX_LOAD` | 3 | 최대 작업 부하 | .env.deepx |
| `NFH_INPUT_WORKER_THREADS` | 2 | 입력 워커 스레드 | .env.deepx |
| `NFH_OUTPUT_WORKER_THREADS` | 4 | 출력 워커 스레드 | .env.deepx |
| `LD_LIBRARY_PATH` | ${VIRTUAL_ENV}/lib:... | 라이브러리 경로 | deepx_env.sh |

**기본값 (1 2 1 3 2 4):**
모든 기본값은 `.env.deepx`에 정의되며 `deepx_env.sh`가 자동으로 로드합니다.

### 커스터마이징

RT 최적화 값을 3가지 방법으로 커스텀할 수 있습니다:

```bash
# 방법 1: .env.deepx 편집 (권장 - 모든 스크립트에 영향)
vi .env.deepx
# 그 다음 setup 실행하여 deepx_env.sh 재생성
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# 방법 2: 설치 시 옵션 지정
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --inter-threads 2 --intra-threads 4

# 방법 3: deepx_env.sh source 시 파라미터 전달
source deepx_env.sh 1 2 1 3 2 4
```

**참고:** 방법 1이 권장됩니다. 모든 실행 환경(run.sh, VS Code 디버거 등)에서 일관성을 보장합니다.

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
./run.sh --ocr-version v6 --model-size medium
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
    "device": "npu"
  }'
```

#### NPU 모드 (Sync)

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1,
    "device": "npu",
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
        "device": "npu"
    }
)

# NPU OCR (Sync)
response = requests.post(
    "http://localhost:8080/api/v1/ocr",
    json={
        "file": img_base64,
        "fileType": 1,
        "device": "npu",
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
./run.sh --ocr-version v6 --model-size medium

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
python 3.10+ is REQUIRED for DEEPX NPU
```

**해결:**
```bash
# python 3.10 설치
sudo apt install python3.10

# 또는 python 3.10로 재설치
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --python python3.10
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

### 4. dx-engine 미설치

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

### 5. deepx 경로 오류

**증상:**
```
❌ deepx not found at /dataPaddleOCR/deepx
```

**해결:**
```bash
# 경로 확인
ls -la /dataPaddleOCR/deepx/engine/

# 심볼릭 링크 확인
ls -la /dataPaddleOCR/deepx
```

### 6. DEEPX 모델 없음

**증상:**
```
⚠ DEEPX models not found or incomplete
```

**해결:**
```bash
# 모델 확인
ls -la /dataPaddleOCR/deepx/engine/model_files/server/
ls -la /dataPaddleOCR/deepx/engine/model_files/mobile/

# 모델이 없으면 deepx/engine/model_files에 배치 필요
```

### 7. RT 최적화 미적용

**증상:**
- NPU 성능이 예상보다 낮음

**해결:**
```bash
# 환경 변수 확인
echo $CUSTOM_INTER_OP_THREADS_COUNT  # 1
echo $CUSTOM_INTRA_OP_THREADS_COUNT  # 2

# 수동 적용 (.env.deepx의 기본값 사용)
source deepx_env.sh

# 또는 커스텀 값 지정
source deepx_env.sh 1 2 1 3 2 4

# 서비스 재시작
./run.sh --ocr-version v6 --model-size medium
```

### 8. Import 에러

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
deploy/fastapi/
├── ocr_service.py           # NPU 지원 추가
├── local_deepx_setup.sh     # NPU 자동 설치 스크립트
├── .env.deepx               # RT 최적화 기본값 (마스터 설정)
├── deepx_env.sh             # RT 최적화 스크립트 (.env.deepx에서 자동 생성)
├── run.sh                   # 서비스 시작 (deepx_env.sh 자동 적용)
└── docs/
    ├── DEEPX_NPU_GUIDE.md       # 영문 가이드
    └── ko/
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

## 사내망: SSLError 로 모델 다운로드 실패

TLS 를 종단하는 사내망(해당 CA 가 시스템 신뢰 저장소에는 있으나 Python 의 `certifi`
번들에는 없는 환경)에서는 CPU 경로가 기동 중 실패합니다:

```
No model hoster is available! Please check your network connection to one of
the following model hoster: HuggingFace, ModelScope, AIStudio, or BOS.
...
Exception: No available model hosting platforms detected.
```

같은 호스트에 `curl` 은 성공하기 때문에 네트워크 장애처럼 보이지만, 실제로는 신뢰
저장소 불일치입니다. `requests` 가 시스템 저장소를 보게 하면 해결됩니다:

```bash
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

CPU 경로에만 해당합니다 — NPU 경로는 로컬 `.dxnn` 파일을 읽으므로 model hoster 가
필요 없습니다.

## CPU 경로에서의 PP-OCRv6

CPU 쪽 v6 에는 `paddleocr>=3.7.0` 이 필요합니다. 3.3.2 에는 PP-OCRv6 config 가
존재하지 않습니다. `paddlepaddle` 도 `>=3.2.2` 여야 합니다 — 3.0.0 은 새 모델 파일을
읽지 못하고 `Type of attribute: strides is not right` 로 실패합니다.

```python
PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det",
          text_recognition_model_name="PP-OCRv6_medium_rec")
```

크기: `PP-OCRv6_{tiny,small,medium}_{det,rec}`.

CPU 첫 추론은 모델을 내려받느라 수 분이 걸릴 수 있습니다. 서비스 테스트 스위트의
테스트당 300초 타임아웃이 이 콜드 스타트에서 걸릴 수 있습니다.

## 모델 버전·크기 선택

`run.sh` 와 `python ocr_service.py` 모두 버전과 크기를 **함께** 받습니다.

```bash
./run.sh --ocr-version v6 --model-size medium   # v6: medium | small | tiny
./run.sh --ocr-version v5 --model-size server   # v5: server | mobile
./run.sh                                        # 둘 다 생략 -> 대화형 메뉴
```

하나만 지정하면 에러입니다. 터미널이 아닌 환경(Docker, CI)에서는 모델이 지정되지
않으면 기동을 거부합니다 — 어떤 가중치로 서빙 중인지 불분명한 상태를 만들지 않기
위해서입니다. Docker 는 `OCR_VERSION` / `MODEL_SIZE` 환경변수로 주입합니다.

> v6 의 `s` 는 **small**(server 아님), `m` 은 **medium**(mobile 아님) 입니다.

## 연산 장치 선택

장치는 서버 단위가 아니라 **요청 단위**로 선택합니다. 한 서버가 CPU·GPU·NPU 요청을
동시에 처리할 수 있습니다:

```jsonc
{"file": "...", "device": "npu"}   // cpu | gpu | npu
{"file": "..."}                    // 생략 -> 자동: npu > gpu > cpu
```

이 배포가 제공할 수 없는 장치를 지정하면 **503** 과 함께 사용 가능한 장치 목록을
돌려줍니다. 조용히 내려가지 않습니다 — npu 에서 cpu 로 말없이 떨어지면 그저 느린
서버처럼 보일 뿐이기 때문입니다.

모든 응답에 실제 실행 장치가 담깁니다:

```jsonc
{"device_used": "npu", "device_requested": null, ...}
```

`USE_GPU` 는 **제거되었습니다**. 프로세스 전역 설정이라 "이 서버는 NPU 를 쓰지만 이
요청만 GPU 로" 를 표현할 수 없었습니다. 설정되어 있으면 deprecated 안내를 출력하고
무시합니다.

`deepx` 는 **deprecated** 이지만 계속 동작합니다: `true` 는 `device="npu"`, `false` 는
자동 선택으로 매핑되며 각각 안내를 출력합니다.

### 사용 가능한 장치 확인

```bash
./run.sh --sanity-check
```

설치된 패키지, 사용 가능한 장치(불가한 경우 사유와 해결 방법), 다운로드된 모델을
보고합니다.

### GPU 지원 현황

2026-09-15 에 실제 추론으로 검증했습니다 (RTX 5060 Ti, CUDA 13.0, Python 3.12).
PP-OCRv6 는
`paddlepaddle >= 3.2.2` 가 필요한데, `paddlepaddle-gpu` 3.x 는 PyPI 가 아니라 Paddle
자체 휠 인덱스에 배포됩니다 — PyPI 는 2.6.2 에서 멈춰 있고 이는 PP-OCRv6 이전
버전입니다. 사내망에서 해당 인덱스가 막혀 있으면 GPU 지원을 설치할 수 없으며,
`--sanity-check` 가 CPU 전용 빌드로 보고합니다.

### GPU 설치 (2026-09-15 검증)

`paddlepaddle-gpu` 3.x 는 PyPI 에 없습니다. Paddle 자체 인덱스에서 설치합니다:

```bash
pip install -r requirements-gpu.txt \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
python -c "import paddle; paddle.utils.run_check()"   # "works well on 1 GPU"
```

`cuXXX` 는 `nvidia-smi` 의 CUDA 버전에 맞춥니다. `paddlepaddle_gpu` 휠만 offline 으로
받아서는 부족합니다 — `nvidia-*` CUDA 런타임 패키지를 함께 요구하며 이들은 PyPI 에서
받습니다.

RTX 5060 Ti (CUDA 13.0, paddlepaddle-gpu 3.2.2) 에서 이 서비스의 실제 코드 경로로,
warm 상태 1장 3회 중 최소값 실측:

| 선택 | GPU | CPU |
|---|---|---|
| v5 / server | 0.093 s | 1.425 s |
| v6 / medium | 0.051 s | 0.921 s |
| v6 / small | 0.036 s | 0.732 s |
| v6 / tiny | 0.034 s | 0.815 s |

CPU 열도 같은 GPU 설치본에서 측정한 값입니다. 3.2.2 는 하나의 환경에서 `cpu`, `gpu`,
`npu` 를 모두 제공하며, 이는 자동 fallback (npu → gpu → cpu) 이 전제하는 조건입니다.

> **paddlepaddle-gpu 3.3.0 으로 올리지 마십시오.** GPU 추론은 되지만 `device: "cpu"` 가
> `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support` 로 실패합니다.
> 이 서비스 없이 순수 paddleocr 로도 재현되므로 Paddle 쪽 regression 이며 3.2.2 는
> 영향이 없습니다. `requirements-gpu.txt` 가 3.2.2 로 고정된 이유입니다.

> **`import paddle` 이 `ImportError: libnvrtc.so.13` 으로 실패한다면?** 3.2.2 의
> `libpaddle.so` 는 구형 분리 CUDA 레이아웃(`nvidia/cuda_nvrtc/lib`) 기준 RUNPATH 를
> 갖는데, CUDA 13 휠은 전부 `nvidia/cu13/lib` 아래에 설치됩니다. `run.sh` 는 이 경로를
> `LD_LIBRARY_PATH` 에 자동으로 추가합니다. `run.sh` 밖에서는 직접 지정하십시오:
> ```bash
> export LD_LIBRARY_PATH="$(python -c 'import nvidia,os;print(os.path.dirname(nvidia.__file__))')/cu13/lib:$LD_LIBRARY_PATH"
> ```
