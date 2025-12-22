# PaddleOCR FastAPI OCR Service

PaddlePaddle 3.0 기반의 FastAPI를 사용한 OCR REST API 서비스입니다. PP-OCRv5 모델을 사용합니다.

## 🎯 빠른 실행 가이드

### 로컬 환경 (개발/테스트)
```bash
cd PaddleOCR/deploy/docker/fastapi

# 1. 환경 설정 (최초 1회만)
./local_setup.sh

# 2. 서버 실행
./run.sh
```

### 로컬 환경 with DEEPX NPU (하드웨어 가속)
```bash
cd PaddleOCR/deploy/docker/fastapi

# 1. NPU 환경 설정 (최초 1회만)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# 2. 서버 실행 (자동으로 NPU 설정 적용)
./run.sh
```

> **참고**: DEEPX NPU 하드웨어 가속을 사용하려면 [DEEPX NPU 지원 가이드](docs/ko/DEEPX_NPU_GUIDE_ko.md)를 참고하세요

### Docker 환경 (운영/배포)
```bash
cd PaddleOCR/deploy/docker/fastapi

# 빌드 + 실행을 한 번에
./docker_run.sh
```

---

## 🚀 상세 가이드

### 방법 1: 로컬 환경에서 실행

#### 1.1 환경 설정 (local_setup.sh)

**기본 설정 (CPU + Mobile 모델):**
```bash
cd PaddleOCR/deploy/docker/fastapi
chmod +x local_setup.sh
./local_setup.sh
```

**GPU 버전 (Mobile 모델):**
```bash
./local_setup.sh --gpu
```

**Server 모델 사용 (고정밀):**
```bash
./local_setup.sh --use-server
```

**GPU + Server 모델:**
```bash
./local_setup.sh --gpu --use-server
```

**모델 다운로드 생략 (런타임에 다운로드):**
```bash
./local_setup.sh --no-models
```

#### 1.1.1 DEEPX NPU 설정 (local_deepx_setup.sh)

**DEEPX NPU 하드웨어 가속을 사용하려면** `local_setup.sh` 대신 `local_deepx_setup.sh`를 사용하세요:

```bash
# 기본 NPU 설정 (필수: dx_rt 경로)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# NPU + Mobile 모델
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --use-mobile

# NPU + Server 모델
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --use-server
```

> **📖 전체 NPU 가이드**: 자세한 내용은 [DEEPX NPU 지원 가이드](docs/ko/DEEPX_NPU_GUIDE_ko.md)를 참고하세요:
> - 자동 설치 및 설정
> - RT 최적화 구성
> - NPU 모델 관리
> - 성능 튜닝
> - 문제 해결

#### 1.2 서버 실행 (run.sh)

**간편한 실행:**
```bash
./run.sh
```

**커스텀 포트:**
```bash
./run.sh --port 9000
```

**환경 변수 사용:**
```bash
PORT=9000 USE_GPU=true ./run.sh
```

**수동 실행 (가상환경 직접 사용):**

```bash
# 가상환경 활성화
source venv/bin/activate

# 서버 실행
python ocr_service.py
```

또는 가상환경 활성화 없이:
```bash
venv/bin/python ocr_service.py
```

서비스는 http://localhost:8080 에서 실행됩니다.
- **API 문서**: http://localhost:8080/docs

#### 1.3 local_setup.sh 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--gpu` | GPU 버전 PaddlePaddle 설치 | CPU |
| `--use-mobile` | Mobile 모델 사용 (빠름, 작음) | Mobile (기본값) |
| `--use-server` | Server 모델 사용 (고정밀) | - |
| `--no-models` | 모델 다운로드 생략 | 다운로드 |
| `--python VERSION` | Python 버전 지정 | python3.10 |
| `--version VERSION` | PaddleOCR 버전 지정 | 3.3.2 |
| `--help` | 도움말 표시 | - |

#### 1.4 run.sh 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--port PORT` | 서비스 포트 | 8080 |
| `--host HOST` | 바인딩 호스트 | 0.0.0.0 |
| `--use-mobile` | Mobile 모델 사용 | Mobile (기본값) |
| `--use-server` | Server 모델 사용 | - |
| `--help` | 도움말 표시 | - |

**환경 변수:**
- `PORT`: 서비스 포트
- `HOST`: 바인딩 호스트  
- `USE_GPU`: GPU 사용 여부 (true/false)
- `USE_MOBILE`: Mobile 모델 사용 여부 (true/false)

#### 1.5 Server vs Mobile 모델

| 모델 타입 | 정확도 | 속도 | 파일 크기 | 사용 권장 |
|---------|-------|------|----------|----------|
| **Mobile** (기본) | 중간 ⭐⭐ | 빠름 | 작음 | 실시간, 엣지 디바이스, 일반 용도 |
| **Server** | 높음 ⭐⭐⭐ | 느림 | 큼 | 서버, 고정밀 OCR |

---

### 방법 2: Docker 컨테이너로 실행

#### 2.1 빌드 + 실행 한 번에 (docker_run.sh) ⭐ 권장

**기본 실행 (CPU + Mobile 모델):**
```bash
cd PaddleOCR/deploy/docker/fastapi
chmod +x docker_run.sh
./docker_run.sh
```

이미지가 없으면 자동으로 빌드한 후 컨테이너를 실행합니다!

**GPU 버전 (Mobile 모델):**
```bash
./docker_run.sh --gpu
```

**Server 모델 사용 (고정밀):**
```bash
./docker_run.sh --use-server
```

**GPU + Server 모델:**
```bash
./docker_run.sh --gpu --use-server
```

**DEEPX NPU 지원 (하드웨어 가속):**
```bash
./docker_run.sh --deepx
```

**DEEPX NPU + GPU:**
```bash
./docker_run.sh --gpu --deepx
```

**커스텀 포트:**
```bash
./docker_run.sh --port 9000
```

#### 2.2 수동 빌드 (docker_build.sh)

**CPU 버전 (모델 포함):**
```bash
cd PaddleOCR/deploy/docker/fastapi
chmod +x docker_build.sh
./docker_build.sh
```

**GPU 버전 (CUDA 11.8, Mobile 모델):**
```bash
./docker_build.sh --gpu
```

**Server 모델 사용 (고정밀):**
```bash
./docker_build.sh --use-server
```

**GPU + Server 모델:**
```bash
./docker_build.sh --gpu --use-server
```

**모델 제외 (런타임에 다운로드):**
```bash
./docker_build.sh --no-models
```

**DEEPX NPU 지원 (하드웨어 가속):**
```bash
./docker_build.sh --deepx
```

**DEEPX NPU + GPU:**
```bash
./docker_build.sh --gpu --deepx
```

#### 2.3 수동 컨테이너 실행

**CPU 버전:**
```bash
docker run -d -p 8081:8080 --name ocr-fastapi paddleocr-fastapi-service:latest
```

**GPU 버전:**
```bash
docker run -d --gpus all -p 8081:8080 --name ocr-fastapi paddleocr-fastapi-service:latest-gpu
```

> **참고**: 로컬 실행방식과 포트 충돌을 피하기 위해 8081 포트를 사용합니다. 로컬은 8080, 컨테이너는 8081 포트를 사용하여 두 서비스를 동시에 비교할 수 있습니다.

**환경 변수 설정:**
```bash
docker run -d -p 8081:8080 \
  -e USE_GPU=false \
  -e PORT=8080 \
  -e HOST=0.0.0.0 \
  --name ocr-fastapi \
  paddleocr-fastapi-service:latest
```

#### 2.4 docker_run.sh 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--gpu` | GPU 모드로 실행 | CPU |
| `--use-mobile` | Mobile 모델 사용 (이미지 없으면 빌드) | Mobile (기본값) |
| `--use-server` | Server 모델 사용 (이미지 없으면 빌드) | - |
| `--deepx` | DEEPX NPU 지원 활성화 (이미지 없으면 빌드) | 비활성화 |
| `--port PORT` | 호스트 포트 | 8081 |
| `--name NAME` | 컨테이너 이름 | ocr-fastapi |
| `--image IMAGE` | 이미지 이름 | paddleocr-fastapi-service |
| `--tag TAG` | 이미지 태그 | latest (GPU: latest-gpu) |
| `--help` | 도움말 표시 | - |

**주요 기능:**
- ✅ 이미지 자동 빌드 (없을 때만)
- ✅ 기존 컨테이너 자동 정리
- ✅ 실행 후 상태 확인
- ✅ 유용한 명령어 안내

#### 2.5 docker_build.sh 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--gpu` | GPU 지원 활성화 | CPU |
| `--use-mobile` | Mobile 모델 사용 (빠름, 작음) | Mobile (기본값) |
| `--use-server` | Server 모델 사용 (고정밀) | - |
| `--deepx` | DEEPX NPU 지원 활성화 (하드웨어 가속) | 비활성화 |
| `--no-models` | 빌드 시 모델 다운로드 생략 | 다운로드 |
| `--version VERSION` | PaddleOCR 버전 지정 | 3.3.2 |
| `--tag NAME` | Docker 이미지 이름 | paddleocr-fastapi-service |
| `--tag-version VER` | 이미지 버전 태그 | latest |

**빌드 예제:**
```bash
# CPU + Mobile 모델 (기본)
./docker_build.sh

# GPU + Mobile 모델 (기본 GPU)
./docker_build.sh --gpu

# CPU + Server 모델 (고정밀)
./docker_build.sh --use-server

# GPU + Server 모델 (고정밀 GPU)
./docker_build.sh --gpu --use-server

# 모델 제외 빌드 (이미지 크기 축소)
./docker_build.sh --no-models

# 커스텀 태그
./docker_build.sh --tag my-ocr --tag-version v1.0
```

---

## 📋 API 사용법

### API 문서 확인

FastAPI는 자동으로 대화형 API 문서를 제공합니다:

**로컬 환경:**
- **Swagger UI**: http://localhost:8080/docs

**Docker 환경 (포트 8081 매핑 시):**
- **Swagger UI**: http://localhost:8081/docs

### API 엔드포인트 예제

> **포트 주의**: 로컬 환경은 8080, Docker는 8081 포트를 사용하는 예제입니다. 실제 사용 중인 포트에 맞게 변경하세요.

**Health Check:**
```bash
# 로컬 환경
curl http://localhost:8080/health

# Docker 환경
curl http://localhost:8081/health
```

**OCR (Hubserving 호환 형식) - `/predict/ocr_system`:**
```bash
# images 배열 형식 (hubserving과 동일)
IMAGE_BASE64=$(base64 -w 0 your_image.jpg)

# 로컬 환경
curl -X POST http://localhost:8080/predict/ocr_system \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMAGE_BASE64\"]}"

# Docker 환경
curl -X POST http://localhost:8081/predict/ocr_system \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMAGE_BASE64\"]}"
```

**OCR (URL) - `/ocr`:**
```bash
# 로컬 환경
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
  }'

# Docker 환경
curl -X POST http://localhost:8081/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
  }'
```

**OCR (Base64 이미지):**
```bash
# 이미지를 base64로 인코딩
IMAGE_BASE64=$(base64 -w 0 your_image.jpg)

# 로컬 환경
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\"}"

# Docker 환경
curl -X POST http://localhost:8081/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\"}"
```

**OCR (파일 업로드):**
```bash
# 로컬 환경
curl -X POST http://localhost:8080/ocr \
  -F "file=@/path/to/image.jpg"

# Docker 환경
curl -X POST http://localhost:8081/ocr \
  -F "file=@/path/to/image.jpg"
```

**Batch OCR:**
```bash
# 로컬 환경
curl -X POST http://localhost:8080/batch_ocr \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["'$IMAGE_BASE64_1'", "'$IMAGE_BASE64_2'"]
  }'

# Docker 환경
curl -X POST http://localhost:8081/batch_ocr \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["'$IMAGE_BASE64_1'", "'$IMAGE_BASE64_2'"]
  }'
```

---

## 📋 API 엔드포인트

### GET /health
서비스 상태 확인

**응답:**
```json
{
  "status": "healthy"
}
```

### POST /ocr
단일 이미지 OCR

**요청 파라미터 (택일):**
- `url`: 이미지 URL (JSON body)
- `image`: Base64 인코딩된 이미지 (JSON body)
- `file`: 멀티파트 파일 업로드 (form-data)

**JSON 요청 예시:**
```json
{
  "url": "https://example.com/image.jpg"
}
```
또는
```json
{
  "image": "base64_encoded_image_string"
}
```

**응답:**
```json
{
  "success": true,
  "results": [
    {
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "text": "인식된 텍스트",
      "confidence": 0.95
    }
  ]
}
```

### POST /batch_ocr
여러 이미지 일괄 OCR

**요청:**
```json
{
  "images": ["base64_image_1", "base64_image_2", ...]
}
```

**응답:**
```json
{
  "success": true,
  "results": [
    [/* image 1 results */],
    [/* image 2 results */]
  ]
}
```

## 🎯 FastAPI의 장점

### 1. 자동 API 문서화
- **Swagger UI** (`/docs`): 대화형 API 테스트 가능

### 2. 데이터 검증
- Pydantic 모델을 통한 자동 요청/응답 검증
- 타입 안정성 보장
- 자동 오류 응답 생성

### 3. 성능
- 비동기(async) 지원으로 높은 처리량
- Uvicorn ASGI 서버 사용

### 4. 개발자 경험
- 타입 힌트 지원
- IDE 자동완성
- 명확한 에러 메시지

## 🔧 빌드 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--gpu` | GPU 지원 활성화 | CPU |
| `--no-models` | 빌드 시 모델 다운로드 생략 | 다운로드 |
| `--version VERSION` | PaddleOCR 버전 지정 | 3.3.2 |
| `--tag NAME` | Docker 이미지 이름 | paddleocr-fastapi-service |
| `--tag-version VER` | 이미지 버전 태그 | latest |

## 🐍 Python 클라이언트 예제

```python
import requests
import base64

# 이미지 파일을 base64로 인코딩
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# OCR 요청 (로컬 환경)
response = requests.post(
    "http://localhost:8080/ocr",
    json={"image": image_base64}
)

# OCR 요청 (Docker 환경)
# response = requests.post(
#     "http://localhost:8081/ocr",
#     json={"image": image_base64}
# )

# 결과 출력
result = response.json()
if result['success']:
    for item in result['results']:
        print(f"Text: {item['text']}")
        print(f"Confidence: {item['confidence']:.2f}")
        print(f"BBox: {item['bbox']}")
        print("-" * 50)
```

### OpenAPI 클라이언트 자동 생성

FastAPI는 OpenAPI 스펙을 제공하므로, 다양한 언어의 클라이언트를 자동 생성할 수 있습니다:

```bash
# OpenAPI 스펙 다운로드 (로컬 환경)
curl http://localhost:8080/openapi.json > openapi.json

# OpenAPI 스펙 다운로드 (Docker 환경)
curl http://localhost:8081/openapi.json > openapi.json

# Python 클라이언트 생성 (openapi-generator 사용)
openapi-generator generate -i openapi.json -g python -o ./client
```

---

## 🔧 환경별 정보

### 로컬 환경

- **베이스**: Python 3.10+ venv
- **웹 프레임워크**: FastAPI 0.109.0
- **ASGI 서버**: Uvicorn 0.27.0
- **PaddlePaddle**: 3.0.0
- **PaddleOCR**: 3.3.2 (기본값)
- **모델**: PP-OCRv5 server/mobile (선택 가능)
- **포트**: 8080 (기본값)
- **모델 저장 위치**: `~/.paddlex/official_models/`

### Docker 환경

- **베이스 이미지**: python:3.10-slim
- **웹 프레임워크**: FastAPI 0.109.0
- **ASGI 서버**: Uvicorn 0.27.0
- **PaddlePaddle**: 3.0.0
- **PaddleOCR**: 3.3.2 (기본값)
- **모델**: PP-OCRv5 server/mobile (빌드 시 선택 가능)
- **포트**: 8080 (컨테이너 내부), 8081 (호스트 매핑 권장)
- **모델 저장 위치**: `/home/paddleocr/.paddlex/official_models/`

---

## ⚙️ 환경 변수

### 로컬 환경
로컬에서는 `ocr_service.py` 파일에서 직접 설정하거나 환경 변수로 설정:
```bash
export USE_GPU=false
export PORT=8080
export HOST=0.0.0.0
python ocr_service.py
```

### Docker 환경

- `USE_GPU`: GPU 사용 여부 (true/false, 기본값: false)
- `PORT`: 서비스 포트 (기본값: 8080)
- `HOST`: 바인딩 호스트 (기본값: 0.0.0.0)

---

## 🆚 비교

### 로컬 vs Docker 실행

| 항목 | 로컬 환경 | Docker 환경 |
|------|----------|-------------|
| **설정 속도** | 빠름 (venv 생성) | 느림 (이미지 빌드) |
| **의존성 관리** | venv로 격리 | 완전 격리 |
| **디버깅** | 쉬움 | 중간 |
| **배포** | 수동 설정 필요 | 이미지 배포만으로 완료 |
| **리소스** | 낮음 | 약간 높음 (컨테이너 오버헤드) |
| **개발 권장** | ⭐⭐⭐ | ⭐⭐ |
| **운영 권장** | ⭐⭐ | ⭐⭐⭐ |

---

## 📝 주의사항

### 로컬 환경

1. **시스템 의존성 설치 필요:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1 libglib2.0-0 libgomp1
   ```

2. **Python 버전:**
   - Python 3.10 이상 권장
   - 다른 버전 사용 시: `./setup.sh --python python3.10`

3. **메모리 요구사항:**
   - CPU + Server 모델: 최소 2GB RAM
   - CPU + Mobile 모델: 최소 1.5GB RAM
   - GPU: 최소 4GB VRAM

4. **모델 저장 위치:**
   - `~/.paddlex/official_models/`
   - 디스크 공간: Server 모델 ~500MB, Mobile 모델 ~200MB

### Docker 환경

1. GPU 버전 사용 시 NVIDIA Docker Runtime 필요:
   ```bash
   # nvidia-docker2 설치 확인
   docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. 메모리 요구사항:
   - CPU: 최소 2GB RAM
   - GPU: 최소 4GB VRAM

3. 이미지 크기 제한:
   - 기본적으로 FastAPI는 큰 파일 업로드를 지원하지만,
   - 필요시 `MAX_UPLOAD_SIZE` 환경 변수로 조정 가능

---

## 🔍 모니터링 및 로깅

### 로컬 환경
```bash
# 서버 실행 시 로그가 터미널에 직접 출력됨
python ocr_service.py

# 백그라운드 실행 + 로그 파일
nohup python ocr_service.py > ocr_service.log 2>&1 &

# 로그 확인
tail -f ocr_service.log
```

### Docker 환경

```bash
# 컨테이너 로그 확인
docker logs -f ocr-fastapi

# 로그 레벨 조정 (uvicorn)
docker run -d -p 8081:8080 \
  -e LOG_LEVEL=debug \
  paddleocr-fastapi-service:latest
```

---

## 🚀 성능 최적화

### 로컬 환경

**1. 다중 워커 실행 (Gunicorn + Uvicorn):**
```bash
# gunicorn 설치
pip install gunicorn

# 4개 워커로 실행
gunicorn ocr_service:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080
```

**2. Mobile 모델 사용:**
```bash
# 빠른 추론이 필요한 경우
./local_setup.sh --use-mobile
```

**3. GPU 활용:**
```bash
./local_setup.sh --gpu
```

### Docker 환경
```bash
# Dockerfile 수정하여 gunicorn + uvicorn workers 사용
CMD ["gunicorn", "ocr_service:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080"]
```

### 비동기 처리 (공통)
현재 구현은 동기 방식이지만, 필요시 비동기로 전환 가능:
```python
@app.post('/ocr')
async def ocr_image(...):
    result = await asyncio.to_thread(ocr.predict, img)
```

---

## 🎓 사용 시나리오별 권장사항

| 시나리오 | 권장 방법 | 명령어 |
|---------|----------|--------|
| **개발/테스트** | 로컬 + Server 모델 | `./local_setup.sh && ./run.sh` |
| **빠른 프로토타입** | 로컬 + Mobile 모델 | `./local_setup.sh --use-mobile && ./run.sh` |
| **고정밀 OCR** | 로컬/Docker + Server + GPU | `./local_setup.sh --gpu` 또는 `./docker_run.sh --gpu` |
| **실시간 처리** | 로컬/Docker + Mobile + GPU | `./local_setup.sh --gpu --use-mobile` 또는 `./docker_run.sh --gpu --use-mobile` |
| **운영 배포** | Docker + Server 모델 | `./docker_run.sh` |
| **엣지/임베디드** | 로컬/Docker + Mobile + CPU | `./local_setup.sh --use-mobile` 또는 `./docker_run.sh --use-mobile` |

---

## 📚 추가 리소스

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Uvicorn 문서](https://www.uvicorn.org/)
- [Pydantic 문서](https://docs.pydantic.dev/)
- [PaddleOCR 공식 문서](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 모델 정보](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/ppocr_introduction.md)

---

## � 스크립트 파일 구조

```
deploy/docker/fastapi/
├── local_setup.sh      # 로컬 환경 설정 (venv 생성, 의존성 설치)
├── run.sh              # 로컬 서버 실행
├── docker_build.sh     # Docker 이미지 빌드
├── docker_run.sh       # Docker 빌드 + 실행 (자동화)
├── ocr_service.py      # FastAPI 서버 코드
├── Dockerfile          # Docker 이미지 정의
└── README.md           # 이 문서
```

### 스크립트 선택 가이드

| 목적 | 사용 스크립트 | 설명 |
|------|-------------|------|
| **로컬 최초 설정** | `local_setup.sh` | Python venv 생성 및 모든 의존성 설치 |
| **로컬 서버 실행** | `run.sh` | 설정된 환경에서 OCR 서비스 시작 |
| **Docker 이미지 빌드** | `docker_build.sh` | Docker 이미지만 빌드 |
| **Docker 서비스 실행** | `docker_run.sh` ⭐ | 이미지 자동 빌드 + 컨테이너 실행 (권장) |

---

## �🔧 문제 해결 (Troubleshooting)

### 로컬 환경

**문제: ImportError - paddle 관련**
```bash
# 가상환경이 활성화되었는지 확인
which python  # venv/bin/python이어야 함

# PaddlePaddle 재설치
pip install --force-reinstall paddlepaddle==3.0.0
```

**문제: 모델 다운로드 실패**
```bash
# 수동으로 모델 다운로드 위치 확인
ls ~/.paddlex/official_models/

# local_setup.sh를 모델 다운로드 없이 실행 후 런타임에 자동 다운로드
./local_setup.sh --no-models
```

**문제: 포트 이미 사용 중**
```bash
# 8080 포트 사용 중인 프로세스 확인
lsof -i :8080

# 다른 포트로 실행
PORT=8888 python ocr_service.py
```

### Docker 환경

**문제: GPU 인식 안 됨**
```bash
# NVIDIA Docker 런타임 확인
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# nvidia-docker2 설치
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**문제: 컨테이너가 시작되지 않음**
```bash
# 로그 확인
docker logs ocr-fastapi

# 컨테이너 내부 접속하여 디버깅
docker run -it --rm paddleocr-fastapi-service:latest /bin/bash
```

**문제: Mobile 모델 vs Server 모델 선택**
```bash
# Mobile 모델로 재빌드 (더 빠른 추론)
./docker_build.sh --use-mobile
# 또는 자동 빌드 + 실행
./docker_run.sh --use-mobile

# Server 모델로 재빌드 (더 높은 정확도)
./docker_build.sh  # 기본값
# 또는 자동 빌드 + 실행
./docker_run.sh
```

**문제: 이미지 크기가 너무 큼**
```bash
# 모델 없이 빌드하여 이미지 크기 축소
./docker_build.sh --no-models

# 런타임에 필요한 모델만 자동 다운로드됨
docker run -d -p 8081:8080 --name ocr-fastapi paddleocr-fastapi-service:latest
```
