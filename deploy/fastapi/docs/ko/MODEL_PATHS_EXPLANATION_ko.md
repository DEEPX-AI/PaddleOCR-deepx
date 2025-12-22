# PP-OCR V5 모델 경로 설정 방법

## 개요
PaddleOCR FastAPI 서비스는 `USE_MOBILE` 환경변수를 통해 mobile 모델과 server 모델을 선택하여 사용할 수 있습니다. **기본적으로 mobile 모델이 사용됩니다** (엣지 디바이스에서 더 나은 성능을 위해). `--use-server` 옵션을 사용하거나 `USE_MOBILE=false`를 설정하여 server 모델로 전환할 수 있습니다.

## 모델 다운로드 위치

모델은 다음 경로에 다운로드됩니다:
```
$HOME/.paddlex/official_models/
├── PP-OCRv5_server_det/     # Server detection model
├── PP-OCRv5_server_rec/     # Server recognition model
├── PP-OCRv5_mobile_det/     # Mobile detection model
├── PP-OCRv5_mobile_rec/     # Mobile recognition model
├── PP-LCNet_x1_0_doc_ori/   # Document orientation classification
├── UVDoc/                    # Document unwarping
└── PP-LCNet_x1_0_textline_ori/  # Text line orientation
```

## 모델 선택 방식

### 1. 환경변수를 통한 선택

**Mobile 모델 (기본값)**:
```bash
# 환경변수 없음 또는 USE_MOBILE=true (기본값)
./run.sh
# 또는
./run.sh --use-mobile
# 또는
docker run -d -p 8081:8080 paddleocr-fastapi-service
```

**Server 모델**:
```bash
# run.sh 사용
./run.sh --use-server

# 환경변수 직접 설정
USE_MOBILE=false ./run.sh

# Docker 실행
docker run -d -p 8081:8080 -e USE_MOBILE=false paddleocr-fastapi-service
```

### 2. Docker 빌드 시 모델 선택

**Mobile 모델로 빌드 (기본값)**:
```bash
./docker_build.sh
# 또는 명시적으로
./docker_build.sh --use-mobile
```

**Server 모델로 빌드**:
```bash
./docker_build.sh --use-server
```

**Docker run 스크립트**:
```bash
# Mobile 모델 (기본값)
./docker_run.sh
# 또는 명시적으로
./docker_run.sh --use-mobile

# Server 모델
./docker_run.sh --use-server
```

## 코드 레벨에서의 모델 경로 설정

### ocr_service.py의 구현

#### 1. 모델 경로 결정 함수 (`get_model_paths()`)

```python
def get_model_paths():
    """
    Get model paths based on USE_MOBILE environment variable
    Returns dict with detection and recognition model paths
    """
    use_mobile = os.getenv('USE_MOBILE', 'false').lower() == 'true'
    models_dir = Path.home() / '.paddlex' / 'official_models'
    
    if use_mobile:
        det_model_name = 'PP-OCRv5_mobile_det'
        rec_model_name = 'PP-OCRv5_mobile_rec'
    else:
        det_model_name = 'PP-OCRv5_server_det'
        rec_model_name = 'PP-OCRv5_server_rec'
    
    det_model_path = models_dir / det_model_name
    rec_model_path = models_dir / rec_model_name
    
    result = {
        'det_model_dir': str(det_model_path) if det_model_path.exists() else None,
        'rec_model_dir': str(rec_model_path) if rec_model_path.exists() else None,
        'model_type': 'mobile' if use_mobile else 'server'
    }
    
    return result
```

#### 2. PaddleOCR 초기화 (`get_ocr_instance()`)

```python
def get_ocr_instance(...) -> PaddleOCR:
    # Get model paths based on USE_MOBILE environment variable
    model_paths = get_model_paths()
    
    # Build PaddleOCR init parameters
    ocr_params = {
        'use_textline_orientation': use_textline_orientation,
        'text_det_limit_side_len': det_limit_side_len,
        'text_det_limit_type': det_limit_type,
        'text_det_thresh': det_db_thresh,
        'text_det_box_thresh': det_db_box_thresh,
        'text_det_unclip_ratio': det_db_unclip_ratio,
        'lang': 'ch',
        'device': device
    }
    
    # Add model directories if they exist
    if model_paths['det_model_dir']:
        ocr_params['text_detection_model_dir'] = model_paths['det_model_dir']
    if model_paths['rec_model_dir']:
        ocr_params['text_recognition_model_dir'] = model_paths['rec_model_dir']
    
    # Initialize PaddleOCR with model paths
    ocr_instances[cache_key] = PaddleOCR(**ocr_params)
```

## PaddleOCR 파라미터 매핑

PaddleOCR 3.x는 다음 파라미터로 모델 경로를 지정합니다:

| 구버전 파라미터 (2.x) | 신버전 파라미터 (3.x) | 설명 |
|---------------------|---------------------|------|
| `det_model_dir` | `text_detection_model_dir` | Detection 모델 경로 |
| `rec_model_dir` | `text_recognition_model_dir` | Recognition 모델 경로 |
| `cls_model_dir` | `textline_orientation_model_dir` | Text line orientation 모델 경로 |

### PaddleOCR 내부 코드 참조

`paddleocr/_pipelines/ocr.py`:
```python
class PaddleOCR(PaddleXPipelineWrapper):
    def __init__(
        self,
        text_detection_model_name=None,
        text_detection_model_dir=None,    # <- 여기서 모델 경로 지정
        text_recognition_model_name=None,
        text_recognition_model_dir=None,  # <- 여기서 모델 경로 지정
        ...
    ):
        # model_dir이 None이면 model_name이나 lang/ocr_version으로 자동 선택
        # model_dir이 지정되면 해당 경로의 모델 사용
```

## 동작 확인 방법

### 1. 로그 확인
서비스 시작 시 다음과 같은 로그가 출력됩니다:

```
🔧 Initializing PaddleOCR with mobile models...
   Detection model: /home/user/.paddlex/official_models/PP-OCRv5_mobile_det
   Recognition model: /home/user/.paddlex/official_models/PP-OCRv5_mobile_rec
✅ PaddleOCR initialized successfully with mobile models
```

또는:

```
🔧 Initializing PaddleOCR with server models...
   Detection model: /home/user/.paddlex/official_models/PP-OCRv5_server_det
   Recognition model: /home/user/.paddlex/official_models/PP-OCRv5_server_rec
✅ PaddleOCR initialized successfully with server models
```

### 2. 컨테이너 내에서 확인

```bash
# 컨테이너 접속
docker exec -it ocr-fastapi bash

# 모델 디렉토리 확인
ls -la ~/.paddlex/official_models/

# 환경변수 확인
echo $USE_MOBILE

# 로그 확인
docker logs ocr-fastapi
```

### 3. Python 코드로 확인

```python
import os
from pathlib import Path

use_mobile = os.getenv('USE_MOBILE', 'false').lower() == 'true'
models_dir = Path.home() / '.paddlex' / 'official_models'

if use_mobile:
    det_model = models_dir / 'PP-OCRv5_mobile_det'
    rec_model = models_dir / 'PP-OCRv5_mobile_rec'
else:
    det_model = models_dir / 'PP-OCRv5_server_det'
    rec_model = models_dir / 'PP-OCRv5_server_rec'

print(f"Detection model exists: {det_model.exists()}")
print(f"Recognition model exists: {rec_model.exists()}")
print(f"Detection model path: {det_model}")
print(f"Recognition model path: {rec_model}")
```

## 일관성 체크리스트

✅ **모든 실행 방식에서 모델 선택 옵션 지원**
- [x] `run.sh` - 로컬 실행 (기본값: mobile, 옵션: --use-mobile, --use-server)
- [x] `docker_run.sh` - Docker 실행 (기본값: mobile, 옵션: --use-mobile, --use-server)
- [x] `docker_build.sh` - Docker 빌드 (기본값: mobile, 옵션: --use-mobile, --use-server)
- [x] `local_setup.sh` - 로컬 환경 설정 (기본값: mobile, 옵션: --use-mobile, --use-server)
- [x] `local_deepx_setup.sh` - 로컬 NPU 설정 (기본값: mobile, 옵션: --use-mobile, --use-server)

✅ **모델 경로 자동 설정**
- [x] `USE_MOBILE` 환경변수 기반 자동 분기
- [x] PaddleOCR 초기화 시 `text_detection_model_dir`, `text_recognition_model_dir` 파라미터 전달
- [x] 모델 존재 여부 확인
- [x] 초기화 로그 출력

## 문제 해결

### 모델이 다운로드되지 않았을 때

모델이 없으면 PaddleOCR가 자동으로 기본 모델을 다운로드하지만, 원하는 버전(mobile/server)이 아닐 수 있습니다.

**해결방법**:
```bash
# 로컬 환경
./local_setup.sh                # mobile 모델 다운로드 (기본값)
# 또는
./local_setup.sh --use-mobile   # 명시적으로 mobile 모델 다운로드
# 또는
./local_setup.sh --use-server   # server 모델 다운로드

# DEEPX NPU 환경
./local_deepx_setup.sh --dx_rt /path/to/dx_rt                # mobile 모델 (기본값)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --use-server   # server 모델

# Docker
./docker_build.sh               # mobile 모델 포함 이미지 빌드 (기본값)
# 또는
./docker_build.sh --use-server  # server 모델 포함 이미지 빌드
```

### 모델 경로를 확인하고 싶을 때

```bash
# 서비스 API 호출
curl http://localhost:8080/health

# 컨테이너 로그 확인
docker logs ocr-fastapi | grep "Initializing PaddleOCR"
```

## 참고 자료

- [PaddleOCR 3.x 문서](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 모델 다운로드](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)
- `paddleocr/_pipelines/ocr.py` - PaddleOCR 클래스 구현
- `deploy/fastapi/Dockerfile` - 모델 다운로드 로직
