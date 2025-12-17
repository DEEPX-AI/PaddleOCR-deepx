# PP-OCR V5 Model Path Configuration Guide

## Overview
PaddleOCR FastAPI service can select between mobile and server models using the `USE_MOBILE` environment variable.

## Model Download Location

Models are downloaded to the following path:
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

## Model Selection Method

### 1. Selection via Environment Variable

**Server Model (default)**:
```bash
# No environment variable or USE_MOBILE=false
./run.sh
# or
docker run -d -p 8081:8080 paddleocr-fastapi-service
```

**Mobile Model**:
```bash
# Using run.sh
./run.sh --use-mobile

# Setting environment variable directly
USE_MOBILE=true ./run.sh

# Docker execution
docker run -d -p 8081:8080 -e USE_MOBILE=true paddleocr-fastapi-service
```

### 2. Model Selection During Docker Build

**Build with Server Model**:
```bash
./docker_build.sh
```

**Build with Mobile Model**:
```bash
./docker_build.sh --use-mobile
```

**Docker run script**:
```bash
# Server model
./docker_run.sh

# Mobile model
./docker_run.sh --use-mobile
```

## Model Path Configuration at Code Level

### Implementation in ocr_service.py

#### 1. Model Path Determination Function (`get_model_paths()`)

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

#### 2. PaddleOCR Initialization (`get_ocr_instance()`)

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

## PaddleOCR Parameter Mapping

PaddleOCR 3.x specifies model paths with the following parameters:

| Old Parameter (2.x) | New Parameter (3.x) | Description |
|---------------------|---------------------|-------------|
| `det_model_dir` | `text_detection_model_dir` | Detection model path |
| `rec_model_dir` | `text_recognition_model_dir` | Recognition model path |
| `cls_model_dir` | `textline_orientation_model_dir` | Text line orientation model path |

### PaddleOCR Internal Code Reference

`paddleocr/_pipelines/ocr.py`:
```python
class PaddleOCR(PaddleXPipelineWrapper):
    def __init__(
        self,
        text_detection_model_name=None,
        text_detection_model_dir=None,    # <- Model path specified here
        text_recognition_model_name=None,
        text_recognition_model_dir=None,  # <- Model path specified here
        ...
    ):
        # If model_dir is None, auto-select using model_name or lang/ocr_version
        # If model_dir is specified, use model at that path
```

## How to Verify Operation

### 1. Check Logs
When the service starts, logs like the following are displayed:

```
🔧 Initializing PaddleOCR with mobile models...
   Detection model: /home/user/.paddlex/official_models/PP-OCRv5_mobile_det
   Recognition model: /home/user/.paddlex/official_models/PP-OCRv5_mobile_rec
✅ PaddleOCR initialized successfully with mobile models
```

Or:

```
🔧 Initializing PaddleOCR with server models...
   Detection model: /home/user/.paddlex/official_models/PP-OCRv5_server_det
   Recognition model: /home/user/.paddlex/official_models/PP-OCRv5_server_rec
✅ PaddleOCR initialized successfully with server models
```

### 2. Check Inside Container

```bash
# Access container
docker exec -it ocr-fastapi bash

# Check model directory
ls -la ~/.paddlex/official_models/

# Check environment variable
echo $USE_MOBILE

# Check logs
docker logs ocr-fastapi
```

### 3. Verify with Python Code

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

## Consistency Checklist

✅ **--use-mobile option supported in all execution methods**
- [x] `run.sh` - Local execution
- [x] `docker_run.sh` - Docker execution
- [x] `docker_build.sh` - Docker build
- [x] `local_setup.sh` - Local environment setup

✅ **Automatic model path configuration**
- [x] Automatic branching based on `USE_MOBILE` environment variable
- [x] Pass `text_detection_model_dir`, `text_recognition_model_dir` parameters during PaddleOCR initialization
- [x] Verify model existence
- [x] Output initialization logs

## Troubleshooting

### When Models Are Not Downloaded

If models don't exist, PaddleOCR will automatically download default models, but they may not be the desired version (mobile/server).

**Solution**:
```bash
# Local environment
./local_setup.sh --use-mobile  # Download mobile models
# or
./local_setup.sh                # Download server models

# Docker
./docker_build.sh --use-mobile  # Build image with mobile models
```

### When You Want to Verify Model Paths

```bash
# Call service API
curl http://localhost:8080/health

# Check container logs
docker logs ocr-fastapi | grep "Initializing PaddleOCR"
```

## References

- [PaddleOCR 3.x Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 Model Downloads](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)
- `paddleocr/_pipelines/ocr.py` - PaddleOCR class implementation
- `deploy/docker/fastapi/Dockerfile` - Model download logic
