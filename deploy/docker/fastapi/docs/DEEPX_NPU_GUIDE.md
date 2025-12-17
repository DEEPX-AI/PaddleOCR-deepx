# DEEPX NPU Support - Complete Guide

Complete guide for adding DEEPX NPU hardware acceleration to PaddleOCR FastAPI service

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Automatic Installation](#automatic-installation)
3. [Components](#components)
4. [Environment Variables](#environment-variables)
5. [Model Structure](#model-structure)
6. [Usage](#usage)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## Quick Start

### Step 1: Automatic Installation

Use the DEEPX NPU dedicated setup script to automatically install all dependencies:

```bash
# Automatic setup of complete NPU environment
./local_deepx_setup.sh --dx_rt /path/to/dx_rt
```

**What the setup script automatically does:**
- ✅ Create and activate Python 3.11+ venv
- ✅ Build DX_RT (auto-install dx-engine)
- ✅ Install PaddlePaddle 3.0.0
- ✅ Install PyTorch 2.3.0 + torchvision + torchaudio
- ✅ Install ONNX Runtime 1.18.0
- ✅ Install additional dependencies (scikit-image, imgaug, shapely, pyclipper, jiwer)
- ✅ Download PaddleOCR v5 models (server/mobile)
- ✅ Verify DEEPX NPU models
- ✅ Create RT optimization environment variable file (deepx_env.sh)
- ✅ Validate deepx/engine path
- ✅ Verify dependency installation

### Step 2: Start Service

```bash
# Activate venv and configure NPU environment
source venv/bin/activate
source deepx_env.sh  # Apply RT optimization environment variables

# Or use run.sh (automatically applies deepx_env.sh)
./run.sh
```

**What run.sh automatically does:**
- ✅ Activate venv
- ✅ Auto-detect and apply `deepx_env.sh`
- ✅ Set RT optimization environment variables
- ✅ Run OCR service

### Step 3: Test CPU OCR

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1
  }'
```

### Step 4: Test DEEPX NPU OCR

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

## Automatic Installation

### Overview

`local_deepx_setup.sh` is a script that ports the setup method from the deepx project to the FastAPI service.

### Ported Key Features

| Feature | deepx | FastAPI Port |
|---------|-------|--------------|
| **Environment Setup** | startup.sh | local_deepx_setup.sh |
| **venv Creation** | ✅ | ✅ |
| **DX_RT Build** | build.sh | ✅ (auto-called) |
| **PyTorch Installation** | requirements.txt (2.3.0) | ✅ (2.3.0) |
| **dx-engine Installation** | ✅ (1.1.2) | ✅ (via dx_rt build) |
| **ONNX Runtime** | requirements.txt (1.18.0) | ✅ (1.18.0) |
| **Model Verification** | - | ✅ (deepx/engine/model_files) |
| **RT Optimization** | set_env.sh | deepx_env.sh |
| **Environment Variable Application** | source set_env.sh | source deepx_env.sh |

### Basic Usage

```bash
# Required: specify dx_rt path
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# Use Server model (default)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# Use Mobile model
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --use-mobile
```

### Advanced Options

```bash
# Skip NPU setup (CPU only)
./local_deepx_setup.sh --no-npu

# Download only PaddleOCR models (exclude DEEPX models)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --no-deepx-models

# Customize RT optimization thread count
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --inter-threads 2 --intra-threads 4

# Use GPU version PaddlePaddle
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --gpu

# Specify Python version
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --python python3.11
```

### All Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dx_rt PATH` | dx_rt directory path (required for NPU) | - |
| `--gpu` | Install GPU version PaddlePaddle | cpu |
| `--use-mobile` | Use Mobile model | server |
| `--no-models` | Skip PaddleOCR model download | false |
| `--no-deepx-models` | Skip DEEPX model download | false |
| `--no-npu` | Skip NPU setup (CPU only) | false |
| `--python VERSION` | Specify Python version | python3.11 |
| `--version VERSION` | Specify PaddleOCR version | 3.3.2 |
| `--inter-threads N` | CUSTOM_INTER_OP_THREADS_COUNT | 1 |
| `--intra-threads N` | CUSTOM_INTRA_OP_THREADS_COUNT | 3 |

---

## Components

### 1. Required Python Packages

```txt
# Core OCR dependencies
opencv-python-headless==4.7.0.72
numpy==1.26.4
torch==2.3.0                    ← NPU required
torchvision==0.18.0             ← NPU required
torchaudio==2.3.0               ← NPU required
onnxruntime==1.18.0             ← NPU required

# Image processing dependencies
opencv-contrib-python==4.7.0.72
scikit-image                    ← NPU required
imgaug                          ← NPU required
shapely                         ← NPU required
pyclipper                       ← NPU required

# Utility dependencies
jiwer                           ← NPU required
```

### 2. DX_RT Build

Automatically installs dx-engine by calling dx_rt's `build.sh`:

```bash
# Automatically executed:
cd /path/to/dx_rt
./build.sh
```

Build results:
- dx-engine automatically installed in venv
- Required libraries compiled

### 3. RT Optimization Environment Setup

Ported deepx's environment configuration logic to `deepx_env.sh`:

**Generated file (deepx_env.sh):**
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

### 4. Automatic Environment Application (run.sh)

`run.sh` automatically applies `deepx_env.sh` on startup:

```bash
# Inside run.sh:
DEEPX_ENV_FILE="$SCRIPT_DIR/deepx_env.sh"
if [ -f "$DEEPX_ENV_FILE" ]; then
    echo "🔧 Applying DEEPX NPU environment settings..."
    source "$DEEPX_ENV_FILE"
    echo ""
fi
```

---

## Environment Variables

### RT Optimization Variables

| Variable | Default | Description | Source |
|----------|---------|-------------|--------|
| `CUSTOM_INTER_OP_THREADS_COUNT` | 1 | Inter-op thread count | deepx/set_env.sh |
| `CUSTOM_INTRA_OP_THREADS_COUNT` | 3 | Intra-op thread count | deepx/set_env.sh |
| `DXRT_DYNAMIC_CPU_THREAD` | - | Dynamic CPU thread | deepx/set_env.sh |
| `DXRT_TASK_MAX_LOAD` | - | Maximum task load | deepx/set_env.sh |
| `NFH_INPUT_WORKER_THREADS` | - | Input worker threads | deepx/set_env.sh |
| `NFH_OUTPUT_WORKER_THREADS` | - | Output worker threads | deepx/set_env.sh |
| `LD_LIBRARY_PATH` | ${VIRTUAL_ENV}/lib:... | Library path | deepx/startup.sh |

### Customization

Specify as options during installation or edit `deepx_env.sh` directly:

```bash
# Method 1: Specify during installation
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --inter-threads 2 --intra-threads 4

# Method 2: Edit deepx_env.sh and apply
vi deepx_env.sh
source deepx_env.sh
```

---

## Model Structure

### PaddleOCR Models ($HOME/.paddlex/official_models)

```
~/.paddlex/official_models/
├── PP-OCRv5_server_det/          # Server detection model
├── PP-OCRv5_server_rec/          # Server recognition model
├── PP-OCRv5_mobile_det/          # Mobile detection model
├── PP-OCRv5_mobile_rec/          # Mobile recognition model
├── PP-LCNet_x1_0_doc_ori/        # Document orientation classification
├── UVDoc/                        # Document unwarping
└── PP-LCNet_x1_0_textline_ori/   # Text line orientation
```

### DEEPX Models (deepx/engine/model_files)

```
deepx/engine/model_files/
├── server/                       # DEEPX Server models
│   ├── det/
│   ├── rec/
│   └── cls/
├── mobile/                       # DEEPX Mobile models
│   ├── det/
│   ├── rec/
│   └── cls/
└── *.txt                         # Dictionary files
```

---

## Usage

### Start Service

```bash
# Method 1: Manual activation
source venv/bin/activate
source deepx_env.sh
python ocr_service.py

# Method 2: Use run.sh (recommended)
./run.sh
```

### API Usage

#### CPU Mode (default)

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1
  }'
```

#### NPU Mode (Async)

```bash
curl -X POST http://localhost:8080/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 test.jpg)'",
    "fileType": 1,
    "deepx": true
  }'
```

#### NPU Mode (Sync)

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

### Python Client Example

```python
import requests
import base64

# Read image
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

## Testing

### Run Test Scripts

```bash
# Start service
./run.sh

# Run tests
./run_tests.sh --all

# Test with NPU
./run_tests.sh --all --deepx true

# Test with CPU
./run_tests.sh --all --deepx false

# Test with Sync mode
./run_tests.sh --all --deepx true --sync
```

### Check Logs

#### CPU Mode
```
🚀 Using CPU for inference
✅ CPU OCR completed: 10 results
```

#### NPU Mode (Async)
```
🚀 Using DEEPX NPU for inference (async mode)
🔧 Initializing DEEPX NPU OCR engine...
✅ DEEPX NPU OCR initialized successfully
✅ NPU OCR completed: 10 results
```

#### NPU Mode (Sync)
```
🚀 Using DEEPX NPU for inference (sync mode)
🔧 Initializing DEEPX NPU OCR engine...
✅ DEEPX NPU OCR initialized successfully
✅ NPU OCR completed: 10 results
```

---

## Troubleshooting

### 1. Python Version Error

**Symptoms:**
```
Error: Python 3.10 is too old for DEEPX NPU support
Python 3.11+ is REQUIRED for DEEPX NPU
```

**Solution:**
```bash
# Install Python 3.11
sudo apt install python3.11

# Or reinstall with Python 3.11
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --python python3.11
```

### 2. dx_rt Path Error

**Symptoms:**
```
Error: --dx_rt option is required for NPU setup
```

**Solution:**
```bash
# Must specify dx_rt path
./local_deepx_setup.sh --dx_rt /path/to/dx_rt
```

### 3. dx_rt Build Failure

**Symptoms:**
```
Error: DX_RT build failed with exit code 1
```

**Solution:**
```bash
# Manually verify dx_rt build
cd /path/to/dx_rt
./build.sh

# Check build log
cat build.log
```

### 4. PyTorch Not Installed

**Symptoms:**
```
❌ PyTorch not found
```

**Solution:**
```bash
source venv/bin/activate
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
```

### 5. dx-engine Not Installed

**Symptoms:**
```
❌ dx-engine not found
```

**Solution:**
```bash
# Automatically installed via dx_rt build
cd /path/to/dx_rt
./build.sh

# Or manual installation
source venv/bin/activate
pip install dx-engine==1.1.2
```

### 6. deepx Path Error

**Symptoms:**
```
❌ deepx not found at /data/home/dhyang/git/github/PaddleOCR/deepx
```

**Solution:**
```bash
# Verify path
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx/engine/

# Check symbolic link
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx
```

### 7. DEEPX Models Missing

**Symptoms:**
```
⚠ DEEPX models not found or incomplete
```

**Solution:**
```bash
# Verify models
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx/engine/model_files/server/
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx/engine/model_files/mobile/

# If models are missing, they need to be placed in deepx/engine/model_files
```

### 8. RT Optimization Not Applied

**Symptoms:**
- NPU performance lower than expected

**Solution:**
```bash
# Verify environment variables
echo $CUSTOM_INTER_OP_THREADS_COUNT  # 1
echo $CUSTOM_INTRA_OP_THREADS_COUNT  # 3

# Manual application
source deepx_env.sh

# Restart service
./run.sh
```

### 9. Import Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'dx_engine'
```

**Solution:**
```bash
# Verify venv activation
source venv/bin/activate

# Re-run dx_rt build
cd /path/to/dx_rt
./build.sh

# Check package
pip list | grep dx-engine
```

---

## References

### Directory Structure

```
deploy/docker/fastapi/
├── ocr_service.py           # NPU support added
├── local_deepx_setup.sh     # NPU automatic installation script
├── deepx_env.sh             # RT optimization environment variables (auto-generated)
├── run.sh                   # Service startup (auto-applies deepx_env.sh)
└── docs/
    ├── DEEPX_NPU_GUIDE.md       # This file
    └── ko/
        └── DEEPX_NPU_GUIDE_ko.md    # Korean version
```

### Code Changes Summary

#### 1. Request Model

```python
class BaiduOCRRequest(BaseModel):
    # ... existing fields ...
    deepx: Optional[bool] = Field(False, description="Use DEEPX NPU")
    sync: Optional[bool] = Field(False, description="Use sync mode (PaddleOcr)")
```

#### 2. NPU Initialization (Async)

```python
def get_npu_ocr_instance():
    """Initialize DEEPX NPU OCR engine (Async)"""
    import sys
    sys.path.insert(0, str(DEEPX_ENGINE_PATH))
    from engine.async_paddleocr import AsyncPipelineOCR
    return AsyncPipelineOCR(use_doc_orientation=False, use_doc_unwarping=False)
```

#### 3. NPU Initialization (Sync)

```python
def get_npu_ocr_instance_sync():
    """Initialize DEEPX NPU OCR engine (Sync)"""
    import sys
    sys.path.insert(0, str(DEEPX_ENGINE_PATH))
    from engine.paddleocr import PaddleOcr
    return PaddleOcr(use_doc_orientation=False, use_doc_unwarping=False)
```

#### 4. Branching Logic

```python
if request.deepx:
    if request.sync:
        # NPU Sync mode
        result = process_image_with_npu_sync(img_np, request)
    else:
        # NPU Async mode
        result = process_image_with_npu(img_np, request)
else:
    # CPU mode
    result = process_image_with_cpu(img_np, request)
```

### Porting Summary

| Item | deepx | FastAPI Port | File |
|------|-------|--------------|------|
| Environment Setup | startup.sh | local_deepx_setup.sh | ✅ |
| DX_RT Build | build.sh | Auto-called | ✅ |
| Dependencies | requirements.txt | Embedded in script | ✅ |
| Model Verification | - | Path validation | ✅ |
| RT Optimization | set_env.sh | deepx_env.sh | ✅ |
| Auto-application | Manual source | run.sh automatic | ✅ |
| NPU Initialization | Python code | ocr_service.py | ✅ |
| Async/Sync | AsyncPipelineOCR/PaddleOcr | Both supported | ✅ |
| Version Management | requirements.txt | Explicit versions | ✅ |

### Key Features

✅ **Completed Implementation**
1. Added `deepx` parameter to `/api/v1/ocr` endpoint
2. `deepx: true` → Use DEEPX NPU
3. `deepx: false` or omitted → Use CPU
4. `sync: true` → Sync mode (PaddleOcr)
5. `sync: false` or omitted → Async mode (AsyncPipelineOCR)
6. CPU/NPU branching logic implemented
7. NPU Async/Sync initialization and inference functions implemented
8. Automatic environment setup (deepx_env.sh)
9. RT optimization environment variables applied

All NPU-related settings from deepx have been fully ported to the FastAPI service! 🎉
