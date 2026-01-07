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
- ✅ Create and activate python 3.10+ venv
- ✅ Build DX_RT (auto-install dx-engine)
- ✅ Install PaddlePaddle 3.0.0
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
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --python python3.10
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
| `--python VERSION` | Specify Python version | python3.10 |
| `--version VERSION` | Specify PaddleOCR version | 3.3.2 |
| `--inter-threads N` | CUSTOM_INTER_OP_THREADS_COUNT | 1 |
| `--intra-threads N` | CUSTOM_INTRA_OP_THREADS_COUNT | 3 |

---

## Components

### 1. Required Python Packages

- [cpu requirement](../requirements.txt)
- [gpu requirement](../requirements-gpu.txt)

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

**Configuration Files:**
- `.env.deepx`: Master configuration file with default RT optimization values
- `deepx_env.sh`: Auto-generated script that reads defaults from `.env.deepx`

**Generated file (.env.deepx):**
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

**Generated file (deepx_env.sh):**
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
# ... (similar logic for other variables)

# Library paths
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
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
| `CUSTOM_INTER_OP_THREADS_COUNT` | 1 | Inter-op thread count | .env.deepx |
| `CUSTOM_INTRA_OP_THREADS_COUNT` | 2 | Intra-op thread count | .env.deepx |
| `DXRT_DYNAMIC_CPU_THREAD` | 1 | Dynamic CPU thread | .env.deepx |
| `DXRT_TASK_MAX_LOAD` | 3 | Maximum task load | .env.deepx |
| `NFH_INPUT_WORKER_THREADS` | 2 | Input worker threads | .env.deepx |
| `NFH_OUTPUT_WORKER_THREADS` | 4 | Output worker threads | .env.deepx |
| `LD_LIBRARY_PATH` | ${VIRTUAL_ENV}/lib:... | Library path | deepx_env.sh |

**Default Values (1 2 1 3 2 4):**
All default values are defined in `.env.deepx` and automatically loaded by `deepx_env.sh`.

### Customization

You can customize RT optimization values in three ways:

```bash
# Method 1: Edit .env.deepx (Recommended - affects all scripts)
vi .env.deepx
# Then run setup to regenerate deepx_env.sh
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# Method 2: Specify during installation
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --inter-threads 2 --intra-threads 4

# Method 3: Pass parameters when sourcing deepx_env.sh
source deepx_env.sh 1 2 1 3 2 4
```

**Note:** Method 1 is recommended as it ensures consistency across all execution environments (run.sh, VS Code debugger, etc.).

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
python 3.10+ is REQUIRED for DEEPX NPU
```

**Solution:**
```bash
# Install python 3.10
sudo apt install python3.10

# Or reinstall with python 3.10
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --python python3.10
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

### 4. dx-engine Not Installed

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

### 5. deepx Path Error

**Symptoms:**
```
❌ deepx not found at /dataPaddleOCR/deepx
```

**Solution:**
```bash
# Verify path
ls -la /dataPaddleOCR/deepx/engine/

# Check symbolic link
ls -la /dataPaddleOCR/deepx
```

### 6. DEEPX Models Missing

**Symptoms:**
```
⚠ DEEPX models not found or incomplete
```

**Solution:**
```bash
# Verify models
ls -la /dataPaddleOCR/deepx/engine/model_files/server/
ls -la /dataPaddleOCR/deepx/engine/model_files/mobile/

# If models are missing, they need to be placed in deepx/engine/model_files
```

### 7. RT Optimization Not Applied

**Symptoms:**
- NPU performance lower than expected

**Solution:**
```bash
# Verify environment variables
echo $CUSTOM_INTER_OP_THREADS_COUNT  # 1
echo $CUSTOM_INTRA_OP_THREADS_COUNT  # 2

# Manual application (uses defaults from .env.deepx)
source deepx_env.sh

# Or with custom values
source deepx_env.sh 1 2 1 3 2 4

# Restart service
./run.sh
```

### 8. Import Error

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
deploy/fastapi/
├── ocr_service.py           # NPU support added
├── local_deepx_setup.sh     # NPU automatic installation script
├── .env.deepx               # RT optimization default values (master config)
├── deepx_env.sh             # RT optimization script (auto-generated from .env.deepx)
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
