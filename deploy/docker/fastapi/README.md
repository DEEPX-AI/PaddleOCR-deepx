# PaddleOCR FastAPI OCR Service

An OCR REST API service using FastAPI based on PaddlePaddle 3.0. Uses PP-OCRv5 models.

## 🎯 Quick Start Guide

### Local Environment (Development/Testing)
```bash
cd PaddleOCR/deploy/docker/fastapi

# 1. Environment setup (first time only)
./local_setup.sh

# 2. Start server
./run.sh
```

### Local Environment with DEEPX NPU (Hardware Acceleration)
```bash
cd PaddleOCR/deploy/docker/fastapi

# 1. NPU environment setup (first time only)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# 2. Start server (automatically applies NPU settings)
./run.sh
```

> **Note**: For DEEPX NPU hardware acceleration, see [DEEPX NPU Support Guide](docs/DEEPX_NPU_GUIDE.md)

### Docker Environment (Production/Deployment)
```bash
cd PaddleOCR/deploy/docker/fastapi

# Build + Run in one command
./docker_run.sh
```

---

## 🚀 Detailed Guide

### Method 1: Running in Local Environment

#### 1.1 Environment Setup (local_setup.sh)

**Basic setup (CPU + Mobile model):**
```bash
cd PaddleOCR/deploy/docker/fastapi
chmod +x local_setup.sh
./local_setup.sh
```

**GPU version (Mobile model):**
```bash
./local_setup.sh --gpu
```

**Use Server model (high precision):**
```bash
./local_setup.sh --use-server
```

**GPU + Server model:**
```bash
./local_setup.sh --gpu --use-server
```

**Skip model download (download at runtime):**
```bash
./local_setup.sh --no-models
```

#### 1.1.1 DEEPX NPU Setup (local_deepx_setup.sh)

**For DEEPX NPU hardware acceleration**, use `local_deepx_setup.sh` instead of `local_setup.sh`:

```bash
# Basic NPU setup (required: dx_rt path)
./local_deepx_setup.sh --dx_rt /path/to/dx_rt

# NPU + Mobile model
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --use-mobile

# NPU + Server model
./local_deepx_setup.sh --dx_rt /path/to/dx_rt --use-server
```

> **📖 Complete NPU Guide**: See [DEEPX NPU Support Guide](docs/DEEPX_NPU_GUIDE.md) for:
> - Automatic installation and setup
> - RT optimization configuration
> - NPU model management
> - Performance tuning
> - Troubleshooting

#### 1.2 Start Server (run.sh)

**Simple execution:**
```bash
./run.sh
```

**Custom port:**
```bash
./run.sh --port 9000
```

**Using environment variables:**
```bash
PORT=9000 USE_GPU=true ./run.sh
```

**Manual execution (using virtual environment directly):**

```bash
# Activate virtual environment
source venv/bin/activate

# Start server
python ocr_service.py
```

Or without activating virtual environment:
```bash
venv/bin/python ocr_service.py
```

The service runs at http://localhost:8080
- **API Documentation**: http://localhost:8080/docs

#### 1.3 local_setup.sh Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu` | Install GPU version of PaddlePaddle | CPU |
| `--use-mobile` | Use Mobile model (fast, small) | Mobile (default) |
| `--use-server` | Use Server model (high precision) | - |
| `--no-models` | Skip model download | Download |
| `--python VERSION` | Specify Python version | python3.10 |
| `--version VERSION` | Specify PaddleOCR version | 3.3.2 |
| `--help` | Show help message | - |

#### 1.4 run.sh Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port PORT` | Service port | 8080 |
| `--host HOST` | Binding host | 0.0.0.0 |
| `--use-mobile` | Use Mobile model | Mobile (default) |
| `--use-server` | Use Server model | - |
| `--help` | Show help message | - |

**Environment variables:**
- `PORT`: Service port
- `HOST`: Binding host  
- `USE_GPU`: Whether to use GPU (true/false)
- `USE_MOBILE`: Whether to use Mobile model (true/false)

#### 1.5 Server vs Mobile Models

| Model Type | Accuracy | Speed | File Size | Recommended Use |
|------------|----------|-------|-----------|-----------------|
| **Mobile** (default) | Medium ⭐⭐ | Fast | Small | Real-time, Edge devices, General use |
| **Server** | High ⭐⭐⭐ | Slow | Large | Server, High-precision OCR |

---

### Method 2: Running in Docker Container

#### 2.1 Build + Run in One Command (docker_run.sh) ⭐ Recommended

**Basic execution (CPU + Mobile model):**
```bash
cd PaddleOCR/deploy/docker/fastapi
chmod +x docker_run.sh
./docker_run.sh
```

If the image doesn't exist, it automatically builds and then runs the container!

**GPU version (Mobile model):**
```bash
./docker_run.sh --gpu
```

**Use Server model (high precision):**
```bash
./docker_run.sh --use-server
```

**GPU + Server model:**
```bash
./docker_run.sh --gpu --use-server
```

**DEEPX NPU support (Hardware Acceleration):**
```bash
./docker_run.sh --deepx
```

**DEEPX NPU + GPU:**
```bash
./docker_run.sh --gpu --deepx
```

**Custom port:**
```bash
./docker_run.sh --port 9000
```

#### 2.2 Manual Build (docker_build.sh)

**CPU version (with models):**
```bash
cd PaddleOCR/deploy/docker/fastapi
chmod +x docker_build.sh
./docker_build.sh
```

**GPU version (CUDA 11.8, Mobile model):**
```bash
./docker_build.sh --gpu
```

**Use Server model (high precision):**
```bash
./docker_build.sh --use-server
```

**GPU + Server model:**
```bash
./docker_build.sh --gpu --use-server
```

**Exclude models (download at runtime):**
```bash
./docker_build.sh --no-models
```

**DEEPX NPU support (Hardware Acceleration):**
```bash
./docker_build.sh --deepx
```

**DEEPX NPU + GPU:**
```bash
./docker_build.sh --gpu --deepx
```

#### 2.3 Manual Container Execution

**CPU version:**
```bash
docker run -d -p 8081:8080 --name ocr-fastapi paddleocr-fastapi-service:latest
```

**GPU version:**
```bash
docker run -d --gpus all -p 8081:8080 --name ocr-fastapi paddleocr-fastapi-service:latest-gpu
```

> **Note**: Port 8081 is used to avoid port conflicts with local execution. Local uses port 8080, container uses port 8081, allowing you to compare both services simultaneously.

**Setting environment variables:**
```bash
docker run -d -p 8081:8080 \
  -e USE_GPU=false \
  -e PORT=8080 \
  -e HOST=0.0.0.0 \
  --name ocr-fastapi \
  paddleocr-fastapi-service:latest
```

#### 2.4 docker_run.sh Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu` | Run in GPU mode | CPU |
| `--use-mobile` | Use Mobile model (builds if image doesn't exist) | Mobile (default) |
| `--use-server` | Use Server model (builds if image doesn't exist) | - |
| `--deepx` | Enable DEEPX NPU support (builds if image doesn't exist) | Disabled |
| `--port PORT` | Host port | 8081 |
| `--name NAME` | Container name | ocr-fastapi |
| `--image IMAGE` | Image name | paddleocr-fastapi-service |
| `--tag TAG` | Image tag | latest (GPU: latest-gpu) |
| `--help` | Show help message | - |

**Key Features:**
- ✅ Automatic image build (only when not exists)
- ✅ Automatic cleanup of existing containers
- ✅ Post-execution status check
- ✅ Useful command suggestions

#### 2.5 docker_build.sh Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu` | Enable GPU support | CPU |
| `--use-mobile` | Use Mobile model (fast, small) | Mobile (default) |
| `--use-server` | Use Server model (high precision) | - |
| `--deepx` | Enable DEEPX NPU support (Hardware Acceleration) | Disabled |
| `--no-models` | Skip model download during build | Download |
| `--version VERSION` | Specify PaddleOCR version | 3.3.2 |
| `--tag NAME` | Docker image name | paddleocr-fastapi-service |
| `--tag-version VER` | Image version tag | latest |

**Build Examples:**
```bash
# CPU + Mobile model (default)
./docker_build.sh

# GPU + Mobile model (default GPU)
./docker_build.sh --gpu

# CPU + Server model (high precision)
./docker_build.sh --use-server

# GPU + Server model (high precision GPU)
./docker_build.sh --gpu --use-server

# Build without models (reduce image size)
./docker_build.sh --no-models

# Custom tag
./docker_build.sh --tag my-ocr --tag-version v1.0
```

---

## 📋 API Usage

### Accessing API Documentation

FastAPI automatically provides interactive API documentation:

**Local environment:**
- **Swagger UI**: http://localhost:8080/docs

**Docker environment (when port 8081 is mapped):**
- **Swagger UI**: http://localhost:8081/docs

### API Endpoint Examples

> **Port Note**: Examples use port 8080 for local and 8081 for Docker. Adjust according to your actual port.

**Health Check:**
```bash
# Local environment
curl http://localhost:8080/health

# Docker environment
curl http://localhost:8081/health
```

**OCR (Hubserving compatible format) - `/predict/ocr_system`:**
```bash
# images array format (same as hubserving)
IMAGE_BASE64=$(base64 -w 0 your_image.jpg)

# Local environment
curl -X POST http://localhost:8080/predict/ocr_system \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMAGE_BASE64\"]}"

# Docker environment
curl -X POST http://localhost:8081/predict/ocr_system \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMAGE_BASE64\"]}"
```

**OCR (URL) - `/ocr`:**
```bash
# Local environment
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
  }'

# Docker environment
curl -X POST http://localhost:8081/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
  }'
```

**OCR (Base64 image):**
```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 your_image.jpg)

# Local environment
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\"}"

# Docker environment
curl -X POST http://localhost:8081/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\"}"
```

**OCR (File upload):**
```bash
# Local environment
curl -X POST http://localhost:8080/ocr \
  -F "file=@/path/to/image.jpg"

# Docker environment
curl -X POST http://localhost:8081/ocr \
  -F "file=@/path/to/image.jpg"
```

**Batch OCR:**
```bash
# Local environment
curl -X POST http://localhost:8080/batch_ocr \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["'$IMAGE_BASE64_1'", "'$IMAGE_BASE64_2'"]
  }'

# Docker environment
curl -X POST http://localhost:8081/batch_ocr \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["'$IMAGE_BASE64_1'", "'$IMAGE_BASE64_2'"]
  }'
```

---

## 📋 API Endpoints

### GET /health
Check service status

**Response:**
```json
{
  "status": "healthy"
}
```

### POST /ocr
Single image OCR

**Request parameters (choose one):**
- `url`: Image URL (JSON body)
- `image`: Base64 encoded image (JSON body)
- `file`: Multipart file upload (form-data)

**JSON request example:**
```json
{
  "url": "https://example.com/image.jpg"
}
```
or
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "text": "Recognized text",
      "confidence": 0.95
    }
  ]
}
```

### POST /batch_ocr
Batch OCR for multiple images

**Request:**
```json
{
  "images": ["base64_image_1", "base64_image_2", ...]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    [/* image 1 results */],
    [/* image 2 results */]
  ]
}
```

## 🎯 Advantages of FastAPI

### 1. Automatic API Documentation
- **Swagger UI** (`/docs`): Interactive API testing

### 2. Data Validation
- Automatic request/response validation through Pydantic models
- Type safety guarantee
- Automatic error response generation

### 3. Performance
- High throughput with async support
- Uses Uvicorn ASGI server

### 4. Developer Experience
- Type hint support
- IDE auto-completion
- Clear error messages

## 🔧 Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu` | Enable GPU support | CPU |
| `--no-models` | Skip model download during build | Download |
| `--version VERSION` | Specify PaddleOCR version | 3.3.2 |
| `--tag NAME` | Docker image name | paddleocr-fastapi-service |
| `--tag-version VER` | Image version tag | latest |

## 🐍 Python Client Example

```python
import requests
import base64

# Encode image file to base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# OCR request (local environment)
response = requests.post(
    "http://localhost:8080/ocr",
    json={"image": image_base64}
)

# OCR request (Docker environment)
# response = requests.post(
#     "http://localhost:8081/ocr",
#     json={"image": image_base64}
# )

# Print results
result = response.json()
if result['success']:
    for item in result['results']:
        print(f"Text: {item['text']}")
        print(f"Confidence: {item['confidence']:.2f}")
        print(f"BBox: {item['bbox']}")
        print("-" * 50)
```

### Auto-generating OpenAPI Clients

FastAPI provides OpenAPI specifications, allowing you to auto-generate clients for various languages:

```bash
# Download OpenAPI spec (local environment)
curl http://localhost:8080/openapi.json > openapi.json

# Download OpenAPI spec (Docker environment)
curl http://localhost:8081/openapi.json > openapi.json

# Generate Python client (using openapi-generator)
openapi-generator generate -i openapi.json -g python -o ./client
```

---

## 🔧 Environment-specific Information

### Local Environment

- **Base**: Python 3.10+ venv
- **Web Framework**: FastAPI 0.109.0
- **ASGI Server**: Uvicorn 0.27.0
- **PaddlePaddle**: 3.0.0
- **PaddleOCR**: 3.3.2 (default)
- **Models**: PP-OCRv5 server/mobile (selectable)
- **Port**: 8080 (default)
- **Model storage location**: `~/.paddlex/official_models/`

### Docker Environment

- **Base Image**: python:3.10-slim
- **Web Framework**: FastAPI 0.109.0
- **ASGI Server**: Uvicorn 0.27.0
- **PaddlePaddle**: 3.0.0
- **PaddleOCR**: 3.3.2 (default)
- **Models**: PP-OCRv5 server/mobile (selectable at build time)
- **Port**: 8080 (inside container), 8081 (recommended host mapping)
- **Model storage location**: `/home/paddleocr/.paddlex/official_models/`

---

## ⚙️ Environment Variables

### Local Environment
In local environment, set directly in `ocr_service.py` file or via environment variables:
```bash
export USE_GPU=false
export PORT=8080
export HOST=0.0.0.0
python ocr_service.py
```

### Docker Environment

- `USE_GPU`: Whether to use GPU (true/false, default: false)
- `PORT`: Service port (default: 8080)
- `HOST`: Binding host (default: 0.0.0.0)

---

## 🆚 Comparison

### Local vs Docker Execution

| Item | Local Environment | Docker Environment |
|------|-------------------|-------------------|
| **Setup Speed** | Fast (venv creation) | Slow (image build) |
| **Dependency Management** | Isolated with venv | Complete isolation |
| **Debugging** | Easy | Moderate |
| **Deployment** | Manual setup required | Just deploy image |
| **Resources** | Low | Slightly higher (container overhead) |
| **Development Recommendation** | ⭐⭐⭐ | ⭐⭐ |
| **Production Recommendation** | ⭐⭐ | ⭐⭐⭐ |

---

## 📝 Notes

### Local Environment

1. **System dependencies installation required:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1 libglib2.0-0 libgomp1
   ```

2. **Python version:**
   - Python 3.10 or higher recommended
   - For other versions: `./setup.sh --python python3.10`

3. **Memory requirements:**
   - CPU + Server model: Minimum 2GB RAM
   - CPU + Mobile model: Minimum 1.5GB RAM
   - GPU: Minimum 4GB VRAM

4. **Model storage location:**
   - `~/.paddlex/official_models/`
   - Disk space: Server model ~500MB, Mobile model ~200MB

### Docker Environment

1. NVIDIA Docker Runtime required for GPU version:
   ```bash
   # Check nvidia-docker2 installation
   docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. Memory requirements:
   - CPU: Minimum 2GB RAM
   - GPU: Minimum 4GB VRAM

3. Image size limitations:
   - FastAPI supports large file uploads by default,
   - Adjust with `MAX_UPLOAD_SIZE` environment variable if needed

---

## 🔍 Monitoring and Logging

### Local Environment
```bash
# Logs output directly to terminal when running server
python ocr_service.py

# Background execution + log file
nohup python ocr_service.py > ocr_service.log 2>&1 &

# Check logs
tail -f ocr_service.log
```

### Docker Environment

```bash
# Check container logs
docker logs -f ocr-fastapi

# Adjust log level (uvicorn)
docker run -d -p 8081:8080 \
  -e LOG_LEVEL=debug \
  paddleocr-fastapi-service:latest
```

---

## 🚀 Performance Optimization

### Local Environment

**1. Multi-worker execution (Gunicorn + Uvicorn):**
```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn ocr_service:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080
```

**2. Use Mobile model:**
```bash
# For faster inference
./local_setup.sh --use-mobile
```

**3. Utilize GPU:**
```bash
./local_setup.sh --gpu
```

### Docker Environment
```bash
# Modify Dockerfile to use gunicorn + uvicorn workers
CMD ["gunicorn", "ocr_service:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080"]
```

### Asynchronous Processing (Common)
Current implementation is synchronous, but can be converted to async if needed:
```python
@app.post('/ocr')
async def ocr_image(...):
    result = await asyncio.to_thread(ocr.predict, img)
```

---

## 🎓 Scenario-based Recommendations

| Scenario | Recommended Method | Command |
|----------|-------------------|---------|
| **Development/Testing** | Local + Server model | `./local_setup.sh && ./run.sh` |
| **Quick Prototype** | Local + Mobile model | `./local_setup.sh --use-mobile && ./run.sh` |
| **High-precision OCR** | Local/Docker + Server + GPU | `./local_setup.sh --gpu` or `./docker_run.sh --gpu` |
| **Real-time Processing** | Local/Docker + Mobile + GPU | `./local_setup.sh --gpu --use-mobile` or `./docker_run.sh --gpu --use-mobile` |
| **Production Deployment** | Docker + Server model | `./docker_run.sh` |
| **Edge/Embedded** | Local/Docker + Mobile + CPU | `./local_setup.sh --use-mobile` or `./docker_run.sh --use-mobile` |

---

## 📚 Additional Resources

- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [PaddleOCR Official Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 Model Information](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/ppocr_introduction.md)

---

## 📁 Script File Structure

```
deploy/docker/fastapi/
├── local_setup.sh      # Local environment setup (venv creation, dependency installation)
├── run.sh              # Local server execution
├── docker_build.sh     # Docker image build
├── docker_run.sh       # Docker build + run (automated)
├── ocr_service.py      # FastAPI server code
├── Dockerfile          # Docker image definition
└── README.md           # This document
```

### Script Selection Guide

| Purpose | Script to Use | Description |
|---------|--------------|-------------|
| **Local Initial Setup** | `local_setup.sh` | Create Python venv and install all dependencies |
| **Local Server Run** | `run.sh` | Start OCR service in configured environment |
| **Docker Image Build** | `docker_build.sh` | Build Docker image only |
| **Docker Service Run** | `docker_run.sh` ⭐ | Auto-build image + run container (recommended) |

---

## 🔧 Troubleshooting

### Local Environment

**Issue: ImportError - paddle related**
```bash
# Check if virtual environment is activated
which python  # Should be venv/bin/python

# Reinstall PaddlePaddle
pip install --force-reinstall paddlepaddle==3.0.0
```

**Issue: Model download failure**
```bash
# Check model download location manually
ls ~/.paddlex/official_models/

# Run local_setup.sh without models and auto-download at runtime
./local_setup.sh --no-models
```

**Issue: Port already in use**
```bash
# Check process using port 8080
lsof -i :8080

# Run on different port
PORT=8888 python ocr_service.py
```

### Docker Environment

**Issue: GPU not recognized**
```bash
# Check NVIDIA Docker runtime
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Install nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Issue: Container won't start**
```bash
# Check logs
docker logs ocr-fastapi

# Access container shell for debugging
docker run -it --rm paddleocr-fastapi-service:latest /bin/bash
```

**Issue: Mobile model vs Server model selection**
```bash
# Rebuild with Mobile model (faster inference)
./docker_build.sh --use-mobile
# Or auto-build + run
./docker_run.sh --use-mobile

# Rebuild with Server model (higher accuracy)
./docker_build.sh  # default
# Or auto-build + run
./docker_run.sh
```

**Issue: Image size too large**
```bash
# Build without models to reduce image size
./docker_build.sh --no-models

# Required models will be auto-downloaded at runtime
docker run -d -p 8081:8080 --name ocr-fastapi paddleocr-fastapi-service:latest
```
