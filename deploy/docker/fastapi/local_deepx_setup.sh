#!/bin/bash

# PaddleOCR FastAPI Service with DEEPX NPU Support - Local Setup Script
# This script creates a virtual environment and sets up the OCR service with NPU acceleration

set -e  # Exit on error

# Configuration
PYTHON_VERSION="python3.11"
VENV_DIR="venv"
DEVICE_TYPE="cpu"  # cpu or gpu
MODEL_TYPE="server"  # server or mobile
PADDLEOCR_VERSION="3.3.2"
DOWNLOAD_MODELS="true"
DOWNLOAD_DEEPX_MODELS="true"
SETUP_NPU="true"

# DEEPX Configuration
DX_RT_PATH=""  # Required for NPU setup
TORCH_VERSION="2.3.0"
TORCHVISION_VERSION="0.18.0"
ONNXRUNTIME_VERSION="1.18.0"

# RT Optimization defaults (from dx_baidu_gui/set_env.sh)
CUSTOM_INTER_OP_THREADS_COUNT="1"
CUSTOM_INTRA_OP_THREADS_COUNT="3"
DXRT_DYNAMIC_CPU_THREAD=""
DXRT_TASK_MAX_LOAD=""
NFH_INPUT_WORKER_THREADS=""
NFH_OUTPUT_WORKER_THREADS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dx_rt)
            DX_RT_PATH="$2"
            shift 2
            ;;
        --gpu)
            DEVICE_TYPE="gpu"
            shift
            ;;
        --no-models)
            DOWNLOAD_MODELS="false"
            shift
            ;;
        --no-deepx-models)
            DOWNLOAD_DEEPX_MODELS="false"
            shift
            ;;
        --no-npu)
            SETUP_NPU="false"
            shift
            ;;
        --use-mobile)
            MODEL_TYPE="mobile"
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --version)
            PADDLEOCR_VERSION="$2"
            shift 2
            ;;
        --inter-threads)
            CUSTOM_INTER_OP_THREADS_COUNT="$2"
            shift 2
            ;;
        --intra-threads)
            CUSTOM_INTRA_OP_THREADS_COUNT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dx_rt PATH         Path to dx_rt directory (REQUIRED for NPU setup)"
            echo "  --gpu                Install GPU version of PaddlePaddle"
            echo "  --use-mobile         Use mobile version models instead of server models"
            echo "  --no-models          Skip downloading PaddleOCR models"
            echo "  --no-deepx-models    Skip downloading DEEPX models"
            echo "  --no-npu             Skip NPU setup (CPU only)"
            echo "  --python VERSION     Specify Python version (default: python3.11, 3.11+ required for NPU)"
            echo "  --version VERSION    Specify PaddleOCR version (default: 3.3.2)"
            echo "  --inter-threads N    Set CUSTOM_INTER_OP_THREADS_COUNT (default: 1)"
            echo "  --intra-threads N    Set CUSTOM_INTRA_OP_THREADS_COUNT (default: 3)"
            echo "  --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --dx_rt /path/to/dx_rt"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== PaddleOCR FastAPI Service with DEEPX NPU Setup ===${NC}"
echo "Python: $PYTHON_VERSION"
echo "Device: $DEVICE_TYPE"
echo "Model Type: $MODEL_TYPE"
echo "PaddleOCR Version: $PADDLEOCR_VERSION"
echo "Download Models: $DOWNLOAD_MODELS"
echo "Download DEEPX Models: $DOWNLOAD_DEEPX_MODELS"
echo "Setup NPU: $SETUP_NPU"
if [ "$SETUP_NPU" = "true" ]; then
    echo "  - DX_RT Path: $DX_RT_PATH"
    echo "  - PyTorch: $TORCH_VERSION"
    echo "  - RT Optimization: INTER=$CUSTOM_INTER_OP_THREADS_COUNT, INTRA=$CUSTOM_INTRA_OP_THREADS_COUNT"
fi
echo ""

# Validate NPU setup requirements
if [ "$SETUP_NPU" = "true" ]; then
    if [ -z "$DX_RT_PATH" ]; then
        echo -e "${RED}Error: --dx_rt option is required for NPU setup${NC}"
        echo "Usage: $0 --dx_rt /path/to/dx_rt"
        echo "Use --help for more information"
        exit 1
    fi
fi

# Check if Python is available
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_VERSION is not installed${NC}"
    if [ "$SETUP_NPU" = "true" ]; then
        echo "Python 3.11+ is REQUIRED for DEEPX NPU support (StrEnum compatibility)"
        echo "Please install Python 3.11 or higher"
    else
        echo "Please install Python 3.11 or specify a different version with --python"
    fi
    exit 1
fi

# Verify Python version (3.11+ required for NPU due to StrEnum)
if [ "$SETUP_NPU" = "true" ]; then
    PYTHON_VERSION_NUM=$($PYTHON_VERSION -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${YELLOW}Detected Python version: $PYTHON_VERSION_NUM${NC}"
    
    # Check if version is 3.11 or higher
    if [[ $(echo -e "$PYTHON_VERSION_NUM\n3.11" | sort -V | head -n 1) != "3.11" ]]; then
        echo -e "${RED}Error: Python $PYTHON_VERSION_NUM is too old for DEEPX NPU support${NC}"
        echo "Python 3.11+ is REQUIRED for DEEPX NPU (dx_baidu_gui uses StrEnum)"
        echo ""
        echo "Options:"
        echo "  1. Install Python 3.11+: sudo apt install python3.11"
        echo "  2. Run without NPU: $0 --no-npu"
        exit 1
    fi
    echo -e "${GREEN}✓ Python version $PYTHON_VERSION_NUM meets requirements (3.11+)${NC}"
fi

# Check system dependencies
echo -e "${YELLOW}Checking system dependencies...${NC}"
MISSING_DEPS=()

if ! dpkg -l | grep -q libgl1; then
    MISSING_DEPS+=("libgl1")
fi
if ! dpkg -l | grep -q libglib2.0-0; then
    MISSING_DEPS+=("libglib2.0-0")
fi
if ! dpkg -l | grep -q libgomp1; then
    MISSING_DEPS+=("libgomp1")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing system dependencies: ${MISSING_DEPS[*]}${NC}"
    echo "Install them with:"
    echo "  sudo apt-get update && sudo apt-get install -y ${MISSING_DEPS[*]}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check deepx path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPX_PATH="$SCRIPT_DIR/../../../deepx"

if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${YELLOW}Checking deepx path...${NC}"
    if [ ! -d "$DEEPX_PATH" ]; then
        echo -e "${RED}Error: deepx not found at $DEEPX_PATH${NC}"
        echo "Expected path: $(realpath "$DEEPX_PATH" 2>/dev/null || echo "$DEEPX_PATH")"
        echo ""
        echo "Please ensure deepx is available at the correct location."
        exit 1
    fi
    
    if [ ! -d "$DEEPX_PATH/engine" ]; then
        echo -e "${RED}Error: deepx/engine directory not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ deepx found at: $(realpath "$DEEPX_PATH")${NC}"
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"

# Check existing venv Python version
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at: $VENV_DIR${NC}"
    
    # Check venv Python version
    if [ -f "$VENV_DIR/bin/python" ]; then
        VENV_PYTHON_VERSION=$($VENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        echo "Existing venv Python version: $VENV_PYTHON_VERSION"
        
        # Check if NPU setup requires Python 3.11+
        if [ "$SETUP_NPU" = "true" ]; then
            if [[ $(echo -e "$VENV_PYTHON_VERSION\n3.11" | sort -V | head -n 1) != "3.11" ]]; then
                echo -e "${RED}⚠ Warning: Existing venv uses Python $VENV_PYTHON_VERSION${NC}"
                echo -e "${RED}  DEEPX NPU requires Python 3.11+ (StrEnum compatibility)${NC}"
                echo ""
                echo "The virtual environment needs to be recreated with Python 3.11+"
                read -p "Remove existing venv and recreate with $PYTHON_VERSION? (Y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Nn]$ ]]; then
                    echo -e "${RED}Setup aborted.${NC}"
                    echo "Cannot proceed with NPU setup using Python $VENV_PYTHON_VERSION"
                    echo "Options:"
                    echo "  1. Remove venv manually: rm -rf $VENV_DIR"
                    echo "  2. Run without NPU: $0 --no-npu"
                    exit 1
                else
                    echo -e "${YELLOW}Removing old venv...${NC}"
                    rm -rf "$VENV_DIR"
                fi
            else
                echo -e "${GREEN}✓ Existing venv Python version is compatible (3.11+)${NC}"
                read -p "Remove and recreate anyway? (y/N) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo -e "${YELLOW}Removing venv...${NC}"
                    rm -rf "$VENV_DIR"
                else
                    echo -e "${YELLOW}Keeping existing venv...${NC}"
                fi
            fi
        else
            # CPU-only mode: just ask if user wants to recreate
            read -p "Remove existing venv and recreate? (Y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                echo -e "${YELLOW}Keeping existing venv...${NC}"
            else
                echo -e "${YELLOW}Removing venv...${NC}"
                rm -rf "$VENV_DIR"
            fi
        fi
    fi
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating new virtual environment with $PYTHON_VERSION...${NC}"
    
    # Create virtual environment with Python 3.11 (if available) or specified version
    if command -v python3.11 &> /dev/null && [ "$PYTHON_VERSION" = "python3.11" ]; then
        python3.11 -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Using Python 3.11 for virtual environment${NC}"
    else
        $PYTHON_VERSION -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Using $PYTHON_VERSION for virtual environment${NC}"
    fi
else
    echo -e "${GREEN}✓ Using existing virtual environment${NC}"
fi

source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install PaddlePaddle
echo -e "${YELLOW}Installing PaddlePaddle 3.0.0 (${DEVICE_TYPE})...${NC}"
if [ "$DEVICE_TYPE" = "gpu" ]; then
    python -m pip install paddlepaddle-gpu==3.0.0
else
    python -m pip install paddlepaddle==3.0.0
fi

# Build DX_RT and install NPU dependencies
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${BLUE}=== Building DX_RT and Installing NPU Dependencies ===${NC}"
    
    # Validate dx_rt path
    if [ ! -d "$DX_RT_PATH" ]; then
        echo -e "${RED}Error: DX_RT path does not exist: $DX_RT_PATH${NC}"
        exit 1
    fi
    
    if [ ! -f "$DX_RT_PATH/build.sh" ]; then
        echo -e "${RED}Error: build.sh not found in DX_RT path: $DX_RT_PATH/build.sh${NC}"
        exit 1
    fi
    
    # Build dx_rt
    echo -e "${YELLOW}Building DX_RT from: $DX_RT_PATH${NC}"
    CURRENT_DIR="$(pwd)"
    cd "$DX_RT_PATH"
    ./build.sh
    BUILD_EXIT_CODE=$?
    cd "$CURRENT_DIR"
    
    if [ $BUILD_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}Error: DX_RT build failed with exit code $BUILD_EXIT_CODE${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ DX_RT build completed successfully${NC}"
    
    # Install PyTorch
    echo -e "${YELLOW}Installing PyTorch ${TORCH_VERSION}...${NC}"
    python -m pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCH_VERSION}
    
    # Install ONNX Runtime
    echo -e "${YELLOW}Installing ONNX Runtime ${ONNXRUNTIME_VERSION}...${NC}"
    python -m pip install onnxruntime==${ONNXRUNTIME_VERSION}
    
    # Install additional dependencies for deepx engine
    echo -e "${YELLOW}Installing additional NPU dependencies...${NC}"
    python -m pip install \
        scikit-image \
        imgaug \
        shapely \
        pyclipper \
        jiwer
    
    echo -e "${GREEN}✓ NPU dependencies installed (dx-engine installed via dx_rt build)${NC}"
fi

# Install PaddleOCR and FastAPI dependencies
echo -e "${YELLOW}Installing PaddleOCR ${PADDLEOCR_VERSION} and FastAPI dependencies...${NC}"
python -m pip install \
    paddleocr==${PADDLEOCR_VERSION} \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    pillow \
    opencv-python-headless

# Download PP-OCRv5 models
if [ "$DOWNLOAD_MODELS" = "true" ]; then
    echo -e "${YELLOW}Downloading PP-OCRv5 models...${NC}"
    
    MODELS_DIR="$HOME/.paddlex/official_models"
    mkdir -p "$MODELS_DIR"
    cd "$MODELS_DIR"
    
    # Model URLs
    if [ "$MODEL_TYPE" = "mobile" ]; then
        echo -e "${YELLOW}Using mobile models...${NC}"
        MODELS=(
            "PP-OCRv5_mobile_det:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar"
            "PP-OCRv5_mobile_rec:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar"
            "PP-LCNet_x1_0_doc_ori:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar"
            "UVDoc:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar"
            "PP-LCNet_x1_0_textline_ori:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar"
        )
    else
        echo -e "${YELLOW}Using server models...${NC}"
        MODELS=(
            "PP-OCRv5_server_det:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar"
            "PP-OCRv5_server_rec:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar"
            "PP-LCNet_x1_0_doc_ori:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar"
            "UVDoc:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar"
            "PP-LCNet_x1_0_textline_ori:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar"
        )
    fi
    
    for model in "${MODELS[@]}"; do
        MODEL_NAME="${model%%:*}"
        MODEL_URL="${model##*:}"
        TAR_FILE="${MODEL_URL##*/}"
        
        if [ ! -d "$MODEL_NAME" ]; then
            echo -e "${YELLOW}Downloading $MODEL_NAME...${NC}"
            wget -q --show-progress "$MODEL_URL"
            tar -xf "$TAR_FILE"
            # Remove _infer suffix from extracted directory
            EXTRACTED_DIR="${TAR_FILE%.tar}"
            if [ -d "$EXTRACTED_DIR" ] && [ "$EXTRACTED_DIR" != "$MODEL_NAME" ]; then
                mv "$EXTRACTED_DIR" "$MODEL_NAME"
            fi
            rm -f "$TAR_FILE"
            echo -e "${GREEN}✓ $MODEL_NAME downloaded${NC}"
        else
            echo -e "${GREEN}✓ $MODEL_NAME already exists${NC}"
        fi
    done
    
    cd - > /dev/null
fi

# Download DEEPX models
if [ "$DOWNLOAD_DEEPX_MODELS" = "true" ] && [ "$SETUP_NPU" = "true" ]; then
    echo -e "${BLUE}=== Checking DEEPX NPU Models ===${NC}"
    
    DEEPX_MODELS_DIR="$DEEPX_PATH/engine/model_files"
    DXNN_DIR="$DEEPX_MODELS_DIR/server"
    DXNN_MOBILE_DIR="$DEEPX_MODELS_DIR/mobile"
    
    NEED_SETUP=0
    if [ ! -d "$DXNN_DIR" ]; then
        echo -e "${YELLOW}⚠ Missing model folder: $DXNN_DIR${NC}"
        NEED_SETUP=1
    fi
    if [ ! -d "$DXNN_MOBILE_DIR" ]; then
        echo -e "${YELLOW}⚠ Missing model folder: $DXNN_MOBILE_DIR${NC}"
        NEED_SETUP=1
    fi
    
    if [ $NEED_SETUP -eq 1 ]; then
        # Try to find setup.sh in deepx directories
        SETUP_SCRIPT=""
        if [ -f "$DEEPX_PATH/setup.sh" ]; then
            SETUP_SCRIPT="$DEEPX_PATH/setup.sh"
        fi
        
        if [ -n "$SETUP_SCRIPT" ]; then
            echo -e "${YELLOW}Running setup.sh to fetch DEEPX models...${NC}"
            echo "Setup script: $SETUP_SCRIPT"
            echo "Target directory: $DEEPX_MODELS_DIR"
            
            CURRENT_DIR="$(pwd)"
            cd "$(dirname "$SETUP_SCRIPT")"
            bash "$(basename "$SETUP_SCRIPT")" --dest="$DEEPX_MODELS_DIR"
            SETUP_EXIT_CODE=$?
            cd "$CURRENT_DIR"
            
            if [ $SETUP_EXIT_CODE -eq 0 ]; then
                echo -e "${GREEN}✓ setup.sh completed successfully${NC}"
            else
                echo -e "${RED}Error: setup.sh failed to prepare models (exit code: $SETUP_EXIT_CODE)${NC}"
                exit 1
            fi
        else
            echo -e "${RED}Error: setup.sh not found${NC}"
            echo "Searched locations:"
            echo "  - $DEEPX_PATH/../dx_baidu_gui/setup.sh"
            echo "  - $DEEPX_PATH/setup.sh"
            echo ""
            echo "Please provide DEEPX models manually under $DEEPX_MODELS_DIR/"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ DEEPX model folders present${NC}"
    fi
    
    # Verify models were downloaded successfully
    if [ -d "$DXNN_DIR" ] && [ -d "$DXNN_MOBILE_DIR" ]; then
        echo -e "${GREEN}✓ DEEPX models verified in $DEEPX_MODELS_DIR${NC}"
        echo "  Server models: ✓"
        echo "  Mobile models: ✓"
    else
        echo -e "${YELLOW}⚠ DEEPX models verification incomplete${NC}"
        echo "  Server models: $([ -d "$DXNN_DIR" ] && echo "✓" || echo "✗")"
        echo "  Mobile models: $([ -d "$DXNN_MOBILE_DIR" ] && echo "✓" || echo "✗")"
    fi
fi

# Create environment configuration file
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${YELLOW}Creating NPU environment configuration...${NC}"
    
    ENV_FILE="$SCRIPT_DIR/deepx_env.sh"
    cat > "$ENV_FILE" << EOF
#!/bin/bash
# DEEPX NPU Environment Configuration
# Source this file before running the FastAPI service with NPU support
# Usage: source ./deepx_env.sh

# RT Optimization Settings (from dx_baidu_gui/set_env.sh)
export CUSTOM_INTER_OP_THREADS_COUNT=${CUSTOM_INTER_OP_THREADS_COUNT}
export CUSTOM_INTRA_OP_THREADS_COUNT=${CUSTOM_INTRA_OP_THREADS_COUNT}

# Optional settings (uncomment and set values if needed)
# export DXRT_DYNAMIC_CPU_THREAD=3
# export DXRT_TASK_MAX_LOAD=4
# export NFH_INPUT_WORKER_THREADS=5
# export NFH_OUTPUT_WORKER_THREADS=6

# Library paths
export LD_LIBRARY_PATH="\${VIRTUAL_ENV}/lib:\${LD_LIBRARY_PATH}"

echo "✓ DEEPX NPU environment configured"
echo "  CUSTOM_INTER_OP_THREADS_COUNT=\${CUSTOM_INTER_OP_THREADS_COUNT}"
echo "  CUSTOM_INTRA_OP_THREADS_COUNT=\${CUSTOM_INTRA_OP_THREADS_COUNT}"
EOF
    
    chmod +x "$ENV_FILE"
    echo -e "${GREEN}✓ Environment configuration saved to: deepx_env.sh${NC}"
    
    # Apply environment settings
    source "$ENV_FILE"
fi

# Verify installation
echo -e "${BLUE}=== Verifying Installation ===${NC}"

echo -e "${YELLOW}Checking PaddleOCR...${NC}"
python -c "import paddleocr; print(f'PaddleOCR version: {paddleocr.__version__}')" && \
    echo -e "${GREEN}✓ PaddleOCR installed${NC}" || \
    echo -e "${RED}✗ PaddleOCR not found${NC}"

if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${YELLOW}Checking NPU dependencies...${NC}"
    
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
        echo -e "${GREEN}✓ PyTorch installed${NC}" || \
        echo -e "${RED}✗ PyTorch not found${NC}"
    
    python -c "import dx_engine; print(f'dx-engine version: {dx_engine.__version__}')" && \
        echo -e "${GREEN}✓ dx-engine installed${NC}" || \
        echo -e "${RED}✗ dx-engine not found${NC}"
    
    if [ -d "$DEEPX_PATH/engine" ]; then
        echo -e "${GREEN}✓ deepx/engine available${NC}"
    else
        echo -e "${RED}✗ deepx/engine not found${NC}"
    fi
fi

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Virtual environment created at: $VENV_DIR"
echo ""
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${BLUE}NPU Support Enabled:${NC}"
    echo "  - DX_RT: $DX_RT_PATH"
    echo "  - PyTorch: $TORCH_VERSION"
    echo "  - dx-engine: installed via dx_rt build"
    echo "  - Environment config: deepx_env.sh"
    echo ""
fi
echo "To activate the virtual environment and NPU settings:"
echo -e "${YELLOW}  source $VENV_DIR/bin/activate${NC}"
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${YELLOW}  source deepx_env.sh${NC}"
fi
echo ""
echo "To start the OCR service:"
echo -e "${YELLOW}  ./run.sh${NC}"
if [ "$SETUP_NPU" = "true" ]; then
    echo ""
    echo "To test NPU functionality:"
    echo -e "${YELLOW}  ./run_tests.sh${NC}"
fi
echo ""
echo "The service will be available at:"
echo -e "${GREEN}  http://localhost:8080${NC}"
echo -e "${GREEN}  API docs: http://localhost:8080/docs${NC}"
echo ""
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${BLUE}NPU API Usage:${NC}"
    echo "  CPU mode: POST /api/v1/ocr with deepx=false (default)"
    echo "  NPU mode: POST /api/v1/ocr with deepx=true"
    echo ""
fi
