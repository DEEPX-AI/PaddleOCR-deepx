#!/bin/bash

# PaddleOCR FastAPI Service with DEEPX NPU Support - Local Setup Script
# This script creates a virtual environment and sets up the OCR service with NPU acceleration

set -e  # Exit on error

# Configuration
PYTHON_VERSION="python3"  # Will auto-detect best available version
VENV_DIR="venv"
DEVICE_TYPE="cpu"  # cpu or gpu
PADDLEOCR_VERSION="3.3.2"
DOWNLOAD_MODELS="true"
DOWNLOAD_DEEPX_MODELS="true"
SETUP_NPU="true"

# Default values
MODEL_TYPE="mobile"  # default: mobile
USE_MOBILE_FLAG=false
USE_SERVER_FLAG=false

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
        --use-mobile)
            USE_MOBILE_FLAG=true
            MODEL_TYPE="mobile"
            export USE_MOBILE="true"
            shift
            ;;
        --use-server)
            USE_SERVER_FLAG=true
            MODEL_TYPE="server"
            export USE_MOBILE="false"
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
        -h|--help)
            echo -e "${BLUE}Usage:${NC}"
            echo -e "  $0 [OPTIONS]"
            echo ""
            echo -e "${BLUE}Options:${NC}"
            echo -e "  ${GREEN}--dx_rt${NC} PATH         ${YELLOW}Path to dx_rt directory (REQUIRED for NPU setup)${NC}"
            echo -e "  ${GREEN}--gpu${NC}                ${YELLOW}Install GPU version of PaddlePaddle${NC}"
            echo -e "  ${GREEN}--use-mobile${NC}         ${YELLOW}Use mobile version models instead of server models${NC}"
            echo -e "  ${GREEN}--use-server${NC}         ${YELLOW}Use server models${NC}"
            echo -e "  ${GREEN}--no-models${NC}          ${YELLOW}Skip downloading PaddleOCR models${NC}"
            echo -e "  ${GREEN}--no-deepx-models${NC}    ${YELLOW}Skip downloading DEEPX models${NC}"
            echo -e "  ${GREEN}--no-npu${NC}             ${YELLOW}Skip NPU setup (CPU only)${NC}"
            echo -e "  ${GREEN}--python${NC} VERSION     ${YELLOW}Specify Python version (default: auto-detect, 3.10+ required for NPU)${NC}"
            echo -e "  ${GREEN}--version${NC} VERSION    ${YELLOW}Specify PaddleOCR version (default: 3.3.2)${NC}"
            echo -e "  ${GREEN}--inter-threads${NC} N    ${YELLOW}Set CUSTOM_INTER_OP_THREADS_COUNT (default: 1)${NC}"
            echo -e "  ${GREEN}--intra-threads${NC} N    ${YELLOW}Set CUSTOM_INTRA_OP_THREADS_COUNT (default: 3)${NC}"
            echo -e "  ${GREEN}-h, --help${NC}           ${YELLOW}Show this help message${NC}"
            echo ""
            echo -e "${BLUE}Example:${NC}"
            echo -e "  ${GREEN}$0 --dx_rt /path/to/dx_rt${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for conflicting options
if [ "$USE_MOBILE_FLAG" = true ] && [ "$USE_SERVER_FLAG" = true ]; then
    echo -e "${RED}❌ Error: --use-mobile and --use-server cannot be used together${NC}"
    echo -e "${YELLOW}Please specify only one model type option.${NC}"
    exit 1
fi

# Set default if no flag was specified
if [ "$USE_MOBILE_FLAG" = false ] && [ "$USE_SERVER_FLAG" = false ]; then
    export USE_MOBILE="true"  # Default to mobile
fi

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${BLUE}PaddleOCR FastAPI Service with DEEPX NPU Setup${NC}       ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo -e "${YELLOW}Python:${NC}               $PYTHON_VERSION"
echo -e "${YELLOW}Device:${NC}               $DEVICE_TYPE"
echo -e "${YELLOW}Model Type:${NC}           $MODEL_TYPE"
echo -e "${YELLOW}PaddleOCR Version:${NC}    $PADDLEOCR_VERSION"
echo -e "${YELLOW}Download Models:${NC}      $DOWNLOAD_MODELS"
echo -e "${YELLOW}Download DEEPX Models:${NC} $DOWNLOAD_DEEPX_MODELS"
echo -e "${YELLOW}Setup NPU:${NC}            $SETUP_NPU"
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "  ${BLUE}├─${NC} DX_RT Path: $DX_RT_PATH"
    echo -e "  ${BLUE}├─${NC} PyTorch: $TORCH_VERSION"
    echo -e "  ${BLUE}└─${NC} RT Optimization: INTER=$CUSTOM_INTER_OP_THREADS_COUNT, INTRA=$CUSTOM_INTRA_OP_THREADS_COUNT"
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
        echo "python 3.10+ is REQUIRED for DEEPX NPU support (StrEnum compatibility)"
        echo "Please install python 3.10 or higher"
    else
        echo "Please install python 3.10 or specify a different version with --python"
    fi
    exit 1
fi

# Verify Python version (3.10+ required for NPU due to StrEnum)
if [ "$SETUP_NPU" = "true" ]; then
    PYTHON_VERSION_NUM=$($PYTHON_VERSION -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${YELLOW}Detected Python version: $PYTHON_VERSION_NUM${NC}"
    
    # Check if version is 3.10 or higher
    if [[ $(echo -e "$PYTHON_VERSION_NUM\n3.10" | sort -V | head -n 1) != "3.10" ]]; then
        echo -e "${RED}Error: Python $PYTHON_VERSION_NUM is too old for DEEPX NPU support${NC}"
        echo "python 3.10+ is REQUIRED for DEEPX NPU (dx_baidu_gui uses StrEnum)"
        echo ""
        echo "Options:"
        echo "  1. Install Python 3.10+: sudo apt install python3.10"
        echo "  2. Run without NPU: $0 --no-npu"
        exit 1
    fi
    echo -e "${GREEN}✓ Python version $PYTHON_VERSION_NUM meets requirements (3.10+)${NC}"
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
DEEPX_PATH="$SCRIPT_DIR/deepx"

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
        
        # Check if NPU setup requires python 3.10+
        if [ "$SETUP_NPU" = "true" ]; then
            if [[ $(echo -e "$VENV_PYTHON_VERSION\n3.10" | sort -V | head -n 1) != "3.10" ]]; then
                echo -e "${RED}⚠ Warning: Existing venv uses Python $VENV_PYTHON_VERSION${NC}"
                echo -e "${RED}  DEEPX NPU requires python 3.10+ (StrEnum compatibility)${NC}"
                echo ""
                echo "The virtual environment needs to be recreated with python 3.10+"
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
                echo -e "${GREEN}✓ Existing venv Python version is compatible (3.10+)${NC}"
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
    # Auto-detect best Python version if using default
    if [ "$PYTHON_VERSION" = "python3" ]; then
        DETECTED_VERSION=$($PYTHON_VERSION -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        echo -e "${YELLOW}Creating new virtual environment with $PYTHON_VERSION (version $DETECTED_VERSION)...${NC}"
    else
        echo -e "${YELLOW}Creating new virtual environment with $PYTHON_VERSION...${NC}"
    fi
    
    $PYTHON_VERSION -m venv "$VENV_DIR"
    
    # Get actual venv Python version for confirmation
    VENV_VERSION=$($VENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}✓ Created virtual environment with Python $VENV_VERSION${NC}"
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
    
    # Check if sudo is required and validate credentials early
    echo -e "${YELLOW}DX_RT build requires sudo privileges...${NC}"
    if ! sudo -n true 2>/dev/null; then
        echo -e "${YELLOW}Please enter your sudo password:${NC}"
        if ! sudo -v; then
            echo -e "${RED}Error: Failed to obtain sudo privileges${NC}"
            echo -e "${RED}DX_RT build cannot proceed without sudo access${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Sudo credentials validated${NC}"
    else
        echo -e "${GREEN}✓ Sudo credentials already cached${NC}"
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
        echo -e "${BLUE}Using mobile models...${NC}"
        MODELS=(
            "PP-OCRv5_mobile_det:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar"
            "PP-OCRv5_mobile_rec:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar"
            "PP-LCNet_x1_0_doc_ori:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar"
            "UVDoc:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar"
            "PP-LCNet_x1_0_textline_ori:https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar"
        )
    else
        echo -e "${BLUE}Using server models...${NC}"
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
        MODEL_URL="${model#*:}"
        TAR_FILE="${MODEL_URL##*/}"
        
        if [ ! -d "$MODEL_NAME" ]; then
            echo -e "${YELLOW}Downloading $MODEL_NAME...${NC}"
            if ! wget --progress=bar:force:noscroll "$MODEL_URL" 2>&1; then
                echo -e "${RED}Error: Failed to download $MODEL_NAME${NC}"
                echo -e "${RED}URL: $MODEL_URL${NC}"
                exit 1
            fi
            if [ ! -f "$TAR_FILE" ]; then
                echo -e "${RED}Error: Downloaded file $TAR_FILE not found${NC}"
                exit 1
            fi
            echo -e "${YELLOW}Extracting $MODEL_NAME...${NC}"
            if ! tar -xf "$TAR_FILE"; then
                echo -e "${RED}Error: Failed to extract $TAR_FILE${NC}"
                exit 1
            fi
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
    # Use common setup script
    SETUP_MODELS_SCRIPT="$SCRIPT_DIR/setup_deepx_models.sh"
    
    if [ ! -f "$SETUP_MODELS_SCRIPT" ]; then
        echo -e "${RED}Error: setup_deepx_models.sh not found at $SETUP_MODELS_SCRIPT${NC}"
        exit 1
    fi
    
    bash "$SETUP_MODELS_SCRIPT" --deepx-path "$DEEPX_PATH"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: DEEPX models setup failed${NC}"
        exit 1
    fi
fi

# Create environment configuration file
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${YELLOW}Creating NPU environment configuration...${NC}"
    
    # Read default values from .env.deepx if it exists
    ENV_DEEPX_FILE="$SCRIPT_DIR/.env.deepx"
    if [ -f "$ENV_DEEPX_FILE" ]; then
        echo -e "${YELLOW}Reading default values from .env.deepx...${NC}"
        DEFAULT_INTER=$(grep "^CUSTOM_INTER_OP_THREADS_COUNT=" "$ENV_DEEPX_FILE" | cut -d'=' -f2)
        DEFAULT_INTRA=$(grep "^CUSTOM_INTRA_OP_THREADS_COUNT=" "$ENV_DEEPX_FILE" | cut -d'=' -f2)
        DEFAULT_DYNAMIC=$(grep "^DXRT_DYNAMIC_CPU_THREAD=" "$ENV_DEEPX_FILE" | cut -d'=' -f2)
        DEFAULT_MAX_LOAD=$(grep "^DXRT_TASK_MAX_LOAD=" "$ENV_DEEPX_FILE" | cut -d'=' -f2)
        DEFAULT_INPUT=$(grep "^NFH_INPUT_WORKER_THREADS=" "$ENV_DEEPX_FILE" | cut -d'=' -f2)
        DEFAULT_OUTPUT=$(grep "^NFH_OUTPUT_WORKER_THREADS=" "$ENV_DEEPX_FILE" | cut -d'=' -f2)
        
        # Use extracted values or fallback to hardcoded defaults
        DEFAULT_INTER=${DEFAULT_INTER:-1}
        DEFAULT_INTRA=${DEFAULT_INTRA:-2}
        DEFAULT_DYNAMIC=${DEFAULT_DYNAMIC:-1}
        DEFAULT_MAX_LOAD=${DEFAULT_MAX_LOAD:-3}
        DEFAULT_INPUT=${DEFAULT_INPUT:-2}
        DEFAULT_OUTPUT=${DEFAULT_OUTPUT:-4}
    else
        echo -e "${YELLOW}.env.deepx not found, using hardcoded defaults...${NC}"
        DEFAULT_INTER=1
        DEFAULT_INTRA=2
        DEFAULT_DYNAMIC=1
        DEFAULT_MAX_LOAD=3
        DEFAULT_INPUT=2
        DEFAULT_OUTPUT=4
    fi
    
    ENV_FILE="$SCRIPT_DIR/deepx_env.sh"
    cat > "$ENV_FILE" << ENVEOF
#!/bin/bash
# DEEPX NPU Environment Configuration
# Source this file before running the FastAPI service with NPU support
# Usage: source ./deepx_env.sh [CUSTOM_INTER_OP_THREADS_COUNT] [CUSTOM_INTRA_OP_THREADS_COUNT] [DXRT_DYNAMIC_CPU_THREAD] [DXRT_TASK_MAX_LOAD] [NFH_INPUT_WORKER_THREADS] [NFH_OUTPUT_WORKER_THREADS]
# Example: source ./deepx_env.sh 1 2 1 3 2 4
# Default values (from .env.deepx): $DEFAULT_INTER $DEFAULT_INTRA $DEFAULT_DYNAMIC $DEFAULT_MAX_LOAD $DEFAULT_INPUT $DEFAULT_OUTPUT
# Use -1 to unset a specific variable: source ./deepx_env.sh 1 -1 3 4 5 6

# Set default values (loaded from .env.deepx)
DEFAULT_CUSTOM_INTER_OP_THREADS_COUNT=$DEFAULT_INTER
DEFAULT_CUSTOM_INTRA_OP_THREADS_COUNT=$DEFAULT_INTRA
DEFAULT_DXRT_DYNAMIC_CPU_THREAD=$DEFAULT_DYNAMIC
DEFAULT_DXRT_TASK_MAX_LOAD=$DEFAULT_MAX_LOAD
DEFAULT_NFH_INPUT_WORKER_THREADS=$DEFAULT_INPUT
DEFAULT_NFH_OUTPUT_WORKER_THREADS=$DEFAULT_OUTPUT

if [ "\$1" = "-1" ]; then
    unset CUSTOM_INTER_OP_THREADS_COUNT
    echo "CUSTOM_INTER_OP_THREADS_COUNT unset"
elif [ -n "\$1" ]; then
    export CUSTOM_INTER_OP_THREADS_COUNT=\$1
    echo "CUSTOM_INTER_OP_THREADS_COUNT=\$CUSTOM_INTER_OP_THREADS_COUNT"
else
    export CUSTOM_INTER_OP_THREADS_COUNT=\$DEFAULT_CUSTOM_INTER_OP_THREADS_COUNT
    echo "CUSTOM_INTER_OP_THREADS_COUNT=\$CUSTOM_INTER_OP_THREADS_COUNT (default)"
fi
if [ "\$2" = "-1" ]; then
    unset CUSTOM_INTRA_OP_THREADS_COUNT
    echo "CUSTOM_INTRA_OP_THREADS_COUNT unset"
elif [ -n "\$2" ]; then
    export CUSTOM_INTRA_OP_THREADS_COUNT=\$2
    echo "CUSTOM_INTRA_OP_THREADS_COUNT=\$CUSTOM_INTRA_OP_THREADS_COUNT"
else
    export CUSTOM_INTRA_OP_THREADS_COUNT=\$DEFAULT_CUSTOM_INTRA_OP_THREADS_COUNT
    echo "CUSTOM_INTRA_OP_THREADS_COUNT=\$CUSTOM_INTRA_OP_THREADS_COUNT (default)"
fi
if [ "\$3" = "-1" ]; then
    unset DXRT_DYNAMIC_CPU_THREAD
    echo "DXRT_DYNAMIC_CPU_THREAD unset"
elif [ -n "\$3" ]; then
    export DXRT_DYNAMIC_CPU_THREAD=\$3
    echo "DXRT_DYNAMIC_CPU_THREAD=\$DXRT_DYNAMIC_CPU_THREAD"
else
    export DXRT_DYNAMIC_CPU_THREAD=\$DEFAULT_DXRT_DYNAMIC_CPU_THREAD
    echo "DXRT_DYNAMIC_CPU_THREAD=\$DXRT_DYNAMIC_CPU_THREAD (default)"
fi
if [ "\$4" = "-1" ]; then
    unset DXRT_TASK_MAX_LOAD
    echo "DXRT_TASK_MAX_LOAD unset"
elif [ -n "\$4" ]; then
    export DXRT_TASK_MAX_LOAD=\$4
    echo "DXRT_TASK_MAX_LOAD=\$DXRT_TASK_MAX_LOAD"
else
    export DXRT_TASK_MAX_LOAD=\$DEFAULT_DXRT_TASK_MAX_LOAD
    echo "DXRT_TASK_MAX_LOAD=\$DXRT_TASK_MAX_LOAD (default)"
fi
if [ "\$5" = "-1" ]; then
    unset NFH_INPUT_WORKER_THREADS
    echo "NFH_INPUT_WORKER_THREADS unset"
elif [ -n "\$5" ]; then
    export NFH_INPUT_WORKER_THREADS=\$5
    echo "NFH_INPUT_WORKER_THREADS=\$NFH_INPUT_WORKER_THREADS"
else
    export NFH_INPUT_WORKER_THREADS=\$DEFAULT_NFH_INPUT_WORKER_THREADS
    echo "NFH_INPUT_WORKER_THREADS=\$NFH_INPUT_WORKER_THREADS (default)"
fi
if [ "\$6" = "-1" ]; then
    unset NFH_OUTPUT_WORKER_THREADS
    echo "NFH_OUTPUT_WORKER_THREADS unset"
elif [ -n "\$6" ]; then
    export NFH_OUTPUT_WORKER_THREADS=\$6
    echo "NFH_OUTPUT_WORKER_THREADS=\$NFH_OUTPUT_WORKER_THREADS"
else
    export NFH_OUTPUT_WORKER_THREADS=\$DEFAULT_NFH_OUTPUT_WORKER_THREADS
    echo "NFH_OUTPUT_WORKER_THREADS=\$NFH_OUTPUT_WORKER_THREADS (default)"
fi

# Library paths
export LD_LIBRARY_PATH="\${VIRTUAL_ENV}/lib:\${LD_LIBRARY_PATH}"
ENVEOF
    
    chmod +x "$ENV_FILE"
    echo -e "${GREEN}✓ Environment configuration saved to: deepx_env.sh${NC}"
    echo -e "${GREEN}  Default values: $DEFAULT_INTER $DEFAULT_INTRA $DEFAULT_DYNAMIC $DEFAULT_MAX_LOAD $DEFAULT_INPUT $DEFAULT_OUTPUT${NC}"
    
    # Apply environment settings with values from .env.deepx
    source "$ENV_FILE" $DEFAULT_INTER $DEFAULT_INTRA $DEFAULT_DYNAMIC $DEFAULT_MAX_LOAD $DEFAULT_INPUT $DEFAULT_OUTPUT
fi

# Prepare fonts for OCR visualization
echo ""
echo -e "${YELLOW}Preparing fonts for OCR visualization...${NC}"
FONTS_DEST_DIR="$SCRIPT_DIR/deepx/engine/fonts"
mkdir -p "$FONTS_DEST_DIR"

# Copy fonts from doc/fonts to deepx/engine/fonts
DOC_FONTS_DIR="$SCRIPT_DIR/../../doc/fonts"
if [ -d "$DOC_FONTS_DIR" ]; then
    cp -v "$DOC_FONTS_DIR"/*.ttf "$FONTS_DEST_DIR/" 2>/dev/null || true
    echo -e "${GREEN}✓ Fonts copied to $FONTS_DEST_DIR${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: doc/fonts directory not found at $DOC_FONTS_DIR${NC}"
    echo -e "${YELLOW}   Skipping font copy. System fonts will be used as fallback.${NC}"
fi
echo ""

# Verify installation
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${BLUE}Verifying Installation${NC}                               ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"

echo -e "${YELLOW}Checking PaddleOCR...${NC}"
python -c "import paddleocr; print(f'PaddleOCR version: {paddleocr.__version__}')" && \
    echo -e "${GREEN}✓ PaddleOCR installed${NC}" || \
    echo -e "${RED}✗ PaddleOCR not found${NC}"

if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${YELLOW}Checking NPU dependencies...${NC}"
    
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
        echo -e "${GREEN}✓ PyTorch installed${NC}" || \
        echo -e "${RED}✗ PyTorch not found${NC}"
    
    python -c "import dx_engine; print('dx-engine: installed')" && \
        echo -e "${GREEN}✓ dx-engine installed${NC}" || \
        echo -e "${RED}✗ dx-engine not found${NC}"
    
    if [ -d "$DEEPX_PATH/engine" ]; then
        echo -e "${GREEN}✓ deepx/engine available${NC}"
    else
        echo -e "${RED}✗ deepx/engine not found${NC}"
    fi
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${BLUE}✓ Setup Complete!${NC}                                     ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Virtual environment:${NC} $VENV_DIR"
echo ""
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${BLUE}NPU Support Enabled:${NC}"
    echo -e "  ${GREEN}├─${NC} DX_RT: $DX_RT_PATH"
    echo -e "  ${GREEN}├─${NC} PyTorch: $TORCH_VERSION"
    echo -e "  ${GREEN}├─${NC} dx-engine: installed via dx_rt build"
    echo -e "  ${GREEN}└─${NC} Environment config: deepx_env.sh"
    echo ""
fi
echo -e "${BLUE}To activate the virtual environment and NPU settings:${NC}"
echo -e "  ${GREEN}source $VENV_DIR/bin/activate${NC}"
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "  ${GREEN}source deepx_env.sh${NC}"
fi
echo ""
echo -e "${BLUE}To start the OCR service:${NC}"
echo -e "  ${GREEN}./run.sh${NC}"
if [ "$SETUP_NPU" = "true" ]; then
    echo ""
    echo -e "${BLUE}To test NPU functionality:${NC}"
    echo -e "  ${GREEN}./run_tests.sh${NC}"
fi
echo ""
echo -e "${BLUE}The service will be available at:${NC}"
echo -e "  ${GREEN}http://localhost:8080${NC}"
echo -e "  ${GREEN}http://localhost:8080/docs${NC} (API documentation)"
echo ""
if [ "$SETUP_NPU" = "true" ]; then
    echo -e "${BLUE}NPU API Usage:${NC}"
    echo -e "  ${YELLOW}CPU mode:${NC} POST /api/v1/ocr with deepx=false (default)"
    echo -e "  ${YELLOW}NPU mode:${NC} POST /api/v1/ocr with deepx=true"
    echo ""
fi
