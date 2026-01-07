#!/bin/bash

# PaddleOCR FastAPI Service Local Setup Script
# This script creates a virtual environment and sets up the OCR service locally

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
PYTHON_VERSION="python3"  # Will auto-detect best available version
VENV_DIR="venv"
DEVICE_TYPE="cpu"  # cpu or gpu
DOWNLOAD_MODELS="true"

# Default values
MODEL_TYPE="mobile"  # default: mobile
USE_MOBILE_FLAG=false
USE_SERVER_FLAG=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo -e "${BLUE}Usage:${NC}"
            echo -e "  $0 [OPTIONS]"
            echo ""
            echo -e "${BLUE}Options:${NC}"
            echo -e "  ${GREEN}--gpu${NC}              ${YELLOW}Install GPU version of PaddlePaddle${NC}"
            echo -e "  ${GREEN}--use-mobile${NC}       ${YELLOW}Use mobile version models instead of server models${NC}"
            echo -e "  ${GREEN}--use-server${NC}       ${YELLOW}Use server models${NC}"
            echo -e "  ${GREEN}--no-models${NC}        ${YELLOW}Skip downloading models${NC}"
            echo -e "  ${GREEN}--python${NC} VERSION   ${YELLOW}Specify Python version (default: auto-detect)${NC}"
            echo -e "  ${GREEN}-h, --help${NC}         ${YELLOW}Show this help message${NC}"
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

# Disable NPU for local_setup.sh (CPU only)
export SETUP_NPU="false"

# Remove deepx_env.sh if it exists (from previous NPU setup)
DEEPX_ENV_FILE="$SCRIPT_DIR/deepx_env.sh"
if [ -f "$DEEPX_ENV_FILE" ]; then
    echo -e "${YELLOW}Removing existing deepx_env.sh (CPU-only setup)...${NC}"
    rm -f "$DEEPX_ENV_FILE"
fi

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${BLUE}PaddleOCR FastAPI Service Setup${NC}                     ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo -e "${YELLOW}Python:${NC}           $PYTHON_VERSION"
echo -e "${YELLOW}Device:${NC}           $DEVICE_TYPE"
echo -e "${YELLOW}Model Type:${NC}       $MODEL_TYPE"
echo -e "${YELLOW}Download Models:${NC}  $DOWNLOAD_MODELS"
echo ""

# Check if Python is available
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_VERSION is not installed${NC}"
    echo "Please install Python 3.10 or specify a different version with --python"
    exit 1
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

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"

# Check existing venv
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at: $VENV_DIR${NC}"
    
    # Check venv Python version
    if [ -f "$VENV_DIR/bin/python" ]; then
        VENV_PYTHON_VERSION=$($VENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        echo "Existing venv Python version: $VENV_PYTHON_VERSION"
    fi
    
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${GREEN}✓ Keeping existing virtual environment${NC}"
        echo -e "${YELLOW}Skipping to package installation...${NC}"
        source "$VENV_DIR/bin/activate"
        SKIP_VENV_CREATION=true
    fi
fi

# Create venv if it doesn't exist or was removed
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
    
    source "$VENV_DIR/bin/activate"
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install dependencies from requirements file
if [ "$DEVICE_TYPE" = "gpu" ]; then
    echo -e "${YELLOW}Installing PaddlePaddle and dependencies (GPU)...${NC}"
    python -m pip install -r "${SCRIPT_DIR}/requirements-gpu.txt"
else
    echo -e "${YELLOW}Installing PaddlePaddle and dependencies (CPU)...${NC}"
    python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

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

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${BLUE}✓ Setup Complete!${NC}                                     ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Virtual environment:${NC} $VENV_DIR"
echo ""
echo -e "${BLUE}To activate the virtual environment:${NC}"
echo -e "  ${GREEN}source $VENV_DIR/bin/activate${NC}"
echo ""
echo -e "${BLUE}To start the OCR service:${NC}"
echo -e "  ${GREEN}./run.sh${NC}"
echo ""
echo -e "${BLUE}The service will be available at:${NC}"
echo -e "  ${GREEN}http://localhost:8080${NC}"
echo -e "  ${GREEN}http://localhost:8080/docs${NC} (API documentation)"
echo ""
