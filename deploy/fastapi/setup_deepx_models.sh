#!/bin/bash

# DEEPX NPU Models Setup Script
# Downloads DEEPX models by calling setup.sh
# Can be used by both local_deepx_setup.sh and Dockerfile

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DEEPX_PATH=""
FORCE_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --deepx-path)
            DEEPX_PATH="$2"
            shift 2
            ;;
        --force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        -h|--help)
            echo -e "${BLUE}Usage:${NC}"
            echo -e "  $0 --deepx-path PATH [OPTIONS]"
            echo ""
            echo -e "${BLUE}Options:${NC}"
            echo -e "  ${GREEN}--deepx-path PATH${NC}    ${YELLOW}Path to deepx directory (REQUIRED)${NC}"
            echo -e "  ${GREEN}--force${NC}               ${YELLOW}Force download even if models exist${NC}"
            echo -e "  ${GREEN}-h, --help${NC}            ${YELLOW}Show this help message${NC}"
            echo ""
            echo -e "${BLUE}Example:${NC}"
            echo -e "  ${GREEN}$0 --deepx-path /app/deepx${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DEEPX_PATH" ]; then
    echo -e "${RED}Error: --deepx-path is required${NC}"
    echo "Usage: $0 --deepx-path PATH"
    exit 1
fi

# Validate deepx path
if [ ! -d "$DEEPX_PATH" ]; then
    echo -e "${RED}Error: DEEPX path does not exist: $DEEPX_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}=== DEEPX NPU Models Setup ===${NC}"
echo -e "${YELLOW}DEEPX Path:${NC} $DEEPX_PATH"

DEEPX_MODELS_DIR="$DEEPX_PATH/engine/model_files"
DXNN_DIR="$DEEPX_MODELS_DIR/server"
DXNN_MOBILE_DIR="$DEEPX_MODELS_DIR/mobile"

# Check if models already exist
NEED_SETUP=0
if [ "$FORCE_DOWNLOAD" = true ]; then
    echo -e "${YELLOW}Force download enabled${NC}"
    NEED_SETUP=1
else
    if [ ! -d "$DXNN_DIR" ]; then
        echo -e "${YELLOW}⚠ Missing model folder: $DXNN_DIR${NC}"
        NEED_SETUP=1
    fi
    if [ ! -d "$DXNN_MOBILE_DIR" ]; then
        echo -e "${YELLOW}⚠ Missing model folder: $DXNN_MOBILE_DIR${NC}"
        NEED_SETUP=1
    fi
fi

if [ $NEED_SETUP -eq 0 ]; then
    echo -e "${GREEN}✓ DEEPX model folders already present${NC}"
    echo -e "${GREEN}  Server models: $DXNN_DIR${NC}"
    echo -e "${GREEN}  Mobile models: $DXNN_MOBILE_DIR${NC}"
    exit 0
fi

# Find setup.sh script
SETUP_SCRIPT=""
if [ -f "$DEEPX_PATH/setup.sh" ]; then
    SETUP_SCRIPT="$DEEPX_PATH/setup.sh"
fi

if [ -z "$SETUP_SCRIPT" ]; then
    echo -e "${RED}Error: setup.sh not found${NC}"
    echo "Searched locations:"
    echo "  - $DEEPX_PATH/setup.sh"
    echo ""
    echo "Please ensure setup.sh is available in the deepx directory."
    echo "Or provide DEEPX models manually under $DEEPX_MODELS_DIR/"
    exit 1
fi

# Run setup.sh to download models
echo -e "${YELLOW}Running setup.sh to fetch DEEPX models...${NC}"
echo -e "${YELLOW}Setup script:${NC} $SETUP_SCRIPT"
echo -e "${YELLOW}Target directory:${NC} $DEEPX_MODELS_DIR"

CURRENT_DIR="$(pwd)"
cd "$(dirname "$SETUP_SCRIPT")"

# Execute setup.sh with destination parameter
bash "$(basename "$SETUP_SCRIPT")" --dest="$DEEPX_MODELS_DIR"
SETUP_EXIT_CODE=$?

cd "$CURRENT_DIR"

if [ $SETUP_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Error: setup.sh failed to prepare models (exit code: $SETUP_EXIT_CODE)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ setup.sh completed successfully${NC}"

# Verify models were downloaded successfully
if [ -d "$DXNN_DIR" ] && [ -d "$DXNN_MOBILE_DIR" ]; then
    echo -e "${GREEN}✓ DEEPX models verified in $DEEPX_MODELS_DIR${NC}"
    echo -e "${GREEN}  Server models: ✓${NC}"
    echo -e "${GREEN}  Mobile models: ✓${NC}"
    
    # Show model file counts
    SERVER_COUNT=$(find "$DXNN_DIR" -name "*.dxnn" 2>/dev/null | wc -l || echo "0")
    MOBILE_COUNT=$(find "$DXNN_MOBILE_DIR" -name "*.dxnn" 2>/dev/null | wc -l || echo "0")
    echo -e "${BLUE}  Server .dxnn files: $SERVER_COUNT${NC}"
    echo -e "${BLUE}  Mobile .dxnn files: $MOBILE_COUNT${NC}"
else
    echo -e "${YELLOW}⚠ DEEPX models verification incomplete${NC}"
    echo "  Server models: $([ -d "$DXNN_DIR" ] && echo "✓" || echo "✗")"
    echo "  Mobile models: $([ -d "$DXNN_MOBILE_DIR" ] && echo "✓" || echo "✗")"
    exit 1
fi

echo -e "${GREEN}✓ DEEPX models setup complete${NC}"
