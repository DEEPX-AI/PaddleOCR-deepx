#!/bin/bash

# PaddleOCR FastAPI Service Local Run Script
# Runs the OCR service using local virtual environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
SERVICE_SCRIPT="$SCRIPT_DIR/ocr_service.py"

# Default values
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
MODEL_TYPE="mobile"  # default: mobile
USE_MOBILE_FLAG=false
USE_SERVER_FLAG=false
LAZY_LOAD_FLAG=false
PDF_DPI=""
PDF_THREAD_COUNT=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
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
        --lazy-load)
            LAZY_LOAD_FLAG=true
            export LAZY_LOAD="true"
            shift
            ;;
        --pdf-dpi)
            PDF_DPI="$2"
            shift 2
            ;;
        --pdf-thread-count)
            PDF_THREAD_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo -e "  ${GREEN}--port PORT${NC}      Service port (default: 8080)"
            echo -e "  ${GREEN}--host HOST${NC}      Bind host (default: 0.0.0.0)"
            echo -e "  ${GREEN}--use-mobile${NC}     Use mobile models (default)"
            echo -e "  ${GREEN}--use-server${NC}     Use server models"
            echo -e "  ${GREEN}--lazy-load${NC}      Enable lazy loading (load models on first request)"
            echo -e "  ${GREEN}--pdf-dpi DPI${NC}    PDF conversion DPI (default: 200, higher = better quality but slower)"
            echo -e "  ${GREEN}--pdf-thread-count N${NC}  PDF conversion threads (default: auto-detect based on CPU cores, max 4)"
            echo -e "  ${GREEN}-h, --help${NC}       Show this help message"
            echo ""
            echo -e "${YELLOW}Environment Variables:${NC}"
            echo -e "  ${GREEN}PORT${NC}             Service port"
            echo -e "  ${GREEN}HOST${NC}             Bind host"
            echo -e "  ${GREEN}USE_GPU${NC}          Use GPU (true/false)"
            echo -e "  ${GREEN}USE_MOBILE${NC}       Use mobile models (true/false)"
            echo -e "  ${GREEN}LAZY_LOAD${NC}        Enable lazy loading (true/false, default: false)"
            echo -e "  ${GREEN}PDF_DPI${NC}          PDF conversion DPI (default: 200)"
            echo -e "  ${GREEN}PDF_THREAD_COUNT${NC}  PDF conversion threads (default: auto-detect)"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  $0                       # Run on default port 8080 (mobile models)"
            echo "  $0 --port 9000           # Run on port 9000"
            echo "  USE_GPU=true $0          # Run with GPU"
            echo "  $0 --use-mobile          # Run with mobile models (default)"
            echo "  $0 --use-server          # Run with server models"
            echo "  $0 --lazy-load           # Enable lazy loading"
            echo "  $0 --pdf-dpi 150         # Use lower DPI for PDF (faster)"
            echo "  $0 --pdf-thread-count 2  # Use 2 threads for PDF conversion"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check for conflicting options
if [ "$USE_MOBILE_FLAG" = true ] && [ "$USE_SERVER_FLAG" = true ]; then
    echo -e "${RED}❌ Error: --use-mobile and --use-server cannot be used together${NC}"
    echo "Please specify only one model type option."
    exit 1
fi

# Set default if no flag was specified
if [ "$USE_MOBILE_FLAG" = false ] && [ "$USE_SERVER_FLAG" = false ]; then
    export USE_MOBILE="true"  # Default to mobile
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PaddleOCR FastAPI Local Service${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Host:${NC} $HOST"
echo -e "${YELLOW}Port:${NC} $PORT"
echo -e "${YELLOW}Model Type:${NC} $MODEL_TYPE"
if [ "$LAZY_LOAD_FLAG" = true ]; then
    echo -e "${YELLOW}Lazy Load:${NC} Enabled (models load on first request)"
else
    echo -e "${YELLOW}Lazy Load:${NC} Disabled (models preload at startup)"
fi
if [ -n "$PDF_DPI" ]; then
    echo -e "${YELLOW}PDF DPI:${NC} $PDF_DPI"
fi
if [ -n "$PDF_THREAD_COUNT" ]; then
    echo -e "${YELLOW}PDF Thread Count:${NC} $PDF_THREAD_COUNT"
fi
echo -e "${GREEN}========================================${NC}"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Virtual environment not found at: $VENV_DIR${NC}"
    echo ""
    echo "Please run local_setup.sh first:"
    echo -e "  ${GREEN}./local_setup.sh${NC} or ${GREEN}./local_deepx_setup.sh${NC} (for DEEPX NPU support)"
    echo ""
    exit 1
fi

# Check if service script exists
if [ ! -f "$SERVICE_SCRIPT" ]; then
    echo ""
    echo -e "${RED}❌ Error: ocr_service.py not found at: $SERVICE_SCRIPT${NC}"
    exit 1
fi

# Check if DEEPX NPU is available and set SETUP_NPU accordingly
DEEPX_ENV_FILE="$SCRIPT_DIR/deepx_env.sh"
if [ -f "$DEEPX_ENV_FILE" ]; then
    echo -e "${YELLOW}🔧 Applying DEEPX NPU environment settings...${NC}"
    # Source with default values (1 2 1 3 2 4)
    source "$DEEPX_ENV_FILE"
    # Enable NPU support
    export SETUP_NPU="true"
    echo ""
else
    # No deepx_env.sh file means CPU only
    export SETUP_NPU="false"
fi

# Export environment variables
export PORT
export HOST
if [ -n "$PDF_DPI" ]; then
    export PDF_DPI
fi
if [ -n "$PDF_THREAD_COUNT" ]; then
    export PDF_THREAD_COUNT
fi

echo ""
echo -e "${BLUE}🚀 Starting OCR service...${NC}"
echo ""
echo -e "${YELLOW}Service will be available at:${NC}"
echo -e "  ${GREEN}http://localhost:${PORT}${NC}"
echo -e "  ${GREEN}API Docs: http://localhost:${PORT}/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the service${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"

# Run the service
"$VENV_DIR/bin/python" "$SERVICE_SCRIPT"
