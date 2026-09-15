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
OCR_VERSION_OPT="${OCR_VERSION:-}"
MODEL_SIZE_OPT="${MODEL_SIZE:-}"
NO_INTERACTIVE=false
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
        --sanity-check|--doctor)
            "$VENV_DIR/bin/python" "$SCRIPT_DIR/device_selection.py" --report
            exit 0
            ;;
        --ocr-version)
            OCR_VERSION_OPT="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE_OPT="$2"
            shift 2
            ;;
        --no-interactive)
            NO_INTERACTIVE=true
            shift
            ;;
        --use-mobile)
            USE_MOBILE_FLAG=true
            MODEL_TYPE="mobile"
            # Legacy alias: implies PP-OCRv5 + mobile, counts as fully specified.
            OCR_VERSION_OPT="${OCR_VERSION_OPT:-v5}"
            MODEL_SIZE_OPT="${MODEL_SIZE_OPT:-mobile}"
            shift
            ;;
        --use-server)
            USE_SERVER_FLAG=true
            MODEL_TYPE="server"
            # Legacy alias: implies PP-OCRv5 + server, counts as fully specified.
            OCR_VERSION_OPT="${OCR_VERSION_OPT:-v5}"
            MODEL_SIZE_OPT="${MODEL_SIZE_OPT:-server}"
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
            echo -e "  ${GREEN}--ocr-version${NC}    ${YELLOW}v5 | v6${NC} (required with --model-size)"
            echo -e "  ${GREEN}--model-size${NC}     ${YELLOW}v6: medium|small|tiny   v5: server|mobile${NC}"
            echo -e "  ${GREEN}--no-interactive${NC} ${YELLOW}Never prompt; the model must be specified${NC}"
            echo -e "  ${GREEN}--sanity-check${NC}   ${YELLOW}Report installed packages, usable devices and models, then exit${NC}"
            echo -e ""
            echo -e "  ${YELLOW}Omit both --ocr-version and --model-size to choose interactively.${NC}"
            echo -e "  ${YELLOW}Giving only one of them is an error.${NC}"
            echo -e ""
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

# Model version/size are resolved by model_selection.py (shared with
# ocr_service.py) so both entry points follow identical rules: both options
# given -> start; exactly one -> error with a hint; neither -> interactive on a
# terminal, documented defaults otherwise.

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

# paddlepaddle-gpu 3.2.2 was linked against the old split CUDA layout
# ($ORIGIN/../../nvidia/cuda_nvrtc/lib and friends), but the CUDA 13 wheels it
# now pulls in install everything under nvidia/cu13/lib. The RUNPATH therefore
# misses libnvrtc.so.13 and "import paddle" dies before it can report anything
# useful. Adding the directory when it exists is harmless for CPU-only and for
# builds whose RUNPATH is already correct (3.3.0).
for _cuda_lib in "$VENV_DIR"/lib/python*/site-packages/nvidia/cu13/lib; do
    [ -d "$_cuda_lib" ] && export LD_LIBRARY_PATH="$_cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

# Check if service script exists
if [ ! -f "$SERVICE_SCRIPT" ]; then
    echo ""
    echo -e "${RED}❌ Error: ocr_service.py not found at: $SERVICE_SCRIPT${NC}"
    exit 1
fi

# Resolve OCR model version/size (shared logic with ocr_service.py).
RESOLVE_ARGS=()
[ -n "$OCR_VERSION_OPT" ] && RESOLVE_ARGS+=(--ocr-version "$OCR_VERSION_OPT")
[ -n "$MODEL_SIZE_OPT" ]  && RESOLVE_ARGS+=(--model-size  "$MODEL_SIZE_OPT")
[ "$NO_INTERACTIVE" = true ] && RESOLVE_ARGS+=(--no-interactive)

if ! SELECTION="$("$VENV_DIR/bin/python" "$SCRIPT_DIR/model_selection.py" --resolve "${RESOLVE_ARGS[@]}")"; then
    # model_selection.py already printed the error and usage hint on stderr.
    exit 2
fi
# Extract each key by name rather than splitting lines: under a pty the
# interactive prompt echoes onto this same stream, so a line can arrive as
# "Select [1]: OCR_VERSION=v6" and a naive split would export an empty value.
_pick() { printf '%s\n' "$SELECTION" | sed -n "s/.*\\b$1=\\([A-Za-z0-9_]*\\).*/\\1/p" | tail -1; }
for _key in OCR_VERSION MODEL_SIZE V6_MODEL_SIZE USE_MOBILE; do
    _val="$(_pick "$_key")"
    [ -n "$_val" ] && export "$_key=$_val"
done
echo -e "${YELLOW}   OCR_VERSION=${OCR_VERSION}  ${V6_MODEL_SIZE:+V6_MODEL_SIZE=$V6_MODEL_SIZE}${USE_MOBILE:+USE_MOBILE=$USE_MOBILE}${NC}"

# DEEPX NPU environment.
# An explicit SETUP_NPU from the caller always wins. This block used to set it
# unconditionally, so `SETUP_NPU=true ./run.sh` was silently forced to CPU-only
# whenever deepx_env.sh was absent - and deepx_env.sh only exists after
# local_deepx_setup.sh has been run.
# DX-RT tuning defaults. Without DXRT_TASK_MAX_LOAD the PP-OCRv5 server set
# (11 engines) fails at load time with "Failed to register memory cache for
# task 12". benchmark_npu.py reads this same file; run.sh used to skip it.
ENV_DEEPX_FILE="$SCRIPT_DIR/.env.deepx"
if [ -f "$ENV_DEEPX_FILE" ]; then
    while IFS='=' read -r _k _v; do
        case "$_k" in ''|\#*) continue ;; esac
        _k="$(echo "$_k" | tr -d '[:space:]')"
        [ -n "$_k" ] && [ -z "$(eval "echo \${$_k:-}")" ] && export "$_k=$(echo "$_v" | tr -d '[:space:]')"
    done < "$ENV_DEEPX_FILE"
fi

DEEPX_ENV_FILE="$SCRIPT_DIR/deepx_env.sh"
if [ -f "$DEEPX_ENV_FILE" ]; then
    echo -e "${YELLOW}🔧 Applying DEEPX NPU environment settings...${NC}"
    # Source with default values (1 2 1 3 2 4)
    source "$DEEPX_ENV_FILE"
    echo ""
fi

if [ -z "${SETUP_NPU:-}" ]; then
    # Not specified: enable NPU when the runtime is actually importable.
    if "$VENV_DIR/bin/python" -c "import dx_engine" >/dev/null 2>&1; then
        export SETUP_NPU="true"
    else
        export SETUP_NPU="false"
    fi
fi
echo -e "${YELLOW}   SETUP_NPU=${SETUP_NPU}${NC}"

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
