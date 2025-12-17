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

# Parse arguments
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT      Service port (default: 8080)"
            echo "  --host HOST      Bind host (default: 0.0.0.0)"
            echo "  --use-mobile     Use mobile models (default)"
            echo "  --use-server     Use server models"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  PORT             Service port"
            echo "  HOST             Bind host"
            echo "  USE_GPU          Use GPU (true/false)"
            echo "  USE_MOBILE       Use mobile models (true/false)"
            echo ""
            echo "Examples:"
            echo "  $0                       # Run on default port 8080 (mobile models)"
            echo "  $0 --port 9000           # Run on port 9000"
            echo "  USE_GPU=true $0          # Run with GPU"
            echo "  $0 --use-mobile          # Run with mobile models (default)"
            echo "  $0 --use-server          # Run with server models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check for conflicting options
if [ "$USE_MOBILE_FLAG" = true ] && [ "$USE_SERVER_FLAG" = true ]; then
    echo "❌ Error: --use-mobile and --use-server cannot be used together"
    echo "Please specify only one model type option."
    exit 1
fi

# Set default if no flag was specified
if [ "$USE_MOBILE_FLAG" = false ] && [ "$USE_SERVER_FLAG" = false ]; then
    export USE_MOBILE="true"  # Default to mobile
fi

echo "========================================"
echo "PaddleOCR FastAPI Local Service"
echo "========================================"
echo "Host: $HOST"
echo "Port: $PORT"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "⚠️  Virtual environment not found at: $VENV_DIR"
    echo ""
    echo "Please run local_setup.sh first:"
    echo "  ./local_setup.sh or ./local_deepx_setup.sh (for DEEPX NPU support)"
    echo ""
    exit 1
fi

# Check if service script exists
if [ ! -f "$SERVICE_SCRIPT" ]; then
    echo ""
    echo "❌ Error: ocr_service.py not found at: $SERVICE_SCRIPT"
    exit 1
fi

# Apply DEEPX NPU environment settings if available
DEEPX_ENV_FILE="$SCRIPT_DIR/deepx_env.sh"
if [ -f "$DEEPX_ENV_FILE" ]; then
    echo "🔧 Applying DEEPX NPU environment settings..."
    source "$DEEPX_ENV_FILE"
    echo ""
fi

# Export environment variables
export PORT
export HOST

echo ""
echo "🚀 Starting OCR service..."
echo ""
echo "Service will be available at:"
echo "  http://localhost:${PORT}"
echo "  API Docs: http://localhost:${PORT}/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""
echo "========================================"

# Run the service
"$VENV_DIR/bin/python" "$SERVICE_SCRIPT"
