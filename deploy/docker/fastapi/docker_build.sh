#!/usr/bin/env bash
# Build script for PaddleOCR FastAPI Service Docker Image

set -e

# Default values
DEVICE_TYPE="cpu"
USE_MOBILE="false"
DOWNLOAD_MODELS="true"
PADDLEOCR_VERSION="3.3.2"
TAG_NAME="paddleocr-fastapi-service"
TAG_VERSION="latest"

# Parse arguments
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
            USE_MOBILE="true"
            shift
            ;;
        --version)
            PADDLEOCR_VERSION="$2"
            shift 2
            ;;
        --tag)
            TAG_NAME="$2"
            shift 2
            ;;
        --tag-version)
            TAG_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu                 Build with GPU support (default: CPU)"
            echo "  --use-mobile          Use mobile models instead of server models"
            echo "  --no-models           Don't download models during build"
            echo "  --version VERSION     PaddleOCR version (default: 3.3.2)"
            echo "  --tag NAME            Docker image name (default: paddleocr-fastapi-service)"
            echo "  --tag-version VER     Docker image version tag (default: latest)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # CPU version with server models"
            echo "  $0 --gpu                        # GPU version with server models"
            echo "  $0 --use-mobile                 # CPU version with mobile models"
            echo "  $0 --gpu --use-mobile           # GPU version with mobile models"
            echo "  $0 --gpu --no-models            # GPU version without models"
            echo "  $0 --version 3.3.0 --tag my-ocr # Custom version and tag"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set full tag
if [ "$DEVICE_TYPE" = "gpu" ]; then
    FULL_TAG="${TAG_NAME}:${TAG_VERSION}-gpu"
else
    FULL_TAG="${TAG_NAME}:${TAG_VERSION}"
fi

echo "========================================"
echo "Building PaddleOCR FastAPI Service Docker Image"
echo "========================================"
echo "Device Type:       $DEVICE_TYPE"
echo "Model Type:        $([ "$USE_MOBILE" = "true" ] && echo "mobile" || echo "server")"
echo "Download Models:   $DOWNLOAD_MODELS"
echo "PaddleOCR Version: $PADDLEOCR_VERSION"
echo "Image Tag:         $FULL_TAG"
echo "========================================"

# Build the image
docker build \
    --build-arg DEVICE_TYPE="$DEVICE_TYPE" \
    --build-arg USE_MOBILE="$USE_MOBILE" \
    --build-arg DOWNLOAD_MODELS="$DOWNLOAD_MODELS" \
    --build-arg PADDLEOCR_VERSION="$PADDLEOCR_VERSION" \
    -t "$FULL_TAG" \
    -f Dockerfile \
    .

echo ""
echo "✅ Build completed successfully!"
echo "Image: $FULL_TAG"
echo ""
echo "To run the container:"
if [ "$DEVICE_TYPE" = "gpu" ]; then
    echo "  docker run --gpus all -p 8081:8080 --name ocr-fastapi $FULL_TAG"
else
    echo "  docker run -d -p 8081:8080 --name ocr-fastapi $FULL_TAG"
fi
echo ""
echo "Test the service:"
echo "  curl http://localhost:8081/health"
echo ""
echo "API Documentation:"
echo "  http://localhost:8081/docs (Swagger UI)"
echo "  http://localhost:8081/redoc (ReDoc)"
echo ""
echo "Note: Using port 8081 to avoid conflict with Flask service on 8080"
