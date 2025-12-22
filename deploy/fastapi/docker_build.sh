#!/usr/bin/env bash
# Build script for PaddleOCR FastAPI Service Docker Image

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEVICE_TYPE="cpu"
USE_MOBILE="true"  # default: mobile
USE_SERVER="false"
DOWNLOAD_MODELS="true"
SETUP_DEEPX="false"
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
            USE_SERVER="false"
            shift
            ;;
        --use-server)
            USE_SERVER="true"
            USE_MOBILE="false"
            shift
            ;;
        --deepx)
            SETUP_DEEPX="true"
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
            echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo -e "  ${GREEN}--gpu${NC}                 Build with GPU support (default: CPU)"
            echo -e "  ${GREEN}--use-mobile${NC}          Use mobile models (fast, small) (default)"
            echo -e "  ${GREEN}--use-server${NC}          Use server models (high precision)"
            echo -e "  ${GREEN}--deepx${NC}               Enable DEEPX NPU support"
            echo -e "  ${GREEN}--no-models${NC}           Don't download models during build"
            echo -e "  ${GREEN}--version VERSION${NC}     PaddleOCR version (default: 3.3.2)"
            echo -e "  ${GREEN}--tag NAME${NC}            Docker image name (default: paddleocr-fastapi-service)"
            echo -e "  ${GREEN}--tag-version VER${NC}     Docker image version tag (default: latest)"
            echo -e "  ${GREEN}-h, --help${NC}            Show this help message"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  $0                              # CPU version with mobile models (default)"
            echo "  $0 --gpu                        # GPU version with mobile models"
            echo "  $0 --use-server                 # CPU version with server models"
            echo "  $0 --gpu --use-server           # GPU version with server models"
            echo "  $0 --deepx                      # CPU version with DEEPX NPU support"
            echo "  $0 --gpu --no-models            # GPU version without models"
            echo "  $0 --version 3.3.0 --tag my-ocr # Custom version and tag"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Determine model type
if [ "$USE_MOBILE" = "true" ]; then
    MODEL_TYPE="mobile"
else
    MODEL_TYPE="server"
fi

# Set full tag
if [ "$SETUP_DEEPX" = "true" ]; then
    if [ "$DEVICE_TYPE" = "gpu" ]; then
        FULL_TAG="${TAG_NAME}:${TAG_VERSION}-deepx-gpu"
    else
        FULL_TAG="${TAG_NAME}:${TAG_VERSION}-deepx"
    fi
elif [ "$DEVICE_TYPE" = "gpu" ]; then
    FULL_TAG="${TAG_NAME}:${TAG_VERSION}-gpu"
else
    FULL_TAG="${TAG_NAME}:${TAG_VERSION}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building PaddleOCR FastAPI Service Docker Image${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Device Type:${NC}       $DEVICE_TYPE"
echo -e "${YELLOW}Model Type:${NC}        $MODEL_TYPE"
echo -e "${YELLOW}DEEPX NPU:${NC}         $([ "$SETUP_DEEPX" = "true" ] && echo "enabled" || echo "disabled")"
echo -e "${YELLOW}Download Models:${NC}   $DOWNLOAD_MODELS"
echo -e "${YELLOW}PaddleOCR Version:${NC} $PADDLEOCR_VERSION"
echo -e "${YELLOW}Image Tag:${NC}         $FULL_TAG"
echo -e "${GREEN}========================================${NC}"

# Prepare fonts for Docker build
echo ""
echo -e "${YELLOW}Preparing fonts for OCR visualization...${NC}"
FONTS_DEST_DIR="deepx/engine/fonts"
mkdir -p "$FONTS_DEST_DIR"

# Copy fonts from doc/fonts to deepx/engine/fonts
if [ -d "../../doc/fonts" ]; then
    cp -v ../../doc/fonts/*.ttf "$FONTS_DEST_DIR/" 2>/dev/null || true
    echo -e "${GREEN}✓ Fonts copied to $FONTS_DEST_DIR${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: doc/fonts directory not found, skipping font copy${NC}"
fi
echo ""

# Build the image
docker build \
    --build-arg DEVICE_TYPE="$DEVICE_TYPE" \
    --build-arg USE_MOBILE="$USE_MOBILE" \
    --build-arg DOWNLOAD_MODELS="$DOWNLOAD_MODELS" \
    --build-arg SETUP_DEEPX="$SETUP_DEEPX" \
    --build-arg PADDLEOCR_VERSION="$PADDLEOCR_VERSION" \
    -t "$FULL_TAG" \
    -f Dockerfile \
    .

echo ""
echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo -e "${YELLOW}Image:${NC} $FULL_TAG"
echo ""
echo -e "${BLUE}To run the container with docker_run.sh (recommended):${NC}"
if [ "$SETUP_DEEPX" = "true" ]; then
    if [ "$DEVICE_TYPE" = "gpu" ]; then
        echo -e "  ${GREEN}./docker_run.sh --gpu --deepx${NC}"
    else
        echo -e "  ${GREEN}./docker_run.sh --deepx${NC}"
    fi
else
    if [ "$DEVICE_TYPE" = "gpu" ]; then
        echo -e "  ${GREEN}./docker_run.sh --gpu${NC}"
    else
        echo -e "  ${GREEN}./docker_run.sh${NC}"
    fi
fi
echo ""
echo ""
echo -e "${BLUE}Or run manually with docker:${NC}"
if [ "$SETUP_DEEPX" = "true" ]; then
    echo -e "${YELLOW}⚠️  Warning: DEEPX NPU requires special permissions and device mounts${NC}"
    echo -e "${YELLOW}   Please use docker_run.sh instead for proper NPU setup:${NC}"
    if [ "$DEVICE_TYPE" = "gpu" ]; then
        echo -e "   ${GREEN}./docker_run.sh --gpu --deepx${NC}"
    else
        echo -e "   ${GREEN}./docker_run.sh --deepx${NC}"
    fi
elif [ "$DEVICE_TYPE" = "gpu" ]; then
    echo -e "  ${GREEN}docker run -d --gpus all -p 8081:8080 --name ocr-fastapi $FULL_TAG${NC}"
else
    echo -e "  ${GREEN}docker run -d -p 8081:8080 --name ocr-fastapi $FULL_TAG${NC}"
fi
echo ""
echo -e "${BLUE}Test the service:${NC}"
echo -e "  ${GREEN}curl http://localhost:8081/health${NC}"
echo ""
echo -e "${BLUE}API Documentation:${NC}"
echo -e "  ${GREEN}http://localhost:8081/docs${NC} (Swagger UI)"
echo ""
echo -e "${YELLOW}Note: Using port 8081 to avoid conflict with local service on 8080${NC}"
