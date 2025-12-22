#!/bin/bash

# PaddleOCR FastAPI Service Docker Run Script
# Automatically builds image if not exists, then runs container

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="paddleocr-fastapi-service"
IMAGE_TAG="latest"
CONTAINER_NAME="ocr-fastapi"
HOST_PORT="8081"
CONTAINER_PORT="8080"
GPU_MODE="false"
DEEPX_MODE="false"
BUILD_OPTIONS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_MODE="true"
            IMAGE_TAG="latest-gpu"
            BUILD_OPTIONS="--gpu"
            shift
            ;;
        --use-mobile)
            BUILD_OPTIONS="$BUILD_OPTIONS --use-mobile"
            shift
            ;;
        --use-server)
            BUILD_OPTIONS="$BUILD_OPTIONS --use-server"
            shift
            ;;
        --deepx)
            DEEPX_MODE="true"
            BUILD_OPTIONS="$BUILD_OPTIONS --deepx"
            if [ "$GPU_MODE" = "true" ]; then
                IMAGE_TAG="latest-deepx-gpu"
            else
                IMAGE_TAG="latest-deepx"
            fi
            shift
            ;;
        --port)
            HOST_PORT="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo -e "  ${GREEN}--gpu${NC}              Run with GPU support"
            echo -e "  ${GREEN}--use-mobile${NC}       Use mobile models (requires rebuild if not exists)"
            echo -e "  ${GREEN}--use-server${NC}       Use server models (requires rebuild if not exists)"
            echo -e "  ${GREEN}--deepx${NC}            Enable DEEPX NPU support (requires rebuild if not exists)"
            echo -e "  ${GREEN}--port PORT${NC}        Host port to expose (default: 8081)"
            echo -e "  ${GREEN}--name NAME${NC}        Container name (default: ocr-fastapi)"
            echo -e "  ${GREEN}--image IMAGE${NC}      Image name (default: paddleocr-fastapi-service)"
            echo -e "  ${GREEN}--tag TAG${NC}          Image tag (default: latest or latest-gpu)"
            echo -e "  ${GREEN}-h, --help${NC}         Show this help message"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  $0                        # Run CPU version on port 8081 (mobile models)"
            echo "  $0 --gpu                  # Run GPU version (mobile models)"
            echo "  $0 --use-server           # Run with server models"
            echo "  $0 --deepx                # Run with DEEPX NPU support"
            echo "  $0 --port 9000            # Run on custom port"
            echo "  $0 --gpu --use-server     # Run GPU with server models"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PaddleOCR FastAPI Docker Run${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Image:${NC}          $FULL_IMAGE"
echo -e "${YELLOW}Container:${NC}      $CONTAINER_NAME"
echo -e "${YELLOW}Port Mapping:${NC}   $HOST_PORT -> $CONTAINER_PORT"
echo -e "${YELLOW}GPU Mode:${NC}       $GPU_MODE"
echo -e "${GREEN}========================================${NC}"

# Check if image exists
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${FULL_IMAGE}$"; then
    echo ""
    echo -e "${YELLOW}⚠️  Image '$FULL_IMAGE' not found!${NC}"
    echo -e "${BLUE}🔨 Building image automatically...${NC}"
    echo ""
    
    # Run build script
    if [ -f "./docker_build.sh" ]; then
        ./docker_build.sh $BUILD_OPTIONS
    else
        echo -e "${RED}Error: docker_build.sh not found!${NC}"
        exit 1
    fi
    
    echo ""
fi

# Stop and remove existing container if exists
if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}🛑 Stopping and removing existing container '$CONTAINER_NAME'...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Run container
echo ""
echo -e "${BLUE}🚀 Starting container...${NC}"
echo ""

if [ "$GPU_MODE" = "true" ]; then
    docker run -d \
        --gpus all \
        -p "${HOST_PORT}:${CONTAINER_PORT}" \
        --name "$CONTAINER_NAME" \
        "$FULL_IMAGE"
else
    docker run -d \
        -p "${HOST_PORT}:${CONTAINER_PORT}" \
        --name "$CONTAINER_NAME" \
        "$FULL_IMAGE"
fi

# Wait a moment for container to start
sleep 2

# Check if container is running
if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✅ Container started successfully!${NC}"
    echo ""
    echo -e "${BLUE}Service URL:${NC}  ${GREEN}http://localhost:${HOST_PORT}${NC}"
    echo -e "${BLUE}API Docs:${NC}     ${GREEN}http://localhost:${HOST_PORT}/docs${NC}"
    echo ""
    echo -e "${YELLOW}Useful commands:${NC}"
    echo -e "  ${GREEN}View logs:${NC}    docker logs -f $CONTAINER_NAME"
    echo -e "  ${GREEN}Stop:${NC}         docker stop $CONTAINER_NAME"
    echo -e "  ${GREEN}Restart:${NC}      docker restart $CONTAINER_NAME"
    echo -e "  ${GREEN}Remove:${NC}       docker rm -f $CONTAINER_NAME"
    echo ""
    echo -e "${BLUE}Test the service:${NC}"
    echo -e "  ${GREEN}curl http://localhost:${HOST_PORT}/health${NC}"
    echo ""
else
    echo -e "${RED}❌ Failed to start container!${NC}"
    echo "Check logs with: docker logs $CONTAINER_NAME"
    exit 1
fi
