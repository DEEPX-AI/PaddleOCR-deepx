#!/bin/bash

# PaddleOCR FastAPI Service Docker Run Script
# Automatically builds image if not exists, then runs container

set -e

# Default values
IMAGE_NAME="paddleocr-fastapi-service"
IMAGE_TAG="latest"
CONTAINER_NAME="ocr-fastapi"
HOST_PORT="8081"
CONTAINER_PORT="8080"
GPU_MODE="false"
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
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu              Run with GPU support"
            echo "  --use-mobile       Use mobile models (requires rebuild if not exists)"
            echo "  --port PORT        Host port to expose (default: 8081)"
            echo "  --name NAME        Container name (default: ocr-fastapi)"
            echo "  --image IMAGE      Image name (default: paddleocr-fastapi-service)"
            echo "  --tag TAG          Image tag (default: latest or latest-gpu)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Run CPU version on port 8081"
            echo "  $0 --gpu                  # Run GPU version"
            echo "  $0 --port 9000            # Run on custom port"
            echo "  $0 --gpu --use-mobile     # Run GPU with mobile models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "========================================"
echo "PaddleOCR FastAPI Docker Run"
echo "========================================"
echo "Image:          $FULL_IMAGE"
echo "Container:      $CONTAINER_NAME"
echo "Port Mapping:   $HOST_PORT -> $CONTAINER_PORT"
echo "GPU Mode:       $GPU_MODE"
echo "========================================"

# Check if image exists
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${FULL_IMAGE}$"; then
    echo ""
    echo "⚠️  Image '$FULL_IMAGE' not found!"
    echo "🔨 Building image automatically..."
    echo ""
    
    # Run build script
    if [ -f "./docker_build.sh" ]; then
        ./docker_build.sh $BUILD_OPTIONS
    else
        echo "Error: docker_build.sh not found!"
        exit 1
    fi
    
    echo ""
fi

# Stop and remove existing container if exists
if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 Stopping and removing existing container '$CONTAINER_NAME'..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Run container
echo ""
echo "🚀 Starting container..."
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
    echo "✅ Container started successfully!"
    echo ""
    echo "Service URL:  http://localhost:${HOST_PORT}"
    echo "API Docs:     http://localhost:${HOST_PORT}/docs"
    echo "ReDoc:        http://localhost:${HOST_PORT}/redoc"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker logs -f $CONTAINER_NAME"
    echo "  Stop:         docker stop $CONTAINER_NAME"
    echo "  Restart:      docker restart $CONTAINER_NAME"
    echo "  Remove:       docker rm -f $CONTAINER_NAME"
    echo ""
    echo "Test the service:"
    echo "  curl http://localhost:${HOST_PORT}/health"
    echo ""
else
    echo "❌ Failed to start container!"
    echo "Check logs with: docker logs $CONTAINER_NAME"
    exit 1
fi
