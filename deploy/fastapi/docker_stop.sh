#!/bin/bash

# PaddleOCR FastAPI Service Docker Stop Script
# Stops and optionally removes the Docker container

set -e

# Default values
CONTAINER_NAME="ocr-fastapi"
REMOVE_CONTAINER="false"
FORCE_STOP="false"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --remove|-rm)
            REMOVE_CONTAINER="true"
            shift
            ;;
        --force|-f)
            FORCE_STOP="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --name NAME        Container name (default: ocr-fastapi)"
            echo "  --remove, -rm      Remove container after stopping"
            echo "  --force, -f        Force stop (kill) the container"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                 # Stop the container"
            echo "  $0 --remove        # Stop and remove the container"
            echo "  $0 --force         # Force stop the container"
            echo "  $0 -rm -f          # Force stop and remove the container"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PaddleOCR FastAPI Docker - Stop${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}⚠ Container '${CONTAINER_NAME}' does not exist${NC}"
    exit 0
fi

# Check if container is running
CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "not-found")

if [ "$CONTAINER_STATUS" == "not-found" ]; then
    echo -e "${YELLOW}⚠ Container '${CONTAINER_NAME}' not found${NC}"
    exit 0
elif [ "$CONTAINER_STATUS" == "running" ]; then
    echo -e "${YELLOW}Container '${CONTAINER_NAME}' is running${NC}"
    
    if [ "$FORCE_STOP" == "true" ]; then
        echo -e "${YELLOW}Force stopping container...${NC}"
        docker kill "$CONTAINER_NAME"
        echo -e "${GREEN}✓ Container force stopped${NC}"
    else
        echo -e "${YELLOW}Stopping container gracefully...${NC}"
        docker stop "$CONTAINER_NAME"
        echo -e "${GREEN}✓ Container stopped${NC}"
    fi
else
    echo -e "${YELLOW}Container '${CONTAINER_NAME}' is already stopped (status: ${CONTAINER_STATUS})${NC}"
fi

echo ""

# Remove container if requested
if [ "$REMOVE_CONTAINER" == "true" ]; then
    echo -e "${YELLOW}Removing container...${NC}"
    docker rm "$CONTAINER_NAME"
    echo -e "${GREEN}✓ Container removed${NC}"
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Stop Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Show final status
if [ "$REMOVE_CONTAINER" == "true" ]; then
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}✗ Container still exists${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Container '${CONTAINER_NAME}' has been stopped and removed${NC}"
    fi
else
    FINAL_STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "not-found")
    if [ "$FINAL_STATUS" == "running" ]; then
        echo -e "${RED}✗ Container is still running${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Container '${CONTAINER_NAME}' is stopped${NC}"
        echo ""
        echo -e "${BLUE}To start the container again:${NC}"
        echo "  docker start $CONTAINER_NAME"
        echo ""
        echo -e "${BLUE}To remove the container:${NC}"
        echo "  $0 --remove"
    fi
fi

echo ""
