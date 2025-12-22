#!/bin/bash

# PaddleOCR FastAPI Service Local Stop Script
# Stops the locally running OCR service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="ocr_service.py"
PID_FILE="$SCRIPT_DIR/.ocr_service.pid"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PaddleOCR FastAPI Service - Stop${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to stop process by PID
stop_by_pid() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}Stopping OCR service (PID: $pid)...${NC}"
        kill "$pid" 2>/dev/null || true
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}✓ Service stopped gracefully${NC}"
                return 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Force stopping service...${NC}"
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}✓ Service force stopped${NC}"
                return 0
            else
                echo -e "${RED}✗ Failed to stop service${NC}"
                return 1
            fi
        fi
    else
        echo -e "${YELLOW}Process (PID: $pid) is not running${NC}"
        return 1
    fi
}

# Check if PID file exists
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo -e "${YELLOW}Found PID file: $PID${NC}"
    
    if stop_by_pid "$PID"; then
        rm -f "$PID_FILE"
        echo -e "${GREEN}✓ PID file removed${NC}"
    else
        rm -f "$PID_FILE"
        echo -e "${YELLOW}⚠ PID file removed (process was not running)${NC}"
    fi
    echo ""
fi

# Find and stop any running OCR service processes
echo -e "${YELLOW}Searching for running OCR service processes...${NC}"

# Get PIDs, excluding Docker container processes
PIDS=$(pgrep -f "$SERVICE_NAME" | while read -r pid; do
    # Check if process is running inside Docker container
    if [ -f "/proc/$pid/cgroup" ]; then
        if ! grep -q "docker" "/proc/$pid/cgroup" 2>/dev/null; then
            echo "$pid"
        fi
    else
        echo "$pid"
    fi
done)

if [ -z "$PIDS" ]; then
    echo -e "${GREEN}✓ No local OCR service processes found${NC}"
    echo -e "${YELLOW}ℹ Docker container processes are managed by docker_stop.sh${NC}"
else
    echo -e "${YELLOW}Found local OCR service processes:${NC}"
    echo "$PIDS" | while read -r pid; do
        echo "  PID: $pid"
    done
    echo ""
    
    # Stop all found processes
    echo "$PIDS" | while read -r pid; do
        stop_by_pid "$pid"
    done
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Stop Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Show status
REMAINING=$(pgrep -f "$SERVICE_NAME" | while read -r pid; do
    if [ -f "/proc/$pid/cgroup" ]; then
        if ! grep -q "docker" "/proc/$pid/cgroup" 2>/dev/null; then
            echo "$pid"
        fi
    else
        echo "$pid"
    fi
done)

if [ -z "$REMAINING" ]; then
    echo -e "${GREEN}✓ All local OCR service processes stopped${NC}"
else
    echo -e "${RED}✗ Some local processes may still be running:${NC}"
    echo "$REMAINING"
    exit 1
fi

echo ""
echo -e "${BLUE}Note: This script stops only local processes.${NC}"
echo -e "${BLUE}To stop Docker containers, use: ./docker_stop.sh${NC}"
echo ""
