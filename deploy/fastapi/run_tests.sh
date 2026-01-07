#!/bin/bash
#
# Run OCR tests and generate HTML report
# 
# Usage: 
#   ./run_tests.sh                       # Use DEEPX NPU for inference (default: NPU | async mode)
#   ./run_tests.sh --deepx true          # Use DEEPX NPU for inference (NPU | async mode)
#   ./run_tests.sh --deepx false         # Use CPU for inference       (CPU | sync mode, cpu only supported sync mode)
#   ./run_tests.sh --deepx true --sync   # Use DEEPX NPU for inference (NPU | sync mode)
#
#   ./run_tests.sh                       # Run single test (test_baidu_ocr_all_parameters)
#   ./run_tests.sh --all                 # Run all test cases
#   ./run_tests.sh --list                # List all available test cases
#   ./run_tests.sh --tc <test_name>      # Run specific test case
#   ./run_tests.sh --input <input_path>  # Run all test cases with custom input images
#   ./run_tests.sh --port 8081           # Use custom port (default: 8080)
#   ./run_tests.sh --limit 5             # Process only first 5 images
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Setup virtual environment
VENV_PATH="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo ""
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Parse command line arguments
RUN_ALL=false
LIST_TESTS=false
TEST_NAME=""
CUSTOM_INPUT_PATH=""
SERVICE_PORT=8080
USE_DEEPX=true
USE_SYNC=false
CLEAN_OUTPUT=false
IMAGE_LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        -l|--list)
            LIST_TESTS=true
            shift
            ;;
        --tc)
            if [ -z "$2" ]; then
                echo -e "${RED}Error: --tc option requires a test case name${NC}"
                echo "Usage: ./run_tests.sh --tc <test_name>"
                exit 1
            fi
            TEST_NAME="$2"
            shift 2
            ;;
        -i|--input)
            if [ -z "$2" ]; then
                echo -e "${RED}Error: -i/--input option requires an input path${NC}"
                echo "Usage: ./run_tests.sh --i <input_path>"
                exit 1
            fi
            CUSTOM_INPUT_PATH="$2"
            if [ ! -e "$CUSTOM_INPUT_PATH" ]; then
                echo -e "${RED}Error: Input path does not exist: $CUSTOM_INPUT_PATH${NC}"
                exit 1
            fi
            shift 2
            ;;
        -p|--port)
            if [ -z "$2" ]; then
                echo -e "${RED}Error: -p/--port option requires a port number${NC}"
                echo "Usage: ./run_tests.sh -p <port_number>"
                exit 1
            fi
            SERVICE_PORT="$2"
            shift 2
            ;;
        --deepx)
            # Optional value: if no value or next arg starts with --, default to true
            if [ -z "$2" ] || [[ "$2" == --* ]] || [[ "$2" == -* ]]; then
                USE_DEEPX=true
                shift
            else
                if [ "$2" = "false" ] || [ "$2" = "False" ] || [ "$2" = "FALSE" ]; then
                    USE_DEEPX=false
                else
                    USE_DEEPX=true
                fi
                shift 2
            fi
            ;;
        --sync)
            USE_SYNC=true
            shift
            ;;
        --clean-output)
            # Optional value: if no value or next arg starts with --, default to true
            if [ -z "$2" ] || [[ "$2" == --* ]] || [[ "$2" == -* ]]; then
                CLEAN_OUTPUT=true
                shift
            else
                if [ "$2" = "false" ] || [ "$2" = "False" ] || [ "$2" = "FALSE" ]; then
                    CLEAN_OUTPUT=false
                else
                    CLEAN_OUTPUT=true
                fi
                shift 2
            fi
            ;;
        -n|--limit)
            if [ -z "$2" ]; then
                echo -e "${RED}Error: -n/--limit option requires a number${NC}"
                echo "Usage: ./run_tests.sh --limit <number>"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Error: --limit value must be a positive integer${NC}"
                exit 1
            fi
            IMAGE_LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
            echo ""
            echo -e "${YELLOW}DEEPX NPU / CPU Options:${NC}"
            echo -e "  ${GREEN}--deepx <true|false>${NC}  Use DEEPX NPU (true) or CPU (false) for inference"
            echo "                        Default: true (NPU | async mode)"
            echo "                        When false: CPU | sync mode (CPU only supports sync mode)"
            echo -e "  ${GREEN}--sync${NC}                Use synchronous mode (PaddleOcr)"
            echo "                        Default: async mode (AsyncPipelineOCR)"
            echo ""
            echo -e "${YELLOW}Test Execution Options:${NC}"
            echo -e "  ${GREEN}--all${NC}                 Run all test cases"
            echo -e "  ${GREEN}--list, -l${NC}            List all available test cases"
            echo -e "  ${GREEN}--tc <test_name>${NC}      Run specific test case"
            echo -e "  ${GREEN}-i, --input <path>${NC}    Use custom input images (file or directory)"
            echo -e "  ${GREEN}-p, --port <port>${NC}     Service port (default: 8080)"
            echo -e "  ${GREEN}-n, --limit <number>${NC}  Limit number of images to process (default: all)"
            echo -e "  ${GREEN}--clean-output <bool>${NC} Clean previous test outputs (default: true)"
            echo -e "  ${GREEN}-h, --help${NC}            Show this help message"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  $0                           # Use NPU (async), run single default test"
            echo "  $0 --deepx true              # Explicitly use NPU (async mode)"
            echo "  $0 --deepx false             # Use CPU (sync mode)"
            echo "  $0 --deepx true --sync       # Use NPU (sync mode)"
            echo ""
            echo "  $0 --all                     # Run all test cases"
            echo "  $0 --list                    # List available tests"
            echo "  $0 --tc test_name            # Run specific test"
            echo "  $0 --input /path/to/images   # Use custom images"
            echo "  $0 --port 8081               # Use custom port"
            echo "  $0 --limit 5                 # Process only first 5 images"
            echo ""
            echo "  $0 --deepx false --all       # Run all tests with CPU"
            echo "  $0 --deepx true --sync --all # Run all tests with NPU (sync)"
            echo "  $0 --clean-output false      # Keep previous test results"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Usage: ./run_tests.sh [--all | --list | --tc <test_name> | -i <input_path> | -p <port_number> | -n <number> | --deepx <true|false> | --sync | --clean-output <true|false>]"
            echo "Use -h or --help for more information"
            exit 1
            ;;
    esac
done

# Install test requirements (needed for --list as well)
if [ -f "test_requirements.txt" ]; then
    echo -e "${YELLOW}Installing test requirements...${NC}"
    pip install -q -r test_requirements.txt
    echo -e "${GREEN}✓ Test requirements installed${NC}"
    echo ""
fi

# If listing tests, just show them and exit
if [ "$LIST_TESTS" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Available Test Cases${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    echo -e "${YELLOW}test_ocr_service.py:${NC}"
    pytest test_ocr_service.py --collect-only 2>&1 | grep -E "<Function" | sed 's/.*<Function /  /' | sed 's/>//'
    echo ""
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Total Test Cases:${NC}"
    TOTAL_TESTS=$(pytest test_ocr_service.py --collect-only 2>&1 | grep "<Function" | wc -l)
    echo -e "  test_ocr_service.py: ${GREEN}${TOTAL_TESTS}${NC} tests"
    echo -e "  Total: ${GREEN}${TOTAL_TESTS}${NC} tests"
    echo ""
    echo -e "${BLUE}Usage:${NC}"
    echo "  ./run_tests.sh                       # Use DEEPX NPU for inference (default: NPU | async mode)"
    echo "  ./run_tests.sh --deepx true          # Use DEEPX NPU for inference (NPU | async mode)"
    echo "  ./run_tests.sh --deepx false         # Use CPU for inference       (CPU | sync mode, cpu only supported sync mode)"
    echo "  ./run_tests.sh --deepx true --sync   # Use DEEPX NPU for inference (NPU | sync mode)"
    echo ""
    echo "  ./run_tests.sh                       # Run single test (test_baidu_ocr_all_parameters)"
    echo "  ./run_tests.sh --all                 # Run all test cases"
    echo "  ./run_tests.sh --list                # List all available test cases"
    echo "  ./run_tests.sh --tc <test_name>      # Run specific test case"
    echo "  ./run_tests.sh --input <input_path>  # Run all test cases with custom input images"
    echo "  ./run_tests.sh --port 8081           # Use custom port (default: 8080)"
    echo ""
    
    exit 0
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PaddleOCR FastAPI Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if service is running
echo -e "${YELLOW}Checking service health...${NC}"
SERVICE_URL="http://localhost:$SERVICE_PORT"
if ! curl -s "$SERVICE_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: OCR service is not running on port $SERVICE_PORT${NC}"
    echo "Please start the service first:"
    echo "  docker start ocr-fastapi"
    exit 1
fi
echo -e "${GREEN}✓ Service is healthy (port: $SERVICE_PORT)${NC}"
echo ""

# Clean previous test outputs
if [ "$CLEAN_OUTPUT" = true ]; then
    echo -e "${YELLOW}Cleaning previous test outputs...${NC}"
    rm -rf test_outputs/
    mkdir -p test_outputs
    echo -e "${GREEN}✓ Test outputs directory cleaned${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping test outputs cleanup (keeping previous results)${NC}"
    mkdir -p test_outputs
    echo ""
fi

# Set custom input path if provided
if [ -n "$CUSTOM_INPUT_PATH" ]; then
    # Convert to absolute path
    CUSTOM_INPUT_PATH="$(cd "$(dirname "$CUSTOM_INPUT_PATH")" && pwd)/$(basename "$CUSTOM_INPUT_PATH")"
    echo -e "${YELLOW}Using custom input path: $CUSTOM_INPUT_PATH${NC}"
    export TEST_CUSTOM_INPUT_PATH="$CUSTOM_INPUT_PATH"
    
    # Auto-enable run all tests when custom input is provided (unless specific test is selected)
    if [ -z "$TEST_NAME" ]; then
        RUN_ALL=true
        echo -e "${YELLOW}Automatically running all test cases with custom input${NC}"
    fi
    echo ""
fi

# Export service port for pytest
export TEST_SERVICE_PORT="$SERVICE_PORT"

# Export image limit if specified
if [ -n "$IMAGE_LIMIT" ]; then
    export TEST_IMAGE_LIMIT="$IMAGE_LIMIT"
    echo -e "${YELLOW}Limiting to first $IMAGE_LIMIT images${NC}"
    echo ""
fi

# Build pytest command with optional --deepx and --sync flags
PYTEST_ARGS=""
if [ "$USE_DEEPX" = true ]; then
    # Check if NPU is actually available
    if [ "$SETUP_NPU" = "false" ]; then
        echo -e "${RED}Error: --deepx flag specified but NPU is not available${NC}"
        echo -e "${YELLOW}Please run local_deepx_setup.sh first to setup NPU environment${NC}"
        echo -e "${YELLOW}Or run without --deepx flag to use CPU mode${NC}"
        echo ""
        exit 1
    else
        PYTEST_ARGS="--deepx"
        echo -e "${YELLOW}Using DEEPX NPU for inference${NC}"
        if [ "$USE_SYNC" = true ]; then
            PYTEST_ARGS="$PYTEST_ARGS --sync"
            echo -e "${YELLOW}Using sync mode (PaddleOcr)${NC}"
        else
            echo -e "${YELLOW}Using async mode (AsyncPipelineOCR)${NC}"
        fi
        echo ""
    fi
fi

# Run tests with pytest-html report
if [ "$RUN_ALL" = true ]; then
    echo -e "${YELLOW}Running all test suites...${NC}"
    TEST_SPEC="test_ocr_service.py"
elif [ -n "$TEST_NAME" ]; then
    echo -e "${YELLOW}Running specific test: $TEST_NAME...${NC}"
    # Auto-detect test class based on test name
    if [[ "$TEST_NAME" == *"::"* ]]; then
        # Full path provided (e.g., TestBatchOCR::test_fastapi_batch_ocr)
        TEST_SPEC="test_ocr_service.py::$TEST_NAME"
    elif [[ "$TEST_NAME" == test_fastapi_batch_ocr* ]] || [[ "$TEST_NAME" == test_hubserving* ]] || [[ "$TEST_NAME" == test_batch_ocr* ]]; then
        TEST_SPEC="test_ocr_service.py::TestBatchOCR::$TEST_NAME"
    elif [[ "$TEST_NAME" == test_fastapi_ocr* ]]; then
        TEST_SPEC="test_ocr_service.py::TestFastAPIOCR::$TEST_NAME"
    elif [[ "$TEST_NAME" == test_swagger* ]] || [[ "$TEST_NAME" == test_redoc* ]] || [[ "$TEST_NAME" == test_openapi* ]]; then
        TEST_SPEC="test_ocr_service.py::TestAPIDocumentation::$TEST_NAME"
    elif [[ "$TEST_NAME" == test_concurrent* ]] || [[ "$TEST_NAME" == test_large* ]] || [[ "$TEST_NAME" == test_ocr_instance* ]]; then
        TEST_SPEC="test_ocr_service.py::TestPerformanceAndEdgeCases::$TEST_NAME"
    elif [[ "$TEST_NAME" == test_health* ]]; then
        TEST_SPEC="test_ocr_service.py::TestHealthCheck::$TEST_NAME"
    else
        # Default to TestBaiduOCRAPI for backward compatibility
        TEST_SPEC="test_ocr_service.py::TestBaiduOCRAPI::$TEST_NAME"
    fi
else
    echo -e "${YELLOW}Running single test: test_baidu_ocr_all_parameters...${NC}"
    TEST_SPEC="test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_all_parameters"
fi
echo ""

REPORT_FILE="test_outputs/test_report_$(date +%Y%m%d_%H%M%S).html"

pytest $TEST_SPEC \
    $PYTEST_ARGS \
    -v \
    --html="$REPORT_FILE" \
    --self-contained-html \
    --tb=short \
    --color=yes \
    2>&1 | tee test_outputs/test_run.log

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Execution Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Summary
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed (exit code: $TEST_EXIT_CODE)${NC}"
fi

echo ""
echo -e "${YELLOW}Test Report:${NC}"
echo "  HTML: $REPORT_FILE"
echo "  Log:  test_outputs/test_run.log"
echo ""

# Count test results
TOTAL_TESTS=$(grep -c "PASSED\|FAILED" test_outputs/test_run.log || echo "0")
PASSED_TESTS=$(grep -c "PASSED" test_outputs/test_run.log || echo "0")
FAILED_TESTS=$(grep -c "FAILED" test_outputs/test_run.log || echo "0")

echo -e "${YELLOW}Test Statistics:${NC}"
echo "  Total:  $TOTAL_TESTS"
echo -e "  Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "  Failed: ${RED}$FAILED_TESTS${NC}"
echo ""

# List generated output files
OUTPUT_COUNT=$(find test_outputs -type f -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
echo -e "${YELLOW}Generated Files:${NC}"
echo "  Images: $OUTPUT_COUNT"
echo ""

# Show output directory structure
echo -e "${YELLOW}Output Directory Structure:${NC}"
tree -L 2 test_outputs/ 2>/dev/null || ls -R test_outputs/
echo ""

echo -e "${GREEN}Done!${NC}"
echo ""
echo "To view the HTML report, open:"
echo "  file://$SCRIPT_DIR/$REPORT_FILE"
echo ""

exit $TEST_EXIT_CODE
