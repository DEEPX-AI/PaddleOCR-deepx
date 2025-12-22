# PaddleOCR FastAPI Service - Test Suite

## Overview

Comprehensive pytest test suite for the PaddleOCR FastAPI service, covering all endpoints and Baidu AI Studio compatible parameters.

## Test Coverage

### Endpoints Tested
- ✅ `GET /health` - Health check
- ✅ `POST /api/v1/ocr` - Baidu AI Studio compatible OCR API
- ✅ `POST /fastapi/ocr` - FastAPI native OCR (URL/Base64/Array)
- ✅ `POST /fastapi/ocr/upload` - File upload OCR
- ✅ `POST /fastapi/batch_ocr` - Batch processing
- ✅ `POST /predict/ocr_system` - Hubserving compatible endpoint
- ✅ `GET /docs` - Swagger UI documentation
- ✅ `GET /redoc` - ReDoc documentation

### Baidu AI Studio 12 Parameters Tested
1. `file` - Base64 encoded file content
2. `fileType` - File type (0=PDF, 1=Image)
3. `useDocOrientationClassify` - Document orientation classification
4. `useDocUnwarping` - Document distortion correction
5. `useTextlineOrientation` - Text line orientation
6. `textDetLimitSideLen` - Detection image side length limit
7. `textDetLimitType` - Limit type (min/max)
8. `textDetThresh` - Text detection threshold
9. `textDetBoxThresh` - Text box threshold
10. `textDetUnclipRatio` - Text box unclip ratio
11. `textRecScoreThresh` - Recognition score threshold
12. `visualize` - Return visualization images

## Installation

### 1. Install Test Dependencies

```bash
cd /dataPaddleOCR/deploy/docker/fastapi
pip install -r test_requirements.txt
```

Or install manually:
```bash
pip install pytest pytest-timeout pytest-cov requests
```

### 2. Ensure Service is Running

The service must be running before executing tests. Default port is `8080` (can be customized with `--port` option).

**Start the service:**

```bash
# Using Docker
docker run -d -p 8080:8080 --name ocr-fastapi paddleocr-fastapi-service:latest

# Or run locally
cd /dataPaddleOCR/deploy/docker/fastapi
./run.sh
```

**Verify service is running:**

```bash
curl http://localhost:8080/health
# Expected: {"status":"healthy"}
```

**Note:** If using a custom port, specify it when running tests:
```bash
./run_tests.sh --port 8080
```

## Running Tests

### Using run_tests.sh Script (Recommended)

The easiest way to run tests is using the provided test script:

```bash
# Show help and all available options
./run_tests.sh --help

# Run default test (test_baidu_ocr_all_parameters)
./run_tests.sh

# Run all tests
./run_tests.sh --all

# List all available test cases
./run_tests.sh --list

# Run specific test case
./run_tests.sh --tc test_baidu_ocr_basic

# Run with custom input images
./run_tests.sh --input /path/to/images/

# Use custom port
./run_tests.sh --port 8081

# Clean previous outputs before running
./run_tests.sh --clean-output

# Test with DEEPX NPU (async mode, default)
./run_tests.sh --deepx true

# Test with DEEPX NPU (sync mode)
./run_tests.sh --deepx true --sync

# Test with CPU only
./run_tests.sh --deepx false
```

### Script Options

- `-h, --help` - Show help message
- `--all` - Run all test cases
- `-l, --list` - List all available test cases
- `--tc <test_name>` - Run specific test case
- `-i, --input <path>` - Run tests with custom input images
- `-p, --port <port>` - Use custom port (default: 8080)
- `--deepx <true|false>` - Use DEEPX NPU (default: true)
- `--sync` - Use sync mode (default: async)
- `--clean-output` - Clean previous test outputs before running (default: keep previous outputs)

### Direct pytest Usage

You can also run tests directly with pytest:

```bash
# Run all tests
pytest test_ocr_service.py -v

# Run with DEEPX NPU
pytest test_ocr_service.py -v --deepx

# Run with DEEPX NPU in sync mode
pytest test_ocr_service.py -v --deepx --sync
```

### Run Specific Test Categories

```bash
# Using run_tests.sh
./run_tests.sh --tc test_health_check

# Using pytest
pytest test_ocr_service.py -v -k "test_health"
pytest test_ocr_service.py -v -k "TestBaiduOCRAPI"
pytest test_ocr_service.py -v -k "TestFastAPIOCR"
pytest test_ocr_service.py -v -k "TestBatchOCR"
pytest test_ocr_service.py -v -k "TestPerformanceAndEdgeCases"
```

### Run Specific Tests

```bash
# Using run_tests.sh
./run_tests.sh --tc test_baidu_ocr_all_parameters
./run_tests.sh --tc test_baidu_ocr_with_visualization
./run_tests.sh --tc test_concurrent_requests

# Using pytest
pytest test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_all_parameters -v
pytest test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_with_visualization -v
pytest test_ocr_service.py::TestPerformanceAndEdgeCases::test_concurrent_requests -v
```

### Advanced Options

```bash
# Run tests in parallel (4 workers)
pytest test_ocr_service.py -v -n 4

# Generate HTML coverage report
pytest test_ocr_service.py -v --cov=ocr_service --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# The run_tests.sh script automatically generates HTML report
./run_tests.sh --all  # Creates test_outputs/test_report_*.html
```

## Test Structure

### Test Classes

```
TestHealthCheck
├── test_health_check

TestBaiduOCRAPI
├── test_baidu_ocr_basic
├── test_baidu_ocr_with_doc_orientation
├── test_baidu_ocr_with_doc_unwarping
├── test_baidu_ocr_with_textline_orientation
├── test_baidu_ocr_detection_params
├── test_baidu_ocr_recognition_params
├── test_baidu_ocr_with_visualization
├── test_baidu_ocr_all_parameters
├── test_baidu_ocr_invalid_base64
└── test_baidu_ocr_pdf_not_supported

TestFastAPIOCR
├── test_fastapi_ocr_with_url
├── test_fastapi_ocr_with_base64
├── test_fastapi_ocr_with_images_array
├── test_fastapi_ocr_no_image
├── test_fastapi_ocr_upload
└── test_fastapi_ocr_result_format

TestBatchOCR
├── test_fastapi_batch_ocr
├── test_fastapi_batch_ocr_empty
├── test_hubserving_batch_ocr
└── test_batch_ocr_result_consistency

TestAPIDocumentation
├── test_swagger_docs
├── test_redoc_docs
└── test_openapi_schema

TestPerformanceAndEdgeCases
├── test_concurrent_requests
├── test_large_image_handling
└── test_ocr_instance_caching
```

## Test Data

Tests automatically load **all PNG images** from:
```
/dataPaddleOCR/deepx/images/
```

Currently **20 images** are processed per test:
- `image_1.png` through `image_20.png`

All Baidu API parameter tests iterate over all images, ensuring comprehensive coverage.

## Visualization Output

All Baidu API tests automatically save visualization images to:
```
deploy/docker/fastapi/test_outputs/{TC_NAME}/{BACKEND}/
```

Where:
- `{TC_NAME}` - Test case name (e.g., `test_basic`, `test_all_parameters`)
- `{BACKEND}` - Backend used (`cpu` or `deepx-npu`)

**Directory Structure:**
```
test_outputs/
├── test_basic/
│   ├── cpu/
│   │   ├── image_1_1_input.jpg
│   │   ├── image_1_3_output.jpg
│   │   ├── image_2_1_input.jpg
│   │   ├── image_2_3_output.jpg
│   │   └── ...
│   └── deepx-npu/
│       ├── image_1_1_input.jpg
│       ├── image_1_3_output.jpg
│       └── ...
├── test_all_parameters/
│   ├── cpu/
│   │   ├── image_1_1_input.jpg
│   │   ├── image_1_2_preprocessing.jpg
│   │   ├── image_1_3_output.jpg
│   │   └── ...
│   └── deepx-npu/
│       └── ...
└── test_detection_params/
    ├── cpu/
    └── deepx-npu/
```

**File Naming Convention:**
- `{IMAGE_NAME}_1_input.jpg` - Original input image
- `{IMAGE_NAME}_2_preprocessing.jpg` - Preprocessing result (when doc orientation/unwarping is used)
- `{IMAGE_NAME}_3_output.jpg` - OCR result with bounding boxes

**Note:** Preprocessing image (step 2) is only generated when `useDocOrientationClassify` or `useDocUnwarping` is enabled.

**Output Management:**
- By default, previous test outputs are preserved
- Use `./run_tests.sh --clean-output` to clean outputs before running tests

## Expected Output

### Using run_tests.sh

```bash
$ ./run_tests.sh --all

========================================
PaddleOCR FastAPI Test Suite
========================================

Checking service health...
✓ Service is healthy (port: 8080)

Skipping test outputs cleanup (keeping previous results)

Using DEEPX NPU for inference
Using async mode (AsyncPipelineOCR)

Running all test suites...

test_ocr_service.py::TestHealthCheck::test_health_check PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_basic PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_with_doc_orientation PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_with_doc_unwarping PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_with_textline_orientation PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_detection_params PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_recognition_params PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_with_visualization PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_all_parameters PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_invalid_base64 PASSED
test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_pdf_not_supported PASSED
test_ocr_service.py::TestFastAPIOCR::test_fastapi_ocr_with_url PASSED
test_ocr_service.py::TestFastAPIOCR::test_fastapi_ocr_with_base64 PASSED
test_ocr_service.py::TestFastAPIOCR::test_fastapi_ocr_with_images_array PASSED
test_ocr_service.py::TestFastAPIOCR::test_fastapi_ocr_no_image PASSED
test_ocr_service.py::TestFastAPIOCR::test_fastapi_ocr_upload PASSED
test_ocr_service.py::TestFastAPIOCR::test_fastapi_ocr_result_format PASSED
test_ocr_service.py::TestBatchOCR::test_fastapi_batch_ocr PASSED
test_ocr_service.py::TestBatchOCR::test_fastapi_batch_ocr_empty PASSED
test_ocr_service.py::TestBatchOCR::test_hubserving_batch_ocr PASSED
test_ocr_service.py::TestBatchOCR::test_batch_ocr_result_consistency PASSED
test_ocr_service.py::TestAPIDocumentation::test_swagger_docs PASSED
test_ocr_service.py::TestAPIDocumentation::test_redoc_docs PASSED
test_ocr_service.py::TestAPIDocumentation::test_openapi_schema PASSED
test_ocr_service.py::TestPerformanceAndEdgeCases::test_concurrent_requests PASSED
test_ocr_service.py::TestPerformanceAndEdgeCases::test_large_image_handling PASSED
test_ocr_service.py::TestPerformanceAndEdgeCases::test_ocr_instance_caching PASSED

========================================
Test Execution Complete
========================================

✓ All tests passed!

Test Report:
  HTML: test_outputs/test_report_20231219_143052.html
  Log:  test_outputs/test_run.log

Test Statistics:
  Total:  27
  Passed: 27
  Failed: 0

Generated Files:
  Images: 240

Output Directory Structure:
test_outputs/
├── test_all_parameters/
├── test_basic/
├── test_report_20231219_143052.html
└── test_run.log

Done!
```

### Test Output Files

The script automatically generates:
- **HTML Report**: `test_outputs/test_report_*.html` - Detailed test results with pass/fail status
- **Log File**: `test_outputs/test_run.log` - Complete test execution log
- **Visualization Images**: Organized by test case and backend (cpu/deepx-npu)

## Troubleshooting

### Service Not Running
```
ERROR: Connection refused
Error: OCR service is not running on port 8080
```
**Solution:** Make sure the service is running on the correct port

```bash
# Check if service is running
docker ps | grep ocr-fastapi

# If not running, start it:
docker run -d -p 8080:8080 --name ocr-fastapi paddleocr-fastapi-service:latest

# Or run locally
./run.sh

# For custom port
./run.sh --port 8081
./run_tests.sh --port 8081
```

### Test Images Not Found
```
AssertionError: Test image not found
```
**Solution:** Verify test images exist in `deepx/images/` directory

```bash
ls -la /dataPaddleOCR/deepx/images/

# Or use custom images
./run_tests.sh --input /path/to/your/images/
```

### Import Errors
```
ModuleNotFoundError: No module named 'pytest'
```
**Solution:** Install test requirements

```bash
pip install -r test_requirements.txt
```

### Clean Output Issues
**Issue:** Previous test outputs interfering with new tests

**Solution:** Use `--clean-output` flag to clean before running

```bash
./run_tests.sh --clean-output --all
```

**Note:** By default, previous outputs are preserved to allow comparison between runs.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test OCR Service

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Build Docker image
      run: |
        cd deploy/docker/fastapi
        bash build.sh --no-models
    
    - name: Start service
      run: |
        docker run -d -p 8081:8080 --name ocr-fastapi paddleocr-fastapi-service:latest
        sleep 10
    
    - name: Install test dependencies
      run: |
        cd deploy/docker/fastapi
        pip install -r test_requirements.txt
    
    - name: Run tests
      run: |
        cd deploy/docker/fastapi
        pytest test_ocr_service.py -v --cov=ocr_service --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Contributing

When adding new endpoints or parameters:

1. Add corresponding test cases to `test_ocr_service.py`
2. Update this README with new test coverage
3. Run all tests to ensure nothing broke:
   ```bash
   ./run_tests.sh --all
   ```
4. Check HTML report for any failures:
   ```bash
   # Report is auto-generated at:
   # test_outputs/test_report_*.html
   ```

## Quick Reference

### Common Commands

```bash
# Show all options
./run_tests.sh --help

# Run default test
./run_tests.sh

# Run all tests
./run_tests.sh --all

# List available tests
./run_tests.sh --list

# Run specific test
./run_tests.sh --tc test_baidu_ocr_basic

# Clean and run
./run_tests.sh --clean-output --all

# Test with DEEPX NPU
./run_tests.sh --deepx true --all

# Test with CPU only
./run_tests.sh --deepx false --all

# Custom input and port
./run_tests.sh --input /path/to/images/ --port 8081
```

### Backend Options

- **DEEPX NPU** (default): Hardware acceleration via NPU
  - Async mode (default): `AsyncPipelineOCR`
  - Sync mode: `PaddleOcr` (use `--sync`)
- **CPU**: Software-based processing (use `--deepx false`)

```bash
# NPU async (default)
./run_tests.sh --deepx true

# NPU sync
./run_tests.sh --deepx true --sync

# CPU
./run_tests.sh --deepx false
```

## License

Same as PaddleOCR project.
