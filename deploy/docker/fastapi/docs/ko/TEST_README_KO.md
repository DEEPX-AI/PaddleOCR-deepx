# PaddleOCR FastAPI 서비스 - 테스트 스위트

## 개요

PaddleOCR FastAPI 서비스의 모든 엔드포인트와 Baidu AI Studio 호환 파라미터를 커버하는 포괄적인 pytest 테스트 스위트입니다.

## 테스트 커버리지

### 테스트된 엔드포인트
- ✅ `GET /health` - 헬스 체크
- ✅ `POST /api/v1/ocr` - Baidu AI Studio 호환 OCR API
- ✅ `POST /fastapi/ocr` - FastAPI 네이티브 OCR (URL/Base64/Array)
- ✅ `POST /fastapi/ocr/upload` - 파일 업로드 OCR
- ✅ `POST /fastapi/batch_ocr` - 배치 처리
- ✅ `POST /predict/ocr_system` - Hubserving 호환 엔드포인트
- ✅ `GET /docs` - Swagger UI 문서
- ✅ `GET /redoc` - ReDoc 문서

### 테스트된 Baidu AI Studio 12개 파라미터
1. `file` - Base64 인코딩된 파일 콘텐츠
2. `fileType` - 파일 타입 (0=PDF, 1=Image)
3. `useDocOrientationClassify` - 문서 방향 분류
4. `useDocUnwarping` - 문서 왜곡 보정
5. `useTextlineOrientation` - 텍스트 라인 방향
6. `textDetLimitSideLen` - 검출 이미지 측면 길이 제한
7. `textDetLimitType` - 제한 타입 (min/max)
8. `textDetThresh` - 텍스트 검출 임계값
9. `textDetBoxThresh` - 텍스트 박스 임계값
10. `textDetUnclipRatio` - 텍스트 박스 언클립 비율
11. `textRecScoreThresh` - 인식 점수 임계값
12. `visualize` - 시각화 이미지 반환

## 설치

### 1. 테스트 의존성 설치

```bash
cd /data/home/dhyang/git/github/PaddleOCR/deploy/docker/fastapi
pip install -r test_requirements.txt
```

또는 수동 설치:
```bash
pip install pytest pytest-timeout pytest-cov requests
```

### 2. 서비스 실행 확인

테스트 실행 전에 서비스가 실행 중이어야 합니다. 기본 포트는 `8080`입니다 (`--port` 옵션으로 커스터마이징 가능).

**서비스 시작:**

```bash
# Docker 사용
docker run -d -p 8080:8080 --name ocr-fastapi paddleocr-fastapi-service:latest

# 또는 로컬 실행
cd /data/home/dhyang/git/github/PaddleOCR/deploy/docker/fastapi
./run.sh
```

**서비스 실행 확인:**

```bash
curl http://localhost:8080/health
# 예상 결과: {"status":"healthy"}
```

**참고:** 커스텀 포트 사용 시 테스트 실행 시 포트를 지정하세요:
```bash
./run_tests.sh --port 8080
```

## 테스트 실행

### run_tests.sh 스크립트 사용 (권장)

제공된 테스트 스크립트를 사용하는 것이 가장 쉬운 방법입니다:

```bash
# 도움말 및 모든 옵션 표시
./run_tests.sh --help

# 기본 테스트 실행 (test_baidu_ocr_all_parameters)
./run_tests.sh

# 모든 테스트 실행
./run_tests.sh --all

# 사용 가능한 모든 테스트 케이스 나열
./run_tests.sh --list

# 특정 테스트 케이스 실행
./run_tests.sh --tc test_baidu_ocr_basic

# 커스텀 입력 이미지로 실행
./run_tests.sh --input /path/to/images/

# 커스텀 포트 사용
./run_tests.sh --port 8081

# 실행 전 이전 출력 삭제
./run_tests.sh --clean-output

# DEEPX NPU로 테스트 (async 모드, 기본값)
./run_tests.sh --deepx true

# DEEPX NPU로 테스트 (sync 모드)
./run_tests.sh --deepx true --sync

# CPU 전용 테스트
./run_tests.sh --deepx false
```

### 스크립트 옵션

- `-h, --help` - 도움말 메시지 표시
- `--all` - 모든 테스트 케이스 실행
- `-l, --list` - 사용 가능한 모든 테스트 케이스 나열
- `--tc <test_name>` - 특정 테스트 케이스 실행
- `-i, --input <path>` - 커스텀 입력 이미지로 테스트 실행
- `-p, --port <port>` - 커스텀 포트 사용 (기본값: 8080)
- `--deepx <true|false>` - DEEPX NPU 사용 (기본값: true)
- `--sync` - sync 모드 사용 (기본값: async)
- `--clean-output` - 실행 전 이전 테스트 출력 삭제 (기본값: 이전 출력 유지)

### pytest 직접 사용

pytest를 직접 사용하여 테스트를 실행할 수도 있습니다:

```bash
# 모든 테스트 실행
pytest test_ocr_service.py -v

# DEEPX NPU로 실행
pytest test_ocr_service.py -v --deepx

# DEEPX NPU로 sync 모드 실행
pytest test_ocr_service.py -v --deepx --sync
```

### 특정 테스트 카테고리 실행

```bash
# run_tests.sh 사용
./run_tests.sh --tc test_health_check

# pytest 사용
pytest test_ocr_service.py -v -k "test_health"
pytest test_ocr_service.py -v -k "TestBaiduOCRAPI"
pytest test_ocr_service.py -v -k "TestFastAPIOCR"
pytest test_ocr_service.py -v -k "TestBatchOCR"
pytest test_ocr_service.py -v -k "TestPerformanceAndEdgeCases"
```

### 특정 테스트 실행

```bash
# run_tests.sh 사용
./run_tests.sh --tc test_baidu_ocr_all_parameters
./run_tests.sh --tc test_baidu_ocr_with_visualization
./run_tests.sh --tc test_concurrent_requests

# pytest 사용
pytest test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_all_parameters -v
pytest test_ocr_service.py::TestBaiduOCRAPI::test_baidu_ocr_with_visualization -v
pytest test_ocr_service.py::TestPerformanceAndEdgeCases::test_concurrent_requests -v
```

### 고급 옵션

```bash
# 병렬로 테스트 실행 (4 워커)
pytest test_ocr_service.py -v -n 4

# HTML 커버리지 리포트 생성
pytest test_ocr_service.py -v --cov=ocr_service --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# run_tests.sh 스크립트는 자동으로 HTML 리포트 생성
./run_tests.sh --all  # test_outputs/test_report_*.html 생성
```

## 테스트 구조

### 테스트 클래스

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

## 테스트 데이터

테스트는 다음 경로에서 **모든 PNG 이미지**를 자동으로 로드합니다:
```
/data/home/dhyang/git/github/PaddleOCR/deepx/images/
```

현재 테스트당 **20개 이미지**가 처리됩니다:
- `image_1.png`부터 `image_20.png`까지

모든 Baidu API 파라미터 테스트는 모든 이미지를 반복하여 포괄적인 커버리지를 보장합니다.

## 시각화 출력

모든 Baidu API 테스트는 시각화 이미지를 자동으로 저장합니다:
```
deploy/docker/fastapi/test_outputs/{TC_NAME}/{BACKEND}/
```

여기서:
- `{TC_NAME}` - 테스트 케이스 이름 (예: `test_basic`, `test_all_parameters`)
- `{BACKEND}` - 사용된 백엔드 (`cpu` 또는 `deepx-npu`)

**디렉토리 구조:**
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

**파일 명명 규칙:**
- `{IMAGE_NAME}_1_input.jpg` - 원본 입력 이미지
- `{IMAGE_NAME}_2_preprocessing.jpg` - 전처리 결과 (문서 방향/왜곡 보정 사용 시)
- `{IMAGE_NAME}_3_output.jpg` - 바운딩 박스가 포함된 OCR 결과

**참고:** 전처리 이미지(2단계)는 `useDocOrientationClassify` 또는 `useDocUnwarping`이 활성화된 경우에만 생성됩니다.

**출력 관리:**
- 기본적으로 이전 테스트 출력이 보존됩니다
- `./run_tests.sh --clean-output`을 사용하여 테스트 실행 전 출력을 삭제할 수 있습니다

## 예상 출력

### run_tests.sh 사용

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

### 테스트 출력 파일

스크립트는 자동으로 다음을 생성합니다:
- **HTML 리포트**: `test_outputs/test_report_*.html` - 통과/실패 상태가 포함된 상세 테스트 결과
- **로그 파일**: `test_outputs/test_run.log` - 완전한 테스트 실행 로그
- **시각화 이미지**: 테스트 케이스 및 백엔드별로 정리 (cpu/deepx-npu)

## 문제 해결

### 서비스 미실행
```
ERROR: Connection refused
Error: OCR service is not running on port 8080
```
**해결책:** 올바른 포트에서 서비스가 실행 중인지 확인하세요

```bash
# 서비스 실행 확인
docker ps | grep ocr-fastapi

# 실행 중이 아니면 시작:
docker run -d -p 8080:8080 --name ocr-fastapi paddleocr-fastapi-service:latest

# 또는 로컬 실행
./run.sh

# 커스텀 포트의 경우
./run.sh --port 8081
./run_tests.sh --port 8081
```

### 테스트 이미지를 찾을 수 없음
```
AssertionError: Test image not found
```
**해결책:** `deepx/images/` 디렉토리에 테스트 이미지가 있는지 확인하세요

```bash
ls -la /data/home/dhyang/git/github/PaddleOCR/deepx/images/

# 또는 커스텀 이미지 사용
./run_tests.sh --input /path/to/your/images/
```

### 임포트 에러
```
ModuleNotFoundError: No module named 'pytest'
```
**해결책:** 테스트 요구사항 설치

```bash
pip install -r test_requirements.txt
```

### 출력 삭제 문제
**문제:** 이전 테스트 출력이 새 테스트를 방해함

**해결책:** 실행 전 삭제하려면 `--clean-output` 플래그 사용

```bash
./run_tests.sh --clean-output --all
```

**참고:** 기본적으로 이전 출력은 실행 간 비교를 위해 보존됩니다.

## CI/CD 통합

### GitHub Actions 예제

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

## 기여하기

새로운 엔드포인트나 파라미터를 추가할 때:

1. `test_ocr_service.py`에 해당 테스트 케이스 추가
2. 이 README를 새로운 테스트 커버리지로 업데이트
3. 모든 테스트를 실행하여 문제가 없는지 확인:
   ```bash
   ./run_tests.sh --all
   ```
4. 실패한 것이 있는지 HTML 리포트 확인:
   ```bash
   # 리포트는 다음 위치에 자동 생성됩니다:
   # test_outputs/test_report_*.html
   ```

## 빠른 참조

### 일반 명령어

```bash
# 모든 옵션 표시
./run_tests.sh --help

# 기본 테스트 실행
./run_tests.sh

# 모든 테스트 실행
./run_tests.sh --all

# 사용 가능한 테스트 나열
./run_tests.sh --list

# 특정 테스트 실행
./run_tests.sh --tc test_baidu_ocr_basic

# 삭제 후 실행
./run_tests.sh --clean-output --all

# DEEPX NPU로 테스트
./run_tests.sh --deepx true --all

# CPU 전용 테스트
./run_tests.sh --deepx false --all

# 커스텀 입력 및 포트
./run_tests.sh --input /path/to/images/ --port 8081
```

### 백엔드 옵션

- **DEEPX NPU** (기본값): NPU를 통한 하드웨어 가속
  - Async 모드 (기본값): `AsyncPipelineOCR`
  - Sync 모드: `PaddleOcr` (`--sync` 사용)
- **CPU**: 소프트웨어 기반 처리 (`--deepx false` 사용)

```bash
# NPU async (기본값)
./run_tests.sh --deepx true

# NPU sync
./run_tests.sh --deepx true --sync

# CPU
./run_tests.sh --deepx false
```

## 라이선스

PaddleOCR 프로젝트와 동일합니다.
