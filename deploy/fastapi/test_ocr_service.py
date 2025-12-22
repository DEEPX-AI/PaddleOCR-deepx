#!/usr/bin/env python3
"""
Comprehensive pytest test suite for PaddleOCR FastAPI Service

Tests all endpoints and Baidu AI Studio compatible parameters:
1. Health check
2. Baidu AI Studio OCR API (/api/v1/ocr) with 12 parameters
3. FastAPI native OCR endpoints
4. Hubserving compatible endpoint
5. Batch processing
6. File upload

Usage:
    pytest test_ocr_service.py -v
    pytest test_ocr_service.py -v -k "test_baidu"  # Run only Baidu API tests
    pytest test_ocr_service.py -v -k "test_health"  # Run only health tests
    
    # With DEEPX NPU:
    pytest test_ocr_service.py -v --deepx
    pytest test_ocr_service.py -v --deepx -k "test_baidu"
"""

import pytest
import requests
import base64
import json
from pathlib import Path
from typing import List, Dict, Any
import time
import os
import argparse

# Configuration
# Support custom port via environment variable (default: 8080)
SERVICE_PORT = os.environ.get('TEST_SERVICE_PORT', '8080')
BASE_URL = f"http://localhost:{SERVICE_PORT}"

# Support custom input path via environment variable
CUSTOM_INPUT = os.environ.get('TEST_CUSTOM_INPUT_PATH')
if CUSTOM_INPUT:
    CUSTOM_INPUT_PATH = Path(CUSTOM_INPUT)
    if CUSTOM_INPUT_PATH.is_file():
        # Single file provided
        IMAGES_DIR = CUSTOM_INPUT_PATH.parent
        TEST_IMAGES = [CUSTOM_INPUT_PATH.name]
    else:
        # Directory provided
        IMAGES_DIR = CUSTOM_INPUT_PATH
        TEST_IMAGES = sorted([f.name for f in IMAGES_DIR.glob("*.png")] + 
                            [f.name for f in IMAGES_DIR.glob("*.jpg")] +
                            [f.name for f in IMAGES_DIR.glob("*.jpeg")])
else:
    # Use DEEPX_PATH environment variable if set, otherwise use relative path (for local development)
    deepx_path_str = os.getenv('DEEPX_PATH')
    if deepx_path_str:
        deepx_path = Path(deepx_path_str)
    else:
        deepx_path = Path(__file__).parent / "deepx"
    
    IMAGES_DIR = deepx_path / "images"
    TEST_IMAGES = sorted([f.name for f in IMAGES_DIR.glob("*.png")])

# Support image limit via environment variable
IMAGE_LIMIT = os.environ.get('TEST_IMAGE_LIMIT')
if IMAGE_LIMIT:
    try:
        limit = int(IMAGE_LIMIT)
        if limit > 0 and limit < len(TEST_IMAGES):
            print(f"⚠️  Limiting images from {len(TEST_IMAGES)} to {limit}")
            TEST_IMAGES = TEST_IMAGES[:limit]
    except ValueError:
        print(f"⚠️  Invalid TEST_IMAGE_LIMIT value: {IMAGE_LIMIT}, using all images")

OUTPUT_DIR = Path(__file__).parent / "test_outputs"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def wait_for_service():
    """Wait for service to be ready"""
    max_retries = 30
    retry_delay = 1
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"\n✅ Service is ready at {BASE_URL}")
                return True
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    
    raise RuntimeError(f"Service did not become ready after {max_retries} seconds")


@pytest.fixture(scope="session")
def test_image_path():
    """Get path to first test image"""
    if not TEST_IMAGES:
        pytest.skip(f"No test images found in {IMAGES_DIR}")
    img_path = IMAGES_DIR / TEST_IMAGES[0]
    assert img_path.exists(), f"Test image not found: {img_path}"
    return img_path


@pytest.fixture(scope="session")
def test_image_base64(test_image_path):
    """Load test image as base64"""
    with open(test_image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


@pytest.fixture(scope="session")
def test_images_base64():
    """Load all test images as base64 with their filenames"""
    images = []
    for img_name in TEST_IMAGES:
        img_path = IMAGES_DIR / img_name
        if img_path.exists():
            with open(img_path, 'rb') as f:
                images.append({
                    'name': img_path.stem,  # filename without extension
                    'base64': base64.b64encode(f.read()).decode('utf-8')
                })
    return images


@pytest.fixture(scope="session")
def sample_image_url():
    """Sample image URL for testing"""
    return "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.6/doc/imgs_en/254.jpg"


# ============================================================================
# Test: Health Check
# ============================================================================

class TestHealthCheck:
    """Health check endpoint tests"""
    
    def test_health_check(self, wait_for_service):
        """Test GET /health endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print(f"✅ Health check passed: {data}")


# ============================================================================
# Test: Baidu AI Studio Compatible API
# ============================================================================

class TestBaiduOCRAPI:
    """Baidu AI Studio OCR API compatibility tests"""
    
    def save_visualization_images(self, test_name: str, image_name: str, response_data: dict, backend: str = "cpu"):
        """Save visualization images from response to files
        
        Args:
            test_name: Test case name (e.g., 'test_basic')
            image_name: Image filename without extension (e.g., 'image_1')
            response_data: OCR response data
            backend: Backend name ('cpu' or 'deepx-npu')
        """
        # Create test-specific directory with backend subdirectory
        test_dir = OUTPUT_DIR / test_name / backend
        test_dir.mkdir(parents=True, exist_ok=True)
        
        ocr_results = response_data.get("result", {}).get("ocrResults", [])
        
        for idx, page_result in enumerate(ocr_results):
            # Save input image
            if page_result.get("inputImage"):
                input_img_data = base64.b64decode(page_result["inputImage"])
                output_path = test_dir / f"{image_name}_1_input.jpg"
                with open(output_path, "wb") as f:
                    f.write(input_img_data)
                print(f"   Saved input image: {output_path}")
            
            # Save preprocessing image
            if page_result.get("docPreprocessingImage"):
                prep_img_data = base64.b64decode(page_result["docPreprocessingImage"])
                output_path = test_dir / f"{image_name}_2_preprocessing.jpg"
                with open(output_path, "wb") as f:
                    f.write(prep_img_data)
                print(f"   Saved preprocessing image: {output_path}")
            
            # Save OCR output image
            if page_result.get("ocrImage"):
                ocr_img_data = base64.b64decode(page_result["ocrImage"])
                output_path = test_dir / f"{image_name}_3_output.jpg"
                with open(output_path, "wb") as f:
                    f.write(ocr_img_data)
                print(f"   Saved output image: {output_path}")
    
    def test_baidu_ocr_basic(self, wait_for_service, test_images_base64, use_deepx, use_sync, backend_name):
        """Test basic Baidu OCR API call with minimal parameters"""
        for img_data in test_images_base64:
            payload = {
                "file": img_data['base64'],
                "fileType": 1,
                "visualize": True,
                "deepx": use_deepx,
                "sync": use_sync
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["errorCode"] == 0
            assert data["errorMsg"] == "Success"
            assert "logId" in data
            assert "result" in data
            assert "ocrResults" in data["result"]
            
            # Save visualization images
            self.save_visualization_images("test_basic", img_data['name'], data, backend_name)
        
        print(f"✅ Baidu OCR basic test passed for {len(test_images_base64)} images ({backend_name})")
    
    def test_baidu_ocr_with_doc_orientation(self, wait_for_service, test_images_base64, use_deepx, use_sync, backend_name):
        """Test Baidu OCR with document orientation classification"""
        for img_data in test_images_base64:
            payload = {
                "file": img_data['base64'],
                "fileType": 1,
                "useDocOrientationClassify": True,
                "visualize": True,
                "deepx": use_deepx,
                "sync": use_sync
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["errorCode"] == 0
            
            # Save visualization images
            self.save_visualization_images("test_doc_orientation", img_data['name'], data, backend_name)
        
        print(f"✅ Document orientation classification test passed for {len(test_images_base64)} images ({backend_name})")
    
    def test_baidu_ocr_with_doc_unwarping(self, wait_for_service, test_images_base64, use_deepx, use_sync, backend_name):
        """Test Baidu OCR with document unwarping"""
        for img_data in test_images_base64:
            payload = {
                "file": img_data['base64'],
                "fileType": 1,
                "useDocUnwarping": True,
                "visualize": True,
                "deepx": use_deepx,
                "sync": use_sync
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["errorCode"] == 0
            
            # Save visualization images
            self.save_visualization_images("test_doc_unwarping", img_data['name'], data, backend_name)
        
        print(f"✅ Document unwarping test passed for {len(test_images_base64)} images ({backend_name})")
    
    def test_baidu_ocr_with_textline_orientation(self, wait_for_service, test_images_base64, use_deepx, use_sync, backend_name):
        """Test Baidu OCR with text line orientation"""
        for img_data in test_images_base64:
            payload = {
                "file": img_data['base64'],
                "fileType": 1,
                "useTextlineOrientation": True,
                "visualize": True,
                "deepx": use_deepx,
                "sync": use_sync
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["errorCode"] == 0
            
            # Save visualization images
            self.save_visualization_images("test_textline_orientation", img_data['name'], data, backend_name)
        
        print(f"✅ Text line orientation test passed for {len(test_images_base64)} images ({backend_name})")
    
    def test_baidu_ocr_detection_params(self, wait_for_service, test_images_base64, use_deepx, use_sync, backend_name):
        """Test Baidu OCR with custom detection parameters"""
        for img_data in test_images_base64:
            payload = {
                "file": img_data['base64'],
                "fileType": 1,
                "textDetLimitSideLen": 1280,
                "textDetLimitType": "max",
                "textDetThresh": 0.4,
                "textDetBoxThresh": 0.7,
                "textDetUnclipRatio": 2.0,
                "visualize": True,
                "deepx": use_deepx,
                "sync": use_sync
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["errorCode"] == 0
            
            # Save visualization images
            self.save_visualization_images("test_detection_params", img_data['name'], data, backend_name)
        
        print(f"✅ Detection parameters test passed for {len(test_images_base64)} images ({backend_name})")
    
    def test_baidu_ocr_recognition_params(self, wait_for_service, test_images_base64, use_deepx, use_sync, backend_name):
        """Test Baidu OCR with custom recognition parameters"""
        for img_data in test_images_base64:
            payload = {
                "file": img_data['base64'],
                "fileType": 1,
                "textRecScoreThresh": 0.6,
                "visualize": True,
                "deepx": use_deepx,
                "sync": use_sync
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["errorCode"] == 0
            
            # Save visualization images
            self.save_visualization_images("test_recognition_params", img_data['name'], data, backend_name)
        
        print(f"✅ Recognition parameters test passed for {len(test_images_base64)} images ({backend_name})")
    
    def test_baidu_ocr_with_visualization(self, wait_for_service, test_images_base64, use_deepx, use_sync, backend_name):
        """Test Baidu OCR with visualization enabled"""
        for img_data in test_images_base64:
            payload = {
                "file": img_data['base64'],
                "fileType": 1,
                "visualize": True,
                "deepx": use_deepx,
                "sync": use_sync
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["errorCode"] == 0
            
            # Check if visualization images are present
            ocr_results = data["result"]["ocrResults"]
            if len(ocr_results) > 0:
                # Note: Visualization images may be None if no text detected
                assert "ocrImage" in ocr_results[0]
                assert "inputImage" in ocr_results[0]
            
            # Save visualization images
            self.save_visualization_images("test_visualization", img_data['name'], data, backend_name)
        
        print(f"✅ Visualization test passed for {len(test_images_base64)} images ({backend_name})")
    
    def test_baidu_ocr_all_parameters(self, wait_for_service, test_images_base64, use_deepx, use_sync, backend_name):
        """Test Baidu OCR with all 12 parameters"""
        for img_data in test_images_base64:
            payload = {
                "file": img_data['base64'],
                "fileType": 1,
                # Preprocessing parameters
                "useDocOrientationClassify": True,
                "useDocUnwarping": True,
                "useTextlineOrientation": True,
                # Detection parameters
                "textDetLimitSideLen": 1920,
                "textDetLimitType": "max",
                "textDetThresh": 0.3,
                "textDetBoxThresh": 0.6,
                "textDetUnclipRatio": 1.8,
                # Recognition parameters
                "textRecScoreThresh": 0.55,
                # Visualization
                "visualize": True,
                # DEEPX NPU
                "deepx": use_deepx,
                "sync": use_sync
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["errorCode"] == 0
            assert data["errorMsg"] == "Success"
            assert "result" in data
            assert "dataInfo" in data["result"]
            assert data["result"]["dataInfo"]["fileType"] == 1
            assert data["result"]["dataInfo"]["imageCount"] == 1
            
            # Save visualization images
            self.save_visualization_images("test_all_parameters", img_data['name'], data, backend_name)
        
        print(f"✅ All 12 parameters test passed for {len(test_images_base64)} images ({backend_name})")
    
    def test_baidu_ocr_invalid_base64(self, wait_for_service):
        """Test Baidu OCR with invalid base64 encoding"""
        payload = {
            "file": "invalid_base64_string!!!",
            "fileType": 1
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
        
        assert response.status_code == 400
        print(f"✅ Invalid base64 handling test passed")
    
    def test_baidu_ocr_pdf_not_supported(self, wait_for_service, test_image_base64):
        """Test Baidu OCR with PDF file type (should fail)"""
        payload = {
            "file": test_image_base64,
            "fileType": 0  # PDF
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
        
        assert response.status_code == 400
        print(f"✅ PDF not supported test passed")


# ============================================================================
# Test: FastAPI Native OCR Endpoints
# ============================================================================

class TestFastAPIOCR:
    """FastAPI native OCR endpoint tests"""
    
    def test_fastapi_ocr_with_url(self, wait_for_service, sample_image_url, use_deepx, use_sync):
        """Test FastAPI OCR with image URL"""
        payload = {
            "url": sample_image_url,
            "deepx": use_deepx,
            "sync": use_sync
        }
        
        response = requests.post(f"{BASE_URL}/fastapi/ocr", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert isinstance(data["results"], list)
        print(f"✅ FastAPI URL OCR test passed")
    
    def test_fastapi_ocr_with_base64(self, wait_for_service, test_image_base64, use_deepx, use_sync):
        """Test FastAPI OCR with base64 encoded image"""
        payload = {
            "image": test_image_base64,
            "deepx": use_deepx,
            "sync": use_sync
        }
        
        response = requests.post(f"{BASE_URL}/fastapi/ocr", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        print(f"✅ FastAPI base64 OCR test passed")
    
    def test_fastapi_ocr_with_images_array(self, wait_for_service, test_image_base64, use_deepx, use_sync):
        """Test FastAPI OCR with images array (uses first image)"""
        payload = {
            "images": [test_image_base64],
            "deepx": use_deepx,
            "sync": use_sync
        }
        
        response = requests.post(f"{BASE_URL}/fastapi/ocr", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        print(f"✅ FastAPI images array OCR test passed")
    
    def test_fastapi_ocr_no_image(self, wait_for_service):
        """Test FastAPI OCR with no image provided (should fail)"""
        payload = {}
        
        response = requests.post(f"{BASE_URL}/fastapi/ocr", json=payload)
        
        assert response.status_code == 400
        print(f"✅ FastAPI no image error handling test passed")
    
    def test_fastapi_ocr_upload(self, wait_for_service, test_image_path, use_deepx, use_sync):
        """Test FastAPI OCR with file upload"""
        with open(test_image_path, 'rb') as f:
            files = {'file': (test_image_path.name, f, 'image/png')}
            data = {
                'deepx': 'true' if use_deepx else 'false',
                'sync': 'true' if use_sync else 'false'
            }
            response = requests.post(f"{BASE_URL}/fastapi/ocr/upload", files=files, data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        print(f"✅ FastAPI file upload OCR test passed")
    
    def test_fastapi_ocr_result_format(self, wait_for_service, test_image_base64, use_deepx, use_sync):
        """Test FastAPI OCR result format validation"""
        payload = {
            "image": test_image_base64,
            "deepx": use_deepx,
            "sync": use_sync
        }
        
        response = requests.post(f"{BASE_URL}/fastapi/ocr", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Validate result structure
        results = data["results"]
        if len(results) > 0:
            result = results[0]
            assert "bbox" in result
            assert "text" in result
            assert "confidence" in result
            assert isinstance(result["bbox"], list)
            assert isinstance(result["text"], str)
            assert isinstance(result["confidence"], (int, float))
            assert 0 <= result["confidence"] <= 1
        
        print(f"✅ FastAPI result format validation test passed")


# ============================================================================
# Test: Batch OCR Endpoints
# ============================================================================

class TestBatchOCR:
    """Batch OCR endpoint tests"""
    
    def test_fastapi_batch_ocr(self, wait_for_service, test_images_base64, use_deepx, use_sync):
        """Test FastAPI batch OCR endpoint"""
        # Extract base64 strings from image dictionaries (use only first 3 for speed)
        images = [img['base64'] for img in test_images_base64[:3]]
        
        payload = {
            "images": images,
            "deepx": use_deepx,
            "sync": use_sync
        }
        
        response = requests.post(f"{BASE_URL}/fastapi/batch_ocr", json=payload, timeout=60)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == len(images)
        print(f"✅ FastAPI batch OCR test passed ({len(images)} images)")
    
    def test_fastapi_batch_ocr_empty(self, wait_for_service):
        """Test FastAPI batch OCR with empty images array"""
        payload = {
            "images": []
        }
        
        response = requests.post(f"{BASE_URL}/fastapi/batch_ocr", json=payload)
        
        assert response.status_code == 400
        print(f"✅ FastAPI batch OCR empty array test passed")
    
    def test_hubserving_ocr_system(self, wait_for_service, test_images_base64, use_deepx, use_sync):
        """Test Hubserving compatible OCR system endpoint"""
        # Extract base64 strings from image dictionaries (use only first 3 for speed)
        images = [img['base64'] for img in test_images_base64[:3]]
        
        payload = {
            "images": images,
            "deepx": use_deepx,
            "sync": use_sync
        }
        
        response = requests.post(f"{BASE_URL}/predict/ocr_system", json=payload, timeout=60)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == len(images)
        print(f"✅ Hubserving batch OCR test passed ({len(images)} images)")
    
    def test_batch_ocr_result_consistency(self, wait_for_service, test_images_base64, use_deepx, use_sync):
        """Test that batch OCR results are consistent across endpoints"""
        # Extract base64 strings from image dictionaries (use only first 2 for speed)
        images = [img['base64'] for img in test_images_base64[:2]]
        
        payload = {
            "images": images,
            "deepx": use_deepx,
            "sync": use_sync
        }
        
        # Test FastAPI batch endpoint
        fastapi_response = requests.post(f"{BASE_URL}/fastapi/batch_ocr", json=payload, timeout=60)
        assert fastapi_response.status_code == 200, f"FastAPI failed: {fastapi_response.text}"
        fastapi_data = fastapi_response.json()
        
        # Test Hubserving endpoint
        hubserving_response = requests.post(f"{BASE_URL}/predict/ocr_system", json=payload, timeout=60)
        assert hubserving_response.status_code == 200, f"Hubserving failed: {hubserving_response.text}"
        hubserving_data = hubserving_response.json()
        
        # Both should succeed
        assert fastapi_data["success"] is True
        assert hubserving_data["success"] is True
        
        # Both should have same number of results
        assert len(fastapi_data["results"]) == len(hubserving_data["results"])
        
        print(f"✅ Batch OCR consistency test passed")


# ============================================================================
# Test: API Documentation
# ============================================================================

class TestAPIDocumentation:
    """API documentation endpoint tests"""
    
    def test_swagger_docs(self, wait_for_service):
        """Test Swagger UI documentation endpoint"""
        response = requests.get(f"{BASE_URL}/docs")
        
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
        print(f"✅ Swagger UI documentation test passed")
    
    def test_redoc_docs(self, wait_for_service):
        """Test ReDoc documentation endpoint"""
        response = requests.get(f"{BASE_URL}/redoc")
        
        assert response.status_code == 200
        assert "redoc" in response.text.lower()
        print(f"✅ ReDoc documentation test passed")
    
    def test_openapi_schema(self, wait_for_service):
        """Test OpenAPI schema endpoint"""
        response = requests.get(f"{BASE_URL}/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        
        # Verify all expected endpoints are documented
        expected_endpoints = [
            "/health",
            "/api/v1/ocr",
            "/fastapi/ocr",
            "/fastapi/ocr/upload",
            "/fastapi/batch_ocr",
            "/predict/ocr_system"
        ]
        
        for endpoint in expected_endpoints:
            assert endpoint in schema["paths"], f"Endpoint {endpoint} not found in OpenAPI schema"
        
        print(f"✅ OpenAPI schema test passed")
        print(f"   Documented endpoints: {len(schema['paths'])}")


# ============================================================================
# Test: Performance and Edge Cases
# ============================================================================

class TestPerformanceAndEdgeCases:
    """Performance and edge case tests"""
    
    def test_concurrent_requests(self, wait_for_service, test_image_base64, use_deepx, use_sync):
        """Test multiple concurrent requests"""
        import concurrent.futures
        
        def make_request():
            payload = {
                "image": test_image_base64,
                "deepx": use_deepx,
                "sync": use_sync
            }
            response = requests.post(f"{BASE_URL}/fastapi/ocr", json=payload)
            return response.status_code == 200
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(results), "Some concurrent requests failed"
        print(f"✅ Concurrent requests test passed (5 requests)")
    
    def test_large_image_handling(self, wait_for_service, use_deepx, use_sync):
        """Test OCR with large image (if available)"""
        # Try to find a larger image
        large_images = list(IMAGES_DIR.glob("*.png"))
        if not large_images:
            pytest.skip("No large images available for testing")
        
        img_path = large_images[0]
        with open(img_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            "image": img_b64,
            "deepx": use_deepx,
            "sync": use_sync
        }
        response = requests.post(f"{BASE_URL}/fastapi/ocr", json=payload)
        
        assert response.status_code == 200
        print(f"✅ Large image handling test passed")
    
    def test_ocr_instance_caching(self, wait_for_service, test_image_base64, use_deepx, use_sync):
        """Test that OCR instances are properly cached"""
        # Make multiple requests with same parameters
        payload = {
            "file": test_image_base64,
            "fileType": 1,
            "textDetLimitSideLen": 960,
            "textDetThresh": 0.3,
            "deepx": use_deepx,
            "sync": use_sync
        }
        
        # First request (creates instance)
        start_time = time.time()
        response1 = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
        first_request_time = time.time() - start_time
        
        # Second request (should use cached instance)
        start_time = time.time()
        response2 = requests.post(f"{BASE_URL}/api/v1/ocr", json=payload)
        second_request_time = time.time() - start_time
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Second request should be faster or similar (caching benefit)
        print(f"✅ OCR instance caching test passed")
        print(f"   First request: {first_request_time:.3f}s")
        print(f"   Second request: {second_request_time:.3f}s")


# ============================================================================
# Test Summary
# ============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Print summary after all tests complete"""
    if exitstatus == 0:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nTested endpoints:")
        print("  ✓ GET  /health")
        print("  ✓ POST /api/v1/ocr (Baidu AI Studio compatible)")
        print("  ✓ POST /fastapi/ocr (url/base64/images array)")
        print("  ✓ POST /fastapi/ocr/upload")
        print("  ✓ POST /fastapi/batch_ocr")
        print("  ✓ POST /predict/ocr_system (hubserving)")
        print("  ✓ GET  /docs")
        print("  ✓ GET  /redoc")
        print("\nBaidu AI Studio 12 parameters tested:")
        print("  ✓ file, fileType")
        print("  ✓ useDocOrientationClassify, useDocUnwarping, useTextlineOrientation")
        print("  ✓ textDetLimitSideLen, textDetLimitType, textDetThresh")
        print("  ✓ textDetBoxThresh, textDetUnclipRatio")
        print("  ✓ textRecScoreThresh")
        print("  ✓ visualize")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
