#!/usr/bin/env bash
# Test script for FastAPI OCR service
#
# Usage:
#   ./test_ocr_with_curl_cmd                    # Use default port 8080
#   ./test_ocr_with_curl_cmd --port 8081        # Use custom port
#   ./test_ocr_with_curl_cmd -p 8081            # Use custom port (short form)

# Default port
SERVICE_PORT=8080

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            if [ -z "$2" ]; then
                echo "Error: --port option requires a port number"
                echo "Usage: ./test_ocr_with_curl_cmd --port <port_number>"
                exit 1
            fi
            SERVICE_PORT="$2"
            shift 2
            ;;
        *)
            # Backward compatibility: treat first argument as full URL
            if [[ $1 == http* ]]; then
                SERVICE_URL="$1"
                shift
            else
                echo "Error: Unknown option: $1"
                echo "Usage: ./test_ocr_with_curl_cmd [--port <port_number>]"
                exit 1
            fi
            ;;
    esac
done

# Set SERVICE_URL if not already set by backward compatibility
if [ -z "$SERVICE_URL" ]; then
    SERVICE_URL="http://localhost:$SERVICE_PORT"
fi

echo "Testing FastAPI OCR Service at: $SERVICE_URL"
echo "=========================================="

# Sample small base64 image (30x30 white image with black text)
IMAGE_BASE64="/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAeAB4DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlbaWmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooAKKKKACiiigD//2Q=="

# Test 1: Health check
echo ""
echo "Test 1: Health Check - GET /health"
echo "------------------------------------"
RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVICE_URL/health")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Health check passed (HTTP $HTTP_CODE)"
else
    echo "❌ Health check failed (HTTP $HTTP_CODE)"
fi

# Test 2: Baidu AI Studio Compatible API - POST /api/v1/ocr
echo ""
echo "Test 2: Baidu AI Studio Compatible OCR - POST /api/v1/ocr"
echo "----------------------------------------------------------"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL/api/v1/ocr" \
  -H "Content-Type: application/json" \
  -d "{
    \"file\": \"$IMAGE_BASE64\",
    \"fileType\": 1,
    \"useTextlineOrientation\": true,
    \"textDetLimitSideLen\": 960,
    \"textDetThresh\": 0.3,
    \"textDetBoxThresh\": 0.6,
    \"textRecScoreThresh\": 0.5,
    \"visualize\": false
  }")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool | head -30
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Baidu OCR API passed (HTTP $HTTP_CODE)"
else
    echo "❌ Baidu OCR API failed (HTTP $HTTP_CODE)"
fi

# Test 3: FastAPI OCR with URL - /fastapi/ocr
echo ""
echo "Test 3: FastAPI Single Image OCR with URL - POST /fastapi/ocr"
echo "--------------------------------------------------------------"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL/fastapi/ocr" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
  }')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool | head -20
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ FastAPI URL OCR passed (HTTP $HTTP_CODE)"
else
    echo "❌ FastAPI URL OCR failed (HTTP $HTTP_CODE)"
fi

# Test 4: FastAPI OCR with base64 - /fastapi/ocr
echo ""
echo "Test 4: FastAPI Single Image OCR with Base64 - POST /fastapi/ocr"
echo "-----------------------------------------------------------------"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL/fastapi/ocr" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\"}")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool | head -20
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Base64 OCR passed (HTTP $HTTP_CODE)"
else
    echo "❌ Base64 OCR failed (HTTP $HTTP_CODE)"
fi

# Test 4: OCR with images array - /ocr
echo ""
echo "Test 5: FastAPI OCR with images array - POST /fastapi/ocr"
echo "----------------------------------------------------------"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL/fastapi/ocr" \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMAGE_BASE64\"]}")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool | head -20
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Images array OCR passed (HTTP $HTTP_CODE)"
else
    echo "❌ Images array OCR failed (HTTP $HTTP_CODE)"
fi

# Test 6: Hubserving compatible endpoint - /predict/ocr_system
echo ""
echo "Test 6: Hubserving Compatible OCR - POST /predict/ocr_system"
echo "-------------------------------------------------------------"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL/predict/ocr_system" \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMAGE_BASE64\"]}")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool | head -20
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Hubserving format OCR passed (HTTP $HTTP_CODE)"
else
    echo "❌ Hubserving format OCR failed (HTTP $HTTP_CODE)"
fi

# Test 7: Batch OCR - /fastapi/batch_ocr
echo ""
echo "Test 7: FastAPI Batch OCR with multiple images - POST /fastapi/batch_ocr"
echo "-------------------------------------------------------------------------"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL/fastapi/batch_ocr" \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMAGE_BASE64\", \"$IMAGE_BASE64\"]}")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool | head -30
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Batch OCR passed (HTTP $HTTP_CODE)"
else
    echo "❌ Batch OCR failed (HTTP $HTTP_CODE)"
fi

# Test 8: File upload - /fastapi/ocr/upload
echo ""
echo "Test 8: FastAPI OCR with file upload - POST /fastapi/ocr/upload"
echo "---------------------------------------------------------------"
echo "Note: Requires a test image file. Skipping if not available."
if [ -f "test_image.jpg" ] || [ -f "../../../deepx/general_ocr_002.png" ]; then
    TEST_FILE="test_image.jpg"
    if [ ! -f "$TEST_FILE" ]; then
        TEST_FILE="../../../deepx/general_ocr_002.png"
    fi
    
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL/fastapi/ocr/upload" \
      -F "file=@$TEST_FILE")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    echo "$BODY" | python3 -m json.tool | head -20
    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "✅ File upload OCR passed (HTTP $HTTP_CODE)"
    else
        echo "❌ File upload OCR failed (HTTP $HTTP_CODE)"
    fi
else
    echo "⚠️  No test image file found. Skipping file upload test."
fi

# Test 9: API Documentation
echo ""
echo "Test 9: API Documentation Endpoints"
echo "------------------------------------"
echo "Swagger UI: $SERVICE_URL/docs"
echo "ReDoc: $SERVICE_URL/redoc"
RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVICE_URL/docs")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ API documentation accessible (HTTP $HTTP_CODE)"
else
    echo "❌ API documentation failed (HTTP $HTTP_CODE)"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "All endpoint tests completed!"
echo ""
echo "Endpoints tested:"
echo "  ✓ GET  /health"
echo "  ✓ POST /api/v1/ocr (Baidu AI Studio compatible)"
echo "  ✓ POST /fastapi/ocr (url)"
echo "  ✓ POST /fastapi/ocr (image base64)"
echo "  ✓ POST /fastapi/ocr (images array)"
echo "  ✓ POST /predict/ocr_system (hubserving format)"
echo "  ✓ POST /fastapi/batch_ocr"
echo "  ✓ POST /fastapi/ocr/upload (if test image available)"
echo "  ✓ GET  /docs (API documentation)"
echo ""
echo "For interactive API testing, visit:"
echo "  $SERVICE_URL/docs"
