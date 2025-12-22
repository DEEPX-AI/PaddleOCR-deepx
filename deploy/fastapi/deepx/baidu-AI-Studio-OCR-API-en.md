# PP-OCRv5 API Documentation

## PP-OCRv5 Service Deployment & API Usage Example

**PaddleOCR Open-Source Project**: [GitHub Repository](https://github.com/PaddlePaddle/PaddleOCR)  
The code here is completely consistent with the open-source model.

## 1. Introduction to PP-OCRv5 Production Line

**OCR (Optical Character Recognition)** is a technology that converts text content in images into editable text. It is widely used in:
- Document digitization
- Information extraction
- Data processing
- And other scenarios

OCR can recognize various types of text, including printed and handwritten characters, helping users efficiently obtain key information from images.

### PP-OCRv5 Overview

**PP-OCRv5** is the latest generation text recognition solution in the PP-OCR series, designed for multi-scenario and multi-text-type recognition tasks. 

Compared to previous versions, PP-OCRv5 has achieved comprehensive upgrades in text type support and adaptability to various application scenarios. This solution can not only return the coordinates of text lines but also output the corresponding text content and its confidence score, effectively improving the accuracy and practicality of text detection and recognition.

### Key Features

- **Support for five major text types**: 
  - Simplified Chinese
  - Chinese Pinyin
  - Traditional Chinese
  - English
  - Japanese

- **Powerful multi-scenario adaptability**: 
  - Complex Chinese-English handwritten text
  - Vertical text
  - Rare characters
  - Other challenging scenarios

- **Performance Improvement**: 
  - On internal multi-scenario complex evaluation sets, PP-OCRv5 achieves a **13 percentage point improvement** in end-to-end recognition accuracy compared to the previous generation, PP-OCRv4.

### Workflow

The overall workflow of the PP-OCRv5 production line is as follows:

```
Input Image/PDF → Document Preprocessing → Text Detection → Text Recognition → Output Results
```


---

## 2. API Description

This applies to scenarios where parsing tasks are created via API. Users must first apply for a Token.

```python
# Please make sure the requests library is installed
# pip install requests
import os
import base64
import requests

# Please visit https://aistudio.baidu.com/paddleocr/task to obtain the API_URL and TOKEN
API_URL = "<your url>"
TOKEN = "<access token>"

file_path = "<local file path>"
input_filename = os.path.splitext(os.path.basename(file_path))[0]

# Read and encode file
with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

# Set headers
headers = {
    "Authorization": f"token {TOKEN}",
    "Content-Type": "application/json"
}

# Required parameters
required_payload = {
    "file": file_data,
    "fileType": 0,  # For PDF documents, set to 0; for images, set to 1
}

# Optional parameters
optional_payload = {
    "useDocOrientationClassify": False,  # Document orientation correction
    "useDocUnwarping": False,             # Image distortion correction
    "useTextlineOrientation": False,      # Text line orientation correction
}

# Merge payloads
payload = {**required_payload, **optional_payload}

# Send request
response = requests.post(API_URL, json=payload, headers=headers)

# Check response
assert response.status_code == 200
result = response.json()["result"]

# Save results
os.makedirs("output", exist_ok=True)

for i, res in enumerate(result["ocrResults"]):
    print(res["prunedResult"])
    
    # Download and save OCR result image
    image_url = res["ocrImage"]
    img_response = requests.get(image_url)
    
    if img_response.status_code == 200:
        filename = f"output/{input_filename}_{i}.jpg"
        with open(filename, "wb") as f:
            f.write(img_response.content)
        print(f"Image saved to: {filename}")
    else:
        print(f"Failed to download image, status code: {img_response.status_code}")
```

---

## API Operations

### Request Method
- **HTTP Method**: `POST`
- **Request Body**: JSON object
- **Response Body**: JSON object

### Success Response (Status Code: 200)

| Name | Type | Description |
|------|------|-------------|
| `logId` | string | The UUID of the request |
| `errorCode` | integer | Error code. Fixed as `0` |
| `errorMsg` | string | Error description. Fixed as `"Success"` |
| `result` | object | Operation result |

### Error Response

| Name | Type | Description |
|------|------|-------------|
| `logId` | string | The UUID of the request |
| `errorCode` | integer | Error code. Same as the response status code |
| `errorMsg` | string | Error description |

### Main Endpoint

#### `POST /ocr`
Obtain OCR results for images.

**Description**: Infer OCR results from input image or PDF file.

---

### Request Parameters

| Name | Parameter | Type | Required | Description |
|------|-----------|------|----------|-------------|
| **Input File** | `file` | string | ✅ Yes | The URL of an image or PDF file accessible by the server, or the Base64-encoded content of such a file.<br/><br/>**Note**: By default, for PDF files with more than 10 pages, only the first 10 pages will be processed.<br/><br/>To lift the page number limit, add the following configuration:<br/>`Serving:`<br/>&nbsp;&nbsp;`extra:`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`max_num_input_imgs: null` |
| **File Type** | `fileType` | integer \| null | ❌ No | File type:<br/>• `0` = PDF files<br/>• `1` = Image files<br/><br/>If not present, the file type will be inferred from the URL. |
| **Document Orientation** | `useDocOrientationClassify` | boolean \| null | ❌ No | Whether to use document orientation classification module.<br/><br/>When enabled, automatically identifies and corrects images rotated by 0°, 90°, 180°, or 270°. |
| **Image Distortion Correction** | `useDocUnwarping` | boolean \| null | ❌ No | Whether to use text image correction module.<br/><br/>When enabled, automatically corrects distorted images (wrinkled or skewed). |
| **Text Line Orientation** | `useTextlineOrientation` | boolean \| null | ❌ No | Whether to use text line orientation classification module.<br/><br/>When enabled, automatically identifies and corrects text lines at 0° and 180°. |
| **Side Length Limit** | `textDetLimitSideLen` | integer \| null | ❌ No | Image side length limit for text detection.<br/><br/>Any integer > 0. Default: `64` |
| **Limit Type** | `textDetLimitType` | string \| null | ❌ No | Type of image side length limit:<br/>• `min` = ensure shortest side ≥ `textDetLimitSideLen`<br/>• `max` = ensure longest side ≤ `textDetLimitSideLen`<br/><br/>Default: `min` |
| **Detection Pixel Threshold** | `textDetThresh` | number \| null | ❌ No | Text detection pixel threshold.<br/><br/>Only pixels with score > this value are considered text pixels.<br/>Default: `0.3` |
| **Detection Box Threshold** | `textDetBoxThresh` | number \| null | ❌ No | Text detection box threshold.<br/><br/>If average score of pixels in box > this value, area is considered text region.<br/>Default: `0.6` |
| **Expansion Ratio** | `textDetUnclipRatio` | number \| null | ❌ No | Text detection expansion ratio.<br/><br/>Used to expand text regions. Larger value = larger expansion.<br/>Default: `1.5` |
| **Recognition Threshold** | `textRecScoreThresh` | number \| null | ❌ No | Text recognition score threshold.<br/><br/>Only results with score > this value are kept.<br/>Default: `0.0` (no threshold) |
| **Visualization** | `visualize` | boolean \| null | ❌ No | Return visualized result images and intermediate processing images.<br/><br/>• `true` = return images<br/>• `false` = do not return images<br/>• `null` or not provided = follows `Serving.visualize` in config<br/><br/>⚠️ Enabling this increases response time.<br/><br/>Config example:<br/>`Serving:`<br/>&nbsp;&nbsp;`visualize: False` |

### Response Body (Success)

The `result` field contains:

| Name | Type | Description |
|------|------|-------------|
| `ocrResults` | array | OCR results array.<br/>• Length = 1 for image input<br/>• Length = number of pages for PDF input<br/>Each element corresponds to one page result. |
| `dataInfo` | object | Input data information |

### OCR Results Object

Each element in `ocrResults` contains:

| Name | Type | Description |
|------|------|-------------|
| `prunedResult` | object | Simplified OCR result (without `input_path` and `page_index` fields) |
| `ocrImage` | string \| null | OCR result image showing detected text regions (JPEG, Base64 encoded) |
| `docPreprocessingImage` | string \| null | Preprocessing visualization image (JPEG, Base64 encoded) |
| `inputImage` | string \| null | Original input image (JPEG, Base64 encoded) |
  visualize: False
---

## 5. Error Code Description

| Error Code | Description | Suggested Solution |
|------------|-------------|-------------------|
| **403** | Invalid Token | • Check if the Token is correct<br/>• Verify if the URL matches the Token |
| **429** | Exceeded daily maximum page limit | • Use another model<br/>• Try again later (daily limit: 3000 pages) |
| **500** | Parameter error | • Ensure parameter types are correct<br/>• Verify `fileType` is valid (0 or 1) |
| **504** | Gateway timeout | • Try again later<br/>• Consider reducing file size or page count |

---

## Support

If you encounter any issues during use, please feel free to submit feedback in the [PaddleOCR Issues](https://github.com/PaddlePaddle/PaddleOCR/issues) section.

---

## Additional Resources

- **API Access**: [https://aistudio.baidu.com/paddleocr/task](https://aistudio.baidu.com/paddleocr/task)
- **Documentation**: [https://ai.baidu.com/ai-doc/AISTUDIO/7mfz6dgx9](https://ai.baidu.com/ai-doc/AISTUDIO/7mfz6dgx9)
- **GitHub**: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
Name	Type	Description
prunedResult	object	A simplified version of the res field in the JSON result produced by the production object's predict method, with the input_path and page_index fields removed.
ocrImage	string | null	The OCR result image, indicating detected text regions. Image is in JPEG format, Base64 encoded.
docPreprocessingImage	string | null	Visualization result image. Image is in JPEG format, Base64 encoded.
inputImage	string | null	Input image. Image is in JPEG format, Base64 encoded.
5. Error Code Description
Error Code	Description	Suggested Solution
403	Invalid Token	Check whether the Token is correct, or if the URL matches the Token
429	Exceeded daily maximum page limit	Please use another model or try again later
500	Parameter error	Ensure parameter types and fileType are correct
504	Gateway timeout	Please try again later
Note: If you encounter any issues during use, please feel free to submit feedback in the issue section.


