
# PP-OCR v5 API 제공 관련 기술 검토

## PP-OCR v5과 바이두에서 제공한 Baidu AI Studio의 OCR API 차이점

### 1. PP-OCR v5와 Baidu OCR API의 관계이 둘은 **엔진과 완성된 자동차의 대여 서비스**의 관계

- **PP-OCR v5 (PaddleOCR GitHub):**
  - **설명:** Baidu의 딥러닝 프레임워크인 PaddlePaddle을 기반으로 만든 **오픈소스 OCR 모델**의 최신 버전(v5)
  - **역할:** 개발자가 직접 다운로드하여 학습시키거나 서버에 설치해 사용할 수 있는 **오픈 소스 OCR 엔진**입니다.


- **Baidu AI Studio / Baidu Cloud OCR API:**
  - **설명:** Baidu가 PP-OCR v5(혹은 그 이상의 내부 전용 고성능 모델)를 서버에 올려놓고, 사용자가 API 호출만으로 쉽게 사용할 수 있도록 제공하는 **유료/무료 클라우드 서비스(SaaS)**입니다.
  - **관계:** Baidu API는 PP-OCR 기술을 사용하여 만들어졌습니다. 즉, GitHub에 있는 모델이 Baidu API의 **핵심 두뇌** 역할을 합니다.


### 2. OCR API를 실행하는 서버 소스코드가 GitHub에 있는가?

#### A. 제공하는 것 (O): 
"OCR 기능을 수행하는 API 서버 코드" PaddleOCR GitHub 리포지토리에는 PP-OCR v5 모델을 로딩하여 **Rest API 형태로 서비스할 수 있는 배포용 코드**가 포함되어 있습니다. 이를 사용하면 Baidu API와 유사한 기능을 하는 개인용 OCR 서버를 구축할 수 있습니다.

- **관련 경로 (GitHub 내 `deploy` 폴더, v3.3.2 기준):**
  - `deploy/hubserving`: **PaddleHub**를 이용한 가장 간편한 API 서버 구축 방식 (초보자 추천)
- (구버전 한정) `deploy/pdserving`: **Paddle Serving**을 이용한 고성능 배포 방식으로 v2.10 까지 존재했으나, v3.0 이후 버전에서는 제거됨


#### B. 제공하지 않는 것 (X): "Baidu 상용 플랫폼의 전체 백엔드 코드"Baidu AI Studio나 Baidu Cloud에서 실제로 돌아가는 **상용 서비스 자체의 소스코드**는 공개되어 있지 않습니다.

### 요약 (사용자 관점 차이)
1. **직접 서버를 구축하고 싶다면:**
PaddleOCR v5 GitHub의 `deploy` 폴더에 있는 코드를 사용하면 됩니다. 이것이 "OCR API를 실행하는 서버 소스코드"에 해당하며, Baidu API와 기능적으로 거의 동일한 결과를 냅니다.

2. **구축 없이 바로 사용하고 싶다면:**
Baidu AI Studio나 Baidu Cloud의 API 서비스를 계약하여 호출만 하면 됩니다.

---


### 3. 바이두에서 제공한 Baidu AI Studio의 OCR API의 인터페이스와 PaddleOCR GitHub의 OCR API 인터페이스 비교 분석 

PaddleOCR Github에서는 HubServing과 Paddle Serving 두 가지 방식으로 OCR API 서버를 구축할 수 있는 코드를 제공하고 있음을 확인하였음.

Baidu AI Studio의 OCR API와 동일한 이미지 URL을 사용하여 결과를 비교 분석시도 했으나, **아래 이슈사항으로 인해 비교 분석이 불가**하였음.


**<이슈사항>**

- Paddle Serving(`deploy/pdserving`)의 경우 최신버전(v3.3.2)을 지원하지 않고 v2.10.0을 기준으로 삭제되어 구축 불가.

- Hub Serving(`deploy/HubServing`)의 경우 최신버전(v3.3.2)을 기준으로 가이와 코드가 작성은 되어있으나 현재 PP-OCR v5 기준으로 `deploy/hubserving`이 정상적으로 동작하지 않고, 관리되고 있지 않음.

---

### 4. Action Item 및 대안

1. CS 및 영업팀: PaddleOCR GitHub의 API Serving 프레임워크 `deploy/hubserving`와 `deploy/pdserving`가 현재 최신버전으로 구동되지 않음을 알리고, 대안으로 원하는 방향을 문의 필요

2. ATD팀: PaddleOCR GitHub의 `serving` 방식을 사용하지 않고, 대안으로 PP-OCR v5 엔진만 활용하여, Python Flask 기반의 간단한 OCR REST API 서버를 구축을 병행 진행

- onnxruntime 기반의 경량화된 OCR API 서버 구축 (진행중)
- dx_rt python sdk 연동 (TODO)
- CS 및 영업팀 확인 사항에 따라 대응


---

### 5. HubServing(hubserving) VS. Paddle Serving(pdserving) 비교 분석

| 구분 | 1. HubServing | 2. PdServing (Paddle Serving) |
| --- | --- | --- |
| **기반 기술** | Python Wrapper | **C++ / Pipeline Engine** |
| **모델 포맷** | 동적/정적 모델 둘 다 가능 | **정적 모델(Static Graph) 필수** |
| **속도** | 보통 (Python 오버헤드 존재) | **매우 빠름 (C++ 최적화)** |
| **난이도** | 하 | 상 |
| **비고** | v3.3.2 최신버전 지원 | v3.0.0 이후 지원 X (v2.10.0 까지만 지원) |


#### 상세 분석

**1. HubServing**

* **구조:** PaddleHub Serving 위에 PaddleOCR의 파이썬 추론 코드를 래핑한 방식입니다.
* **로직:** 내부적으로 PaddleOCR의 파이썬 라이브러리인 `tools/infer/predict_system.py` 혹은 `PaddleOCR()` 클래스를 **그대로 import해서 사용**합니다.
* **특징:** `deploy/hubserving/ocr_system` 등 모듈 디렉터리에서 `module.py`, `config.json`을 정의하고, `hub install .` 후 `hub serving start -m ocr_system` 식으로 REST API 서버를 띄웁니다.


**2. PdServing (Paddle Serving)**

* **구조:** 이것은 단순한 웹 래퍼가 아니라, **고성능 전용 서빙 프레임워크**입니다. Client-Server 구조로 되어 있습니다.
* **로직:** `tools/infer`에 있는 파이썬 추론 스크립트를 직접 쓰는 대신, 저장된 **inference model**을 `paddle_serving_client.convert`로 변환하여 `ppocr_det_v3_serving`, `ppocr_rec_v3_serving` 같은 **Serving 전용 모델 디렉터리**를 생성합니다.
* 서버 쪽에서는 `python3 web_service.py --config=config.yml` 형태의 **Python Pipeline 서비스**와, `python3 -m paddle_serving_server.serve --model ppocr_det_v3_serving ...` 형태의 **C++ Serving** 두 가지 방식을 제공합니다.
* C++ Serving의 경우 `general_detection_op.cpp` 등 서버 측 전/후처리 코드를 작성·컴파일해 여러 모델을 하나의 파이프라인으로 묶고, 클라이언트에서는 `ocr_cpp_client.py` 같은 스크립트로 요청을 보냅니다.


* **특징:** Python Pipeline 방식은 2.10.0 기준 `deploy/pdserving`에 포함된 `web_service.py`, `pipeline_http_client.py`로 손쉽게 사용할 수 있고, C++ Serving 방식은 대량 트래픽과 낮은 지연 시간을 위해 RPC, 배치 처리(Batching) 등을 최적화한 **별도의 고성능 실행 엔진**을 제공합니다.


---

## HubServing 이용한 OCR API 서버 실행

### 분석 결과

#### local 환경 설치 구축 시도 결과
https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/hubserving/readme_en.md

현재 PaddleOCR GitHub의 `deploy/hubserving/readme_en.md` 파일에는 HubServing을 이용한 OCR API 서버 실행 방법이 상세히 설명되어 있으나, 절차대로 수행하였을때 paddleocr, paddlehub, paddlenlp, aistudio-sdk 등의 의존성 문제로 정상적으로 실행되지 않음.

PP-OCR v3 기준으로 작성된 것으로 보임. PP-OCR v5 기준으로 관리되고 있지 않음.

#### docker 환경 설치 구축 시도 결과

https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/docker/hubserving/README.md
PP-OCR V2 기준으로 작성되어 있고, 최신 버전으로 관리되고 있지 않음.




