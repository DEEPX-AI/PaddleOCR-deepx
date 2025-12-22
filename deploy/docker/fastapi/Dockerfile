# PaddleOCR Simple OCR Service with FastAPI
# Based on PaddlePaddle 3.x and PaddleOCR 3.3.2 with PP-OCRv5
FROM python:3.10-slim

ARG SETUP_DEEPX="false"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
ARG SETUP_DEEPX="false"
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        wget \
        $(if [ "${SETUP_DEEPX}" = "true" ]; then echo "build-essential git"; fi) \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle (choose CPU or GPU version)
ARG DEVICE_TYPE="cpu"
RUN if [ "${DEVICE_TYPE}" = "gpu" ]; then \
        python -m pip install paddlepaddle-gpu==3.0.0; \
    else \
        python -m pip install paddlepaddle==3.0.0; \
    fi

# Install PaddleOCR and FastAPI
ARG PADDLEOCR_VERSION="3.3.2"
ARG SETUP_DEEPX="false"
RUN python -m pip install paddleocr==${PADDLEOCR_VERSION} \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    pillow \
    opencv-python-headless

# Install DEEPX NPU dependencies if enabled
RUN if [ "${SETUP_DEEPX}" = "true" ]; then \
        python -m pip install \
            torch==2.3.0 \
            torchvision==0.18.0 \
            torchaudio==2.3.0 \
            onnxruntime==1.18.0 \
            opencv-contrib-python==4.7.0.72 \
            scikit-image \
            imgaug \
            shapely \
            pyclipper \
            jiwer; \
    fi

# Create working directory
WORKDIR /app

# Create user
RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr \
    && chown -R paddleocr:paddleocr /app

USER paddleocr

# Download PP-OCRv5 models (optional, can be done at runtime)
ARG DOWNLOAD_MODELS="true"
ARG USE_MOBILE="true"
RUN if [ "${DOWNLOAD_MODELS}" = "true" ]; then \
        mkdir -p /home/paddleocr/.paddlex/official_models && cd /home/paddleocr/.paddlex/official_models; \
        if [ "${USE_MOBILE}" = "true" ]; then \
            echo "Downloading mobile models..."; \
            wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar \
            && tar -xf PP-OCRv5_mobile_det_infer.tar && mv PP-OCRv5_mobile_det_infer PP-OCRv5_mobile_det \
            && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar \
            && tar -xf PP-OCRv5_mobile_rec_infer.tar && mv PP-OCRv5_mobile_rec_infer PP-OCRv5_mobile_rec; \
        else \
            echo "Downloading server models..."; \
            wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar \
            && tar -xf PP-OCRv5_server_det_infer.tar && mv PP-OCRv5_server_det_infer PP-OCRv5_server_det \
            && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar \
            && tar -xf PP-OCRv5_server_rec_infer.tar && mv PP-OCRv5_server_rec_infer PP-OCRv5_server_rec; \
        fi; \
        wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar \
        && tar -xf PP-LCNet_x1_0_doc_ori_infer.tar && mv PP-LCNet_x1_0_doc_ori_infer PP-LCNet_x1_0_doc_ori \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar \
        && tar -xf UVDoc_infer.tar && mv UVDoc_infer UVDoc \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar \
        && tar -xf PP-LCNet_x1_0_textline_ori_infer.tar && mv PP-LCNet_x1_0_textline_ori_infer PP-LCNet_x1_0_textline_ori \
        && rm -f *.tar; \
    fi

# Copy service script
COPY --chown=paddleocr:paddleocr ocr_service.py /app/

# Expose port for the service
EXPOSE 8080

# Run OCR service with uvicorn
CMD ["python", "ocr_service.py"]
