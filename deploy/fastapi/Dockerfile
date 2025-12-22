# PaddleOCR Simple OCR Service with FastAPI
# Based on PaddlePaddle 3.x and PaddleOCR 3.3.2 with PP-OCRv5
FROM python:3.10.16-slim

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
        curl \
        $(if [ "${SETUP_DEEPX}" = "true" ]; then echo "build-essential git"; fi) \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle (choose CPU or GPU version)
ARG DEVICE_TYPE="cpu"
RUN if [ "${DEVICE_TYPE}" = "gpu" ]; then \
        python -m pip install "paddlepaddle-gpu==3.0.0" "numpy<2.0.0"; \
    else \
        python -m pip install "paddlepaddle==3.0.0" "numpy<2.0.0"; \
    fi

# Copy requirements.txt and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
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
            jiwer;\
        python -m pip install "numpy<2.0.0"; \
    fi

# Create working directory
WORKDIR /app

# Create user
RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr \
    && chown -R paddleocr:paddleocr /app

# Copy DEEPX engine if enabled
ARG SETUP_DEEPX="false"
RUN if [ "${SETUP_DEEPX}" = "true" ]; then \
        mkdir -p /app/deepx; \
    fi

COPY deepx/ /app/deepx/

# Download DEEPX models if enabled (run as root before USER paddleocr)
RUN if [ "${SETUP_DEEPX}" = "true" ]; then \
        echo "=== DEEPX NPU Models Setup ===" && \
        DEEPX_MODELS_DIR="/app/deepx/engine/model_files" && \
        DXNN_DIR="$DEEPX_MODELS_DIR/server" && \
        DXNN_MOBILE_DIR="$DEEPX_MODELS_DIR/mobile" && \
        NEED_SETUP=0 && \
        if [ ! -d "$DXNN_DIR" ]; then \
            echo "Missing model folder: $DXNN_DIR" && \
            NEED_SETUP=1; \
        fi && \
        if [ ! -d "$DXNN_MOBILE_DIR" ]; then \
            echo "Missing model folder: $DXNN_MOBILE_DIR" && \
            NEED_SETUP=1; \
        fi && \
        if [ $NEED_SETUP -eq 1 ]; then \
            if [ -f "/app/deepx/setup.sh" ]; then \
                echo "Running setup.sh to fetch DEEPX models..." && \
                cd /app/deepx && \
                mkdir -p download .temp/download && \
                rm -rf .temp/download/* && \
                bash setup.sh --dest="$DEEPX_MODELS_DIR" && \
                echo "DEEPX models setup completed" && \
                if [ -d "$DXNN_DIR" ] && [ -d "$DXNN_MOBILE_DIR" ]; then \
                    echo "Models verified: server and mobile"; \
                else \
                    echo "Error: Models verification failed" && exit 1; \
                fi; \
            else \
                echo "Error: setup.sh not found at /app/deepx/setup.sh" && \
                echo "Please ensure deepx folder contains setup.sh" && \
                exit 1; \
            fi; \
        else \
            echo "DEEPX model folders already present"; \
        fi; \
    fi

# Set ownership after model download
RUN chown -R paddleocr:paddleocr /app/deepx

# Set DEEPX_PATH environment variable
ENV DEEPX_PATH=/app/deepx

# Set FONT_PATH environment variable
ENV FONT_PATH=/app/deepx/engine/fonts

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
