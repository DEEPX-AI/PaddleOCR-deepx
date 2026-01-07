# PaddleOCR Simple OCR Service with FastAPI
# Based on PaddlePaddle 3.x and PaddleOCR 3.3.2 with PP-OCRv5
FROM python:3.10.16-slim

ARG SETUP_DEEPX="false"
ARG DX_RT_VERSION="3.1.0"

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
        $(if [ "${SETUP_DEEPX}" = "true" ]; then echo "build-essential git sudo"; fi) \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle and dependencies from requirements file
# CPU/GPU version is determined by the requirements file used
ARG DEVICE_TYPE="cpu"
COPY requirements.txt /tmp/requirements.txt
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN if [ "${DEVICE_TYPE}" = "gpu" ]; then \
        python -m pip install --no-cache-dir -r /tmp/requirements-gpu.txt; \
    else \
        python -m pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi \
    && rm /tmp/requirements.txt /tmp/requirements-gpu.txt

# Install dx_rt if DEEPX is enabled
ARG DX_RT_VERSION="3.1.0"
RUN if [ "${SETUP_DEEPX}" = "true" ]; then \
        echo "=== Installing dx_rt v${DX_RT_VERSION} ===" && \
        cd /tmp && \
        wget --no-check-certificate -O dx_rt.tar.gz "https://github.com/DEEPX-AI/dx_rt/archive/refs/tags/v${DX_RT_VERSION}.tar.gz" && \
        mkdir -p dx_rt && \
        tar -xzf dx_rt.tar.gz -C dx_rt --strip-components=1 && \
        cd dx_rt && \
        chmod +x install.sh build.sh && \
        ./install.sh --all && \
        ./build.sh && \
        echo "/usr/local/lib" > /etc/ld.so.conf.d/dxrt.conf && \
        ldconfig && \
        cd / && \
        rm -rf /tmp/dx_rt /tmp/dx_rt.tar.gz && \
        echo "dx_rt installation completed"; \
    fi

# Set LD_LIBRARY_PATH for dx_rt shared libraries
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

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
        echo "Downloading CPU models for preprocessing (doc orientation, unwarping, textline orientation)..."; \
        wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar \
        && tar -xf PP-LCNet_x1_0_doc_ori_infer.tar && mv PP-LCNet_x1_0_doc_ori_infer PP-LCNet_x1_0_doc_ori \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar \
        && tar -xf UVDoc_infer.tar && mv UVDoc_infer UVDoc \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar \
        && tar -xf PP-LCNet_x1_0_textline_ori_infer.tar && mv PP-LCNet_x1_0_textline_ori_infer PP-LCNet_x1_0_textline_ori; \
        if [ "${USE_MOBILE}" = "true" ]; then \
            echo "Downloading mobile det/rec models..."; \
            wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar \
            && tar -xf PP-OCRv5_mobile_det_infer.tar && mv PP-OCRv5_mobile_det_infer PP-OCRv5_mobile_det \
            && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar \
            && tar -xf PP-OCRv5_mobile_rec_infer.tar && mv PP-OCRv5_mobile_rec_infer PP-OCRv5_mobile_rec; \
        fi; \
        echo "Downloading server det/rec models (always for fallback)..."; \
        wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar \
        && tar -xf PP-OCRv5_server_det_infer.tar && mv PP-OCRv5_server_det_infer PP-OCRv5_server_det \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar \
        && tar -xf PP-OCRv5_server_rec_infer.tar && mv PP-OCRv5_server_rec_infer PP-OCRv5_server_rec \
        && rm -f *.tar; \
    fi

# Copy service script and DEEPX environment configuration
COPY --chown=paddleocr:paddleocr ocr_service.py /app/
COPY --chown=paddleocr:paddleocr .env.deepx /app/.env.deepx

# Load DEEPX NPU environment variables if SETUP_DEEPX is enabled
RUN if [ "${SETUP_DEEPX}" = "true" ]; then \
        echo '#!/bin/bash' > /app/start.sh && \
        echo '# Load DEEPX NPU environment variables' >> /app/start.sh && \
        echo 'echo "🔧 Applying DEEPX NPU environment settings..."' >> /app/start.sh && \
        echo '' >> /app/start.sh && \
        echo '# Load defaults from .env.deepx if environment variables are not already set' >> /app/start.sh && \
        echo 'if [ -f /app/.env.deepx ]; then' >> /app/start.sh && \
        echo '    while IFS="=" read -r key value; do' >> /app/start.sh && \
        echo '        # Skip comments and empty lines' >> /app/start.sh && \
        echo '        if [[ ! "$key" =~ ^[[:space:]]*# ]] && [[ -n "$key" ]]; then' >> /app/start.sh && \
        echo '            # Only set if not already defined (allows runtime override)' >> /app/start.sh && \
        echo '            if [ -z "${!key}" ]; then' >> /app/start.sh && \
        echo '                export "$key=$value"' >> /app/start.sh && \
        echo '            fi' >> /app/start.sh && \
        echo '        fi' >> /app/start.sh && \
        echo '    done < /app/.env.deepx' >> /app/start.sh && \
        echo 'fi' >> /app/start.sh && \
        echo '' >> /app/start.sh && \
        echo '# Display current settings' >> /app/start.sh && \
        echo 'echo "CUSTOM_INTER_OP_THREADS_COUNT=${CUSTOM_INTER_OP_THREADS_COUNT:-not set}"' >> /app/start.sh && \
        echo 'echo "CUSTOM_INTRA_OP_THREADS_COUNT=${CUSTOM_INTRA_OP_THREADS_COUNT:-not set}"' >> /app/start.sh && \
        echo 'echo "DXRT_DYNAMIC_CPU_THREAD=${DXRT_DYNAMIC_CPU_THREAD:-not set}"' >> /app/start.sh && \
        echo 'echo "DXRT_TASK_MAX_LOAD=${DXRT_TASK_MAX_LOAD:-not set}"' >> /app/start.sh && \
        echo 'echo "NFH_INPUT_WORKER_THREADS=${NFH_INPUT_WORKER_THREADS:-not set}"' >> /app/start.sh && \
        echo 'echo "NFH_OUTPUT_WORKER_THREADS=${NFH_OUTPUT_WORKER_THREADS:-not set}"' >> /app/start.sh && \
        echo 'echo ""' >> /app/start.sh && \
        echo '' >> /app/start.sh && \
        echo 'exec python ocr_service.py' >> /app/start.sh && \
        chmod +x /app/start.sh; \
    else \
        echo '#!/bin/bash' > /app/start.sh && \
        echo 'exec python ocr_service.py' >> /app/start.sh && \
        chmod +x /app/start.sh; \
    fi

# Expose port for the service
EXPOSE 8080

# Run OCR service with start script
CMD ["/app/start.sh"]
