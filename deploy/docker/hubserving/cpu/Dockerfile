# Base Image: PaddlePaddle 2.6+ 권장 (v5 모델 호환성 고려)
# 주의: 최신 v5 모델은 PaddlePaddle 버전을 탐(2.6 이상 권장).
# FROM registry.baidubce.com/paddlepaddle/paddle:2.6.0
FROM paddlepaddle/paddle:3.2.2
# (CPU만 쓴다면: FROM registry.baidubce.com/paddlepaddle/paddle:2.6.0-cpu)

# 필수 라이브러리 설치
RUN pip3 install --upgrade pip
RUN pip3 install paddlehub --upgrade

# [FIX 1] Dependency 충돌 해결
# 최신 paddlenlp(2.8.x)는 Paddle 2.6 환경에서 aistudio-sdk 오류를 일으킵니다.
# Paddle Framework 버전(2.6.0)에 맞춰 nlp 라이브러리를 2.6.1로 다운그레이드합니다.
RUN pip3 install paddlenlp==2.6.1 aistudio-sdk --upgrade

RUN pip3 install paddleocr 

# [FIX 2] C++ 심볼 에러 해결: paddle2onnx를 1.2.11으로 다운그레이드
RUN pip3 install "paddle2onnx==1.2.11" --force-reinstall

# (paddleocr 패키지를 설치하면 git clone 없이도 내부 로직 사용 가능하지만, 
# hubserving은 git repo의 구조를 따르므로 clone 유지)

# for setting ssl issue on intranet
# only used 'USE_INTRANET=true'
ARG USE_INTRANET=true
ARG CA_FILE_NAME="intranet_CA_SSL.crt"
COPY ${CA_FILE_NAME} /usr/local/share/ca-certificates/
RUN if [ "$USE_INTRANET" = "true" ]; then \
      git config --global http.sslVerify false && \
      update-ca-certificates ; \
    else \
      echo "Skipping intranet SSL setup"; \
    fi
# RUN git config --global http.sslVerify false

RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git /PaddleOCR

WORKDIR /PaddleOCR
RUN pip3 install -r requirements.txt

# 모델 저장소 생성
RUN mkdir -p /PaddleOCR/inference/

# ---------------------------------------------------------
# [중요] PP-OCR v5 구성을 위한 핵심 3개 모델 다운로드 및 배치
# ---------------------------------------------------------

# 1. Detection Model (PP-OCR v5 Server)
# (주의: 모바일용 경량 모델을 원하면 _mobile_로 변경)
ADD https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar /PaddleOCR/inference/
RUN tar -xf /PaddleOCR/inference/PP-OCRv5_mobile_det_infer.tar -C /PaddleOCR/inference/

# 2. Recognition Model (PP-OCR v5 Server)
ADD https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar /PaddleOCR/inference/
RUN tar -xf /PaddleOCR/inference/PP-OCRv5_mobile_rec_infer.tar -C /PaddleOCR/inference/

# 3. Classification Model (PP-OCR v2 Mobile - Cls는 v5가 따로 없고 이거 씀)
ADD https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar /PaddleOCR/inference/
RUN tar -xf /PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer.tar -C /PaddleOCR/inference/

# ---------------------------------------------------------
# Config 파일(params.py) 수정
# HubServing이 위에서 받은 v5 모델을 바라보도록 설정 파일 덮어쓰기
# (sed 명령어로 파일 경로 바꿔치기)
# ---------------------------------------------------------
RUN sed -i 's|PP-OCRv3_mobile_det_infer|PP-OCRv5_mobile_det_infer|g' deploy/hubserving/ocr_system/params.py
RUN sed -i 's|ch_PP-OCRv3_rec_infer|PP-OCRv5_mobile_rec_infer|g' deploy/hubserving/ocr_system/params.py
# cls 모델은 이름(ch_ppocr_mobile_v2.0_cls_infer)이 같아 수정 불필요
# 보통 params.py에 기본값이 mobile 버전이므로 그대로 둬도 됨.

EXPOSE 8868

# 실행 (config.json 설정이 필요할 수 있음)
CMD ["/bin/bash","-c","hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system"]


### known issue 1: import error
# ```
# (venv) dhyang@dhyang-linux-dev-pc:/data/home/dhyang/git/github/PaddleOCR/deploy/docker/hubserving/cpu$ docker logs -f paddle_ocr 
# Traceback (most recent call last):
#   File "/usr/local/bin/hub", line 3, in <module>
#     from paddlehub.commands.utils import execute
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/__init__.py", line 31, in <module>
#     from paddlehub import datasets
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/datasets/__init__.py", line 16, in <module>
#     from paddlehub.datasets.chnsenticorp import ChnSentiCorp
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/datasets/chnsenticorp.py", line 19, in <module>
#     from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/datasets/base_nlp_dataset.py", line 21, in <module>
#     import paddlenlp
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/__init__.py", line 35, in <module>
#     from . import (
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/data/__init__.py", line 18, in <module>
#     from .data_collator import *
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/data/data_collator.py", line 26, in <module>
#     from ..transformers import BertTokenizer
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/transformers/__init__.py", line 17, in <module>
#     from .model_utils import PretrainedModel, register_base_model
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/transformers/model_utils.py", line 66, in <module>
#     from ..generation import GenerationConfig, GenerationMixin
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/generation/__init__.py", line 33, in <module>
#     from .streamers import BaseStreamer, TextIteratorStreamer, TextStreamer
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/generation/streamers.py", line 18, in <module>
#     from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/transformers/tokenizer_utils.py", line 61, in <module>
#     from .utils import InitTrackerMeta, fn_args_to_dict
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/transformers/utils.py", line 59, in <module>
#     from .aistudio_utils import aistudio_download
#   File "/usr/local/lib/python3.10/dist-packages/paddlenlp/transformers/aistudio_utils.py", line 17, in <module>
#     from aistudio_sdk.hub import download
# ImportError: cannot import name 'download' from 'aistudio_sdk.hub' (/usr/local/lib/python3.10/dist-packages/aistudio_sdk/hub.py)
# ```

### known issue 2: Paddle2ONNX symbol error (after FIX 1, FIX 2 applied)
# ```
# /usr/local/lib/python3.10/dist-packages/paddle/base/framework.py:688: UserWarning: You are using GPU version Paddle, but your CUDA device is not set properly. CPU device will be used by default.
#   warnings.warn(
# Traceback (most recent call last):
#   File "/usr/local/bin/hub", line 3, in <module>
#     from paddlehub.commands.utils import execute
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/__init__.py", line 35, in <module>
#     from paddlehub.utils.paddlex import download, ResourceNotFoundError
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/utils/paddlex.py", line 19, in <module>
#     from paddlehub.server.server import module_server, CacheUpdater
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/server/__init__.py", line 17, in <module>
#     from paddlehub.server.git_source import GitSource
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/server/git_source.py", line 24, in <module>
#     from paddlehub.module.module import RunModule
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/module/__init__.py", line 16, in <module>
#     from .module import Module
#   File "/usr/local/lib/python3.10/dist-packages/paddlehub/module/module.py", line 29, in <module>
#     import paddle2onnx
#   File "/usr/local/lib/python3.10/dist-packages/paddle2onnx/__init__.py", line 85, in <module>
#     from .convert import export  # noqa: F401
#   File "/usr/local/lib/python3.10/dist-packages/paddle2onnx/convert.py", line 18, in <module>
#     import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
# ImportError: /usr/local/lib/python3.10/dist-packages/paddle2onnx/paddle2onnx_cpp2py_export.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3pir6detail14TypeIdResolverINS_14Float8E5M2TypeEE3id_E
# ```

### known issue 2 solution
#
# FROM registry.baidubce.com/paddlepaddle/paddle:2.6.0
# 를
# FROM paddlepaddle/paddle:3.2.2
# 로 변경하여 Paddle 버전을 3.2.2로 올리면 해결됨.
