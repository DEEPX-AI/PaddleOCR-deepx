#!/bin/bash
# DEEPX NPU Environment Configuration
# Source this file before running the FastAPI service with NPU support
# Usage: source ./deepx_env.sh

# RT Optimization Settings (from dx_baidu_gui/set_env.sh)
export CUSTOM_INTER_OP_THREADS_COUNT=1
export CUSTOM_INTRA_OP_THREADS_COUNT=3

# Optional settings (uncomment and set values if needed)
# export DXRT_DYNAMIC_CPU_THREAD=3
# export DXRT_TASK_MAX_LOAD=4
# export NFH_INPUT_WORKER_THREADS=5
# export NFH_OUTPUT_WORKER_THREADS=6

# Library paths
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"

echo "✓ DEEPX NPU environment configured"
echo "  CUSTOM_INTER_OP_THREADS_COUNT=${CUSTOM_INTER_OP_THREADS_COUNT}"
echo "  CUSTOM_INTRA_OP_THREADS_COUNT=${CUSTOM_INTRA_OP_THREADS_COUNT}"
