#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to your local VLM checkpoint directory or chosen model ID}"
: "${TOOL_CALL_PARSER:?Set the vLLM tool parser supported by your chosen model}"
args=(serve "$MODEL_PATH" --host 127.0.0.1 --port "${PORT:-8000}"
  --served-model-name "${SERVED_MODEL_NAME:-local-vlm}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --max-model-len "${MAX_MODEL_LEN:-65536}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
if [[ -n "${LOCAL_MODEL_API_KEY:-}" ]]; then args+=(--api-key "$LOCAL_MODEL_API_KEY"); fi
exec vllm "${args[@]}" "$@"
