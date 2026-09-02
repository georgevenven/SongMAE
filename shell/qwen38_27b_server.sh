#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP=${LLAMA_CPP:-/home/george-vengrovski/src/llama.cpp}
MODEL_DIR=${MODEL_DIR:-/home/george-vengrovski/models/Qwen3.8-27B-GGUF}
CTX_SIZE=${CTX_SIZE:-262144}
PARALLEL=${PARALLEL:-32}

exec "$LLAMA_CPP/build/bin/llama-server" \
  --model "$MODEL_DIR/Qwen3.8-27B-Q8_0.gguf" \
  --mmproj "$MODEL_DIR/mmproj-F16.gguf" \
  --image-min-tokens 1024 \
  --ctx-size "$CTX_SIZE" \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --n-gpu-layers 999 \
  --device CUDA0,CUDA1 \
  --split-mode layer \
  --tensor-split 1,1 \
  --flash-attn on \
  --batch-size 2048 \
  --ubatch-size 512 \
  --parallel "$PARALLEL" \
  --jinja \
  --reasoning-effort high \
  --reasoning-preserve \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0 \
  --host 127.0.0.1 \
  --port 8080
