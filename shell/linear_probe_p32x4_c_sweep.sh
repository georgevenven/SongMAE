#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}" \
MODELS="xcl_micro_100k_p32x4_c005 xcl_micro_100k_p32x4_c010 xcl_micro_100k_p32x4_c020" \
  bash shell/linear_probe_across_models.sh "$@"
