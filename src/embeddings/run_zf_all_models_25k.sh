#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"

SPEC_DIR="${SPEC_DIR:-/media/george-vengrovski/disk2/specs/zf_64hop_32khz}"
WAV_DIR="${WAV_DIR:-/media/george-vengrovski/disk2/raw_data/avn_zf_data}"
ANNOTATION_FILE="${ANNOTATION_FILE:-$ROOT/files/zf_annotations.json}"
OUT_DIR="${OUT_DIR:-$ROOT/results/embeddings/zf_all_models_25k}"
SONGMAE_RUN_DIR="${SONGMAE_RUN_DIR:-$ROOT/runs/zf_songmae_32h10_bs128_10k_e2e}"

MODELS="${MODELS:-songmae,aves,bird_mae,hubert,perch2}"
MAX_POINTS="${MAX_POINTS:-25000}"
UMAP_NEIGHBORS="${UMAP_NEIGHBORS:-200}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.1}"
UMAP_METRIC="${UMAP_METRIC:-cosine}"
SEED="${SEED:-42}"
RECORDING_STEM="${RECORDING_STEM:-}"
BIRD="${BIRD:-}"
REUSE="${REUSE:-0}"

PERCH_PYTHON="${PERCH_PYTHON:-/home/george-vengrovski/anaconda3/envs/perch/bin/python}"
PERCH_CUDNN_DIR="${PERCH_CUDNN_DIR:-/home/george-vengrovski/anaconda3/envs/perch/lib/python3.11/site-packages/nvidia/cudnn/lib}"

cd "$ROOT"
cmd=(
  "$PYTHON_BIN" -m src.embeddings.umap
  --spec_dir "$SPEC_DIR"
  --wav_dir "$WAV_DIR"
  --annotation_file "$ANNOTATION_FILE"
  --out_dir "$OUT_DIR"
  --models "$MODELS"
  --songmae_run_dir "$SONGMAE_RUN_DIR"
  --num_timebins 0
  --max_points "$MAX_POINTS"
  --seed "$SEED"
  --deterministic
  --umap_neighbors "$UMAP_NEIGHBORS"
  --umap_min_dist "$UMAP_MIN_DIST"
  --umap_metric "$UMAP_METRIC"
  --perch_python "$PERCH_PYTHON"
  --perch_cudnn_dir "$PERCH_CUDNN_DIR"
)

if [[ -n "$RECORDING_STEM" ]]; then cmd+=(--recording_stem "$RECORDING_STEM"); fi
if [[ -n "$BIRD" ]]; then cmd+=(--bird "$BIRD"); fi
if [[ "$REUSE" == "1" ]]; then cmd+=(--reuse); fi

PYTHONPATH="$ROOT" "${cmd[@]}"
