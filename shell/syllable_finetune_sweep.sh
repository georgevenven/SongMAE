#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/mae/bin/python}"
source "$(dirname "$0")/syllable_knn_lib.sh"

MODELS="${MODELS:-xcl_base_100k_p32x4_c010 birdaves_biox_base hubert_base_ls960}"
TRAIN_SECONDS="${TRAIN_SECONDS:-32 64 128 256 512 MAX}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/syllable_finetune}"
ENCODER_LRS="${ENCODER_LRS:-1e-5,5e-5,1e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FIXED="${FIXED:-0}"
FINETUNE_NUM_TIMEBINS="${FINETUNE_NUM_TIMEBINS:-720000}"
BIRDS="${BIRDS:-}"
TARGETS=("$@")

selected() {
  [[ "${#TARGETS[@]}" -eq 0 ]] && return
  [[ " ${TARGETS[*]} " == *" $1 "* ]]
}

selected_bird() {
  [[ -z "$BIRDS" ]] || [[ " $BIRDS " == *" $1 "* ]]
}

read -r -a MODEL_LIST <<< "$MODELS"
read -r -a BUDGETS <<< "$TRAIN_SECONDS"
for row in "${DATASETS[@]}"; do
  IFS="|" read -r species annotation spec_dir wav_dir <<< "$row"
  selected "$species" || continue
  while IFS= read -r bird; do
    selected_bird "$bird" || continue
    for model in "${MODEL_LIST[@]}"; do
      for budget in "${BUDGETS[@]}"; do
        out_dir="$OUT_ROOT/$species/$bird/$model/all/train_${budget}s"
        [[ -f "$out_dir/metrics.json" ]] && { echo "exists $out_dir/metrics.json"; continue; }
        rm -rf "$out_dir"
        common=(
          "$PYTHON_BIN" src/evals/syllable_finetune.py
          --spec_dir "$spec_dir" --annotation_file "$annotation" --bird "$bird" --out_dir "$out_dir"
          --max_train_seconds "$budget" --encoder_lrs "$ENCODER_LRS" --head_lr "$HEAD_LR"
          --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"
          --num_timebins "$FINETUNE_NUM_TIMEBINS"
        )
        [[ "$FIXED" == "1" ]] && common+=(--fixed)
        echo "running species=$species bird=$bird model=$model train=$budget tune=all"
        case "$model" in
          birdaves_biox_base)
            if ! "${common[@]}" aves --wav_dir "$wav_dir" \
              --aves_model_path "$BIRDAVES_MODEL_PATH" --aves_config_path "$BIRDAVES_CONFIG_PATH"; then
              echo "failed species=$species bird=$bird model=$model train=$budget" >&2
            fi
            ;;
          hubert_base_ls960)
            if ! "${common[@]}" hubert --wav_dir "$wav_dir" --model_name "$HUBERT_MODEL_NAME"; then
              echo "failed species=$species bird=$bird model=$model train=$budget" >&2
            fi
            ;;
          *)
            if ! "${common[@]}" songmae --songmae_run_dir "$(songmae_run_dir "$model")"; then
              echo "failed species=$species bird=$bird model=$model train=$budget" >&2
            fi
            ;;
        esac
      done
    done
  done < <(birds_in_json "$annotation")
done
