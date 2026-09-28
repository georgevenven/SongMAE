#!/usr/bin/env bash
# SongMAE-Micro 100k-step runs for the revision (Table 2 recipe: global batch 128), one after another.
# CONFIGS entries are <patch>_<mask>: mask = random | time | frequency | cNNN (Voronoi seed %, e.g. c005 = 5%).
# Usage (repo root): CONFIGS="p32x1_time p32x1_frequency" SEEDS="0 1 2" NPROC=1 SongMAE_TMLR_Revision/scripts/train_micro_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

TORCHRUN=${TORCHRUN:-$HOME/miniconda3/bin/torchrun}
NPROC=${NPROC:-2}
DATA_ROOT=${DATA_ROOT:-$PWD/data}
CONFIGS=${CONFIGS:-"p32x1_random p32x1_c0025 p32x1_c005 p32x1_c010"}
SEEDS=${SEEDS:-"1 2"}
export WANDB_PROJECT=${WANDB_PROJECT:-SongMAE-TMLR-revisions} WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-micro_seeds}

for config in $CONFIGS; do
  IFS=_ read -r patch mask <<< "$config"
  height=${patch#p}; height=${height%x*}; width=${patch#*x}
  case $mask in
    random) mask_args=(--mask_type random --mask_c 0.1) ;;
    time) mask_args=(--mask_type time --mask_c 0.05) ;;  # same expected seeds per clip as Voronoi at 5%
    frequency) mask_args=(--mask_type frequency) ;;
    c*) mask_args=(--mask_type voronoi --mask_c "0.${mask#c0}") ;;
    *) echo "unknown mask: $mask" >&2; exit 1 ;;
  esac
  for seed in $SEEDS; do
    name=xcl_micro_100k_${config}_seed${seed}
    [[ -f runs/$name/weights/model_step_099999.pth ]] && { echo "done: $name"; continue; }
    echo "training: $name ${mask_args[*]}"
    WANDB_TAGS="micro,100k,$config,seed$seed" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m src.core.train \
      --train_dir "$DATA_ROOT/XCL_clean" --val_dir "$DATA_ROOT/XCL_val_clean" --run_name "$name" \
      --steps 100000 --model_preset micro --patch_height "$height" --patch_width "$width" \
      "${mask_args[@]}" --seed "$seed"
  done
done
