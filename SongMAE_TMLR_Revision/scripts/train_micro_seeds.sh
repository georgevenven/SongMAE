#!/usr/bin/env bash
# Extra pretraining seeds (R1) for SongMAE-Micro 100k-step runs, same recipe as Table 2: DDP on 2 GPUs, global batch 128.
# Usage (repo root): CONFIGS="p32x1_random p32x1_c0025" SEEDS="1 2" SongMAE_TMLR_Revision/scripts/train_micro_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

TORCHRUN=${TORCHRUN:-$HOME/miniconda3/bin/torchrun}
CONFIGS=${CONFIGS:-"p32x1_random p32x1_c0025 p32x1_c005 p32x1_c010"}
SEEDS=${SEEDS:-"1 2"}
export WANDB_PROJECT=${WANDB_PROJECT:-SongMAE-TMLR-revisions} WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-micro_seeds}

for config in $CONFIGS; do
  IFS=_ read -r patch mask <<< "$config"
  height=${patch#p}; height=${height%x*}; width=${patch#*x}
  if [[ $mask == random ]]; then mask_args=(--mask_type random --mask_c 0.1)
  else mask_args=(--mask_type voronoi --mask_c "0.${mask#c0}"); fi
  for seed in $SEEDS; do
    name=xcl_micro_100k_${config}_seed${seed}
    [[ -f runs/$name/weights/model_step_099999.pth ]] && { echo "done: $name"; continue; }
    echo "training: $name ${mask_args[*]}"
    WANDB_TAGS="micro,100k,$config,seed$seed" "$TORCHRUN" --standalone --nproc_per_node=2 -m src.core.train \
      --train_dir "$PWD/data/XCL_clean" --val_dir "$PWD/data/XCL_val_clean" --run_name "$name" \
      --steps 100000 --model_preset micro --patch_height "$height" --patch_width "$width" \
      "${mask_args[@]}" --seed "$seed"
  done
done
