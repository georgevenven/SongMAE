#!/bin/bash
#SBATCH -J tinybird-xcl-data2vec
#SBATCH -p gpulong
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-80gb,no-mig
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH -o logs/tinybird-data2vec-xcl-%j.out
#SBATCH -e logs/tinybird-data2vec-xcl-%j.err

set -euo pipefail

module load miniconda3/20240410
eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/tinybird

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$SLURM_SUBMIT_DIR"

MY_SCRATCH="/scratch/gardnerlab/georgev"
TRAIN_DIR="${MY_SCRATCH}/XCL_Train"
VAL_DIR="${MY_SCRATCH}/XCL_Test"
mkdir -p logs

RUN_NAME="xcl_data2vec_voronoi_32h_10w"

python -u src/data2vec_train.py \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --run_name "$RUN_NAME" \
  --steps 500000 \
  --batch_size 256 \
  --num_workers 48 \
  --lr 0.0003 \
  --dropout 0.1 \
  --mask_p 0.75 \
  --mask_c 0.1 \
  --mask_type voronoi \
  --eval_every 500 \
  --warmup_steps 1000 \
  --min_lr 0.00001 \
  --weight_decay 0.1 \
  --patch_height 32 \
  --patch_width 10 \
  --num_timebins 1000 \
  --enc_hidden_d 384 \
  --enc_n_head 6 \
  --enc_n_layer 6 \
  --enc_dim_ff 1536 \
  --dec_hidden_d 192 \
  --dec_n_head 6 \
  --dec_n_layer 2 \
  --dec_dim_ff 768 \
  --input_normalization per_file_zscore \
  --amp \
  --wandb
