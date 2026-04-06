#!/bin/bash -l
#SBATCH --account=gardnerlab
#SBATCH --partition=computelong
#SBATCH --job-name=tinybird-xcl2b
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --output=/home/georgev/TinyBird/logs/tinybird-xcl-v2-bench-%j.out
#SBATCH --error=/home/georgev/TinyBird/logs/tinybird-xcl-v2-bench-%j.err

set -euo pipefail

module load miniconda3/20240410
eval "$(conda shell.bash hook)"
conda activate /home/georgev/.conda/envs/tinybird
pip install -r "$HOME/TinyBird/requirements.txt"

PIRG=${PIRG:-gardnerlab}
MY_SCRATCH=/scratch/${PIRG}/${USER}
chmod -R g+rwX "$MY_SCRATCH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1

workers=$(( SLURM_CPUS_PER_TASK * 75 / 100 ))
if [ "$workers" -lt 1 ]; then workers=1; fi

cd "$HOME/TinyBird"

python -u src/audio2spec.py \
--birdset "XCL" \
--birdset_split "train" \
--birdset_detections "human" \
--dst_dir "$MY_SCRATCH/XCL_corpus_v2_bench/" \
--single_threaded false \
--max_workers "$workers" \
--take_n_random 2000

echo "Done. Specs under: $MY_SCRATCH/XCL_corpus_v2_bench"
