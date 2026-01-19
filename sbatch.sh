#!/bin/bash
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --job-name=ChestXray_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail

mkdir -p logs

echo "=== JOB INFO ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Working dir: $(pwd)"
echo

echo "=== GPU ==="
nvidia-smi || true
echo

# (opzionale ma utile) evita di riempire la HOME: cache su /work
export HF_HOME=/work/grana_far2023_fomo/ChestXray/.hf
export HF_DATASETS_CACHE=/work/grana_far2023_fomo/ChestXray/.hf/datasets
export HF_HUB_CACHE=/work/grana_far2023_fomo/ChestXray/.hf/hub
export TORCH_HOME=/work/grana_far2023_fomo/ChestXray/.torch
export WANDB_DIR=/work/grana_far2023_fomo/ChestXray/wandb
export PIP_CACHE_DIR=/work/grana_far2023_fomo/ChestXray/.pip-cache

# Attiva venv
source /work/grana_far2023_fomo/ChestXray/Chest/bin/activate

# (opzionale) thread control: spesso meglio non lasciare default altissimi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Python: $(which python)"
python -V
echo

# Lancia training
python baseline_net.py

echo
echo "End: $(date)"
echo "=== GPU (end) ==="
nvidia-smi || true
echo "=== JOB END ==="