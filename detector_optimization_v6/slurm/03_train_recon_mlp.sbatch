#!/bin/bash
#SBATCH --job-name=tambo_03_recon_mlp
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# Trains the flat MLP reconstruction network (03_train_recon.py) as a
# comparison baseline against the DeepSets recon (03_train_recon_deepsets.py).
# Prerequisites: fnn_electron.pt and fnn_muon.pt must exist in FNN_FOLDER.

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:   $(date)"
echo "========================================"

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6

python 03_train_recon.py

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
