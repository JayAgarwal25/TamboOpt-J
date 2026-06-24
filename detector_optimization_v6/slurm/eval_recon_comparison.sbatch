#!/bin/bash
#SBATCH --job-name=tambo_eval_2x2
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# 2x2 controlled utility evaluation: Jay's layout vs Zlatan's layout
# x flat-MLP recon vs DeepSets recon.
# Diagnoses whether low U in Jay's runs is due to the recon or the layout.

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

python eval_recon_comparison.py

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
