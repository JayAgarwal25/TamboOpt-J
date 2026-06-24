#!/bin/bash
#SBATCH --job-name=tambo_eval_cross
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# Cross-evaluate 7 optimizer-found layouts (all post-NE-fix) x 2 recons.
# Answers: does DS recon just scale all utilities up, or did DS-optimized
# layouts genuinely find better regions of layout space?

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

python eval_cross_comparison.py

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
