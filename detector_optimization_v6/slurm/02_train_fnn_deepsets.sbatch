#!/bin/bash
#SBATCH --job-name=tambo_02_fnn
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# Trains electron + muon FNN surrogates sequentially (both in one job).
# Estimated wall time: ~8-14h for the full 7M-pair corpus on an A100.
#
# gpu_requeue is preemptable. If preempted during Adam, the job restarts
# from epoch 1. If preempted during L-BFGS, it reloads the Adam best
# checkpoint and reruns L-BFGS.  Either way the checkpoint files are safe.
#
# To retrain only one species after a partial run:
#   python 02_train_fnn_deepsets.py --species electron
#   python 02_train_fnn_deepsets.py --species muon

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

python 02_train_fnn_deepsets.py

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
