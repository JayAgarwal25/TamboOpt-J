#!/bin/bash
#SBATCH --job-name=tambo_03_recon
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# Trains the DeepSets recon (max+mean pool, n_dec=3) on frozen dual FNN outputs.
# Architecture: hidden=256 context=128 n_enc=3 n_dec=3 pool=maxmean
# Estimated wall time: ~5-10h on the full 7M-pair corpus on an A100.
#
# Prerequisites: fnn_electron.pt and fnn_muon.pt must exist in FNN_FOLDER
# (output of 02_train_fnn_deepsets.py). The trainer precomputes all FNN
# predictions at startup, so the FNN is not touched during training epochs.
#
# gpu_requeue is preemptable. If preempted during Adam, the job restarts
# from epoch 1. If preempted during L-BFGS, it reloads the Adam best
# checkpoint. Checkpoint files are not corrupted by preemption.

set -euo pipefail

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

python 03_train_recon_deepsets.py

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
