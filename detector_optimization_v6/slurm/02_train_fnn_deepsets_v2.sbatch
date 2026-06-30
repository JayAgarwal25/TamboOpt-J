#!/bin/bash
#SBATCH --job-name=tambo_02_fnn_v2
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# Trains FNN surrogates with the upgraded architecture:
#   hidden=512 (was 256), context=128 (was 64), pool=maxmean (was mean).
# Writes fnn_electron.pt + fnn_muon.pt into FNN_FOLDER (test_v6_run_02_recentered).
# After this completes, retrain the recon with 03_train_recon_deepsets.sbatch
# before running any 04 optimizers.

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

python 02_train_fnn_deepsets.py \
  --output_folder /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/test_v6_run_02_recentered_v2 \
  "$@"

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
