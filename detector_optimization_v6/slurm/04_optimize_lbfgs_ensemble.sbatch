#!/bin/bash
#SBATCH --job-name=tambo_04_lbfgs
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# v6 step-4: L-BFGS ensemble optimizer over detector positions.
# Runs INIT_SCHEMES=("grid","center") × N_CHAINS=15 Adam warm-starts each,
# then L-BFGS refinement on a fixed primary batch, plus a "combined" ensemble.
# Reads surrogates from FNN_FOLDER + RECON_FOLDER in modules_v6/constants.py.
#
# Wall-time estimate: ~8h for 3×15 L-BFGS chains on a V100/A100.
# Outputs: test_v6_run_04_optimize_lbfgs_ensemble_{grid,center,combined}/

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

python 04_optimize_lbfgs_ensemble.py "$@"

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
