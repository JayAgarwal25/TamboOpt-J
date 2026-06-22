#!/bin/bash
#SBATCH --job-name=tambo_v5_cmaes
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# v5b sep-CMA-ES + multi-fidelity detector layout optimizer.
#
# Algorithm: sep-CMA-ES (diagonal covariance adaptation) — learns the shape
# of the fitness landscape, unlike the isotropic (mu+lambda)-ES in 01_run_evolution.py.
#
# Multi-fidelity: 64 primaries/eval when sigma > 50m (explore),
#                512 primaries/eval when sigma <= 50m (refine).
# Best layout per restart is re-evaluated with 512 primaries for fair comparison.
#
# Outputs land in v5_es_runs/cmaes_<timestamp>/.

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

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v5

python 02_run_cmaes.py "$@"

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
