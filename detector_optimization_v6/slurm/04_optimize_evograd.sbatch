#!/bin/bash
#SBATCH --job-name=tambo_04_evograd
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# v6 step-4: EvoGrad detector layout optimizer.
# 5 chains × 2 init schemes + combined, each chain runs 200 EvoGrad generations
# (15 backward passes/gen) then polishes with L-BFGS (1000 iter).
# More expensive per-generation than GES due to K=15 backward passes.
#
# Wall-time estimate: ~4-6h for 10 chains (2 schemes × 5) + combined on a GPU.
# Outputs: test_v6_run_04_optimize_evograd_{grid,center,combined}/

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

python 04_optimize_evograd.py "$@"

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
