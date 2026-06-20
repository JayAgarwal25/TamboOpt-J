#!/bin/bash
#SBATCH --job-name=tambo_01_build
#SBATCH --partition=serial_requeue
#SBATCH --time=0-12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# 64G covers the 35GB dual shower cache (cashed_showers_dual_200000.pt) plus
# training pair tensors in RAM; serial_requeue is preemptable and writes atomically.

set -euo pipefail

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "========================================"

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6

python 01_build_dataset_northeast.py

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
