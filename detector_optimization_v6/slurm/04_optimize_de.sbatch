#!/bin/bash
#SBATCH --job-name=tambo_04_de
#SBATCH --partition=gpu_requeue
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# v6 step-4: Differential Evolution ensemble optimizer over detector positions.
# Runs INIT_SCHEMES=("grid","center") with N_CHAINS=1 DE run per scheme, plus
# a "combined" ensemble — apples-to-apples comparison with the L-BFGS ensemble.
# Reads the same surrogates as 04_optimize_lbfgs_ensemble.py:
#   FNN_FOLDER   = test_v6_run_02_recentered   (DeepSets, val_e=0.408)
#   RECON_FOLDER = test_v6_run_03_recentered_deepsets (DeepSets recon, val=0.001118)
#
# Wall-time estimate: ~12-24h for 2 DE runs (grid+center) on a V100/A100.
# Outputs: test_v6_run_04_optimize_de_ensemble_{grid,center,combined}/

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

python 04_optimize_differential_evolution.py "$@"

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
