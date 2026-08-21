#!/bin/bash
#SBATCH --job-name=tambo_04_bh
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# Basin hopping optimizer: large random perturbation + L-BFGS local polish,
# repeated N_HOPS times with Metropolis accept/reject.
#
# Typical usage — start from the best known layout and explore:
#   sbatch 04_optimize_basinhopping.sbatch \
#     --init_from <path/to/layout_best.pt> \
#     --n_hops 50 --sigma_hop 750
#
# Wall-time estimate: ~3 min/hop × 50 hops = ~2.5h; 8h ceiling gives headroom
# for gpu_requeue preemption restarts (the script re-polishes the start layout
# from scratch on each restart, so a preempted run wastes at most one hop).
#
# Outputs: test_v6_run_04_optimize_basinhopping{opt_suffix}/
#   layout_best.pt     best accepted layout
#   hop_log.json       per-hop U, delta_U, accepted, elapsed
#   layout_basinhopping.png

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

python 04_optimize_basinhopping.py "$@"

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
