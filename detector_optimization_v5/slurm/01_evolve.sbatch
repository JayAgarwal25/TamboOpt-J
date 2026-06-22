#!/bin/bash
#SBATCH --job-name=tambo_v5_es
#SBATCH --partition=gpu_requeue
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --output=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.out
#SBATCH --error=/n/home07/jagarwal/tambo_slurm_logs/%x_%j.err

# v5 (mu+lambda)-ES detector layout optimizer.
#
# Runs 5 independent restarts of the ES on the frozen v6 dual-species
# surrogate (N_DETECTORS=100, North+Up convention).  Each restart uses
# a different random seed so the ensemble spans diverse init conditions.
#
# Wall-time estimate:
#   ~120 candidates/gen × ~10 ms/eval × 200 gen × 5 restarts ≈ 20 min.
#   4 h gives 12× headroom for slower GPU allocation or checkpoint overhead.
#
# gpu_requeue is preemptable.  01_run_evolution.py saves layout_best.pt
# and es_log.json at the END of each restart, so a preemption mid-restart
# discards that restart's result but all completed restarts are safe.
# Re-submit with --n-restart <remaining> and --seed <next> to continue.
#
# Optional arguments:
#   --seed N          base random seed (default 0)
#   --n-gen N         max generations per restart (default 200)
#   --n-restart N     number of restarts (default 5)
#   --output-dir DIR  root output directory

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

export PYTHONUNBUFFERED=1

cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v5

python 01_run_evolution.py "$@"

echo "========================================"
echo "Done:    $(date)"
echo "========================================"
