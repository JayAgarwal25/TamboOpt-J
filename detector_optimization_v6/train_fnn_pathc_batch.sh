#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=100g
#SBATCH --time=12:00:00
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J fnn_pathc

# Path-(c) FNN de-risk experiment, chained:
#   1. lr_range_test.py   — LR finder → writes recommended LR_MAX to FNN_FOLDER
#   2. 02_train_fnn.py    — retrain (auto-loads that LR_MAX; AdamW, no dropout,
#                           corrected OneCycle floor, L-BFGS capped at 800)
#   3. eval_fnn_fired_r2.py — honest metric: prints PASS (keep flat MLP) or
#                           "use path (a)" (DeepSets rewrite) with the failing number
#
# `set -e` stops the chain if any stage fails, so the eval only runs on a
# genuinely-completed training. Submit with:  sbatch train_fnn_pathc_batch.sh
#
# CAVEAT: gpu_requeue is preemptable and this chain is NOT resumable — a
# preemption mid-train restarts stage 2 from scratch (the LR test result on
# disk is reused, so stage 1 won't repeat). Switch -p to a non-preempt GPU
# partition if you have access and want guaranteed completion.

set -euo pipefail

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
echo "================================================================"
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-?}   gpu: ${CUDA_VISIBLE_DEVICES:-?}"
echo "started : $(date)"
echo "================================================================"

echo ">>> [1/3] LR range test"
python -u lr_range_test.py

echo ">>> [2/3] FNN training (Adam + L-BFGS)"
python -u 02_train_fnn.py

echo ">>> [3/3] Conditional-on-fired evaluation (path-c go/no-go)"
python -u eval_fnn_fired_r2.py

echo "================================================================"
echo "finished: $(date)"
echo "Decision printed above by eval_fnn_fired_r2.py:"
echo "  ✅ PASS        -> path (c) worked; keep the flat MLP."
echo "  ❌ BELOW BAR   -> ask Claude to implement path (a): the DeepSets rewrite."
echo "================================================================"
