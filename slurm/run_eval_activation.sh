#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=20g
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J run_eval_activation
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

# Whole untouched heldout reserve (~25k pairs, ~25 GB of clouds streamed in
# --load-block chunks). run_eval_performance did 2 kernel passes over 5120 events
# in 48 s, so 2 h is headroom, not an estimate.
# Both scheme dirs, so the grid- and center-initialized optima sit next to the
# two baselines in one figure.
#
# Pinned to the ACTIVATION runs (04_optimize_lbfgs_activation.py). Retarget this
# suffix by hand when the run being scored changes — stage-4 outputs get archived
# into "<RUN_LOCATION>/run N .../" between experiments, and the previous value
# (_lbfgs_ensemble_full_corpus) turned into a FileNotFoundError the moment that
# happened.
O="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')"
O="${O}_lbfgs_activation"
python -u plots/eval_activation_counts.py \
    --layout "${O}_grid/layout_best.pt" "${O}_center/layout_best.pt"
