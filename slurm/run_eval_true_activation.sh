#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=20g
#SBATCH --time=00:30:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J run_eval_true_activation
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

source slurm/env.sh

# Each scheme against the baseline it was initialized from, so "baseline vs
# optimized" is the actual before/after of that run rather than a cross-comparison.
O="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')"
O="${O}_lbfgs_activation"

python -u plots/eval_true_activation.py --layout "${O}_grid/layout_best.pt"
echo
python -u plots/eval_true_activation.py --layout "${O}_center/layout_best.pt" \
    --center-layout
