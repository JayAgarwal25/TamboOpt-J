#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J check_softcaps
#SBATCH -o slurm_logs/slurm-%j-%x.out

module load python
conda deactivate
conda deactivate
conda activate multiproc_env
export PYTHONUNBUFFERED=1

RUN="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/07_750k_primaires_meanvar/run 5 penalty edge"

python -u check_softcaps.py \
    --run-dir "$RUN/test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_center" \
              "$RUN/test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_grid"
