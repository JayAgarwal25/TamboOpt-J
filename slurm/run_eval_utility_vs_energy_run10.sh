#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=40g
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J eval_u_vs_e_r10
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

# all .pyc under one tree instead of __pycache__/ dirs across the source
export PYTHONPYCACHEPREFIX=/n/home05/zdimitrov/tambo/TambOpt/.pycache

python -u plots/eval_utility_vs_energy.py --run-dir \
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/07_750k_primaires_meanvar/run 10 simple utility coverage/test_v6_run_04_optimize_lbfgs_activation_center" \
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/07_750k_primaires_meanvar/run 10 simple utility coverage/test_v6_run_04_optimize_lbfgs_activation_grid"
