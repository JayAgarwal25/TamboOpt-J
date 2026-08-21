#!/bin/bash
#SBATCH -p gpu_test 	
#SBATCH --mem=20g        			
#SBATCH --time=00:30:00 			
#SBATCH -c 8            			
#SBATCH --gres=gpu:1
#SBATCH -J run_eval_performance
#SBATCH -o slurm_logs/slurm-%j-%x.out

# The pipeline stages refuse to guess a run world (modules_v6/run_world.py).
# Export it before submitting:  TAMBO_RUN_WORLD=/path/to/run sbatch <this>
: "${TAMBO_RUN_WORLD:?set TAMBO_RUN_WORLD to the run world root before submitting}"
export TAMBO_RUN_WORLD

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

python -u plots/eval_true_utility.py --grid-layout