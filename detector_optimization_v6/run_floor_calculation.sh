#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=70g        			
#SBATCH --time=3:00:00 			
#SBATCH -c 32            			
#SBATCH --gres=gpu:1        
#SBATCH --constraint=a100
#SBATCH -J run_floor_calculation
#SBATCH -o slurm_logs/slurm-%j-%x.out

# The pipeline stages refuse to guess a run world (modules_v6/run_world.py).
# Export it before submitting:  TAMBO_RUN_WORLD=/path/to/run sbatch <this>
: "${TAMBO_RUN_WORLD:?set TAMBO_RUN_WORLD to the run world root before submitting}"
export TAMBO_RUN_WORLD

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

python -u plots/compute_aleatoric_floor.py