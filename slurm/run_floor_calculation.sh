#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=70g        			
#SBATCH --time=3:00:00 			
#SBATCH -c 32            			
#SBATCH --gres=gpu:1        
#SBATCH --constraint=a100
#SBATCH -J run_floor_calculation
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

source slurm/env.sh

python -u plots/compute_aleatoric_floor.py