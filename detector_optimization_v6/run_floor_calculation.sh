#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=70g        			
#SBATCH --time=3:00:00 			
#SBATCH -c 32            			
#SBATCH --gres=gpu:1        
#SBATCH --constraint=a100

module load python 

conda deactivate
conda deactivate

conda activate multiproc_env

python -u plots/compute_aleatoric_floor.py