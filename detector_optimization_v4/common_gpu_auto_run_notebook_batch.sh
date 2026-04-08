#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=150g        			
#SBATCH --time=04:00:00 			
#SBATCH -c 4            			
#SBATCH --gres=gpu:1        
#SBATCH --constraint=a100


module load python 

conda activate multiproc_env

python auto_run_notebook.py SWGOLO7_optimization_tr.ipynb
