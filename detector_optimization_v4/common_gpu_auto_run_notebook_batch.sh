#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=50g        			
#SBATCH --time=12:00:00 			
#SBATCH -c 4            			
#SBATCH --gres=gpu:1        
#SBATCH --constraint=a100


module load python 

conda activate multiproc_env

python auto_run_notebook.py SWGOLO7_optimization_tr_20k_center_init_angle_energy_adam_lr1_mean_u.ipynb
