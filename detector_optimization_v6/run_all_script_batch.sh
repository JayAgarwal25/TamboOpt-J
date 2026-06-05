#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=100g        			
#SBATCH --time=1-00:00:00 			
#SBATCH -c 32            			
#SBATCH --gres=gpu:1        
#SBATCH --constraint=a100

module load python 

conda deactivate
conda deactivate

conda activate multiproc_env

export PYTHONUNBUFFERED=1

# python -u 00_generate_data.py
# python -u 01_build_dataset.py
# python -u 02_train_fnn.py
python -u 02_train_fnn_deepsets.py
# python -u 03_train_recon.py
# python -u 04_optimize_hmc_chains.py
# python -u 04_optimize.py
# python -u plots/02_plot_nn_target_vs_pred.py

