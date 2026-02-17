#!/bin/bash
#SBATCH -p arguelles_delgado_gpu_mixed          # partition. Remember to change to a desired partition
#SBATCH --mem=200g                              # memory in GB
#SBATCH --time=24:00:00                         # time in HH:MM:SS
#SBATCH -c 4                                    # number of cores
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb        # requested GPU type


module load python 

conda activate multiproc_env

python auto_run_notebook.py SWGOLO7_optimization.ipynb
