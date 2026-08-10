#!/bin/bash
#SBATCH -p gpu_test 	
#SBATCH --mem=60g        			
#SBATCH --time=00:30:00
#SBATCH -c 8            			
#SBATCH --gres=gpu:1        
#SBATCH -J run_plots
#SBATCH -o slurm_logs/slurm-%j-%x.out


module load python

conda deactivate
conda deactivate

conda activate multiproc_env

# --dual: this run is the dual-species pipeline (02_train_fnn_deepsets.py ->
# fnn_electron.pt + fnn_muon.pt, 03_train_recon_deepsets.py -> deepsets
# recon.pt). Without it the CLI looks for the single-species fnn.pt / flat-MLP
# recon.pt that these runs never wrote.
python -u plots/02_plot_nn_target_vs_pred.py --dual
# --phase both: the default is "adam" only, which drops the L-BFGS half of the
# run (chain 0: 1001 adam frames vs 2043 lbfgs). Those lbfgs frames are closure
# calls, not accepted iterates, so U is not monotonic along them.
python -u plots/04_plot_trajectory_gif.py --phase both --monotonic