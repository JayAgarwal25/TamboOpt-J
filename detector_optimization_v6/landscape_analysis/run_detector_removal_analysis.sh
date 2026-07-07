#!/bin/bash
#SBATCH -J detector_removal_analysis
#SBATCH -p arguelles_delgado_gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 10:00:00
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/detector_removal_analysis_out.txt
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/detector_removal_analysis_err.txt

export PYTHONUNBUFFERED=1
# Extra args ("$@") pass through to the python script -- see
# run_detector_grid_scan.sh for the --layout_path/--layout_tag override pattern.
/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo/bin/python3 \
    /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/detector_removal_analysis.py "$@"
