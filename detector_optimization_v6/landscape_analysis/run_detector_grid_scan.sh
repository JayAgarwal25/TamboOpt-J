#!/bin/bash
#SBATCH -J detector_grid_scan
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/detector_grid_scan_out.txt
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/detector_grid_scan_err.txt

export PYTHONUNBUFFERED=1
# Extra args ("$@") pass through to the python script, e.g.:
#   sbatch --job-name=grid_evograd \
#       -o .../detector_grid_scan_evograd_out.txt -e .../detector_grid_scan_evograd_err.txt \
#       run_detector_grid_scan.sh --layout_path <path> --layout_tag evograd
/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo/bin/python3 \
    /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/detector_grid_scan.py "$@"
