#!/bin/bash
#SBATCH -J critical_detector_segregation_grid
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -t 00:15:00
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/critical_detector_segregation_grid_out.txt
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/critical_detector_segregation_grid_err.txt

export PYTHONUNBUFFERED=1
/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo/bin/python3 \
    /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/critical_detector_segregation_grid.py
