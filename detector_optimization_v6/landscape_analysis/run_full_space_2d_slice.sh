#!/bin/bash
#SBATCH -J full_space_2d_slice
#SBATCH -p arguelles_delgado_gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/full_space_2d_slice_out.txt
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/full_space_2d_slice_err.txt

export PYTHONUNBUFFERED=1
# Extra args ("$@") pass through to the python script -- see
# run_detector_grid_scan.sh for the --layout_path/--layout_tag override pattern.
/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo/bin/python3 \
    /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6/landscape_analysis/full_space_2d_slice.py "$@"
