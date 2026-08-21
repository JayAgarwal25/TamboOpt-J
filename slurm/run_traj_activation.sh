#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH --mem=24g
#SBATCH --time=03:00:00
#SBATCH -c 4
#SBATCH -J traj_activation
#SBATCH --open-mode=append
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

module load python
conda deactivate
conda deactivate
conda activate multiproc_env

# all .pyc under one tree instead of __pycache__/ dirs across the source
export PYTHONPYCACHEPREFIX=/n/home05/zdimitrov/tambo/TambOpt/.pycache

# CPU only: 04_plot_trajectory_gif.py loads trajectory.pt on cpu and never touches
# a model, so this does not need (or wait for) a GPU.
cd /n/home05/zdimitrov/tambo/TambOpt
R="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import RUN_LOCATION; print(RUN_LOCATION)')"
A="$R/test_v6_run_04_optimize_lbfgs_activation"
DIRS=("${A}_center" "${A}_grid")

# serial_requeue is preemptible and the render is not checkpointed, so a job that
# gets requeued restarts from scratch. Skipping outputs that already exist makes
# the restart resume at the render that was interrupted instead of redoing both.
# Delete a file to force it to be regenerated.
render () {
    local out="$1"; shift
    if [ -s "$out" ]; then
        echo ">>> Skipping $(basename "$out") (already written)"
        return 0
    fi
    python -u plots/04_plot_trajectory_gif.py --run-dir "${DIRS[@]}" "$@" -o "$out"
}

# 1) Adam warm-start, first 500 epochs. --only zoom skips the full-trajectory
#    render; the zoom pass appends _first500ep to the -o stem, so the guard has to
#    test THAT name, not the stem passed in.
if [ -s "$R/activation_distinct_trajectory_first500ep.mp4" ]; then
    echo ">>> Skipping activation_distinct_trajectory_first500ep.mp4 (already written)"
else
    python -u plots/04_plot_trajectory_gif.py --run-dir "${DIRS[@]}" \
        --only zoom --zoom-epochs 500 \
        -o "$R/activation_distinct_trajectory.mp4"
fi

# 2) L-BFGS sweep chunks 0-9. --min-per-chunk 0 because the per-chunk floor is
#    pointless over a 10-chunk range, and --fit-view drops the strong-Wolfe probes
#    that fling the layout off-map so the axes can fit the detectors.
render "$R/activation_distinct_lbfgs_chunk0-9.mp4" \
    --phase lbfgs --lbfgs-chunk 0-9 --fit-view --min-per-chunk 0 \
    --zoom-epochs 0 --seconds 20 --fps 60
