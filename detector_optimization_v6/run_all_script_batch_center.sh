#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=64g        			
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -J run_all_script_batch_center
#SBATCH -o slurm_logs/slurm-%j-%x.out

# The pipeline stages refuse to guess a run world (modules_v6/run_world.py).
# Export it before submitting:  TAMBO_RUN_WORLD=/path/to/run sbatch <this>
: "${TAMBO_RUN_WORLD:?set TAMBO_RUN_WORLD to the run world root before submitting}"
export TAMBO_RUN_WORLD

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

export PYTHONUNBUFFERED=1

# --- checkpointing: steps marked done in pipeline_status.json are skipped ---
# Delete the file (or a step's entry) to force a rerun.
# -s not -f: the file can exist but be EMPTY (a preempted gpu_requeue job killed
# between open(...,"w") truncating it and json.dump refilling it). -f accepted
# the 0-byte file, so every json.load below threw and no step was ever marked
# done -- the whole pipeline re-ran from scratch every time.
STATUS_FILE="pipeline_status_center.json"
[ -s "$STATUS_FILE" ] || echo '{}' > "$STATUS_FILE"

run_step () {
    local step="$1"; shift
    if python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS_FILE')).get('$step')=='done' else 1)"; then
        echo ">>> Skipping $step (already done)"
        return 0
    fi
    echo ">>> Running $step $*"
    python -u "$step" "$@" || exit $?
    # write to a temp file then os.replace (atomic): a preemption can no longer
    # leave the status file truncated to 0 bytes.
    python -c "import json,os; d=json.load(open('$STATUS_FILE')); d['$step']='done'; json.dump(d, open('$STATUS_FILE.tmp','w'), indent=2); os.replace('$STATUS_FILE.tmp','$STATUS_FILE')"
}

# Step 0 now resumes automatically (progress.json next to each output corpus,
# per species) — a preempted run just needs the same command re-run, no manual
# row/offset bookkeeping. --n-pairs 0 = all in-band tau events (~751,931);
# Step 0 itself splits off HOLDOUT_FRAC (5%) into a separate corpus before
# generating, so this is 750k-scale total, not 750k into training.
run_step 00_generate_data_dual_species.py --n-pairs 0
run_step 01_build_dataset_northeast.py
run_step 02_train_fnn_deepsets.py
run_step 03_train_recon_deepsets.py
# --chains 1: this run is for the layout trajectory, not the ensemble spread,
# and one chain at POS_LOG_EVERY=1 is what makes the animation smooth. The
# 8-chain run took 11h03m of the 12h limit; one chain is ~1/8 of that.
run_step 04_optimize_lbfgs_ensemble.py --schemes center --chains 1


