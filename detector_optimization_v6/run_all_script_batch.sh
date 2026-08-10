#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=64g        			
#SBATCH --time=1-00:00:00
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --open-mode=append
#SBATCH -J run_all_script_batch
#SBATCH -o slurm_logs/slurm-%j-%x.out

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
STATUS_FILE="pipeline_status.json"
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
run_step 04_optimize_lbfgs_ensemble.py


