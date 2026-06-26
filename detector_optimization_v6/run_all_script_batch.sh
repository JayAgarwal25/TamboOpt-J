#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=64g        			
#SBATCH --time=1-10:00:00 			
#SBATCH -c 32            			
#SBATCH --gres=gpu:1        
#SBATCH --constraint=a100

module load python 

conda deactivate
conda deactivate

conda activate multiproc_env

export PYTHONUNBUFFERED=1

# --- checkpointing: steps marked done in pipeline_status.json are skipped ---
# Delete the file (or a step's entry) to force a rerun.
STATUS_FILE="pipeline_status.json"
[ -f "$STATUS_FILE" ] || echo '{}' > "$STATUS_FILE"

run_step () {
    local step="$1"; shift
    if python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS_FILE')).get('$step')=='done' else 1)"; then
        echo ">>> Skipping $step (already done)"
        return 0
    fi
    echo ">>> Running $step $*"
    python -u "$step" "$@" || exit $?
    python -c "import json; d=json.load(open('$STATUS_FILE')); d['$step']='done'; json.dump(d, open('$STATUS_FILE','w'), indent=2)"
}

# # Step-0 resume: continue the a crashed run (slurm-21376182) into the existing
# # corpus file from this row — electron block complete + 20k muons = last
# # logged "file offset". Set to 0 for a FRESH corpus (re-preallocates the
# # file)
# RESUME_ROW=520000

# run_step 00_generate_data.py
run_step 00_generate_data_dual_species.py
run_step 01_build_dataset_northeast.py
run_step 02_train_fnn_deepsets.py
run_step 03_train_recon.py
run_step 03_train_recon_deepsets.py
# run_step 04_optimize_lbfgs_ensemble.py
run_step 04_optimize_differential_evolution.py
run_step 04_optimize_differential_evolution_pop.py
# run_step plots/02_plot_nn_target_vs_pred.py

