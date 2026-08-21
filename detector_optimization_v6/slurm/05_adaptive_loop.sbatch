#!/bin/bash
# 05_adaptive_loop.sbatch — Submit one round of the adaptive surrogate retraining loop.
#
# Run on the LOGIN NODE (not via sbatch itself):
#
#   bash slurm/05_adaptive_loop.sbatch <ROUND> [<PREV_JOB_C_ID>]
#
# Each round chains three gpu_requeue SLURM jobs:
#   Job A  (~2h)  — step-1 infill dataset build
#   Job B  (~1h)  — step-2 FNN fine-tune on base + infill
#   Job C  (~3h)  — step-4 L-BFGS ensemble re-optimize with fine-tuned surrogate
#
# Round k reads  L_star_r{k-1}.pt  as the infill centre (North, East layout tensor)
#         writes L_star_r{k}.pt    after step-4 Job C completes.
#
# Before running round 1, create L_star_r0.pt from the baseline step-4 output:
#
#   python - << 'EOF'
#   import torch, os
#   D = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs'
#   src = os.path.join(D, 'test_v6_run_04_optimize_lbfgs_ensemble_combined/layout_best.pt')
#   opt = torch.load(src, weights_only=False)
#   layout = torch.stack([opt['x'], opt['y']], dim=-1)  # (N_DETECTORS, 2)
#   torch.save(layout, os.path.join(D, 'L_star_r0.pt'))
#   print('saved', layout.shape, 'to', os.path.join(D, 'L_star_r0.pt'))
#   EOF
#
# To submit all three rounds at once (each round starts after the previous):
#
#   PREV=""
#   for ROUND in 1 2 3; do
#       read -r JOB_C <<< "$(bash slurm/05_adaptive_loop.sbatch $ROUND $PREV | grep 'Job C' | awk '{print $NF}')"
#       PREV=$JOB_C
#   done
#
# Or more simply, just run each round after the previous one finishes:
#
#   bash slurm/05_adaptive_loop.sbatch 1
#   # wait for round 1 to complete, then:
#   bash slurm/05_adaptive_loop.sbatch 2
#   bash slurm/05_adaptive_loop.sbatch 3

# ── Tunable parameters ────────────────────────────────────────────────────────
N_INFILL_LAYOUTS=300
N_SHOWERS_PER_INFILL=1000
INFILL_SIGMA=200.0
FINETUNE_LR=1e-5
FINETUNE_EPOCHS=15

set -euo pipefail

ROUND=${1:?"Usage: bash slurm/05_adaptive_loop.sbatch <ROUND> [<PREV_JOB_C_ID>]"}
PREV_JOB_C=${2:-}   # optional job ID of previous round's step-4, for cross-round chaining

V6_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6
RUN_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs
FNN_BASE=${RUN_DIR}/test_v6_run_02_recentered

CENTER_PT=${RUN_DIR}/L_star_r$((ROUND - 1)).pt
OUTPUT_PT=${RUN_DIR}/L_star_r${ROUND}.pt
INFILL_DIR=${RUN_DIR}/test_v6_run_01_northeast_infill_r${ROUND}
FNN_DIR=${FNN_BASE}_r${ROUND}
RECON_BASE=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/test_v6_run_03_recentered_deepsets
RECON_DIR=${RECON_BASE}_r${ROUND}
OPT_SUFFIX=_r${ROUND}

LOG_DIR=/n/home07/jagarwal/tambo_slurm_logs
MAMBA_ENV=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo

# When chaining rounds (PREV_JOB_C set), CENTER_PT will be written by the
# previous round's Job C before Job A of this round starts — skip the check.
if [ -z "${PREV_JOB_C}" ] && [ ! -f "${CENTER_PT}" ]; then
    echo "ERROR: centre layout not found: ${CENTER_PT}"
    echo "Create it from the baseline step-4 output first (see script header)."
    exit 1
fi

echo "======================================================================"
echo "Adaptive loop — Round ${ROUND}"
echo "  centre   : ${CENTER_PT}"
echo "  infill   : ${INFILL_DIR}"
echo "  FNN out  : ${FNN_DIR}"
echo "  opt out  : test_v6_run_04_optimize_lbfgs_ensemble_r${ROUND}_{scheme}/"
echo "  recon    : ${RECON_DIR}"
echo "  L_star   : ${OUTPUT_PT}"
echo "======================================================================"

# ── Dependency for Job A ──────────────────────────────────────────────────────
if [ -n "${PREV_JOB_C}" ]; then
    DEP_A="--dependency=afterok:${PREV_JOB_C}"
else
    DEP_A=""
fi

# Shared preamble for all three jobs.
PREAMBLE="
set -euo pipefail
export PYTHONUNBUFFERED=1
echo \"Job: \$SLURM_JOB_ID  Node: \$(hostname)  Start: \$(date)\"
module load python
eval \"\$(mamba shell hook --shell bash)\"
mamba config set changeps1 False
mamba activate ${MAMBA_ENV}
cd ${V6_DIR}
"

# ── Job A: step-1 infill dataset build ───────────────────────────────────────
JOB_A=$(sbatch --parsable \
    --job-name=tambo_01_infill_r${ROUND} \
    --partition=gpu_requeue \
    --time=0-04:00:00 \
    --gres=gpu:nvidia_a100-sxm4-80gb:1 \
    --mem=64G \
    --cpus-per-task=4 \
    --requeue \
    --output=${LOG_DIR}/tambo_01_infill_r${ROUND}_%j.out \
    --error=${LOG_DIR}/tambo_01_infill_r${ROUND}_%j.err \
    ${DEP_A} \
    --wrap="${PREAMBLE}
python 01_build_dataset_northeast.py \\
    --infill_center ${CENTER_PT} \\
    --infill_sigma ${INFILL_SIGMA} \\
    --n_infill_layouts ${N_INFILL_LAYOUTS} \\
    --n_showers_per_infill ${N_SHOWERS_PER_INFILL} \\
    --round ${ROUND}
echo \"Done: \$(date)\"
")
echo "Submitted Job A (infill):      ${JOB_A}"

# ── Job B: step-2 FNN fine-tune ───────────────────────────────────────────────
JOB_B=$(sbatch --parsable \
    --job-name=tambo_02_finetune_r${ROUND} \
    --partition=gpu_requeue \
    --time=0-08:00:00 \
    --gres=gpu:nvidia_a100-sxm4-80gb:1 \
    --mem=64G \
    --cpus-per-task=4 \
    --requeue \
    --output=${LOG_DIR}/tambo_02_finetune_r${ROUND}_%j.out \
    --error=${LOG_DIR}/tambo_02_finetune_r${ROUND}_%j.err \
    --dependency=afterok:${JOB_A} \
    --wrap="${PREAMBLE}
python 02_train_fnn_deepsets.py \\
    --finetune_from ${FNN_BASE} \\
    --finetune_lr ${FINETUNE_LR} \\
    --finetune_epochs ${FINETUNE_EPOCHS} \\
    --finetune_dataset ${INFILL_DIR} \\
    --round ${ROUND}
echo \"Done: \$(date)\"
")
echo "Submitted Job B (finetune):    ${JOB_B}"

# ── Job B2: step-3 recon retrain with updated FNN ────────────────────────────
# The recon uses FNN predictions as input features; if the FNN changes the
# recon must be retrained with the updated predictions or optimization runs on
# a distorted landscape (u_theta, u_phi collapse ~3x).
JOB_B2=$(sbatch --parsable \
    --job-name=tambo_03_recon_r${ROUND} \
    --partition=gpu_requeue \
    --time=0-16:00:00 \
    --gres=gpu:nvidia_a100-sxm4-80gb:1 \
    --mem=64G \
    --cpus-per-task=4 \
    --requeue \
    --output=${LOG_DIR}/tambo_03_recon_r${ROUND}_%j.out \
    --error=${LOG_DIR}/tambo_03_recon_r${ROUND}_%j.err \
    --dependency=afterok:${JOB_B} \
    --wrap="${PREAMBLE}
python 03_train_recon_deepsets.py \\
    --fnn_folder ${FNN_DIR} \\
    --output_folder ${RECON_DIR}
echo \"Done: \$(date)\"
")
echo "Submitted Job B2 (recon retrain): ${JOB_B2}"

# ── Job C: step-4 L-BFGS ensemble re-optimize ────────────────────────────────
JOB_C=$(sbatch --parsable \
    --job-name=tambo_04_reopt_r${ROUND} \
    --partition=gpu_requeue \
    --time=0-12:00:00 \
    --gres=gpu:nvidia_a100-sxm4-80gb:1 \
    --mem=64G \
    --cpus-per-task=4 \
    --requeue \
    --output=${LOG_DIR}/tambo_04_reopt_r${ROUND}_%j.out \
    --error=${LOG_DIR}/tambo_04_reopt_r${ROUND}_%j.err \
    --dependency=afterok:${JOB_B2} \
    --wrap="${PREAMBLE}
python 04_optimize_lbfgs_ensemble.py \\
    --fnn_folder ${FNN_DIR} \\
    --recon_folder ${RECON_DIR} \\
    --init_from ${CENTER_PT} \\
    --opt_suffix ${OPT_SUFFIX} \\
    --save_best_layout ${OUTPUT_PT}
echo \"Done: \$(date)\"
")
echo "Submitted Job C (re-optimize): ${JOB_C}"

echo ""
echo "Chain: A=${JOB_A} -> B=${JOB_B} -> B2=${JOB_B2} -> C=${JOB_C}"
echo "L_star_r${ROUND}.pt will be written to ${OUTPUT_PT} when Job C finishes."
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOG_DIR}/tambo_*_r${ROUND}_*.out"
