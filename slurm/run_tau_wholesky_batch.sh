#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH -J tau_wholesky
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# nu_tau injection -> tau propagation -> geometric cuts -> HDF5.
# Drives decay_locations/tau_wholesky.jl.
#
# Usage:
#   sbatch run_tau_wholesky_batch.sh 100             # smoke test (run this first)
#   sbatch run_tau_wholesky_batch.sh 200000          # production corpus
#   NEVENT=100 ./run_tau_wholesky_batch.sh           # run outside slurm
#
# The first job also builds the julia depot (~200 pkgs, incl. C++/Fortran deps),
# which dominates its wall time. Later jobs reuse ~/.julia and skip straight to
# the run -- so do the 100-event smoke test first and let it warm the cache.
#
# No GPU: this is a CPU job (TauRunner + PROPOSAL). Do not request --gres=gpu.

set -euo pipefail

# --- TamboSim + julia ---------------------------------------------------------
# Julia is NOT provided by the module system on this cluster (the only module is
# julia/1.0.0-ncf, which is from 2018 and does not even load). Per
# https://docs.rc.fas.harvard.edu/kb/julia/ you install it yourself with juliaup:
#
#   curl -fsSL https://install.julialang.org | sh     # on a compute node
#   source ~/.bashrc
#
# TAMBOSIM_PATH is the TamboSim repo root (contains resources/ and Project.toml).
# The julia script activates that project itself and reads it from the env, so it
# is exported rather than passed as a flag.
export TAMBOSIM_PATH="${TAMBOSIM_PATH:-$HOME/tambo/TamboSim}"
JULIA_BIN="${JULIA_BIN:-$HOME/.juliaup/bin/julia}"

# The script + malata.jld2 live here in TambOpt, NOT inside the TamboSim repo
# (TamboSim only tracks the colca_valley geometry).
# Hardcoded rather than derived from $0/SLURM_SUBMIT_DIR: this script now lives
# in slurm/, a sibling of decay_locations/, not a parent of it, and
# SLURM_SUBMIT_DIR tracks wherever `sbatch` was invoked from (not this file's
# location), so neither could reliably point at decay_locations/ any more.
BASE_DIR="${BASE_DIR:-/n/home05/zdimitrov/tambo/TambOpt}"
SCRIPT="${SCRIPT:-$BASE_DIR/decay_locations/tau_wholesky.jl}"

# --- run parameters -----------------------------------------------------------
NEVENT=10000000       
SEED=1
MINDIST=1000          # integer m, decay vertex -> observation mesh

# Julia threads follow the cpus slurm actually gave us.
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# TauRunner's PROPOSAL tables go to $TAMBO_DATA_PATH/proposal_tables, and
# taurunner.jl:29 defaults that to tempdir() -- i.e. the compute node's local
# /tmp, which is wiped when the job ends, so every job rebuilds them ("Tables
# are not available and need to be created ... can take some minutes" in the
# log). Point it somewhere persistent so this is paid once.
# (The [proposal] tablespath in the TOML is a SEPARATE cache for the propagation
# stage; it already persists under TamboSim/resources/proposal_tables.)
export TAMBO_DATA_PATH="${TAMBO_DATA_PATH:-$HOME/tambo/tambo_data}"
mkdir -p "$TAMBO_DATA_PATH/proposal_tables"

[ -x "$JULIA_BIN" ]                || { echo "ERROR: no julia at $JULIA_BIN (install via juliaup)"; exit 1; }
[ -f "$SCRIPT" ]                   || { echo "ERROR: no script at $SCRIPT"; exit 1; }
[ -d "$TAMBOSIM_PATH/resources" ]  || { echo "ERROR: no resources/ under TAMBOSIM_PATH=$TAMBOSIM_PATH"; exit 1; }

echo "=== tau_wholesky ==="
echo "  julia   : $JULIA_BIN ($("$JULIA_BIN" --version))"
echo "  tambosim: $TAMBOSIM_PATH"
echo "  script  : $SCRIPT"
echo "  nevent  : $NEVENT"
echo "  seed    : $SEED"
echo "  mindist : $MINDIST m"
echo "  threads : $JULIA_NUM_THREADS"
echo "  tables  : $TAMBO_DATA_PATH/proposal_tables"
echo "  host    : $(hostname)"
date

# --- build (resolve + precompile) ---------------------------------------------
# Done here, on the compute node, NOT on a login node: TamboSim pulls ~200
# packages including PROPOSAL (C++) and Dierckx (Fortran), and precompiling that
# tree is heavily I/O-bound against the networked $HOME. FASRC's julia guide
# explicitly says to install/build packages from a compute node for this reason.
# Both steps are no-ops once ~/.julia is warm, so every later job skips them.
echo "--- build: instantiate + precompile"
time "$JULIA_BIN" --project="$TAMBOSIM_PATH" -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
    @time using TamboSim
    println("build ok: TamboSim loads")
'

echo "--- run"
time "$JULIA_BIN" "$SCRIPT" \
    --nevent  "$NEVENT" \
    --seed    "$SEED" \
    --mindist "$MINDIST"

date
echo "=== done ==="
