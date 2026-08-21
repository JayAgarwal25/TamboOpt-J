# Shared environment for every SLURM wrapper in this directory.
#
# Source it, never execute it. Each wrapper sets `#SBATCH --chdir` to the repo
# root, so the relative path is what the job actually sees:
#
#     source slurm/env.sh
#
# SBATCH directives stay in the individual wrappers: they must be in the
# submitted file's leading comment block, and sbatch stops scanning at the first
# non-comment line, so a sourced file cannot carry them.

# Real path of THIS file, not of the wrapper -- under sbatch the wrapper runs
# from the node spool dir (/var/slurmd/spool/...), so $0 is useless here.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module load python

# Twice: the login profile can leave two envs stacked.
conda deactivate
conda deactivate
conda activate multiproc_env

export PYTHONUNBUFFERED=1

# One bytecode tree at the repo root instead of __pycache__/ dirs scattered
# through the source. Set here rather than in each wrapper so there is a single
# place to change it; `_pathfix.py` sets the same value as a fallback for
# notebooks and interactive sessions that never source this file.
export PYTHONPYCACHEPREFIX="${REPO_ROOT}/.pycache"
