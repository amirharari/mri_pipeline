#!/bin/bash
#
# SBATCH directives -------------------------------------------------
#SBATCH --job-name=ts_extract
#SBATCH --time=12:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err
#SBATCH --mail-user=ah2668@yale.edu
#SBATCH --mail-type=ALL
# -------------------------------------------------------------------
# Usage:
#   sbatch run_timeseries_extract.sh ses-1
#   sbatch run_timeseries_extract.sh ses-2 rest both
#   sbatch run_timeseries_extract.sh ses-3 rest global
#
#   Positional args:
#     $1  session       (required)   ses-1 | ses-2 | ses-3
#     $2  mode          (optional)   rest | task          (default: rest)
#     $3  denoising     (optional)   global | anatomical | both  (default: both)
# -------------------------------------------------------------------

if [ -z "$1" ]; then
  echo "Usage: sbatch run_timeseries_extract.sh ses-1|ses-2|ses-3 [rest|task] [global|anatomical|both]"
  exit 1
fi

SESSION=$1
MODE=${2:-rest}
DENOISING=${3:-both}

# --- load env/modules ---------------------------------------------
ml Python/3.8.6-GCCcore-10.2.0
source ~/venv/first/bin/activate

# No MATLAB / FSL / SPM needed — pure Python pipeline

# --- run extraction -----------------------------------------------
python /home/ah2668/Documents/mcclary_timeseries_extract.py \
    "$SESSION" \
    --mode "$MODE" \
    --denoising "$DENOISING" \
    --workers 4
