"""
Central project configuration — single source of truth for all paths and parameters.

To move to a new machine: edit FMRIPREP_ROOT and NILEARN_CACHE only.
Everything else derives from those two values + PROJECT_ROOT.
"""
import os

# =============================================================================
# Machine-specific — edit only these two lines when moving to a new machine
# =============================================================================
FMRIPREP_ROOT = r"D:\amir_shared_folder\amir-scans-psylo-study\fmriprep_synFmap"
NILEARN_CACHE = r"C:\Users\amirh\Documents\nilearn_cache"

# =============================================================================
# Project layout (derived — do not edit)
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

# ---------------------------------------------------------------------------
# Timeseries output folders — one per denoising config
# ---------------------------------------------------------------------------
#   lean        12 motion + 6 aCompCor (combined WM/CSF)                   18 regressors
#   anatomical  12 motion + 5 WM CompCor + 5 CSF CompCor (separate masks)  22 regressors
#   global      12 motion + 6 aCompCor + GSR + GSR-deriv                   20 regressors
#   strict      24 Friston motion + 10 WM + 10 CSF CompCor (no GSR)        44 regressors

TS_OUTPUT_DIR_LEAN       = os.path.join(OUTPUTS_DIR, "timeseries_lean")
TS_OUTPUT_DIR_ANATOMICAL = os.path.join(OUTPUTS_DIR, "timeseries_anatomical")
TS_OUTPUT_DIR_GLOBAL     = os.path.join(OUTPUTS_DIR, "timeseries_global")
TS_OUTPUT_DIR_STRICT     = os.path.join(OUTPUTS_DIR, "timeseries_strict")

# Default active config (used by pipeline.py and subject_configs.py)
TS_OUTPUT_DIR    = TS_OUTPUT_DIR_LEAN
TS_OUTPUT_SUBDIR = "timeseries_lean"   # kept for pipeline log message

# Config name → output directory mapping (used by extractor __main__)
TS_CONFIG_DIRS = {
    "lean":       TS_OUTPUT_DIR_LEAN,
    "anatomical": TS_OUTPUT_DIR_ANATOMICAL,
    "global":     TS_OUTPUT_DIR_GLOBAL,
    "strict":     TS_OUTPUT_DIR_STRICT,
}

# ---------------------------------------------------------------------------
# QA + connectivity output dirs
# ---------------------------------------------------------------------------
QA_OUTPUT_DIR           = os.path.join(OUTPUTS_DIR, "quality_assurance")
QA_DENOISING_DIR        = os.path.join(QA_OUTPUT_DIR, "denoising_effect")
QA_DENOISING_REST_DIR   = os.path.join(QA_OUTPUT_DIR, "denoising_effect_restonly")

# =============================================================================
# Processing parameters — match these across all scripts
# =============================================================================
SMOOTHING_FWHM = 4       # mm spatial smoothing (FWHM)
HIGH_PASS      = 0.01    # Hz  (0.01 = 100s period)
LOW_PASS       = None    # Hz  — None disables low-pass filtering
FD_THRESHOLD   = 0.40    # mm  — notch-filtered FD scrubbing threshold (current + next vol)
N_ACOMPCOR     = 6       # aCompCor components (standard config)
N_WORKERS      = 4       # parallel processes for extraction

# Study sessions
SESSIONS = ["ses-1", "ses-2", "ses-3"]

# Session labels used in all plots and tables (ses-2 = psilocybin)
SESSION_LABELS = {
    "ses-1": "Baseline",
    "ses-2": "Psilocybin",
    "ses-3": "Follow-up",
}

SESSION_COLORS = {
    "ses-1": "#4A90D9",   # blue
    "ses-2": "#E94E77",   # pink/red
    "ses-3": "#5CB85C",   # green
}

# =============================================================================
# Helpers
# =============================================================================

def ensure_nilearn_cache() -> None:
    """Set NILEARN_DATA env var so all nilearn downloads go to the cache dir."""
    os.environ.setdefault("NILEARN_DATA", NILEARN_CACHE)
