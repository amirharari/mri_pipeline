"""fMRI timeseries extractor.

Generic API
-----------
    from atlases import DEFAULT_ATLASES
    from fmri_timeseries_extractor import BoldRun, create_time_series

    runs = [
        BoldRun("sub-001", "ses-1", "rest", "run-1",
                bold_path="/path/to/bold.nii.gz",
                confounds_path="/path/to/confounds.tsv"),
        ...
    ]
    create_time_series(runs, output_dir="/out", atlases=DEFAULT_ATLASES, use_gsr=False)

fMRIPrep helper
---------------
    Use discover_files() to auto-build the run list from an fMRIPrep output tree.
"""

import datetime
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img, index_img
from nilearn.maskers import NiftiLabelsMasker
from scipy.signal import filtfilt, iirnotch

from atlases import AtlasSpec, DEFAULT_ATLASES, map_labels_to_names
from ts_qc import (  # noqa: E402  — QC helpers live in their own module
    post_dvars                   as _post_dvars,
    ts_qc_metrics                as _ts_qc_metrics,
    confounds_variance_explained as _confounds_variance_explained,
    louvain_q                    as _louvain_q,
    fc_network_metrics           as _fc_network_metrics,
)

# Import shared config — single source of truth for all paths and parameters
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    FMRIPREP_ROOT,
    NILEARN_CACHE,
    SMOOTHING_FWHM        as _CFG_SMOOTHING,
    HIGH_PASS             as _CFG_HIGH_PASS,
    LOW_PASS              as _CFG_LOW_PASS,
    FD_THRESHOLD          as _CFG_FD_THRESHOLD,
    N_ACOMPCOR            as _CFG_N_ACOMPCOR,
    N_WORKERS             as _CFG_N_WORKERS,
    TS_OUTPUT_DIR,
    TS_OUTPUT_DIR_LEAN,
    TS_OUTPUT_DIR_ANATOMICAL,
    TS_OUTPUT_DIR_GLOBAL,
    TS_OUTPUT_DIR_STRICT,
    TS_CONFIG_DIRS,
    ensure_nilearn_cache,
)

ensure_nilearn_cache()

# -------------------------------------------------------------------------
# --- parameters (defaults sourced from config.py) ------------------------
# -------------------------------------------------------------------------
STANDARDIZE: str       = "zscore_sample"
SMOOTHING_FWHM: float  = _CFG_SMOOTHING
DETREND: bool          = True
HIGH_PASS: float       = _CFG_HIGH_PASS
LOW_PASS: Optional[float] = _CFG_LOW_PASS
NUM_VOLS_TO_REMOVE: int = 0
N_ACOMPCOR: int        = _CFG_N_ACOMPCOR
DO_SCRUBBING: bool     = True
N_WORKERS: int         = _CFG_N_WORKERS

SCRUB_REPORT_CSV = "scrubbing_report_filtered.csv"

# Legacy aliases kept for any external code that imports them directly
OUTPUT_DIR_GSR_OFF  = TS_OUTPUT_DIR_LEAN
OUTPUT_DIR_RESEARCH = TS_OUTPUT_DIR_STRICT

# Config 4 (Research Standard) — Friston 24-parameter motion model
# 6 raw + 6 deriv + 6 squared + 6 deriv-squared
_MOTION_RAW    = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
_MOTION_DERIV  = [f + "_derivative1" for f in _MOTION_RAW]
_MOTION_POW2   = [f + "_power2"      for f in _MOTION_RAW]
_MOTION_D_POW2 = [f + "_derivative1_power2" for f in _MOTION_RAW]
MOTION_COLS_24 = _MOTION_RAW + _MOTION_DERIV + _MOTION_POW2 + _MOTION_D_POW2

# Config 4 CompCor: 10 WM + 10 CSF components (separate masks, not combined)
RESEARCH_N_WM  = 10
RESEARCH_N_CSF = 10

# Motion thresholds — sourced from config.py (change there, not here)
FD_THRESHOLD                  = _CFG_FD_THRESHOLD   # mm, notch-filtered FD
SCRUBBED_VOLS_RATIO_THRESHOLD = 0.3

BACKGROUND_LABEL_VALUE = 0


# =========================================================================
# --- Motion Filtering  (Fair et al. 2020) --------------------------------
# =========================================================================

def apply_motion_filter(motion_params: pd.DataFrame, tr: float) -> pd.DataFrame:
    """Notch-filter motion parameters to remove respiratory pseudo-motion.

    Adult resting respiratory rate: ~0.2-0.5 Hz.
    Notch centre: 0.35 Hz, bandwidth: +/-0.1 Hz.  filtfilt => zero phase shift.

    If the TR is too long (Nyquist < breathing band centre), the notch cannot
    be applied — respiratory frequencies are already aliased into the signal.
    In that case the original params are returned unchanged with a warning.
    """
    fs          = 1.0 / tr
    nyquist     = fs / 2.0
    center_freq = 0.35   # Hz — mid-point of adult breathing band

    if center_freq >= nyquist:
        print(f"  [Motion filter] TR={tr:.3f}s -> Nyquist={nyquist:.3f} Hz < "
              f"{center_freq} Hz breathing band — notch filter skipped.")
        return motion_params.copy()

    Q    = center_freq / 0.2
    b, a = iirnotch(center_freq, Q, fs)
    filtered = motion_params.copy()
    for col in filtered.columns:
        filtered[col] = filtfilt(b, a, motion_params[col].values)
    return filtered


# =========================================================================
# --- BoldRun  (generic input data structure) -----------------------------
# =========================================================================

class BoldRun:
    """Paths and metadata for a single BOLD run.

    This is the input data structure for create_time_series().
    Build it yourself or use discover_files() / discover_files_all_tasks()
    for fMRIPrep trees.

    Parameters
    ----------
    acq : optional acquisition label (e.g. "glass", "bailero"). Included in
          the output CSV filename when present so runs can be distinguished.
    """

    def __init__(self, subject: str, session: str, task: str, run: str,
                 bold_path: str, confounds_path: str,
                 acq: Optional[str] = None):
        self.subject        = subject
        self.session        = session
        self.task           = task
        self.run            = run
        self.acq            = acq
        self.bold_path      = bold_path
        self.confounds_path = confounds_path

    @property
    def bids_id(self) -> str:
        """Short identifier used in output filenames and log messages."""
        parts = [f"task-{self.task}"]
        if self.acq:
            parts.append(f"acq-{self.acq}")
        parts.append(self.run)
        return "_".join(parts)

    def __repr__(self):
        return (f"{self.subject} | {self.session} | {self.bids_id}\n"
                f"  BOLD: {self.bold_path}")


# =========================================================================
# --- fMRIPrep helper — not required for generic use ----------------------
# =========================================================================

def discover_files(project_root: str,
                   max_subjects: Optional[int] = None) -> List[BoldRun]:
    """Walk an fMRIPrep output tree and return one BoldRun per rest BOLD run."""
    runs: List[BoldRun] = []
    subjects = sorted(p for p in os.listdir(project_root) if p.startswith("sub-"))
    if max_subjects:
        subjects = subjects[:max_subjects]

    for subj in subjects:
        subj_path = os.path.join(project_root, subj)
        if not os.path.isdir(subj_path):
            continue
        for ses in sorted(p for p in os.listdir(subj_path) if p.startswith("ses-")):
            func_dir = os.path.join(subj_path, ses, "func")
            if not os.path.isdir(func_dir):
                continue

            run_files: Dict[str, dict] = {}
            for fname in sorted(os.listdir(func_dir)):
                if "task-rest" not in fname:
                    continue
                run_match = re.search(r"run-(\w+)", fname)
                run_key   = run_match.group(0) if run_match else "run-1"
                run_files.setdefault(run_key, {})
                if "preproc_bold" in fname and "MNI" in fname and fname.endswith(".nii.gz"):
                    run_files[run_key]["bold"] = os.path.join(func_dir, fname)
                elif fname.endswith("confounds_timeseries.tsv"):
                    run_files[run_key]["conf"] = os.path.join(func_dir, fname)

            for run_key, f in run_files.items():
                if "bold" in f and "conf" in f:
                    runs.append(BoldRun(subj, ses, "rest", run_key, f["bold"], f["conf"]))

    return runs


def discover_files_all_tasks(project_root: str,
                             max_subjects: Optional[int] = None) -> List[BoldRun]:
    """Walk an fMRIPrep output tree and return BoldRun objects for every
    preprocessed BOLD run (all tasks, not just rest).

    Designed for datasets where filenames follow BIDS conventions, e.g.:
      sub-001_ses-1_task-rest_space-MNI..._desc-preproc_bold.nii.gz
      sub-001_ses-1_task-music_acq-glass_run-1_space-MNI..._desc-preproc_bold.nii.gz

    Confounds are located by stripping the _space-... suffix and appending
    _desc-confounds_timeseries.tsv — the standard fMRIPrep naming pattern.

    Runs where no BOLD NII.GZ or no matching confounds TSV exist are silently
    skipped.
    """
    runs: List[BoldRun] = []
    subjects = sorted(p for p in os.listdir(project_root) if p.startswith("sub-"))
    if max_subjects:
        subjects = subjects[:max_subjects]

    for subj in subjects:
        subj_path = os.path.join(project_root, subj)
        if not os.path.isdir(subj_path):
            continue
        for ses in sorted(p for p in os.listdir(subj_path) if p.startswith("ses-")):
            func_dir = os.path.join(subj_path, ses, "func")
            if not os.path.isdir(func_dir):
                continue

            for fname in sorted(os.listdir(func_dir)):
                if not (fname.endswith(".nii.gz") and
                        "preproc_bold" in fname and
                        "MNI" in fname):
                    continue

                bold_path = os.path.join(func_dir, fname)

                # Derive confounds path: strip _space-... suffix
                stem    = re.sub(r"_space-.*$", "", fname)
                conf_fn = stem + "_desc-confounds_timeseries.tsv"
                conf_path = os.path.join(func_dir, conf_fn)
                if not os.path.isfile(conf_path):
                    print(f"  [SKIP] no confounds for {fname}")
                    continue

                # Use [a-zA-Z0-9]+ (not \w+) — underscores are BIDS separators,
                # not part of entity values.
                task_m = re.search(r"task-([a-zA-Z0-9]+)", fname)
                acq_m  = re.search(r"acq-([a-zA-Z0-9]+)",  fname)
                run_m  = re.search(r"run-([a-zA-Z0-9]+)",  fname)
                if not task_m:
                    continue

                task    = task_m.group(1)
                acq     = acq_m.group(1)              if acq_m else None
                run_key = f"run-{run_m.group(1)}"     if run_m else "run-1"

                runs.append(BoldRun(
                    subject=subj, session=ses, task=task, run=run_key,
                    bold_path=bold_path, confounds_path=conf_path, acq=acq,
                ))

    return runs


# =========================================================================
# --- Helpers -------------------------------------------------------------
# =========================================================================

def get_tr_seconds(bold_path: str, img: nib.Nifti1Image) -> float:
    """Read TR from NIfTI header or JSON sidecar.  Raises ValueError if absent.

    Also handles misnamed BOLD files (e.g. a stray extra token before _space-)
    by progressively stripping trailing tokens to locate the JSON sidecar.
    """
    zooms = img.header.get_zooms()
    if len(zooms) > 3 and zooms[3] > 0:
        return float(zooms[3])

    # Try the direct JSON path, then fall back by stripping stray tokens
    folder   = os.path.dirname(bold_path)
    basename = os.path.basename(bold_path)
    prefix   = re.sub(r"_space-.*$", "", basename)   # strip MNI-space suffix

    for _ in range(5):
        json_path = os.path.join(folder, prefix + "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json")
        if os.path.isfile(json_path):
            with open(json_path) as fh:
                meta = json.load(fh)
            if "RepetitionTime" in meta:
                return float(meta["RepetitionTime"])
        new_prefix = re.sub(r"_[^_]+$", "", prefix)
        if new_prefix == prefix:
            break
        prefix = new_prefix

    raise ValueError(f"TR not found in header or JSON sidecar: {bold_path}")


def _get_acompcor_cols(conf_path: str, max_n: int = 10) -> List[str]:
    """Return aCompCor columns to use as confound regressors.

    Strategy: take the first `max_n` WM/CSF-mask components, sorted by index
    (i.e. ranked by variance explained, highest first as fMRIPrep outputs them).

    Why cap at max_n?
      fMRIPrep marks many components as Retained=True (sometimes 100+), but
      including all of them causes severe over-fitting: for a 350-TR run,
      100+ aCompCor + 12 motion = >35% of TRs as regressors, which destroys
      residual signal and produces zero-variance ROIs.
      Standard practice (Ciric 2017, Muschelli 2014) is 5–10 components.

    Only WM/CSF-mask components are selected (Mask: "WM", "CSF", or "combined").
    Brain-mask aCompCor is excluded — it behaves similarly to GSR.
    """
    json_path = conf_path.replace(".tsv", ".json")
    if os.path.isfile(json_path):
        with open(json_path) as fh:
            meta = json.load(fh)
        WM_CSF_MASKS = {"WM", "CSF", "combined"}
        candidates = sorted([
            col for col, info in meta.items()
            if col.startswith("a_comp_cor")
            and isinstance(info, dict)
            and info.get("Retained", False)
            and info.get("Mask", "") in WM_CSF_MASKS
        ])
        if candidates:
            return candidates[:max_n]

    # Sidecar absent — fall back to first max_n by index
    return [f"a_comp_cor_{i:02d}" for i in range(max_n)]


def get_conf_cols(use_gsr: bool, conf_path: Optional[str] = None,
                  config: str = "lean") -> List[str]:
    """Return the confound column list for this run.

    Configs
    -------
    lean        (default) 12 motion + 6 aCompCor combined WM/CSF       = 18 regressors
    anatomical  12 motion + 5 WM CompCor + 5 CSF CompCor (separate)    = 22 regressors
    global      12 motion + 6 aCompCor + GSR + GSR-deriv                = 20 regressors
    strict      24 Friston motion + 10 WM CompCor + 10 CSF CompCor      = 44 regressors

    Backward-compat aliases: "standard" -> "lean", "research" -> "strict"
    """
    # Normalize legacy names
    _ALIAS = {"standard": "lean", "research": "strict"}
    config = _ALIAS.get(config, config)

    _12_motion = [
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
        "rot_x_derivative1",  "rot_y_derivative1",  "rot_z_derivative1",
    ]

    if config == "strict":
        wm_cols  = [f"w_comp_cor_{i:02d}" for i in range(RESEARCH_N_WM)]
        csf_cols = [f"c_comp_cor_{i:02d}" for i in range(RESEARCH_N_CSF)]
        return MOTION_COLS_24 + wm_cols + csf_cols   # 24 + 10 + 10 = 44

    if config == "anatomical":
        wm_cols  = [f"w_comp_cor_{i:02d}" for i in range(5)]
        csf_cols = [f"c_comp_cor_{i:02d}" for i in range(5)]
        return _12_motion + wm_cols + csf_cols       # 12 + 5 + 5 = 22

    if config == "global":
        acompcor = _get_acompcor_cols(conf_path, max_n=N_ACOMPCOR) if conf_path else \
                   [f"a_comp_cor_{i:02d}" for i in range(N_ACOMPCOR)]
        return _12_motion + acompcor + [
            "global_signal", "global_signal_derivative1"
        ]                                            # 12 + 6 + 2 = 20

    # --- lean (default) ---
    n_acomp  = 3 if use_gsr else N_ACOMPCOR
    acompcor = _get_acompcor_cols(conf_path, max_n=n_acomp) if conf_path else \
               [f"a_comp_cor_{i:02d}" for i in range(n_acomp)]

    cols = _12_motion + acompcor

    if use_gsr:
        cols.append("global_signal")
    return cols


# =========================================================================
# --- Confound building  (Fair et al. 2020 motion filter) -----------------
# =========================================================================

def build_confounds(conf_path: str, run: BoldRun, use_gsr: bool,
                    tr: float, config: str = "lean") -> Tuple[pd.DataFrame, dict]:
    """Return (confound_matrix, stats_dict).

    Applies notch filter to the 6 raw motion params before computing FD so that
    respiratory pseudo-motion does not inflate scrubbing rates.
    NOTE: assumes fMRIPrep confound column naming conventions.
    """
    if tr <= 0:
        raise ValueError(f"Invalid TR={tr:.4f}s for {run.subject}/{run.session}/{run.run}.")

    df         = pd.read_csv(conf_path, sep="\t")
    df         = df.loc[NUM_VOLS_TO_REMOVE:].reset_index(drop=True)
    total_vols = len(df)

    # Raw FD (for comparison logging)
    fd_raw     = df["framewise_displacement"].fillna(0)
    raw_spikes = int((fd_raw > FD_THRESHOLD).sum())

    # Notch-filter the 6 raw motion params → recompute FD (always from 6 raw params)
    raw_motion_cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    filt_motion = apply_motion_filter(df[raw_motion_cols].fillna(0), tr)
    diff        = filt_motion.diff().fillna(0)
    diff[["rot_x", "rot_y", "rot_z"]] *= 50   # radians -> mm (50 mm head radius)
    fd_filtered     = diff.abs().sum(axis=1)
    filtered_spikes = int((fd_filtered > FD_THRESHOLD).sum())

    print(f"  FD raw:      mean={fd_raw.mean():.3f}  max={fd_raw.max():.3f}  "
          f"spikes>{FD_THRESHOLD}mm = {raw_spikes}/{total_vols}")
    print(f"  FD filtered: mean={fd_filtered.mean():.3f}  max={fd_filtered.max():.3f}  "
          f"spikes>{FD_THRESHOLD}mm = {filtered_spikes}/{total_vols}  "
          f"(reduction: {raw_spikes - filtered_spikes:+d})")

    # Base confound matrix  (FD is used only for scrubbing, not as a regressor)
    available = [c for c in get_conf_cols(use_gsr, conf_path, config) if c in df.columns]
    conf_df   = df[available].fillna(0).copy()  # derivatives are NaN on row 0

    # Scrubbing: spike + 1 volume after
    scrubbed_volumes = 0
    scrub_fraction   = 0.0
    high_motion_skip = False

    if DO_SCRUBBING:
        to_scrub: set = set()
        for t in fd_filtered[fd_filtered > FD_THRESHOLD].index.tolist():
            for dt in [0, 1]:
                if 0 <= t + dt < total_vols:
                    to_scrub.add(t + dt)

        scrubbed_volumes = len(to_scrub)
        scrub_fraction   = scrubbed_volumes / total_vols if total_vols > 0 else 0.0

        if scrub_fraction > SCRUBBED_VOLS_RATIO_THRESHOLD:
            high_motion_skip = True
            print(f"  [HIGH MOTION] {scrubbed_volumes}/{total_vols} "
                  f"({scrub_fraction:.1%}) -- spike regressors skipped, run flagged.")
        elif scrubbed_volumes > 0:
            spike_regs = pd.DataFrame(
                0, index=df.index,
                columns=[f"motion_spike_{i}" for i in range(scrubbed_volumes)]
            )
            for idx, vol in enumerate(sorted(to_scrub)):
                spike_regs.loc[vol, f"motion_spike_{idx}"] = 1
            conf_df = pd.concat([conf_df, spike_regs], axis=1)
            print(f"  Scrubbing:   {scrubbed_volumes}/{total_vols} vols censored ({scrub_fraction:.1%})")
        else:
            print("  Scrubbing:   0 volumes flagged -- clean run")

    stats = {
        "subject":          run.subject,
        "session":          run.session,
        "task":             run.task,
        "acq":              run.acq or "",
        "run":              run.run,
        "fd_mean_raw":      round(float(fd_raw.mean()),      4),
        "fd_mean_filtered": round(float(fd_filtered.mean()), 4),
        "fd_max_filtered":  round(float(fd_filtered.max()),  4),
        "raw_spikes":       raw_spikes,
        "filtered_spikes":  filtered_spikes,
        "scrubbed_volumes": scrubbed_volumes,
        "scrub_percent":    round(scrub_fraction * 100, 1),
        "total_vols":       total_vols,
        "high_motion_skip": high_motion_skip,
        "gsr_applied":      use_gsr,
    }
    return conf_df, stats


# =========================================================================
# --- Worker  (top-level required for Windows multiprocessing spawn) ------
# =========================================================================

def _process_one_run(args: tuple) -> Optional[dict]:
    """Process one BOLD run.  Returns stats dict, or None on unrecoverable error."""
    run, output_dir, use_gsr, atlas_factories, config = args

    # Load atlases inside the worker — avoids pickling NiBabel objects
    atlas_specs: List[AtlasSpec] = [fn() for fn in atlas_factories]

    masker_kwargs = dict(
        standardize=STANDARDIZE,
        smoothing_fwhm=SMOOTHING_FWHM,
        detrend=DETREND,
        low_pass=LOW_PASS,
        high_pass=HIGH_PASS,
        standardize_confounds=True,
        memory=os.environ.get("NILEARN_DATA", "nilearn_cache"),
        verbose=0,
        background_label=BACKGROUND_LABEL_VALUE,
    )

    # Skip if all output CSVs already exist (allows safe resume after crash)
    expected_paths = [
        os.path.join(output_dir, f"{run.subject}_{run.session}_{run.bids_id}{spec.suffix}")
        for spec in atlas_specs
    ]
    if all(os.path.exists(p) for p in expected_paths):
        print(f"[SKIP already exists] {run.subject} | {run.session} | {run.bids_id}")
        # Still return stats (FD, scrubbing) so scrubbing report stays complete
        try:
            tr = get_tr_seconds(run.bold_path, nib.load(run.bold_path))
            _, stats = build_confounds(run.confounds_path, run, use_gsr, tr, config)
            stats["confound_var_explained"] = np.nan  # not re-computed on skip
            return stats
        except Exception:
            return None

    print(f"Processing {run.subject} | {run.session} | {run.bids_id}")
    try:
        img_full = nib.load(run.bold_path)
        tr       = get_tr_seconds(run.bold_path, img_full)
        img      = index_img(img_full, slice(NUM_VOLS_TO_REMOVE, None))

        conf_df, stats = build_confounds(run.confounds_path, run, use_gsr, tr, config)

        if stats["high_motion_skip"]:
            stats["confound_var_explained"] = np.nan
            print("  -> Skipped (high motion)\n")
            return stats

        conf_arr = conf_df.values.astype(np.float64)
        conf_arr = (conf_arr - np.nanmean(conf_arr, axis=0)) / (np.nanstd(conf_arr, axis=0) + 1e-12)

        # Extract raw (unprocessed) ROI timeseries once for var-explained diagnostic.
        # Done before the atlas loop so we only need one extra masker call total.
        raw_kwargs = dict(
            standardize=False, detrend=False, low_pass=None, high_pass=None,
            smoothing_fwhm=SMOOTHING_FWHM, standardize_confounds=False,
            memory=os.environ.get("NILEARN_DATA", "nilearn_cache"),
            verbose=0, background_label=BACKGROUND_LABEL_VALUE,
        )

        var_explained_max = 0.0
        for spec in atlas_specs:
            atlas_img  = resample_to_img(spec.maps, img, interpolation="nearest", copy=True)
            masker     = NiftiLabelsMasker(labels_img=atlas_img, t_r=tr, **masker_kwargs)
            ts         = masker.fit_transform(img, confounds=conf_df)
            col_names  = map_labels_to_names(masker.labels_, spec.index_to_name)
            df_ts      = pd.DataFrame(ts, columns=col_names)

            # Variance explained (re-uses the already-in-memory img — no extra gzip load)
            raw_masker = NiftiLabelsMasker(labels_img=atlas_img, t_r=tr, **raw_kwargs)
            raw_ts     = raw_masker.fit_transform(img, confounds=None)
            var_exp    = _confounds_variance_explained(raw_ts, conf_arr)
            if not np.isnan(var_exp):
                var_explained_max = max(var_explained_max, var_exp)
                print(f"  [Var explained] {spec.name}: {var_exp:.1%} by confounds", end="")
                if var_exp > 0.7:
                    print("  [OVER-CORRECTION?]")
                else:
                    print()

            corr = np.corrcoef(df_ts.T)
            n    = corr.shape[0]
            gcor = float(np.nanmean(corr[~np.eye(n, dtype=bool)]))

            # Post-denoising DVARS + tSNR (from raw) + lag-1 AC — always (< 1 ms)
            dvars  = _post_dvars(ts)
            tsqc   = _ts_qc_metrics(ts, ts_raw=raw_ts)

            # Extended QC metrics — only meaningful for cortical parcellation atlases
            is_cortical = "schaefer" in spec.name.lower()
            if is_cortical:
                q_val  = _louvain_q(corr)
                nm     = _fc_network_metrics(df_ts, corr)
                # Store for cross-run QC-FC proxy
                if "gcor_schaefer" not in stats:
                    stats["gcor_schaefer"] = round(gcor, 4)
                    stats["dvars_post"]    = round(dvars, 3)

                # Build two-line detail block
                line2 = (
                    f"    tSNR={tsqc['tsnr']:.1f}"
                    + (f"  [LOW]" if tsqc["tsnr"] < 20 else "")
                    + f"  AC={tsqc['lag1_ac']:.3f}"
                    + (f"  [OVER-DENOISE?]" if tsqc["lag1_ac"] < 0.25 else "")
                    + f"  Q={q_val:.3f}"
                    + f"  MotLR={nm['motor_lr']:+.3f}"
                    + f"  SomMot={nm['somm_within']:+.3f}"
                    + f"  DMN={nm['dmn_within']:+.3f}"
                    + f"  DMN-FPN={nm['dmn_fpn']:+.3f}"
                )
            else:
                line2 = (
                    f"    tSNR={tsqc['tsnr']:.1f}"
                    + f"  AC={tsqc['lag1_ac']:.3f}"
                )

            # GCOR + DVARS joint interpretation
            gcor_flag = ""
            if gcor > 0.3:
                if dvars > 1.5:
                    gcor_flag = "  <- GCOR HIGH + DVARS HIGH: likely artifact"
                else:
                    gcor_flag = "  <- GCOR HIGH, DVARS normal: likely biological"

            out_name = f"{run.subject}_{run.session}_{run.bids_id}{spec.suffix}"
            df_ts.to_csv(os.path.join(output_dir, out_name), index=False)
            dvars_flag = "  [DVARS HIGH]" if dvars > 1.5 else ""
            print(f"  Saved {out_name}  shape={df_ts.shape}"
                  f"  GCOR={gcor:.3f}  DVARS={dvars:.3f}{dvars_flag}{gcor_flag}")
            print(line2)

        stats["confound_var_explained"] = round(float(var_explained_max), 4)
        print()
        return stats

    except Exception as exc:
        import traceback
        print(f"  [ERROR] {run.subject}/{run.session}/{run.run}: {exc}")
        traceback.print_exc()
        return None


# =========================================================================
# --- Provenance ----------------------------------------------------------
# =========================================================================

def _write_extraction_config(
    output_dir: str,
    runs: List[BoldRun],
    use_gsr: bool,
    atlas_factories: List[Callable[[], AtlasSpec]],
) -> None:
    """Save extraction_config.json to *output_dir* for downstream traceability.

    Figure scripts read this file to auto-derive display labels (e.g. GSR on/off)
    and to document exactly what parameters produced the timeseries CSVs.
    """
    atlas_names: List[str] = []
    for fn in atlas_factories:
        try:
            atlas_names.append(fn().name)
        except Exception:
            atlas_names.append(getattr(fn, "__name__", str(fn)))

    config = {
        "created_at":                    datetime.datetime.now().isoformat(timespec="seconds"),
        "subjects":                      sorted({r.subject for r in runs}),
        "n_runs":                        len(runs),
        "gsr":                           use_gsr,
        "atlases":                       atlas_names,
        "fd_threshold_mm":               FD_THRESHOLD,
        "scrubbing_window":              [0, 1],
        "scrubbed_vols_ratio_threshold": SCRUBBED_VOLS_RATIO_THRESHOLD,
        "smoothing_fwhm_mm":             SMOOTHING_FWHM,
        "n_vols_removed_start":          NUM_VOLS_TO_REMOVE,
        "n_acompcor":                    3 if use_gsr else N_ACOMPCOR,
        "motion_filter":                 "notch_0.35Hz_Fair2020",
        "standardize":                   STANDARDIZE,
        "high_pass_hz":                  HIGH_PASS,
        "low_pass_hz":                   LOW_PASS,
        "detrend":                       DETREND,
    }

    config_path = os.path.join(output_dir, "extraction_config.json")
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"Extraction config -> {config_path}")


# =========================================================================
# --- Orchestrator --------------------------------------------------------
# =========================================================================

def create_time_series(
    runs: List[BoldRun],
    output_dir: str,
    atlases: List[Callable[[], AtlasSpec]] = DEFAULT_ATLASES,
    use_gsr: bool = False,
    n_workers: int = N_WORKERS,
    config: str = "lean",
) -> None:
    """Extract timeseries for every run in `runs` across all `atlases`.

    Parameters
    ----------
    runs        : list of BoldRun objects (build manually or via discover_files)
    output_dir  : directory where CSV files and scrubbing report are written
    atlases     : list of atlas factory callables
    use_gsr     : whether to include global signal as a confound (lean/global configs)
    n_workers   : number of parallel processes (set to 1 to run sequentially)
    config      : one of lean | anatomical | global | strict
                  lean       — 12 motion + 6 aCompCor (18 regressors)
                  anatomical — 12 motion + 5 WM + 5 CSF CompCor (22 regressors)
                  global     — 12 motion + 6 aCompCor + GSR + GSR-deriv (20 regressors)
                  strict     — 24 Friston motion + 10 WM + 10 CSF CompCor (44 regressors)
    """
    os.makedirs(output_dir, exist_ok=True)

    if not runs:
        print("No runs provided.")
        return

    # Fresh start — wipe all CSV and report files so stale data never contaminates
    # results when parameters change (atlases, FD threshold, config, etc.)
    existing = [f for f in os.listdir(output_dir)
                if f.endswith(".csv") or f.endswith(".json")]
    if existing:
        print(f"Clearing {len(existing)} existing file(s) from {output_dir} ...")
        for f in existing:
            try:
                os.remove(os.path.join(output_dir, f))
            except OSError:
                pass

    _write_extraction_config(output_dir, runs, use_gsr, atlases)

    print(f"Found {len(runs)} BOLD run(s).  Output -> {output_dir}")
    print(f"Config: {config}  |  GSR: {use_gsr}")
    print(f"Atlases: {[fn().__class__.__name__ for fn in atlases]}")
    print(f"Workers: {n_workers}\n")

    args_list = [(run, output_dir, use_gsr, atlases, config) for run in runs]

    if n_workers == 1:
        all_stats = [_process_one_run(a) for a in args_list]
    else:
        all_stats = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_run = {pool.submit(_process_one_run, a): a[0] for a in args_list}
            for future in as_completed(future_to_run):
                run = future_to_run[future]
                try:
                    all_stats.append(future.result())
                except Exception as exc:
                    print(f"[ERROR] {run.subject}/{run.session}: {exc}")
                    all_stats.append(None)

    # Save scrubbing report
    valid = [s for s in all_stats if s is not None]
    if valid:
        report_df   = pd.DataFrame(valid)
        report_path = os.path.join(output_dir, SCRUB_REPORT_CSV)
        try:
            report_df.to_csv(report_path, index=False)
        except PermissionError:
            alt_path = report_path.replace(".csv", "_new.csv")
            report_df.to_csv(alt_path, index=False)
            print(f"[WARNING] Could not overwrite {SCRUB_REPORT_CSV} (file locked). Saved to {os.path.basename(alt_path)}")
            report_path = alt_path
        passed = (~report_df["high_motion_skip"]).sum()
        print(f"\nScrubbing report -> {report_path}")
        print(f"Retained {passed}/{len(report_df)} runs  "
              f"(high-motion skipped: {report_df['high_motion_skip'].sum()})")

        # ── Variance-explained summary table ──────────────────────────────
        if "confound_var_explained" in report_df.columns:
            ve_df = report_df[["subject", "session", "task", "acq", "run",
                                "confound_var_explained", "high_motion_skip"]].copy()
            ve_df = ve_df.sort_values(["subject", "session", "task", "run"])
            ve_df["var_exp_%"] = (ve_df["confound_var_explained"] * 100).round(1)
            ve_df["status"] = ve_df.apply(
                lambda r: "HIGH-MOTION-SKIP" if r["high_motion_skip"]
                          else ("⚠ OVER-CORRECTION?" if r["var_exp_%"] > 70 else "OK"),
                axis=1,
            )

            col_w = {"subject": 10, "session": 8, "task": 22, "acq": 12,
                     "run": 7, "var_exp_%": 10, "status": 18}
            header = (f"{'subject':<{col_w['subject']}}  {'session':<{col_w['session']}}"
                      f"  {'task':<{col_w['task']}}  {'acq':<{col_w['acq']}}"
                      f"  {'run':<{col_w['run']}}  {'var_exp_%':>{col_w['var_exp_%']}}"
                      f"  {'status':<{col_w['status']}}")
            sep = "-" * len(header)

            print(f"\n{'='*len(header)}")
            print("  VARIANCE EXPLAINED BY CONFOUNDS  (max across atlases per run)")
            print(f"{'='*len(header)}")
            print(header)
            print(sep)
            for _, row in ve_df.iterrows():
                flag = "  <-- OVER-CORRECTION?" if row["var_exp_%"] > 70 else ""
                print(
                    f"{row['subject']:<{col_w['subject']}}  {row['session']:<{col_w['session']}}"
                    f"  {row['task']:<{col_w['task']}}  {str(row['acq']):<{col_w['acq']}}"
                    f"  {str(row['run']):<{col_w['run']}}  {row['var_exp_%']:>{col_w['var_exp_%']}.1f}"
                    f"  {row['status']:<{col_w['status']}}{flag}"
                )
            print(sep)
            mean_ve = ve_df["var_exp_%"].mean()
            max_ve  = ve_df["var_exp_%"].max()
            n_over  = (ve_df["var_exp_%"] > 70).sum()
            print(f"  Mean: {mean_ve:.1f}%   Max: {max_ve:.1f}%   Runs >70%: {n_over}/{len(ve_df)}")
            print(f"{'='*len(header)}\n")

        # ── QC-FC proxy: Pearson r(mean_FD, GCOR) across all retained runs ──
        # Interpretation: should be near 0.  Large positive value means high-
        # motion runs also have high global coupling → residual motion artifact.
        if "gcor_schaefer" in report_df.columns:
            qcfc_df = report_df[["fd_mean_filtered", "gcor_schaefer", "dvars_post"]].dropna()
            if len(qcfc_df) >= 3:
                from scipy.stats import pearsonr as _pearsonr
                qcfc_r, qcfc_p = _pearsonr(qcfc_df["fd_mean_filtered"],
                                            qcfc_df["gcor_schaefer"])
                dvars_r, _ = _pearsonr(qcfc_df["fd_mean_filtered"],
                                       qcfc_df["dvars_post"])
                n_art = int(((qcfc_df["gcor_schaefer"] > 0.3) &
                             (qcfc_df["dvars_post"]    > 1.5)).sum())
                flag = "  [motion artifact suspected]" if qcfc_r > 0.3 else ""
                print(f"QC-FC proxy  ({len(qcfc_df)} runs):  "
                      f"r(FD, GCOR)={qcfc_r:+.3f} p={qcfc_p:.3f}  "
                      f"r(FD, DVARS)={dvars_r:+.3f}  "
                      f"runs with GCOR>0.3+DVARS>1.5: {n_art}{flag}\n")


# =========================================================================
# --- Entry point ---------------------------------------------------------
# =========================================================================

if __name__ == "__main__":
    import argparse as _ap
    from discover_psilo_study import discover_psilo_runs, STUDY_ROOT as _PSILO_ROOT

    _CONFIG_DESC = {
        "lean":       "18 regressors  (12 motion + 6 aCompCor combined WM/CSF)",
        "anatomical": "22 regressors  (12 motion + 5 WM CompCor + 5 CSF CompCor)",
        "global":     "20 regressors  (12 motion + 6 aCompCor + GSR + GSR-deriv)",
        "strict":     "44 regressors  (24 Friston motion + 10 WM + 10 CSF CompCor)",
    }

    _parser = _ap.ArgumentParser(description="fMRI timeseries extractor — psilocybin study")
    _parser.add_argument(
        "--config",
        choices=list(_CONFIG_DESC.keys()),
        default="lean",
        help=(
            "Denoising config:\n"
            "  lean       — 12 motion + 6 aCompCor (18 regressors)\n"
            "  anatomical — 12 motion + 5 WM + 5 CSF CompCor (22 regressors)\n"
            "  global     — 12 motion + 6 aCompCor + GSR + deriv (20 regressors)\n"
            "  strict     — 24 Friston motion + 10 WM + 10 CSF CompCor (44 regressors)"
        ),
    )
    _parser.add_argument("--workers",   type=int, default=N_WORKERS)
    _parser.add_argument("--subjects",  nargs="*", default=None,
                         help="Restrict to subject IDs e.g. sub-001 sub-002")
    _parser.add_argument("--rest-only", action="store_true",
                         help="Only process rest + restchecktr2 runs")
    _parser.add_argument("--test",      action="store_true",
                         help="Process only 1 run (smoke test)")
    _pargs = _parser.parse_args()

    _out_dir  = TS_CONFIG_DIRS[_pargs.config]
    _use_gsr  = (_pargs.config == "global")

    print(f"\n=== fMRI Timeseries Extractor ===")
    print(f"  Study : {_PSILO_ROOT}")
    print(f"  Config: {_pargs.config}  —  {_CONFIG_DESC[_pargs.config]}")
    print(f"  Output: {_out_dir}\n")

    # Use the psylocibin-specific discovery (handles restchecktr2, music, glass, etc.)
    _runs = discover_psilo_runs(_PSILO_ROOT, subjects=_pargs.subjects)
    if _pargs.rest_only:
        _runs = [r for r in _runs if "rest" in r.task]
        print(f"Rest-only filter: {len(_runs)} runs kept")
    if _pargs.test:
        _runs = _runs[:1]
        print(f"Test mode: 1 run only")

    create_time_series(
        _runs, _out_dir,
        atlases=DEFAULT_ATLASES,
        use_gsr=_use_gsr,
        n_workers=_pargs.workers,
        config=_pargs.config,
    )
