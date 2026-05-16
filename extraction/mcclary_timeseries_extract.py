#!/usr/bin/env python3
"""
McClary / Gibbs HPC — standalone timeseries extractor.

Everything in one file: atlas definitions, QC helpers, extraction engine,
and the McClary-specific run-discovery logic. No other project files needed.

Parameters (FWHM, FD threshold, scrubbing, notch filter, CompCor counts, etc.)
are defined in the PARAMETERS block near the top — edit there only.

Usage
-----
    python mcclary_timeseries_extract.py ses-2 --mode rest --denoising both

    # full subject list
    python mcclary_timeseries_extract.py ses-2 --mode rest --denoising both \\
        --subjects 010 024 037 096 125 161 163 175 330 433 505 582 590 632 \\
                   644 659 672 693 695 696 739 865 917 971 987 995 1014 1031 1186

    # task mode (largest non-rest raw BOLD per subject → derivatives)
    python mcclary_timeseries_extract.py ses-1 --mode task --denoising global

    # smoke test (1 run only)
    python mcclary_timeseries_extract.py ses-1 --mode rest --denoising both \\
        --subjects 917 --test

Denoising configs
-----------------
    global     — 12 motion + 6 aCompCor + GSR + GSR-deriv         (20 regressors)
    anatomical — 12 motion + 5 WM CompCor + 5 CSF CompCor          (22 regressors)
    both       — run global then anatomical (separate output trees)

Outputs land in:
    <data-dir>/results_ts_mcclary_<session>/timeseries_global/
    <data-dir>/results_ts_mcclary_<session>/timeseries_anatomical/

Dependencies: nilearn, nibabel, scipy, pandas, numpy  (no nipype / SPM needed)
"""
from __future__ import annotations

# ─── standard library ───────────────────────────────────────────────────────
import argparse
import datetime
import glob
import json
import os
import re
import sys
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# ─── third-party ────────────────────────────────────────────────────────────
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import index_img, resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from scipy.signal import filtfilt, iirnotch


# =============================================================================
# PARAMETERS — edit only this block when moving to a new machine / study
# =============================================================================

DATA_DIR      = "/gpfs/gibbs/pi/levy_ifat/Amir/"   # study root (contains BIDS/)
NILEARN_CACHE = "/home/ma2565/palmer_scratch/nilearn_cache"

SMOOTHING_FWHM: float      = 6       # mm FWHM spatial smoothing
HIGH_PASS: float            = 0.01   # Hz  (0.01 = 100 s period)
LOW_PASS: Optional[float]   = None   # Hz  (None = no low-pass)
FD_THRESHOLD: float         = 0.40   # mm  notch-filtered FD scrubbing threshold
N_ACOMPCOR: int             = 6      # aCompCor components for lean / global configs
N_WORKERS: int              = 4      # parallel worker processes

STANDARDIZE: str            = "zscore_sample"
DETREND: bool               = True
NUM_VOLS_TO_REMOVE: int     = 0
DO_SCRUBBING: bool          = True
SCRUBBED_VOLS_RATIO_THRESHOLD: float = 0.30
BACKGROUND_LABEL_VALUE: int = 0
SCRUB_REPORT_CSV: str       = "scrubbing_report_filtered.csv"

# Friston 24-parameter motion model (strict config)
_MOTION_RAW    = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]
_MOTION_DERIV  = [f + "_derivative1"          for f in _MOTION_RAW]
_MOTION_POW2   = [f + "_power2"               for f in _MOTION_RAW]
_MOTION_D_POW2 = [f + "_derivative1_power2"   for f in _MOTION_RAW]
MOTION_COLS_24 = _MOTION_RAW + _MOTION_DERIV + _MOTION_POW2 + _MOTION_D_POW2

RESEARCH_N_WM  = 10   # WM  CompCor components (strict config)
RESEARCH_N_CSF = 10   # CSF CompCor components (strict config)

# =============================================================================
# Atlas / nilearn cache setup
# =============================================================================

_ATLAS_CACHE = os.path.join(NILEARN_CACHE, "tian_s2")
os.makedirs(_ATLAS_CACHE, exist_ok=True)
os.environ.setdefault("NILEARN_DATA", NILEARN_CACHE)


# =============================================================================
# ── SECTION 1: Atlas definitions ─────────────────────────────────────────────
# =============================================================================

@dataclass
class AtlasSpec:
    """Fully picklable atlas descriptor — safe to pass to multiprocessing workers."""
    name:          str
    maps:          str
    index_to_name: Dict[int, str]
    suffix:        str


def map_labels_to_names(masker_labels: list, index_to_name: Dict[int, str]) -> List[str]:
    return [index_to_name.get(int(lbl), f"region_{lbl}") for lbl in masker_labels]


def get_schaefer(n_rois: int = 400) -> AtlasSpec:
    sch    = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    labels = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in sch.labels]
    return AtlasSpec(
        name          = f"schaefer{n_rois}",
        maps          = sch.maps,
        index_to_name = {i + 1: name for i, name in enumerate(labels)},
        suffix        = f"_schaefer{n_rois}_ts.csv",
    )


_TIAN_BASE    = ("https://raw.githubusercontent.com/yetianmed/subcortex/master/"
                 "Group-Parcellation/3T/Subcortex-Only/")
_TIAN_URL_NII = _TIAN_BASE + "Tian_Subcortex_S2_3T_2009cAsym.nii.gz"
_TIAN_URL_TXT = _TIAN_BASE + "Tian_Subcortex_S2_3T_label.txt"


def get_tian_s2() -> Optional[AtlasSpec]:
    """Tian 2020 subcortical Scale-II (32 bilateral ROIs). Downloaded once to cache."""
    nii_path = os.path.join(_ATLAS_CACHE, "Tian_S2_3T_2009cAsym.nii.gz")
    txt_path = os.path.join(_ATLAS_CACHE, "Tian_S2_3T_labels.txt")

    if not os.path.isfile(nii_path):
        try:
            print("  Downloading Tian S2 atlas... ", end="", flush=True)
            urllib.request.urlretrieve(_TIAN_URL_NII, nii_path)
            print("done.", flush=True)
        except Exception as exc:
            print(f"FAILED ({exc}). Skipping Tian S2.", flush=True)
            return None

    if not os.path.isfile(txt_path):
        try:
            urllib.request.urlretrieve(_TIAN_URL_TXT, txt_path)
        except Exception:
            pass

    labels: List[str] = []
    if os.path.isfile(txt_path):
        with open(txt_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    labels.append(line.split("\t")[0] if "\t" in line else line)

    index_to_name = ({i + 1: name for i, name in enumerate(labels)} if labels
                     else {i: str(i) for i in range(1, 33)})

    return AtlasSpec(
        name          = "tian_s2",
        maps          = nii_path,
        index_to_name = index_to_name,
        suffix        = "_tian_s2_ts.csv",
    )


DEFAULT_ATLASES: List[Callable[[], AtlasSpec]] = [
    get_schaefer,   # Schaefer-400, 7 networks
    get_tian_s2,    # Tian subcortical S2, 32 bilateral ROIs
]


# =============================================================================
# ── SECTION 2: QC helpers ────────────────────────────────────────────────────
# =============================================================================

def post_dvars(ts: np.ndarray) -> float:
    """Post-denoising DVARS: mean RMS of temporal derivative (z-scored signal)."""
    if ts.shape[0] < 2:
        return float("nan")
    ts_std = ts / (ts.std(axis=0, keepdims=True) + 1e-12)
    diff   = np.diff(ts_std, axis=0)
    return float(np.mean(np.sqrt(np.mean(diff ** 2, axis=1))))


def ts_qc_metrics(ts_clean: np.ndarray, ts_raw: Optional[np.ndarray] = None) -> dict:
    """tSNR (from raw) + lag-1 autocorrelation (from cleaned signal)."""
    if ts_clean.shape[0] < 4:
        return {"tsnr": float("nan"), "lag1_ac": float("nan")}

    if ts_raw is not None and ts_raw.shape == ts_clean.shape:
        tsnr = float(np.nanmean(
            np.abs(ts_raw.mean(axis=0)) / (ts_raw.std(axis=0) + 1e-12)
        ))
    else:
        tsnr = float("nan")

    lag1_per_roi = np.array([
        float(np.corrcoef(ts_clean[:-1, j], ts_clean[1:, j])[0, 1])
        for j in range(ts_clean.shape[1])
    ])
    return {"tsnr": tsnr, "lag1_ac": float(np.nanmean(lag1_per_roi))}


def confounds_variance_explained(raw_ts: np.ndarray, confounds: np.ndarray) -> float:
    """Mean R² across ROIs: fraction of raw signal variance explained by confounds."""
    n_rois = raw_ts.shape[1]
    if confounds.shape[1] >= raw_ts.shape[0] - 2:
        return np.nan
    r2_list = []
    for j in range(n_rois):
        y = raw_ts[:, j].astype(np.float64)
        y -= np.mean(y)
        ss = np.sum(y ** 2)
        if ss < 1e-12:
            continue
        X = confounds.astype(np.float64)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            continue
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1.0 - np.sum((y - X @ beta) ** 2) / ss
            r2_list.append(float(np.clip(r2, 0.0, 1.0)))
        except (np.linalg.LinAlgError, ValueError):
            continue
    return float(np.mean(r2_list)) if r2_list else np.nan


def louvain_q(corr: np.ndarray) -> float:
    """Louvain modularity Q on the positive FC matrix."""
    try:
        import networkx as nx
        W = np.maximum(corr.copy(), 0)
        np.fill_diagonal(W, 0)
        G = nx.from_numpy_array(W)
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight="weight")
            return community_louvain.modularity(partition, G, weight="weight")
        except ImportError:
            communities = nx.community.greedy_modularity_communities(G, weight="weight")
            return float(nx.community.modularity(G, communities, weight="weight"))
    except Exception:
        return float("nan")


def fc_network_metrics(df_ts: pd.DataFrame, corr: np.ndarray) -> dict:
    """Within-network and between-network FC for Schaefer-400 7-network labels."""
    cols    = list(df_ts.columns)
    lh_mot  = [i for i, c in enumerate(cols) if "SomMot" in c and "LH" in c]
    rh_mot  = [i for i, c in enumerate(cols) if "SomMot" in c and "RH" in c]
    all_mot = lh_mot + rh_mot

    motor_lr    = (float(np.nanmean(corr[np.ix_(lh_mot, rh_mot)]))
                   if lh_mot and rh_mot else float("nan"))

    if len(all_mot) >= 2:
        within = corr[np.ix_(all_mot, all_mot)].copy()
        np.fill_diagonal(within, np.nan)
        somm_within = float(np.nanmean(within))
    else:
        somm_within = float("nan")

    dmn_idx = [i for i, c in enumerate(cols) if "Default" in c]
    if len(dmn_idx) >= 2:
        blk = corr[np.ix_(dmn_idx, dmn_idx)].copy()
        np.fill_diagonal(blk, np.nan)
        dmn_within = float(np.nanmean(blk))
    else:
        dmn_within = float("nan")

    fpn_idx = [i for i, c in enumerate(cols) if "Cont" in c]
    dmn_fpn = (float(np.nanmean(corr[np.ix_(dmn_idx, fpn_idx)]))
               if dmn_idx and fpn_idx else float("nan"))

    return {"motor_lr": motor_lr, "somm_within": somm_within,
            "dmn_within": dmn_within, "dmn_fpn": dmn_fpn}


# =============================================================================
# ── SECTION 3: Core extraction engine ────────────────────────────────────────
# =============================================================================

class BoldRun:
    """Paths + metadata for one preprocessed BOLD run."""

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
        parts = [f"task-{self.task}"]
        if self.acq:
            parts.append(f"acq-{self.acq}")
        parts.append(self.run)
        return "_".join(parts)

    def __repr__(self):
        return (f"{self.subject} | {self.session} | {self.bids_id}\n"
                f"  BOLD: {self.bold_path}")


# ─── Motion filtering (Fair et al. 2020) ─────────────────────────────────────

def apply_motion_filter(motion_params: pd.DataFrame, tr: float) -> pd.DataFrame:
    """Notch-filter raw motion params to remove respiratory pseudo-motion (~0.35 Hz)."""
    fs      = 1.0 / tr
    nyquist = fs / 2.0
    center  = 0.35   # Hz — mid-range adult breathing

    if center >= nyquist:
        print(f"  [Motion filter] TR={tr:.3f}s -> Nyquist={nyquist:.3f} Hz < "
              f"{center} Hz — notch filter skipped.")
        return motion_params.copy()

    b, a     = iirnotch(center, center / 0.2, fs)
    filtered = motion_params.copy()
    for col in filtered.columns:
        filtered[col] = filtfilt(b, a, motion_params[col].values)
    return filtered


# ─── TR detection ─────────────────────────────────────────────────────────────

def get_tr_seconds(bold_path: str, img: nib.Nifti1Image) -> float:
    zooms = img.header.get_zooms()
    if len(zooms) > 3 and zooms[3] > 0:
        return float(zooms[3])

    folder   = os.path.dirname(bold_path)
    basename = os.path.basename(bold_path)
    prefix   = re.sub(r"_space-.*$", "", basename)

    for _ in range(5):
        json_path = os.path.join(
            folder,
            prefix + "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json",
        )
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


# ─── CompCor column selection ─────────────────────────────────────────────────

def _get_acompcor_cols(conf_path: str, max_n: int = 10) -> List[str]:
    json_path = conf_path.replace(".tsv", ".json")
    if os.path.isfile(json_path):
        with open(json_path) as fh:
            meta = json.load(fh)
        WM_CSF = {"WM", "CSF", "combined"}
        candidates = sorted([
            col for col, info in meta.items()
            if col.startswith("a_comp_cor")
            and isinstance(info, dict)
            and info.get("Retained", False)
            and info.get("Mask", "") in WM_CSF
        ])
        if candidates:
            return candidates[:max_n]
    return [f"a_comp_cor_{i:02d}" for i in range(max_n)]


def get_conf_cols(use_gsr: bool, conf_path: Optional[str] = None,
                  config: str = "lean") -> List[str]:
    """Return confound column list for the chosen denoising config."""
    _ALIAS = {"standard": "lean", "research": "strict"}
    config = _ALIAS.get(config, config)

    _12 = [
        "trans_x","trans_y","trans_z","rot_x","rot_y","rot_z",
        "trans_x_derivative1","trans_y_derivative1","trans_z_derivative1",
        "rot_x_derivative1","rot_y_derivative1","rot_z_derivative1",
    ]

    if config == "strict":
        wm  = [f"w_comp_cor_{i:02d}" for i in range(RESEARCH_N_WM)]
        csf = [f"c_comp_cor_{i:02d}" for i in range(RESEARCH_N_CSF)]
        return MOTION_COLS_24 + wm + csf

    if config == "anatomical":
        wm  = [f"w_comp_cor_{i:02d}" for i in range(5)]
        csf = [f"c_comp_cor_{i:02d}" for i in range(5)]
        return _12 + wm + csf

    if config == "global":
        acompcor = (_get_acompcor_cols(conf_path, N_ACOMPCOR) if conf_path
                    else [f"a_comp_cor_{i:02d}" for i in range(N_ACOMPCOR)])
        return _12 + acompcor + ["global_signal", "global_signal_derivative1"]

    # lean (default)
    n       = 3 if use_gsr else N_ACOMPCOR
    acomp   = (_get_acompcor_cols(conf_path, n) if conf_path
               else [f"a_comp_cor_{i:02d}" for i in range(n)])
    cols    = _12 + acomp
    if use_gsr:
        cols.append("global_signal")
    return cols


# ─── Confound building ────────────────────────────────────────────────────────

def build_confounds(conf_path: str, run: BoldRun, use_gsr: bool,
                    tr: float, config: str = "lean") -> Tuple[pd.DataFrame, dict]:
    if tr <= 0:
        raise ValueError(f"Invalid TR={tr:.4f}s for {run.subject}/{run.session}/{run.run}.")

    df         = pd.read_csv(conf_path, sep="\t")
    df         = df.loc[NUM_VOLS_TO_REMOVE:].reset_index(drop=True)
    total_vols = len(df)

    fd_raw     = df["framewise_displacement"].fillna(0)
    raw_spikes = int((fd_raw > FD_THRESHOLD).sum())

    raw_motion_cols = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]
    filt_motion = apply_motion_filter(df[raw_motion_cols].fillna(0), tr)
    diff        = filt_motion.diff().fillna(0)
    diff[["rot_x","rot_y","rot_z"]] *= 50
    fd_filtered     = diff.abs().sum(axis=1)
    filtered_spikes = int((fd_filtered > FD_THRESHOLD).sum())

    print(f"  FD raw:      mean={fd_raw.mean():.3f}  max={fd_raw.max():.3f}  "
          f"spikes>{FD_THRESHOLD}mm = {raw_spikes}/{total_vols}")
    print(f"  FD filtered: mean={fd_filtered.mean():.3f}  max={fd_filtered.max():.3f}  "
          f"spikes>{FD_THRESHOLD}mm = {filtered_spikes}/{total_vols}  "
          f"(reduction: {raw_spikes - filtered_spikes:+d})")

    available = [c for c in get_conf_cols(use_gsr, conf_path, config) if c in df.columns]
    conf_df   = df[available].fillna(0).copy()

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
                  f"({scrub_fraction:.1%}) — spike regressors skipped, run flagged.")
        elif scrubbed_volumes > 0:
            spike_regs = pd.DataFrame(
                0, index=df.index,
                columns=[f"motion_spike_{i}" for i in range(scrubbed_volumes)],
            )
            for idx, vol in enumerate(sorted(to_scrub)):
                spike_regs.loc[vol, f"motion_spike_{idx}"] = 1
            conf_df = pd.concat([conf_df, spike_regs], axis=1)
            print(f"  Scrubbing:   {scrubbed_volumes}/{total_vols} vols censored "
                  f"({scrub_fraction:.1%})")
        else:
            print("  Scrubbing:   0 volumes flagged — clean run")

    stats = {
        "subject":          run.subject,
        "session":          run.session,
        "task":             run.task,
        "acq":              run.acq or "",
        "run":              run.run,
        "fd_mean_raw":      round(float(fd_raw.mean()),       4),
        "fd_mean_filtered": round(float(fd_filtered.mean()),  4),
        "fd_max_filtered":  round(float(fd_filtered.max()),   4),
        "raw_spikes":       raw_spikes,
        "filtered_spikes":  filtered_spikes,
        "scrubbed_volumes": scrubbed_volumes,
        "scrub_percent":    round(scrub_fraction * 100, 1),
        "total_vols":       total_vols,
        "high_motion_skip": high_motion_skip,
        "gsr_applied":      use_gsr,
    }
    return conf_df, stats


# ─── Per-run worker ───────────────────────────────────────────────────────────

def _process_one_run(args: tuple) -> Optional[dict]:
    run, output_dir, use_gsr, atlas_factories, config = args
    atlas_specs: List[AtlasSpec] = [fn() for fn in atlas_factories
                                    if fn() is not None]
    atlas_specs = [s for s in atlas_specs if s is not None]

    masker_kwargs = dict(
        standardize          = STANDARDIZE,
        smoothing_fwhm       = SMOOTHING_FWHM,
        detrend              = DETREND,
        low_pass             = LOW_PASS,
        high_pass            = HIGH_PASS,
        standardize_confounds= True,
        memory               = NILEARN_CACHE,
        verbose              = 0,
        background_label     = BACKGROUND_LABEL_VALUE,
    )

    expected_paths = [
        os.path.join(output_dir,
                     f"{run.subject}_{run.session}_{run.bids_id}{spec.suffix}")
        for spec in atlas_specs
    ]
    if all(os.path.exists(p) for p in expected_paths):
        print(f"[SKIP already exists] {run.subject} | {run.session} | {run.bids_id}")
        try:
            tr = get_tr_seconds(run.bold_path, nib.load(run.bold_path))
            _, stats = build_confounds(run.confounds_path, run, use_gsr, tr, config)
            stats["confound_var_explained"] = np.nan
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
        conf_arr = (conf_arr - np.nanmean(conf_arr, axis=0)) / (
            np.nanstd(conf_arr, axis=0) + 1e-12
        )

        raw_kwargs = dict(
            standardize=False, detrend=False, low_pass=None, high_pass=None,
            smoothing_fwhm=SMOOTHING_FWHM, standardize_confounds=False,
            memory=NILEARN_CACHE, verbose=0,
            background_label=BACKGROUND_LABEL_VALUE,
        )

        var_explained_max = 0.0
        for spec in atlas_specs:
            atlas_img  = resample_to_img(spec.maps, img, interpolation="nearest",
                                         copy=True)
            masker     = NiftiLabelsMasker(labels_img=atlas_img, t_r=tr,
                                           **masker_kwargs)
            ts         = masker.fit_transform(img, confounds=conf_df)
            col_names  = map_labels_to_names(masker.labels_, spec.index_to_name)
            df_ts      = pd.DataFrame(ts, columns=col_names)

            raw_masker = NiftiLabelsMasker(labels_img=atlas_img, t_r=tr, **raw_kwargs)
            raw_ts     = raw_masker.fit_transform(img, confounds=None)
            var_exp    = confounds_variance_explained(raw_ts, conf_arr)
            if not np.isnan(var_exp):
                var_explained_max = max(var_explained_max, var_exp)
                print(f"  [Var explained] {spec.name}: {var_exp:.1%} by confounds",
                      end="  [OVER-CORRECTION?]\n" if var_exp > 0.7 else "\n")

            corr  = np.corrcoef(df_ts.T)
            n     = corr.shape[0]
            gcor  = float(np.nanmean(corr[~np.eye(n, dtype=bool)]))
            dvars = post_dvars(ts)
            tsqc  = ts_qc_metrics(ts, raw_ts)

            is_cortical = "schaefer" in spec.name.lower()
            if is_cortical:
                q_val = louvain_q(corr)
                nm    = fc_network_metrics(df_ts, corr)
                if "gcor_schaefer" not in stats:
                    stats["gcor_schaefer"] = round(gcor, 4)
                    stats["dvars_post"]    = round(dvars, 3)
                line2 = (
                    f"    tSNR={tsqc['tsnr']:.1f}"
                    + ("  [LOW]" if tsqc["tsnr"] < 20 else "")
                    + f"  AC={tsqc['lag1_ac']:.3f}"
                    + ("  [OVER-DENOISE?]" if tsqc["lag1_ac"] < 0.25 else "")
                    + f"  Q={q_val:.3f}"
                    + f"  MotLR={nm['motor_lr']:+.3f}"
                    + f"  SomMot={nm['somm_within']:+.3f}"
                    + f"  DMN={nm['dmn_within']:+.3f}"
                    + f"  DMN-FPN={nm['dmn_fpn']:+.3f}"
                )
            else:
                line2 = f"    tSNR={tsqc['tsnr']:.1f}  AC={tsqc['lag1_ac']:.3f}"

            gcor_flag = ""
            if gcor > 0.3:
                gcor_flag = ("  <- GCOR HIGH + DVARS HIGH: likely artifact"
                             if dvars > 1.5
                             else "  <- GCOR HIGH, DVARS normal: likely biological")

            out_name = (f"{run.subject}_{run.session}_{run.bids_id}{spec.suffix}")
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


def _write_extraction_config(output_dir: str, runs: List[BoldRun],
                              use_gsr: bool,
                              atlas_factories: List[Callable]) -> None:
    atlas_names: List[str] = []
    for fn in atlas_factories:
        try:
            spec = fn()
            atlas_names.append(spec.name if spec else str(fn))
        except Exception:
            atlas_names.append(getattr(fn, "__name__", str(fn)))

    cfg = {
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
        "n_acompcor":                    N_ACOMPCOR,
        "motion_filter":                 "notch_0.35Hz_Fair2020",
        "standardize":                   STANDARDIZE,
        "high_pass_hz":                  HIGH_PASS,
        "low_pass_hz":                   LOW_PASS,
        "detrend":                       DETREND,
    }
    config_path = os.path.join(output_dir, "extraction_config.json")
    with open(config_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Extraction config -> {config_path}")


def create_time_series(
    runs: List[BoldRun],
    output_dir: str,
    atlases: List[Callable[[], AtlasSpec]] = DEFAULT_ATLASES,
    use_gsr: bool = False,
    n_workers: int = N_WORKERS,
    config: str = "lean",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if not runs:
        print("No runs provided.")
        return

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

    valid = [s for s in all_stats if s is not None]
    if not valid:
        return

    report_df   = pd.DataFrame(valid)
    report_path = os.path.join(output_dir, SCRUB_REPORT_CSV)
    try:
        report_df.to_csv(report_path, index=False)
    except PermissionError:
        alt = report_path.replace(".csv", "_new.csv")
        report_df.to_csv(alt, index=False)
        report_path = alt

    passed = (~report_df["high_motion_skip"]).sum()
    print(f"\nScrubbing report -> {report_path}")
    print(f"Retained {passed}/{len(report_df)} runs  "
          f"(high-motion skipped: {report_df['high_motion_skip'].sum()})")

    if "confound_var_explained" in report_df.columns:
        ve_df = report_df[["subject","session","task","acq","run",
                            "confound_var_explained","high_motion_skip"]].copy()
        ve_df["var_exp_%"] = (ve_df["confound_var_explained"] * 100).round(1)
        ve_df["status"]    = ve_df.apply(
            lambda r: ("HIGH-MOTION-SKIP" if r["high_motion_skip"]
                       else ("OVER-CORRECTION?" if r["var_exp_%"] > 70 else "OK")),
            axis=1,
        )
        print(f"\nMean var-explained: {ve_df['var_exp_%'].mean():.1f}%  "
              f"Max: {ve_df['var_exp_%'].max():.1f}%  "
              f"Runs >70%: {(ve_df['var_exp_%'] > 70).sum()}/{len(ve_df)}\n")

    if "gcor_schaefer" in report_df.columns:
        qcfc_df = report_df[["fd_mean_filtered","gcor_schaefer","dvars_post"]].dropna()
        if len(qcfc_df) >= 3:
            from scipy.stats import pearsonr
            r, p = pearsonr(qcfc_df["fd_mean_filtered"], qcfc_df["gcor_schaefer"])
            flag = "  [motion artifact suspected]" if r > 0.3 else ""
            print(f"QC-FC proxy ({len(qcfc_df)} runs):  r(FD,GCOR)={r:+.3f} p={p:.3f}{flag}\n")


# =============================================================================
# ── SECTION 4: McClary run-discovery ─────────────────────────────────────────
# =============================================================================

def _norm_subj(sid: str) -> str:
    sid = sid.strip()
    return sid if sid.startswith("sub-") else f"sub-{sid}"


def _parse_task_acq_run(fname: str) -> Tuple[str, Optional[str], str]:
    task_m = re.search(r"task-([a-zA-Z0-9]+)", fname)
    if not task_m:
        return "", None, "run-1"
    task  = task_m.group(1)
    acq_m = re.search(r"acq-([a-zA-Z0-9]+)", fname)
    run_m = re.search(r"run-([a-zA-Z0-9]+)", fname)
    return (task,
            acq_m.group(1) if acq_m else None,
            f"run-{run_m.group(1)}" if run_m else "run-1")


def discover_rest_runs(subject_ids: List[str], session: str,
                       data_dir: str) -> List[BoldRun]:
    """All task-rest fMRIPrep preproc BOLD + confounds for listed subjects."""
    runs: List[BoldRun] = []
    deriv = os.path.join(data_dir, "BIDS", "derivatives")
    for sid in subject_ids:
        subj     = _norm_subj(sid)
        func_dir = os.path.join(deriv, subj, session, "func")
        if not os.path.isdir(func_dir):
            print(f"  [SKIP] no derivatives func dir: {func_dir}")
            continue
        for fname in sorted(os.listdir(func_dir)):
            if "task-rest" not in fname:
                continue
            if not (fname.endswith(".nii.gz") and "preproc_bold" in fname
                    and "MNI" in fname):
                continue
            bold_path = os.path.join(func_dir, fname)
            stem      = re.sub(r"_space-.*$", "", fname)
            conf_path = os.path.join(func_dir, stem + "_desc-confounds_timeseries.tsv")
            if not os.path.isfile(conf_path):
                print(f"  [SKIP] no confounds for {fname}")
                continue
            task, acq, run_key = _parse_task_acq_run(fname)
            task = task or "rest"
            runs.append(BoldRun(subject=subj, session=session, task=task,
                                run=run_key, bold_path=bold_path,
                                confounds_path=conf_path, acq=acq))
    return runs


def discover_task_runs(subject_ids: List[str], session: str,
                       data_dir: str) -> List[BoldRun]:
    """One BoldRun per subject: largest non-rest raw BOLD (first-level convention)."""
    runs: List[BoldRun] = []
    for sid in subject_ids:
        subj = _norm_subj(sid)
        sid_num = subj.replace("sub-", "")

        bids_func_dir = os.path.join(data_dir, "BIDS", subj, session, "func")
        pattern       = os.path.join(
            bids_func_dir, f"{subj}_{session}_task-*_bold.nii.gz"
        )
        bold_files = glob.glob(pattern)
        if not bold_files:
            print(f"  [SKIP] no raw BIDS bold: {bids_func_dir}")
            continue

        candidates = []
        for bf in bold_files:
            base = os.path.basename(bf)
            try:
                task_label = base.split("task-")[1].split("_")[0]
            except IndexError:
                continue
            candidates.append((os.path.getsize(bf), task_label, bf))

        non_rest = [c for c in candidates if not c[1].lower().startswith("rest")]
        if not non_rest:
            print(f"  [SKIP] only rest in raw BIDS for {subj} {session}")
            continue

        _size, task_label, chosen_raw = max(non_rest, key=lambda x: x[0])

        deriv_func_dir = os.path.join(
            data_dir, "BIDS", "derivatives", subj, session, "func"
        )
        raw_base    = os.path.basename(chosen_raw)
        bids_prefix = raw_base[: -len("_bold.nii.gz")]

        func = os.path.join(
            deriv_func_dir,
            f"{bids_prefix}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        )
        conf = os.path.join(
            deriv_func_dir, f"{bids_prefix}_desc-confounds_timeseries.tsv"
        )

        if not os.path.isfile(func):
            alt = sorted(glob.glob(os.path.join(
                deriv_func_dir,
                f"{subj}_{session}_task-{task_label}_*_space-*_desc-preproc_bold.nii.gz",
            )))
            if len(alt) == 1:
                func = alt[0]
                stem = re.sub(r"_space-.*$", "", os.path.basename(func))
                conf = os.path.join(deriv_func_dir,
                                    f"{stem}_desc-confounds_timeseries.tsv")
            else:
                print(f"  [SKIP] derivatives not found for {subj} {session} "
                      f"task-{task_label}")
                continue

        for path, label in [(func, "func"), (conf, "confounds")]:
            if not os.path.isfile(path):
                print(f"  [SKIP] missing {label}: {path}")
                break
        else:
            task, acq, run_key = _parse_task_acq_run(os.path.basename(func))
            task = task or task_label
            runs.append(BoldRun(subject=subj, session=session, task=task,
                                run=run_key, bold_path=func,
                                confounds_path=conf, acq=acq))
    return runs


# =============================================================================
# ── SECTION 5: CLI entry point ────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", choices=["ses-1","ses-2","ses-3"])
    ap.add_argument("--mode", choices=["rest","task"], default="rest")
    ap.add_argument("--denoising", choices=["global","anatomical","both"],
                    default="both")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default="",
                    help="Output root (default: <data-dir>/results_ts_mcclary_<session>/)")
    ap.add_argument("--nilearn-cache", default=NILEARN_CACHE)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Subject IDs (with or without 'sub-' prefix)")
    ap.add_argument("--test", action="store_true",
                    help="Process only 1 run (smoke test)")
    args = ap.parse_args()

    # ── Subject list ──────────────────────────────────────────────────────────
    SUBJECT_LIST = ["917"]
    # SUBJECT_LIST = [
    #     "010","024","037","096","125","161","163","175","330","433","505","582",
    #     "590","632","644","659","672","693","695","696","739","865","917","971",
    #     "987","995","1014","1031","1186",
    # ]
    subject_ids = args.subjects if args.subjects is not None else SUBJECT_LIST

    # ── Paths ──────────────────────────────────────────────────────────────────
    global NILEARN_CACHE
    NILEARN_CACHE = os.path.expanduser(args.nilearn_cache)
    os.environ["NILEARN_DATA"] = NILEARN_CACHE
    os.makedirs(NILEARN_CACHE, exist_ok=True)

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    out_root = (os.path.abspath(os.path.expanduser(args.out_dir))
                if args.out_dir.strip()
                else os.path.join(data_dir, f"results_ts_mcclary_{args.session}"))

    derivatives_root = os.path.join(data_dir, "BIDS", "derivatives")
    if not os.path.isdir(derivatives_root):
        raise SystemExit(f"Derivatives root not found: {derivatives_root}")

    # ── Discover runs ─────────────────────────────────────────────────────────
    if args.mode == "rest":
        runs = discover_rest_runs(subject_ids, args.session, data_dir)
    else:
        runs = discover_task_runs(subject_ids, args.session, data_dir)

    if args.test and runs:
        runs = runs[:1]
        print(f"Test mode: using 1 run -> {runs[0]}")

    if not runs:
        raise SystemExit("No runs found. Check --subjects, session, and file layout.")

    print(f"\n=== McClary timeseries extraction ===")
    print(f"  Session     : {args.session}")
    print(f"  Mode        : {args.mode}")
    print(f"  Runs found  : {len(runs)}")
    print(f"  Output root : {out_root}")
    print(f"  Derivatives : {derivatives_root}")
    print(f"  Nilearn     : {NILEARN_CACHE}")
    print(f"  Params      : FWHM={SMOOTHING_FWHM}mm  HP={HIGH_PASS}Hz  "
          f"FD={FD_THRESHOLD}mm  (notch + scrubbing)\n")

    # ── Run each requested denoising config ───────────────────────────────────
    jobs: List[Tuple[str, str, bool]] = []
    if args.denoising in ("global", "both"):
        jobs.append(("global",     os.path.join(out_root, "timeseries_global"),     True))
    if args.denoising in ("anatomical", "both"):
        jobs.append(("anatomical", os.path.join(out_root, "timeseries_anatomical"), False))

    for cfg_name, out_dir, use_gsr in jobs:
        print(f"--- Denoising: {cfg_name}  ->  {out_dir} ---")
        create_time_series(
            runs, out_dir,
            atlases   = DEFAULT_ATLASES,
            use_gsr   = use_gsr,
            n_workers = args.workers,
            config    = cfg_name,
        )


if __name__ == "__main__":
    main()
