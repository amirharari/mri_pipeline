"""
QA 04: Denoising Configuration Comparison.

Evaluates four denoising strategies on every BOLD run:

  Config 1: Lean              : 12 motion (6+deriv) + 6 aCompCor WM/CSF      [18 regressors]
  Config 2: Anatomical        : 12 motion + 5 WM CompCor + 5 CSF CompCor     [22 regressors]
  Config 3: Global            : 12 motion + 6 aCompCor + global_signal+deriv [20 regressors]
  Config 4: Research Standard : 24 motion (Friston) + 10 WM + 10 CSF CompCor [44 regressors]

Per-run quality metrics (Ciric et al. 2017 benchmarks):
  QC-FC         : mean |r| between each denoised parcel timeseries and the FD timeseries.
                  Lower = less residual motion contamination.
  DM-FC         : Pearson r between FC edge values and inter-parcel Euclidean distances.
                  Motion inflates short-range connections (negative DM-FC).
                  Good cleaning drives this toward 0.
  DoF Lost      : n_regressors removed (+ scrubbed volumes in full pipeline).
                  Cost of denoising.
  Modularity    : mean within-network FC minus mean between-network FC (Schaefer-400 7 nets).
                  Higher = network structure better preserved.

Cross-run gold-standard (computed after all runs):
  Cross-run QC-FC : for each of 79,800 edges, r(FC across runs, Mean FD across runs).
                    Reports mean |r| and % significant edges.
  Cross-run DM-FC : r(|QC-FC|, inter-parcel distance) — is motion distance-dependent?

Atlas:
  Schaefer-400 (cortical 7-net)  : GCOR + QC-FC + DM-FC + Modularity
  Tian S2 (subcortical 32 ROIs)  : combined into GCOR
  Harvard-Oxford cortical        : A1 (Heschl) for V1/A1 negative control

Usage:
  python 04_denoising_effect.py --config lean     --test              # 1 run, smoke test
  python 04_denoising_effect.py --config all      --workers 4        # all 4 configs
  python 04_denoising_effect.py --config research --rest-only        # Config 4 rest only
"""

import argparse
import gc
import json
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

_QA_DIR        = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT     = os.path.dirname(_QA_DIR)
_EXTRACTOR_DIR = os.path.join(_REPO_ROOT, "extraction")
sys.path.insert(0, _EXTRACTOR_DIR)
sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy import stats as spstats
from nilearn import signal as nls
from nilearn import datasets as nlds
from nilearn.maskers import NiftiLabelsMasker
import networkx as nx
import community as community_louvain
from fmri_timeseries_extractor import get_tr_seconds, apply_motion_filter
from config import (  # noqa: E402
    FMRIPREP_ROOT,
    HIGH_PASS,
    LOW_PASS,
    SMOOTHING_FWHM,
    FD_THRESHOLD      as FD_SCRUB_THRESHOLD,
    N_ACOMPCOR,
    QA_DENOISING_DIR,
    QA_DENOISING_REST_DIR,
    ensure_nilearn_cache,
)

ensure_nilearn_cache()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths — derived from config.py (edit paths there, not here)
# ---------------------------------------------------------------------------
BASE_OUT_DIR  = QA_DENOISING_DIR
ATLAS_CACHE   = os.path.join(_QA_DIR, "atlases")
os.makedirs(ATLAS_CACHE, exist_ok=True)

MOTION_COLS_RAW   = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
MOTION_DERIV_COLS = ["trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
                     "rot_x_derivative1",   "rot_y_derivative1",   "rot_z_derivative1"]
MOTION_COLS       = MOTION_COLS_RAW + MOTION_DERIV_COLS  # 12 total — matches main pipeline

# Friston 24-parameter model: raw + deriv + squared + squared-deriv
MOTION_POWER2_COLS = ["trans_x_power2", "trans_y_power2", "trans_z_power2",
                      "rot_x_power2",   "rot_y_power2",   "rot_z_power2"]
MOTION_DERIV_POWER2_COLS = ["trans_x_derivative1_power2", "trans_y_derivative1_power2",
                             "trans_z_derivative1_power2", "rot_x_derivative1_power2",
                             "rot_y_derivative1_power2",   "rot_z_derivative1_power2"]
MOTION_COLS_24 = MOTION_COLS + MOTION_POWER2_COLS + MOTION_DERIV_POWER2_COLS  # 24 total

# Separate WM + CSF CompCor for Config 4 (10 each = 20 total noise regressors).
# Using separate masks is much more effective than combined a_comp_cor:
# c_comp_cor components explain 4x more variance (cardiac/respiratory in CSF)
# than combined a_comp_cor (which is dominated by lower-variance WM patterns).
RESEARCH_N_WM  = 10   # top-N w_comp_cor components (WM mask)
RESEARCH_N_CSF = 10   # top-N c_comp_cor components (CSF mask)

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
CONFIGS = {
    "lean": {
        "label":  "Config 1: Lean (12 motion + 6 aCompCor WM/CSF)",
        "folder": "config1_lean",
    },
    "anatomical": {
        "label":  "Config 2: Anatomical (12 motion + 5 WM + 5 CSF CompCor)",
        "folder": "config2_anatomical",
    },
    "global": {
        "label":  "Config 3: Global (12 motion + 6 aCompCor + GSR + GSR-deriv)",
        "folder": "config3_global",
    },
    "research": {
        "label":  "Config 4: Research Standard (24 motion Friston + 10 WM + 10 CSF CompCor)",
        "folder": "config4_research",
    },
}

# ---------------------------------------------------------------------------
# Atlas loading (once per process) — shared factories from atlases.py
# ---------------------------------------------------------------------------
from atlases import get_schaefer, get_tian_s2  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _SCHAEFER400 = nlds.fetch_atlas_schaefer_2018(n_rois=400)

SCHAEFER400_MAPS = _SCHAEFER400.maps
_SCH_LABELS      = [l.decode() if isinstance(l, bytes) else l
                    for l in _SCHAEFER400.labels]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _HO = nlds.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
_HO_LABELS = _HO.labels

_HO_HESCHL_IDX  = [i for i, l in enumerate(_HO_LABELS) if "Heschl" in l]
if not _HO_HESCHL_IDX:
    _HO_HESCHL_IDX = [i for i, l in enumerate(_HO_LABELS)
                      if "Planum Temporale" in l or "Superior Temporal Gyrus, posterior" in l]
_HO_HESCHL_COLS = [i - 1 for i in _HO_HESCHL_IDX if i > 0]

_tian_spec      = get_tian_s2()
_TIAN_NII       = _tian_spec.maps if _tian_spec else None
_TIAN_LABELS    = list(_tian_spec.index_to_name.values()) if _tian_spec else []
TIAN_AVAILABLE  = _TIAN_NII is not None and os.path.isfile(_TIAN_NII)

# ---------------------------------------------------------------------------
# Sanity-check parcel indices (Schaefer-400, 0-based columns)
# ---------------------------------------------------------------------------
_LH_MOTOR_COLS = [i for i, l in enumerate(_SCH_LABELS) if "LH_SomMot" in l]
_RH_MOTOR_COLS = [i for i, l in enumerate(_SCH_LABELS) if "RH_SomMot" in l]
_DMN_COLS      = [i for i, l in enumerate(_SCH_LABELS) if "Default"  in l]
_DORSATTN_COLS = [i for i, l in enumerate(_SCH_LABELS) if "DorsAttn" in l]
_VIS_COLS      = [i for i, l in enumerate(_SCH_LABELS)
                  if "_Vis_" in l or l.endswith("_Vis")]

# ---------------------------------------------------------------------------
# Parcel geometry — for QC-FC / DM-FC / Modularity (Ciric et al. 2017)
# ---------------------------------------------------------------------------

def _parcel_centroids(atlas_path: str) -> np.ndarray:
    """MNI centroid for each integer label in the atlas (shape: n_parcels x 3)."""
    img_  = nib.load(atlas_path)
    data_ = img_.get_fdata()
    aff_  = img_.affine
    labels_present = sorted(np.unique(data_[data_ > 0]).astype(int))
    coords = []
    for lbl in labels_present:
        vox = np.argwhere(data_ == lbl)
        mni = nib.affines.apply_affine(aff_, vox.mean(axis=0))
        coords.append(mni)
    return np.array(coords)   # (n_parcels, 3)


_PARCEL_COORDS = _parcel_centroids(SCHAEFER400_MAPS)   # (400, 3) MNI mm
_UTI           = np.triu_indices(400, k=1)             # upper triangle, (2, 79800)
_DISTS_VEC     = squareform(pdist(_PARCEL_COORDS))[_UTI]  # (79800,) Euclidean mm

# 7-network assignment for each parcel (0=Vis … 6=Default, -1=unknown)
_NETWORK_NAMES = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def _net_id(label: str) -> int:
    # Label format: "7Networks_LH_Vis_1" or "LH_Vis_1"
    # Network name is any part that matches _NETWORK_NAMES
    for part in label.split("_"):
        if part in _NETWORK_NAMES:
            return _NETWORK_NAMES.index(part)
    return -1


_NETWORK_IDX  = np.array([_net_id(l) for l in _SCH_LABELS])  # (400,)
_VALID_NET_MASK = _NETWORK_IDX >= 0


# ---------------------------------------------------------------------------
# Print atlas summary
# ---------------------------------------------------------------------------
print("=" * 66, flush=True)
print("  Atlas summary", flush=True)
print("=" * 66, flush=True)
print(f"  Schaefer-400 parcels : {len(_SCH_LABELS)}", flush=True)
print(f"  Tian S2              : {'32 ROIs' if TIAN_AVAILABLE else 'NOT AVAILABLE'}", flush=True)
print(f"  HO Heschl cols       : {_HO_HESCHL_COLS}", flush=True)
print(f"  Parcel coords shape  : {_PARCEL_COORDS.shape}   "
      f"Edge dist vec : {_DISTS_VEC.shape}", flush=True)
print(f"  Network assignments  : {_VALID_NET_MASK.sum()}/400 valid", flush=True)
print(f"  [HIGH] LH SomMot: {len(_LH_MOTOR_COLS)}  RH SomMot: {len(_RH_MOTOR_COLS)}", flush=True)
print(f"  [MED]  DMN: {len(_DMN_COLS)}  DorsAttn: {len(_DORSATTN_COLS)}", flush=True)
print(f"  [LOW]  Vis: {len(_VIS_COLS)}  HO Heschl: {len(_HO_HESCHL_COLS)}", flush=True)
print("=" * 66 + "\n", flush=True)


# ---------------------------------------------------------------------------
# Modularity contrast helper
# ---------------------------------------------------------------------------

def modularity_contrast(fc_mat: np.ndarray) -> float:
    """Mean within-network FC minus mean between-network FC (Schaefer 7-network labels).

    Higher value = better-preserved network structure after denoising.
    fc_mat: (400, 400) Pearson correlation matrix.
    """
    fc_vec   = fc_mat[_UTI]                                        # (79800,)
    same_net = _NETWORK_IDX[_UTI[0]] == _NETWORK_IDX[_UTI[1]]     # bool (79800,)
    valid    = _VALID_NET_MASK[_UTI[0]] & _VALID_NET_MASK[_UTI[1]] # bool (79800,)
    within   = float(fc_vec[same_net & valid].mean())
    between  = float(fc_vec[~same_net & valid].mean())
    return round(within - between, 4)


def louvain_modularity_q(fc_mat: np.ndarray, n_runs: int = 10,
                         random_state: int = 42) -> float:
    """Weighted Louvain modularity Q — matches BCT (Brain Connectome Toolbox) approach.

    Implements graph community detection on the positive-weight FC matrix,
    exactly as described in Ciric et al. 2017 (citing BCT, Rubinov & Sporns 2010).

    Steps:
      1. Keep only positive correlations (negative FC has no agreed null model).
      2. Zero the diagonal.
      3. Build a weighted undirected graph.
      4. Run Louvain community detection n_runs times (stochastic) → take max Q.

    Returns Newman-Girvan Q for the best partition found.
    Higher = stronger community structure preserved after denoising.
    """
    adj = fc_mat.copy().astype(np.float64)
    np.fill_diagonal(adj, 0)
    adj[adj < 0] = 0   # positive weights only (BCT convention for fMRI)

    G = nx.from_numpy_array(adj)

    best_q = -np.inf
    rng = np.random.RandomState(random_state)
    for _ in range(n_runs):
        seed = int(rng.randint(0, 2**31))
        partition = community_louvain.best_partition(G, weight="weight",
                                                     random_state=seed)
        q = community_louvain.modularity(partition, G, weight="weight")
        if q > best_q:
            best_q = q

    return round(float(best_q), 4)


# ---------------------------------------------------------------------------
# Confound loaders
# ---------------------------------------------------------------------------

def _get_acomp_cols(conf_path: str, n: int, prefix: str) -> list:
    json_path = conf_path.replace(".tsv", ".json")
    if prefix == "a_comp_cor" and os.path.isfile(json_path):
        with open(json_path) as fh:
            meta = json.load(fh)
        WM_CSF = {"WM", "CSF", "combined"}
        cands = sorted([
            col for col, info in meta.items()
            if col.startswith("a_comp_cor")
            and isinstance(info, dict)
            and info.get("Retained", False)
            and info.get("Mask", "") in WM_CSF
        ])
        if cands:
            return cands[:n]
    return [f"{prefix}_{i:02d}" for i in range(n)]


def _get_acomp_variance_ranked(conf_path: str, cap: int) -> list:
    """Return top `cap` aCompCor components ranked by variance explained.

    Uses the fMRIPrep JSON sidecar to sort Retained components by VarianceExplained
    (descending) and returns up to `cap` columns. This is the Ciric et al. 2017
    aCompCor50 approach but with a practical DoF cap.

    Falls back to the fixed-N approach if the JSON is unavailable.
    """
    json_path = conf_path.replace(".tsv", ".json")
    if not os.path.isfile(json_path):
        return _get_acomp_cols(conf_path, cap, "a_comp_cor")

    with open(json_path) as fh:
        meta = json.load(fh)

    WM_CSF = {"WM", "CSF", "combined"}
    entries = [
        (col, info.get("VarianceExplained", 0.0))
        for col, info in meta.items()
        if col.startswith("a_comp_cor")
        and isinstance(info, dict)
        and info.get("Retained", False)
        and info.get("Mask", "") in WM_CSF
    ]
    # Sort by variance explained descending (highest-variance components first)
    entries.sort(key=lambda x: x[1], reverse=True)

    selected = [col for col, _ in entries[:cap]]
    return selected if selected else _get_acomp_cols(conf_path, cap, "a_comp_cor")


def load_confounds(conf_path: str, config_name: str, n_vols: int, tr: float,
                   scrub: bool = False):
    """Return (mean_fd_filtered, fd_filtered, X, reg_names, n_scrubbed).

    Matches fmri_timeseries_extractor.py exactly:
      - Notch-filter applied to motion params before computing FD (Fair et al. 2020)
      - Scrubbing via spike regressors (one column per censored volume), not deletion
    """
    df = pd.read_csv(conf_path, sep="\t").fillna(0).iloc[:n_vols].reset_index(drop=True)

    # Config-specific motion + noise regressors
    if config_name == "lean":
        motion = [c for c in MOTION_COLS if c in df.columns]           # 12 params
        acomp  = [c for c in _get_acomp_cols(conf_path, N_ACOMPCOR, "a_comp_cor")
                  if c in df.columns]
        extra  = []
    elif config_name == "anatomical":
        motion = [c for c in MOTION_COLS if c in df.columns]           # 12 params
        wm     = [f"w_comp_cor_{i:02d}" for i in range(N_ACOMPCOR)]
        csf    = [f"c_comp_cor_{i:02d}" for i in range(N_ACOMPCOR)]
        acomp  = [c for c in wm + csf if c in df.columns]
        extra  = []
    elif config_name == "global":
        motion = [c for c in MOTION_COLS if c in df.columns]           # 12 params
        acomp  = [c for c in _get_acomp_cols(conf_path, N_ACOMPCOR, "a_comp_cor")
                  if c in df.columns]
        extra  = [c for c in ["global_signal", "global_signal_derivative1"]
                  if c in df.columns]
    elif config_name == "research":
        # Friston 24-parameter motion model
        motion = [c for c in MOTION_COLS_24 if c in df.columns]        # 24 params
        # Separate WM + CSF CompCor (top 10 each) — much more effective than combined a_comp_cor.
        # c_comp_cor captures cardiac/respiratory variance in CSF (4× higher variance than combined).
        wm_cols  = [f"w_comp_cor_{i:02d}" for i in range(RESEARCH_N_WM)]
        csf_cols = [f"c_comp_cor_{i:02d}" for i in range(RESEARCH_N_CSF)]
        acomp    = [c for c in wm_cols + csf_cols if c in df.columns]  # 20 noise regressors
        extra    = []   # no GSR — protects psilocybin-induced global fluctuations
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Notch-filter raw motion → recompute FD (removes respiratory pseudo-motion)
    raw_motion_df = df[MOTION_COLS_RAW].copy()
    filt_motion   = apply_motion_filter(raw_motion_df, tr)
    diff          = filt_motion.diff().fillna(0)
    diff[["rot_x", "rot_y", "rot_z"]] *= 50   # radians → mm (50 mm head radius)
    fd_filtered   = diff.abs().sum(axis=1).values.astype(np.float64)
    fd_filtered[0] = 0.0   # first volume has no predecessor
    mean_fd = float(fd_filtered[1:].mean())

    selected = motion + acomp + extra
    X        = df[selected].values.astype(np.float64)

    # Scrubbing: spike regressors (matches main pipeline — keeps all timepoints,
    # projects out high-motion volumes via regression rather than deleting them)
    n_scrubbed = 0
    if scrub:
        to_scrub: set = set()
        for t in np.where(fd_filtered > FD_SCRUB_THRESHOLD)[0]:
            to_scrub.add(int(t))
            if int(t) + 1 < n_vols:
                to_scrub.add(int(t) + 1)
        n_scrubbed = len(to_scrub)
        if n_scrubbed > 0:
            spike_regs = np.zeros((n_vols, n_scrubbed), dtype=np.float64)
            for i, vol in enumerate(sorted(to_scrub)):
                spike_regs[vol, i] = 1.0
            X        = np.hstack([X, spike_regs])
            selected = selected + [f"spike_{i}" for i in range(n_scrubbed)]

    return mean_fd, fd_filtered, X, selected, n_scrubbed


# ---------------------------------------------------------------------------
# Atlas timeseries extraction
# ---------------------------------------------------------------------------

def _extract_ts(bold_img, atlas_maps, tr: float, confounds=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        masker = NiftiLabelsMasker(
            labels_img     = atlas_maps,
            standardize    = False,
            smoothing_fwhm = SMOOTHING_FWHM,   # matches main pipeline
            detrend        = True,
            high_pass      = HIGH_PASS,
            t_r            = tr,
        )
        ts = masker.fit_transform(bold_img, confounds=confounds)
    return ts


def extract_all_roi_ts(bold_img, tr: float, confounds=None):
    sch = _extract_ts(bold_img, SCHAEFER400_MAPS, tr, confounds)

    tian = None
    if TIAN_AVAILABLE:
        try:
            tian = _extract_ts(bold_img, _TIAN_NII, tr, confounds)
        except Exception:
            pass

    ho = None
    if _HO_HESCHL_COLS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ho_m = NiftiLabelsMasker(
                    labels_img     = _HO.maps,
                    standardize    = False,
                    smoothing_fwhm = SMOOTHING_FWHM,
                    detrend        = True,
                    high_pass   = HIGH_PASS,
                    t_r         = tr,
                )
                ho = ho_m.fit_transform(bold_img, confounds=confounds)
        except Exception:
            pass

    return sch, tian, ho


# ---------------------------------------------------------------------------
# Connectivity / sanity checks
# ---------------------------------------------------------------------------

def _net_corr(ts, idx_a, idx_b):
    if not idx_a or not idx_b or ts is None:
        return np.nan
    a = ts[:, idx_a].mean(axis=1)
    b = ts[:, idx_b].mean(axis=1)
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def gcor(signals: np.ndarray) -> float:
    if signals is None or signals.shape[1] < 2:
        return np.nan
    c = signals - signals.mean(axis=0)
    n = np.linalg.norm(c, axis=0)
    n = np.where(n < 1e-10, 1e-10, n)
    g = (c / n).mean(axis=1)
    return float(np.dot(g, g))


def compute_sanity_checks(sch_ts, tian_ts, ho_ts):
    combined = np.concatenate([sch_ts, tian_ts], axis=1) if tian_ts is not None else sch_ts
    gc_val   = gcor(combined)
    motor_lr = _net_corr(sch_ts, _LH_MOTOR_COLS, _RH_MOTOR_COLS)
    dmn_da   = _net_corr(sch_ts, _DMN_COLS, _DORSATTN_COLS)

    v1_ts  = sch_ts[:, _VIS_COLS].mean(axis=1) if _VIS_COLS else None
    v1_a1  = np.nan
    if ho_ts is not None and ho_ts.shape[1] > 0 and _HO_HESCHL_COLS:
        valid = [c for c in _HO_HESCHL_COLS if c < ho_ts.shape[1]]
        if valid and v1_ts is not None:
            a1 = ho_ts[:, valid].mean(axis=1)
            v1_a1 = round(float(np.corrcoef(v1_ts, a1)[0, 1]), 4)

    return {"GCOR": round(gc_val, 4), "motor_LR": motor_lr,
            "dmn_da": dmn_da, "v1_a1": v1_a1}


# ---------------------------------------------------------------------------
# tSNR helpers
# ---------------------------------------------------------------------------

def _tsnr_mean(mean_v, std_v):
    std_v = np.where(std_v < 1e-6, 1e-6, std_v)
    return float(np.mean(mean_v / std_v))


# ---------------------------------------------------------------------------
# Core per-run processing
# ---------------------------------------------------------------------------


def process_run(bold_path: str, mask_path, conf_path: str,
                tr: float, config_name: str, scrub: bool = False) -> dict:
    img  = nib.load(bold_path)
    data = img.get_fdata(dtype=np.float32)
    T    = data.shape[-1]

    # load_confounds uses notch-filtered FD and (when scrub=True) spike regressors
    mean_fd, fd_filtered, X, reg_names, n_scrubbed = load_confounds(
        conf_path, config_name, T, tr, scrub=scrub
    )
    n_reg = X.shape[1]   # includes spike regressors when scrubbing

    # X_base = confounds WITHOUT spike regressors, used only for voxel tSNR.
    # Spike regressors zero-out scrubbed timepoints in the residual, collapsing
    # std and inflating tSNR artificially — so tSNR must be measured without them.
    _, _, X_base, _, _ = load_confounds(conf_path, config_name, T, tr, scrub=False)

    # FD vector aligned to BOLD length (for QC-FC correlation)
    fd_ts_qcfc = fd_filtered[:T]

    # -----------------------------------------------------------------------
    # ROI timeseries (pre = detrend+HP+smooth, post = +confound regression)
    # Spike regressors inside X handle scrubbing — no volume deletion needed.
    # -----------------------------------------------------------------------
    sch_pre,  tian_pre,  ho_pre  = extract_all_roi_ts(img, tr, confounds=None)
    sch_post, tian_post, ho_post = extract_all_roi_ts(img, tr, confounds=X)

    # --- GCOR + sanity checks ---
    pre_sc  = compute_sanity_checks(sch_pre,  tian_pre,  ho_pre)
    post_sc = compute_sanity_checks(sch_post, tian_post, ho_post)

    # -----------------------------------------------------------------------
    # Ciric 2017 benchmarks (ROI-level)
    # -----------------------------------------------------------------------

    # 1. QC-FC: mean |r| between each denoised parcel timeseries and FD timeseries
    #    Uses notch-filtered FD — consistent with scrubbing threshold
    with np.errstate(invalid="ignore"):
        qcfc_per_p = np.array([
            np.corrcoef(sch_post[:, i], fd_ts_qcfc)[0, 1]
            for i in range(sch_post.shape[1])
        ])
    qcfc_mean_abs = float(np.nanmean(np.abs(qcfc_per_p)))
    with np.errstate(invalid="ignore"):
        qcfc_pre_p = np.array([
            np.corrcoef(sch_pre[:, i], fd_ts_qcfc)[0, 1]
            for i in range(sch_pre.shape[1])
        ])
    qcfc_pre_mean_abs = float(np.nanmean(np.abs(qcfc_pre_p)))

    # 2. DM-FC: r(FC edge values, inter-parcel Euclidean distances)
    #    Motion inflates short-range connections -> negative r before cleaning
    #    Good cleaning drives this toward 0
    pre_fc_mat  = np.corrcoef(sch_pre.T)    # (400, 400)
    post_fc_mat = np.corrcoef(sch_post.T)   # (400, 400)
    pre_fc_vec  = pre_fc_mat[_UTI].astype(np.float32)    # (79800,)
    post_fc_vec = post_fc_mat[_UTI].astype(np.float32)   # (79800,) — kept for cross-run

    with np.errstate(invalid="ignore"):
        pre_dm_fc  = float(np.corrcoef(pre_fc_vec,  _DISTS_VEC)[0, 1])
        post_dm_fc = float(np.corrcoef(post_fc_vec, _DISTS_VEC)[0, 1])

    # 3. Loss of Degrees of Freedom: n_reg already includes spike regressors
    dof_lost = n_reg

    # 4a. Modularity contrast: within-network minus between-network FC (Schaefer 7-net)
    pre_mod  = modularity_contrast(pre_fc_mat)
    post_mod = modularity_contrast(post_fc_mat)

    # 4b. Louvain Q: graph community detection on weighted positive FC matrix
    #     Matches BCT (Rubinov & Sporns 2010) as used in Ciric et al. 2017
    pre_louvain_q  = louvain_modularity_q(pre_fc_mat)
    post_louvain_q = louvain_modularity_q(post_fc_mat)

    del sch_pre, sch_post, tian_pre, tian_post, ho_pre, ho_post
    del pre_fc_mat, post_fc_mat
    gc.collect()

    # -----------------------------------------------------------------------
    # Voxel-level tSNR + variance explained
    # -----------------------------------------------------------------------
    if mask_path and os.path.isfile(mask_path):
        mask = nib.load(mask_path).get_fdata() > 0.5
    else:
        mask = np.mean(data, axis=-1) > 50

    vox_raw  = data[mask].T.astype(np.float64)
    del data
    gc.collect()

    mean_v   = vox_raw.mean(axis=0)
    std_raw  = vox_raw.std(axis=0)
    pre_tsnr = _tsnr_mean(mean_v, std_raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        vox_clean = nls.clean(
            vox_raw,
            confounds   = X_base,   # no spike regressors — keeps tSNR interpretable
            t_r         = tr,
            high_pass   = HIGH_PASS,
            detrend     = True,
            standardize = False,
        )

    post_tsnr = _tsnr_mean(mean_v, vox_clean.std(axis=0))
    var_raw   = np.var(vox_raw,   axis=0)
    var_clean = np.var(vox_clean, axis=0)
    valid     = var_raw > 1e-10
    var_exp   = float(np.mean(1.0 - var_clean[valid] / var_raw[valid])) * 100.0

    del vox_raw, vox_clean
    gc.collect()

    return {
        # Core metrics
        "mean_fd":               round(mean_fd, 4),
        "pre_tSNR":              round(pre_tsnr, 3),
        "post_tSNR":             round(post_tsnr, 3),
        "delta_tSNR_pct":        round(100.0 * (post_tsnr - pre_tsnr) / max(pre_tsnr, 1e-6), 2),
        "pre_GCOR":              pre_sc["GCOR"],
        "post_GCOR":             post_sc["GCOR"],
        "delta_GCOR":            round(post_sc["GCOR"] - pre_sc["GCOR"], 4),
        # Sanity-check correlations
        "pre_motor_LR":          pre_sc["motor_LR"],
        "post_motor_LR":         post_sc["motor_LR"],
        "pre_dmn_da":            pre_sc["dmn_da"],
        "post_dmn_da":           post_sc["dmn_da"],
        "pre_v1_a1":             pre_sc["v1_a1"],
        "post_v1_a1":            post_sc["v1_a1"],
        # Ciric 2017 benchmarks
        "pre_qcfc":              round(qcfc_pre_mean_abs, 4),    # before cleaning
        "post_qcfc":             round(qcfc_mean_abs, 4),        # after cleaning (LOWER = BETTER)
        "pre_dm_fc":             round(pre_dm_fc, 4),
        "post_dm_fc":            round(post_dm_fc, 4),
        "dof_lost":              dof_lost,
        "pre_modularity":        pre_mod,
        "post_modularity":       post_mod,
        "pre_louvain_q":         pre_louvain_q,
        "post_louvain_q":        post_louvain_q,
        # Variance + regressor info
        "variance_explained_pct": round(var_exp, 2),
        "n_regressors":          n_reg,
        "n_scrubbed":            n_scrubbed,
        "regressors":            ", ".join(reg_names[:4]) + ("..." if n_reg > 4 else ""),
        # FC vector for cross-run QC-FC (stripped before saving to CSV)
        "_fc_vector":            post_fc_vec,     # (79800,) float32 ndarray
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _conf_from_bold(bold_path: str) -> str:
    return re.sub(
        r"_space-[^_]+(?:_res-[^_]+)?_desc-preproc_bold\.nii\.gz",
        "_desc-confounds_timeseries.tsv",
        bold_path,
    )


def find_runs(fmriprep_root: str):
    for subj in sorted(p for p in os.listdir(fmriprep_root) if p.startswith("sub-")):
        subj_path = os.path.join(fmriprep_root, subj)
        if not os.path.isdir(subj_path):
            continue
        for ses in sorted(p for p in os.listdir(subj_path) if p.startswith("ses-")):
            func_dir = os.path.join(subj_path, ses, "func")
            if not os.path.isdir(func_dir):
                continue
            for f in sorted(os.listdir(func_dir)):
                if not (f.endswith("_desc-preproc_bold.nii.gz") and "MNI" in f):
                    continue
                bold_path = os.path.join(func_dir, f)
                conf_path = _conf_from_bold(bold_path)
                if not os.path.isfile(conf_path):
                    print(f"  [SKIP] {f}", flush=True)
                    continue
                mask_path = bold_path.replace(
                    "_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz"
                )
                task_m = re.search(r"task-([a-zA-Z0-9]+)", f)
                acq_m  = re.search(r"acq-([a-zA-Z0-9]+)",  f)
                run_m  = re.search(r"run-([a-zA-Z0-9]+)",  f)
                yield {
                    "subject": subj, "session": ses,
                    "task":    task_m.group(1) if task_m else "unknown",
                    "acq":     acq_m.group(1)  if acq_m  else "",
                    "run":     f"run-{run_m.group(1)}" if run_m else "run-1",
                    "bold_path": bold_path,
                    "mask_path": mask_path if os.path.isfile(mask_path) else None,
                    "conf_path": conf_path,
                }


# ---------------------------------------------------------------------------
# Worker wrapper
# ---------------------------------------------------------------------------

def _worker(args):
    run_dict, config_name, scrub, idx, total = args
    label = (
        f"{run_dict['subject']} | {run_dict['session']} | task-{run_dict['task']}"
        + (f" acq-{run_dict['acq']}" if run_dict["acq"] else "")
        + f" | {run_dict['run']}"
    )
    print(f"[{idx:>2}/{total}] {label}", flush=True)

    try:
        img_hdr = nib.load(run_dict["bold_path"])
        tr      = get_tr_seconds(run_dict["bold_path"], img_hdr)
        del img_hdr
    except Exception as e:
        print(f"       [TR ERROR] {e} — skipping", flush=True)
        return {**{k: run_dict[k] for k in ("subject","session","task","acq","run")},
                **dict.fromkeys(["mean_fd","pre_tSNR","post_tSNR","delta_tSNR_pct",
                                  "pre_GCOR","post_GCOR","delta_GCOR",
                                  "pre_motor_LR","post_motor_LR",
                                  "pre_dmn_da","post_dmn_da","pre_v1_a1","post_v1_a1",
                                  "pre_qcfc","post_qcfc",
                                  "pre_dm_fc","post_dm_fc","dof_lost",
                                  "pre_modularity","post_modularity",
                                  "pre_louvain_q","post_louvain_q",
                                  "variance_explained_pct","n_regressors","n_scrubbed"], np.nan),
                "_fc_vector": None}

    try:
        metrics = process_run(
            run_dict["bold_path"], run_dict["mask_path"],
            run_dict["conf_path"], tr, config_name, scrub=scrub
        )
        scrub_str = f"  scrubbed={metrics['n_scrubbed']}" if scrub else ""
        print(
            f"       FD={metrics['mean_fd']:.3f}{scrub_str}  "
            f"tSNR {metrics['pre_tSNR']:.1f}->{metrics['post_tSNR']:.1f} ({metrics['delta_tSNR_pct']:+.1f}%)  "
            f"GCOR {metrics['pre_GCOR']:.3f}->{metrics['post_GCOR']:.3f}  "
            f"QC-FC {metrics['pre_qcfc']:.3f}->{metrics['post_qcfc']:.3f}  "
            f"DM-FC {metrics['pre_dm_fc']:+.3f}->{metrics['post_dm_fc']:+.3f}  "
            f"Mod {metrics['pre_modularity']:.3f}->{metrics['post_modularity']:.3f}  "
            f"DoF-{metrics['dof_lost']}  "
            f"Motor {metrics['pre_motor_LR']:.3f}->{metrics['post_motor_LR']:.3f}  "
            f"V1/A1 {metrics['pre_v1_a1']:.3f}->{metrics['post_v1_a1']:.3f}",
            flush=True,
        )
        return {**{k: run_dict[k] for k in ("subject","session","task","acq","run")},
                **metrics}
    except Exception as e:
        import traceback
        print(f"       [ERROR] {e}", flush=True)
        traceback.print_exc()
        return {**{k: run_dict[k] for k in ("subject","session","task","acq","run")},
                **dict.fromkeys(["mean_fd","pre_tSNR","post_tSNR","delta_tSNR_pct",
                                  "pre_GCOR","post_GCOR","delta_GCOR",
                                  "pre_motor_LR","post_motor_LR",
                                  "pre_dmn_da","post_dmn_da","pre_v1_a1","post_v1_a1",
                                  "pre_qcfc","post_qcfc",
                                  "pre_dm_fc","post_dm_fc","dof_lost",
                                  "pre_modularity","post_modularity",
                                  "pre_louvain_q","post_louvain_q",
                                  "variance_explained_pct","n_regressors","n_scrubbed"], np.nan),
                "_fc_vector": None}


# ---------------------------------------------------------------------------
# Cross-run QC-FC (gold standard — computed over all N runs)
# ---------------------------------------------------------------------------

def compute_cross_run_qcfc(fd_vec: np.ndarray, fc_matrix: np.ndarray) -> dict:
    """
    fc_matrix : (N, 79800) — one row per run, upper-triangle of Schaefer-400 FC.
    fd_vec    : (N,) — Mean FD per run.

    Returns:
      qcfc_mean_abs   : mean |r| across all 79800 edges
      qcfc_pct_sig    : % edges with |r| significant at p<0.05
      cross_run_dm_fc : r(|QC-FC|, inter-parcel distance)
    """
    N = len(fd_vec)
    if N < 5:
        return {"qcfc_mean_abs": np.nan, "qcfc_pct_sig": np.nan, "cross_run_dm_fc": np.nan}

    # Pearson r for each edge
    fd_z  = (fd_vec - fd_vec.mean()) / (fd_vec.std() + 1e-12)    # (N,)
    fc_z  = fc_matrix - fc_matrix.mean(axis=0)
    fc_z /= (fc_matrix.std(axis=0) + 1e-12)
    qcfc  = (fd_z @ fc_z) / N                                     # (79800,) r values

    # Significance: t = r * sqrt(N-2) / sqrt(1-r^2), df=N-2
    df    = N - 2
    t_val = qcfc * np.sqrt(df) / np.sqrt(np.maximum(1 - qcfc ** 2, 1e-12))
    pvals = 2 * spstats.t.sf(np.abs(t_val), df=df)

    qcfc_mean_abs   = float(np.abs(qcfc).mean())
    qcfc_pct_sig    = float((pvals < 0.05).mean() * 100.0)

    # DM-FC gold standard: do higher-motion edges correlate with shorter distances?
    with np.errstate(invalid="ignore"):
        cross_run_dm_fc = float(np.corrcoef(np.abs(qcfc), _DISTS_VEC)[0, 1])

    return {
        "qcfc_mean_abs":   round(qcfc_mean_abs, 4),
        "qcfc_pct_sig":    round(qcfc_pct_sig, 2),
        "cross_run_dm_fc": round(cross_run_dm_fc, 4),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_gcor_scatter(df, out_path, config_label):
    subjects = sorted(df["subject"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, s in enumerate(subjects):
        sub = df[df["subject"] == s]
        ax.scatter(sub["pre_GCOR"], sub["post_GCOR"],
                   label=s, color=cmap(i % 10), s=60, alpha=0.85)
    lim = max(df["pre_GCOR"].max(), df["post_GCOR"].max()) * 1.15
    ax.plot([0, lim], [0, lim], "k--", lw=0.9, label="no change")
    ax.set_xlabel("Pre-denoising GCOR"); ax.set_ylabel("Post-denoising GCOR")
    ax.set_title(f"GCOR\n{config_label}", fontsize=10); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out_path)}", flush=True)


def plot_tsnr_panels(df, out_path, config_label):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    subjects  = sorted(df["subject"].unique())
    cmap = plt.get_cmap("tab10")
    ax = axes[0]
    for i, s in enumerate(subjects):
        sub = df[df["subject"] == s]
        ax.scatter(sub["pre_tSNR"], sub["post_tSNR"],
                   label=s, color=cmap(i % 10), s=55, alpha=0.85)
    lim = max(df["pre_tSNR"].max(), df["post_tSNR"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    ax.set_xlabel("Pre tSNR"); ax.set_ylabel("Post tSNR")
    ax.set_title("tSNR Before vs After"); ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    vals = df["delta_tSNR_pct"].dropna()
    ax.hist(vals, bins=15, color="#55A868", edgecolor="white")
    ax.axvline(vals.mean(), color="red", linestyle="--", label=f"Mean={vals.mean():+.1f}%")
    ax.axvline(0, color="black", linestyle=":", lw=0.8)
    ax.set_xlabel("tSNR Improvement (%)"); ax.set_ylabel("N runs")
    ax.set_title("tSNR Change Distribution"); ax.legend()
    plt.suptitle(config_label, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out_path)}", flush=True)


def plot_sanity_checks(df, out_path, config_label):
    checks = [
        ("Motor L/R",    "pre_motor_LR",  "post_motor_LR",  "#2196F3"),
        ("DMN/DorsAttn", "pre_dmn_da",    "post_dmn_da",    "#FF9800"),
        ("V1/A1",        "pre_v1_a1",     "post_v1_a1",     "#9C27B0"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(checks)); w = 0.35
    for j, (lab, pre_c, post_c, col) in enumerate(checks):
        ax.bar(x[j] - w/2, df[pre_c].dropna().mean(),  w, color=col, alpha=0.4, edgecolor="k", label="Pre" if j == 0 else "")
        ax.bar(x[j] + w/2, df[post_c].dropna().mean(), w, color=col, alpha=0.9, edgecolor="k", label="Post" if j == 0 else "")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in checks], fontsize=11)
    ax.set_ylabel("Mean Pearson r"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Connectivity Sanity Checks\n{config_label}", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out_path)}", flush=True)


def plot_ciric_benchmarks(df, out_path, config_label, cross_run_qcfc=None):
    """4-panel: QC-FC, DM-FC, DoF, Modularity."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. QC-FC: pre vs post
    ax = axes[0, 0]
    subjects = sorted(df["subject"].unique())
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(subjects):
        sub = df[df["subject"] == s]
        ax.scatter(sub["pre_qcfc"], sub["post_qcfc"],
                   label=s, color=cmap(i % 10), s=55, alpha=0.85)
    lim = max(df["pre_qcfc"].max(), df["post_qcfc"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    ax.set_xlabel("QC-FC pre (mean |r|)"); ax.set_ylabel("QC-FC post (mean |r|)")
    title = "QC-FC: Parcel-FD Correlation [LOWER = BETTER]"
    if cross_run_qcfc:
        title += f"\nCross-run: mean|r|={cross_run_qcfc['qcfc_mean_abs']:.3f}  sig={cross_run_qcfc['qcfc_pct_sig']:.1f}%"
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7, ncol=2)

    # 2. DM-FC: pre vs post (scatter)
    ax = axes[0, 1]
    for i, s in enumerate(subjects):
        sub = df[df["subject"] == s]
        ax.scatter(sub["pre_dm_fc"], sub["post_dm_fc"],
                   label=s, color=cmap(i % 10), s=55, alpha=0.85)
    if len(df["pre_dm_fc"].dropna()) > 0:
        lim_min = min(df["pre_dm_fc"].min(), df["post_dm_fc"].min()) - 0.05
        lim_max = max(df["pre_dm_fc"].max(), df["post_dm_fc"].max()) + 0.05
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=0.9)
    ax.axhline(0, color="red", linestyle=":", lw=1)
    ax.axvline(0, color="red", linestyle=":", lw=1)
    dm_title = "DM-FC: Distance-Dependent FC [TOWARD ZERO = BETTER]"
    if cross_run_qcfc:
        dm_title += f"\nCross-run DM-FC r={cross_run_qcfc['cross_run_dm_fc']:.3f}"
    ax.set_xlabel("DM-FC pre (r)"); ax.set_ylabel("DM-FC post (r)")
    ax.set_title(dm_title, fontsize=9)

    # 3. Degrees of Freedom lost (histogram)
    ax = axes[1, 0]
    dof = df["dof_lost"].dropna()
    ax.hist(dof, bins=range(int(dof.min()), int(dof.max()) + 2), color="#FF7043", edgecolor="white")
    ax.axvline(dof.mean(), color="k", linestyle="--", label=f"Mean={dof.mean():.1f}")
    ax.set_xlabel("Degrees of Freedom Lost (n_regressors)")
    ax.set_ylabel("Number of runs")
    ax.set_title("DoF Lost per Run [LOWER = BETTER statistical power]", fontsize=9)
    ax.legend()

    # 4. Network Modularity: pre vs post
    ax = axes[1, 1]
    for i, s in enumerate(subjects):
        sub = df[df["subject"] == s]
        ax.scatter(sub["pre_modularity"], sub["post_modularity"],
                   label=s, color=cmap(i % 10), s=55, alpha=0.85)
    if len(df["pre_modularity"].dropna()) > 0:
        lim = max(df["pre_modularity"].max(), df["post_modularity"].max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    ax.set_xlabel("Modularity pre (within - between FC)")
    ax.set_ylabel("Modularity post (within - between FC)")
    ax.set_title("Network Modularity [HIGHER = BETTER network structure]", fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    plt.suptitle(f"Ciric 2017 Benchmarks\n{config_label}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out_path)}", flush=True)


# ---------------------------------------------------------------------------
# Column order for CSV
# ---------------------------------------------------------------------------
COL_ORDER = [
    "subject", "session", "task", "acq", "run",
    "mean_fd",
    "pre_tSNR", "post_tSNR", "delta_tSNR_pct",
    "pre_GCOR", "post_GCOR", "delta_GCOR",
    # Sanity checks
    "pre_motor_LR", "post_motor_LR",
    "pre_dmn_da",   "post_dmn_da",
    "pre_v1_a1",    "post_v1_a1",
    # Ciric 2017 benchmarks
    "pre_qcfc",        "post_qcfc",         # QC-FC  (lower post = better)
    "pre_dm_fc",       "post_dm_fc",        # DM-FC  (post toward 0 = better)
    "dof_lost",                             # DoF    (lower = less cost)
    "pre_modularity",  "post_modularity",   # within-between contrast
    "pre_louvain_q",   "post_louvain_q",    # Louvain weighted Q (BCT / Ciric 2017)
    # Other
    "variance_explained_pct", "n_regressors", "n_scrubbed", "regressors",
]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, config_label: str, cross_run: dict = None):
    dv = df.dropna(subset=["pre_tSNR"])
    if dv.empty:
        return

    def _m(c): return dv[c].dropna().mean() if c in dv else np.nan

    print(f"\n{'='*66}", flush=True)
    print(f"  {config_label}", flush=True)
    print(f"  Grand mean over {len(dv)} runs", flush=True)
    print(f"{'='*66}", flush=True)
    print(f"  tSNR            pre/post : {_m('pre_tSNR'):.2f} / {_m('post_tSNR'):.2f}  "
          f"(delta {_m('delta_tSNR_pct'):+.1f}%)", flush=True)
    print(f"  GCOR            pre/post : {_m('pre_GCOR'):.4f} / {_m('post_GCOR'):.4f}  "
          f"(delta {_m('delta_GCOR'):.4f})", flush=True)
    print(f"", flush=True)
    print(f"  --- Ciric 2017 Benchmarks ---", flush=True)
    print(f"  QC-FC           pre/post : {_m('pre_qcfc'):.4f} / {_m('post_qcfc'):.4f}"
          f"  [LOWER = less motion contamination]", flush=True)
    if cross_run:
        print(f"  QC-FC cross-run  mean|r| : {cross_run['qcfc_mean_abs']:.4f}  "
              f"sig edges : {cross_run['qcfc_pct_sig']:.1f}%  "
              f"[gold standard, 0% sig = perfect]", flush=True)
    print(f"  DM-FC           pre/post : {_m('pre_dm_fc'):+.4f} / {_m('post_dm_fc'):+.4f}"
          f"  [TOWARD ZERO = less distance-dependent artifacts]", flush=True)
    if cross_run:
        print(f"  DM-FC cross-run dist cor : {cross_run['cross_run_dm_fc']:+.4f}"
              f"  [TOWARD ZERO = no distance-dependent motion effect]", flush=True)
    n_scr = _m('n_scrubbed')
    scr_str = f" + {n_scr:.1f} scrubbed vols" if not np.isnan(n_scr) and n_scr > 0 else ""
    print(f"  DoF Lost        per run  : {_m('dof_lost'):.1f} regressors{scr_str}"
          f"  [statistical cost]", flush=True)
    print(f"  Modularity (contrast) pre/post : {_m('pre_modularity'):.4f} / {_m('post_modularity'):.4f}"
          f"  [HIGHER = network structure preserved]", flush=True)
    print(f"  Modularity (Louvain Q) pre/post: {_m('pre_louvain_q'):.4f} / {_m('post_louvain_q'):.4f}"
          f"  [HIGHER = stronger community structure, BCT/Ciric2017]", flush=True)
    print(f"", flush=True)
    print(f"  --- Sanity Checks ---", flush=True)
    print(f"  Motor L/R       pre/post : {_m('pre_motor_LR'):.3f} / {_m('post_motor_LR'):.3f}"
          f"  [HIGH ~0.8+]", flush=True)
    print(f"  DMN/DorsAttn    pre/post : {_m('pre_dmn_da'):.3f} / {_m('post_dmn_da'):.3f}"
          f"  [MEDIUM]", flush=True)
    print(f"  V1/A1 neg-ctrl  pre/post : {_m('pre_v1_a1'):.3f} / {_m('post_v1_a1'):.3f}"
          f"  [LOW - negative control]", flush=True)
    print(f"  Var Explained           : {_m('variance_explained_pct'):.1f}%", flush=True)
    print(f"  Mean FD                 : {_m('mean_fd'):.3f} mm", flush=True)
    print(f"{'='*66}\n", flush=True)


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------

def _is_rest_run(r: dict) -> bool:
    """True for resting-state runs across all subjects.

    Includes:
      - task == 'rest'            (all subjects)
      - task == 'restchecktr2'    (sub-004 TR=2 rest scan)
    """
    task = r.get("task", "")
    return task in ("rest", "restchecktr2")


def run_config(config_name: str, n_workers: int = 1, test_mode: bool = False,
               scrub: bool = False, rest_only: bool = False,
               out_base: str = None):
    cfg        = CONFIGS[config_name]
    folder     = cfg["folder"] + ("_scrubbed" if scrub else "")
    base       = out_base or BASE_OUT_DIR
    out_dir    = os.path.join(base, folder)
    os.makedirs(out_dir, exist_ok=True)

    scrub_note = f" | scrub FD>{FD_SCRUB_THRESHOLD} +1vol" if scrub else " | no scrubbing"
    rest_note  = " | REST ONLY" if rest_only else ""
    print(f"\n{'#'*70}", flush=True)
    print(f"  {cfg['label']}{scrub_note}{rest_note}", flush=True)
    print(f"  Output: {out_dir}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    runs = list(find_runs(FMRIPREP_ROOT))
    if rest_only:
        runs = [r for r in runs if _is_rest_run(r)]
        print(f"Rest-only filter: {len(runs)} runs kept "
              f"(task=rest or task=restchecktr2)\n", flush=True)
    if test_mode:
        runs = runs[:1]
        print("*** TEST MODE: 1 run only ***\n", flush=True)

    total     = len(runs)
    args_list = [(r, config_name, scrub, i + 1, total) for i, r in enumerate(runs)]
    results   = []

    print(f"Found {total} BOLD runs to process.\n", flush=True)

    if n_workers <= 1:
        for a in args_list:
            results.append(_worker(a))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, a): a for a in args_list}
            for fut in as_completed(futures):
                results.append(fut.result())

    # Sort results
    df = pd.DataFrame(results)
    key_cols = [c for c in ["subject","session","task","acq","run"] if c in df.columns]
    df = df.sort_values(key_cols).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Cross-run QC-FC (gold standard)
    # -----------------------------------------------------------------------
    cross_run_qcfc = None
    fc_rows = [r for r in results if r.get("_fc_vector") is not None]
    if len(fc_rows) >= 5:
        fd_vec  = np.array([r["mean_fd"] for r in fc_rows], dtype=np.float64)
        fc_mat  = np.vstack([r["_fc_vector"].astype(np.float64)
                             for r in fc_rows])   # (N, 79800)
        valid   = ~np.isnan(fd_vec)
        if valid.sum() >= 5:
            cross_run_qcfc = compute_cross_run_qcfc(fd_vec[valid], fc_mat[valid])
            print(f"\nCross-run QC-FC (n={valid.sum()} runs):", flush=True)
            print(f"  mean |QC-FC r| = {cross_run_qcfc['qcfc_mean_abs']:.4f}", flush=True)
            print(f"  % sig edges    = {cross_run_qcfc['qcfc_pct_sig']:.1f}%", flush=True)
            print(f"  DM-FC r        = {cross_run_qcfc['cross_run_dm_fc']:+.4f}", flush=True)

            # Save cross-run summary to CSV
            cr_df = pd.DataFrame([{**{"config": config_name, "n_runs": int(valid.sum())},
                                   **cross_run_qcfc}])
            cr_csv = os.path.join(out_dir, "cross_run_qcfc.csv")
            cr_df.to_csv(cr_csv, index=False)
            print(f"  Saved: {os.path.basename(cr_csv)}", flush=True)

    # -----------------------------------------------------------------------
    # Save per-run CSV (drop _fc_vector column)
    # -----------------------------------------------------------------------
    col_order = [c for c in COL_ORDER if c in df.columns]
    df_save   = df[col_order]
    csv_path  = os.path.join(out_dir, "denoising_summary.csv")
    try:
        df_save.to_csv(csv_path, index=False)
    except PermissionError:
        csv_path = csv_path.replace(".csv", "_new.csv")
        df_save.to_csv(csv_path, index=False)
    print(f"\nSaved CSV -> {csv_path}\n", flush=True)

    # -----------------------------------------------------------------------
    # Summary + Plots
    # -----------------------------------------------------------------------
    dv = df_save.dropna(subset=["pre_tSNR"])
    if not dv.empty:
        print_summary(dv, cfg["label"], cross_run_qcfc)
        print("Generating plots...", flush=True)
        plot_gcor_scatter(dv, os.path.join(out_dir, "gcor_scatter.png"),      cfg["label"])
        plot_tsnr_panels (dv, os.path.join(out_dir, "tsnr_improvement.png"),  cfg["label"])
        plot_sanity_checks(dv, os.path.join(out_dir, "sanity_checks.png"),    cfg["label"])
        plot_ciric_benchmarks(dv, os.path.join(out_dir, "ciric_benchmarks.png"),
                              cfg["label"], cross_run_qcfc)

    return df_save


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    choices=["lean","anatomical","global","research","all"],
                        default="lean")
    parser.add_argument("--workers",   type=int, default=1)
    parser.add_argument("--test",      action="store_true")
    parser.add_argument("--scrub",     action="store_true",
                        help=f"Enable FD scrubbing (threshold={FD_SCRUB_THRESHOLD} mm, +1 vol rule)")
    parser.add_argument("--rest-only", action="store_true",
                        help="Process only resting-state runs (task=rest + task=restchecktr2)")
    args = parser.parse_args()

    # Output base dir: separate folder for rest-only runs
    if args.rest_only:
        out_base = os.path.join(os.path.dirname(__file__), "..", "outputs",
                                "quality_assurance", "denoising_effect_restonly")
    else:
        out_base = BASE_OUT_DIR

    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    for cfg_name in configs_to_run:
        run_config(cfg_name, n_workers=args.workers, test_mode=args.test,
                   scrub=args.scrub, rest_only=args.rest_only, out_base=out_base)

    print("\nAll requested configs complete.", flush=True)


if __name__ == "__main__":
    main()
