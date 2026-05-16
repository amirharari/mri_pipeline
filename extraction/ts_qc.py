"""Per-run timeseries quality-control metrics.

These functions are called by the extractor to print per-run diagnostics, but
they contain no extraction logic — they live here so they can be imported
independently by QA scripts without pulling in the full extractor stack.

Public API
----------
    post_dvars(ts)                        -> float
    ts_qc_metrics(ts_clean, ts_raw)       -> dict
    confounds_variance_explained(raw, X)  -> float
    louvain_q(corr)                       -> float
    fc_network_metrics(df_ts, corr)       -> dict
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Temporal DVARS
# ─────────────────────────────────────────────────────────────────────────────

def post_dvars(ts: np.ndarray) -> float:
    """Post-denoising DVARS: mean RMS of the temporal derivative of clean parcel timeseries.

    Units: signal-SD per TR (timeseries is z-scored before differentiation,
           so values are comparable across runs and atlases).
    Typical range after good denoising: 0.7–1.3.
    Flag threshold: > 1.5 (printed as [DVARS HIGH]).
    Runtime: < 1 ms.
    """
    if ts.shape[0] < 2:
        return float("nan")
    ts_std = ts / (ts.std(axis=0, keepdims=True) + 1e-12)
    diff   = np.diff(ts_std, axis=0)                        # (T-1) × n_rois
    dvars  = np.sqrt(np.mean(diff ** 2, axis=1))            # RMS across ROIs per frame
    return float(np.mean(dvars))


# ─────────────────────────────────────────────────────────────────────────────
# tSNR + lag-1 autocorrelation
# ─────────────────────────────────────────────────────────────────────────────

def ts_qc_metrics(ts_clean: np.ndarray,
                  ts_raw: Optional[np.ndarray] = None) -> dict:
    """Cheap timeseries-level quality metrics (< 1 ms).

    tSNR
        mean / std across time, averaged over ROIs — computed from the RAW
        (pre-denoising, non-standardised) parcel timeseries so the mean is
        non-zero.  Typical preprocessed BOLD: 30–100.  Below ~20 suggests
        poor data quality.  Passed as ts_raw; falls back to NaN if unavailable.

    lag1_ac
        Mean lag-1 temporal autocorrelation of the CLEANED (z-scored) signal
        across ROIs.  Well-denoised BOLD retains ~0.4–0.7 autocorrelation
        because neural activity is temporally smooth.  Values < 0.25 mean the
        confound model removed too much signal and the residual looks like
        white noise — flag for over-denoising (especially relevant for the
        44-regressor strict config).
    """
    if ts_clean.shape[0] < 4:
        return {"tsnr": float("nan"), "lag1_ac": float("nan")}

    if ts_raw is not None and ts_raw.shape == ts_clean.shape:
        tsnr_per_roi = np.abs(ts_raw.mean(axis=0)) / (ts_raw.std(axis=0) + 1e-12)
        tsnr = float(np.nanmean(tsnr_per_roi))
    else:
        tsnr = float("nan")

    lag1_per_roi = np.array([
        float(np.corrcoef(ts_clean[:-1, j], ts_clean[1:, j])[0, 1])
        for j in range(ts_clean.shape[1])
    ])
    lag1_ac = float(np.nanmean(lag1_per_roi))

    return {"tsnr": tsnr, "lag1_ac": lag1_ac}


# ─────────────────────────────────────────────────────────────────────────────
# Confound variance explained (over-correction diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def confounds_variance_explained(raw_ts: np.ndarray,
                                 confounds: np.ndarray) -> float:
    """Mean R² across ROIs: fraction of variance explained by confound regression.

    High values (>0.6–0.8) suggest over-correction — confounds may be removing
    neural signal. Used as a diagnostic only; does not change the pipeline.
    """
    n_rois = raw_ts.shape[1]
    n_conf = confounds.shape[1]
    if n_conf >= raw_ts.shape[0] - 2:
        return np.nan

    r2_list = []
    for j in range(n_rois):
        y = raw_ts[:, j].astype(np.float64)
        y = y - np.mean(y)
        ss_total = np.sum(y ** 2)
        if ss_total < 1e-12:
            continue
        X = confounds.astype(np.float64)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            continue
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residual = y - (X @ beta)
            r2 = 1.0 - np.sum(residual ** 2) / ss_total
            r2_list.append(float(np.clip(r2, 0.0, 1.0)))
        except (np.linalg.LinAlgError, ValueError):
            continue

    return float(np.mean(r2_list)) if r2_list else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Louvain graph modularity
# ─────────────────────────────────────────────────────────────────────────────

def louvain_q(corr: np.ndarray) -> float:
    """Louvain modularity Q on the positive FC matrix.

    Uses the python-louvain (community) package when available, falls back
    to NetworkX greedy modularity — both give comparable values.
    Negative correlations are zeroed: community detection needs non-negative
    edge weights.  Returns NaN if graph has no edges or a library is missing.
    Runtime: ~0.05-0.2 s for a 400×400 matrix.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Network-level FC metrics (Schaefer-400 7-network labels)
# ─────────────────────────────────────────────────────────────────────────────

def fc_network_metrics(df_ts: pd.DataFrame, corr: np.ndarray) -> dict:
    """Compute Motor LH-RH interhemispheric r and within-SomMot / DMN / DMN-FPN r.

    Works for Schaefer-400 7-network parcellations where columns are labelled
    '7Networks_LH_SomMot_N' / '7Networks_RH_SomMot_N'.  Returns NaN for
    atlases that do not carry these labels (e.g. Tian S2).

    SomMot in the 7-network Yeo parcellation encompasses both primary motor
    (M1) and primary auditory (A1/STG) cortex, so within-SomMot r captures
    M1-A1 coupling alongside bilateral motor coupling — reported as context.
    """
    cols = list(df_ts.columns)

    lh_mot = [i for i, c in enumerate(cols) if "SomMot" in c and "LH" in c]
    rh_mot = [i for i, c in enumerate(cols) if "SomMot" in c and "RH" in c]

    motor_lr    = float("nan")
    somm_within = float("nan")

    if lh_mot and rh_mot:
        motor_lr = float(np.nanmean(corr[np.ix_(lh_mot, rh_mot)]))

    all_mot = lh_mot + rh_mot
    if len(all_mot) >= 2:
        within = corr[np.ix_(all_mot, all_mot)].copy()
        np.fill_diagonal(within, np.nan)
        somm_within = float(np.nanmean(within))

    dmn_idx    = [i for i, c in enumerate(cols) if "Default" in c]
    dmn_within = float("nan")
    if len(dmn_idx) >= 2:
        dmn_blk = corr[np.ix_(dmn_idx, dmn_idx)].copy()
        np.fill_diagonal(dmn_blk, np.nan)
        dmn_within = float(np.nanmean(dmn_blk))

    fpn_idx = [i for i, c in enumerate(cols) if "Cont" in c]
    dmn_fpn = float("nan")
    if dmn_idx and fpn_idx:
        dmn_fpn = float(np.nanmean(corr[np.ix_(dmn_idx, fpn_idx)]))

    return {
        "motor_lr":    motor_lr,
        "somm_within": somm_within,
        "dmn_within":  dmn_within,
        "dmn_fpn":     dmn_fpn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Private aliases for backward-compat with extractor's _-prefixed names
# ─────────────────────────────────────────────────────────────────────────────
_post_dvars                   = post_dvars
_ts_qc_metrics                = ts_qc_metrics
_confounds_variance_explained = confounds_variance_explained
_louvain_q                    = louvain_q
_fc_network_metrics           = fc_network_metrics
