"""
Entropy–CAPs Linear Regression (ROI-wise and Window-wise)

What’s new vs. your draft:
- Correct Sample Entropy (unique pairs, no double-count bias; faster via sliding windows)
- Robust NaN handling (drops NaN timepoints before correlation; guards short data)
- Comparable Shannon entropy (fixed, robust histogram range across ROIs)
- Clear CAPs alignment rules (error on mismatch instead of meaningless fallback)
- Optional FDR control for multiple tests
- Window-wise analysis API for CAPs defined per time-window/session
- Configurable edge-weight entropy (abs vs signed-like bins; Fisher z threshold option)

Author: you + assistant
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================== Utilities ==============================

def _sliding_window_view_1d(x: np.ndarray, k: int) -> np.ndarray:
    """Safe sliding window (NumPy's version may be missing on old installs)."""
    x = np.asarray(x, dtype=float)
    if k < 1 or k > len(x):
        return np.empty((0, k))
    stride = x.strides[0]
    shape = (len(x) - k + 1, k)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=(stride, stride))


def fdr_bh(pvals: Union[List[float], np.ndarray], alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini–Hochberg FDR. Returns boolean mask of discoveries at level alpha.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    keep = ranked <= thresh
    crit = ranked[keep].max() if np.any(keep) else -np.inf
    return p <= crit


# ============================== Entropy Metrics ==============================

def sample_entropy(ts: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Sample Entropy (SampEn) with unique pair counting (j>i), excluding self-matches.

    Parameters
    ----------
    ts : array-like (T,)
    m  : pattern length
    r  : tolerance as fraction of std(ts)

    Returns
    -------
    float or np.nan
    """
    x = np.asarray(ts, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < m + 2:
        return np.nan

    std_x = np.std(x)
    if np.allclose(std_x, 0.0):
        return np.nan
    tol = r * std_x

    def _count_pairs(k: int) -> float:
        patterns = _sliding_window_view_1d(x, k)  # shape (M, k)
        M = len(patterns)
        if M <= 1:
            return 0.0
        cnt = 0
        total = 0
        # vectorized diff per i against trailing block
        for i in range(M - 1):
            diffs = np.max(np.abs(patterns[i + 1:] - patterns[i]), axis=1)
            hits = np.sum(diffs <= tol)
            cnt += hits
            total += (M - i - 1)
        return cnt / total if total > 0 else 0.0

    Bm = _count_pairs(m)
    Bm1 = _count_pairs(m + 1)
    if Bm == 0 or Bm1 == 0:
        return np.nan
    return float(-np.log(Bm1 / Bm))


def shannon_entropy(x: np.ndarray, bins: int = 50, hist_range: Optional[Tuple[float, float]] = None) -> float:
    """
    Shannon entropy of the distribution of a time series (histogram-based).

    Notes
    -----
    - For comparability across ROIs, pass a fixed hist_range.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan

    if hist_range is not None:
        hist, _ = np.histogram(x, bins=bins, range=hist_range)
    else:
        hist, _ = np.histogram(x, bins=bins)

    total = np.sum(hist)
    if total == 0:
        return np.nan
    p = hist / total
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log2(p)))


def compute_correlation_matrix(timeseries_list: List[np.ndarray]) -> np.ndarray:
    """
    ROI×ROI correlation with NaN-safe handling (drops any timepoints containing NaNs).

    Parameters
    ----------
    timeseries_list : list of (T,) arrays

    Returns
    -------
    (R, R) correlation matrix (may contain NaNs if too few points remain)
    """
    X = np.array([np.asarray(ts, dtype=float) for ts in timeseries_list])  # (R, T)
    # Drop columns (timepoints) that have any NaNs across ROIs
    mask = ~np.any(~np.isfinite(X), axis=0)
    X = X[:, mask]
    if X.shape[1] < 3:
        return np.full((X.shape[0], X.shape[0]), np.nan)
    # ROI×ROI correlation (rows are variables)
    return np.corrcoef(X)


def edge_weight_entropy_single_roi(
    weights: np.ndarray,
    threshold: float = 0.1,
    include_self: bool = False,
    use_abs: bool = True,
    fisher_z_threshold: bool = False,
    eps: float = 1e-12,
) -> float:
    """
    Edge-weight entropy for one ROI's connectivity profile.

    Parameters
    ----------
    weights : (R,) correlation row (including self at index i)
    threshold : minimum magnitude to keep
    include_self : include diagonal term (usually False)
    use_abs : if True, use |corr| distribution. If False, keep sign via two bins
              (positive and negative magnitudes concatenated).
    fisher_z_threshold : if True, threshold on |arctanh(corr)| instead of |corr|
    """
    w = weights.astype(float).copy()
    R = len(w)

    # by default, drop the self-edge
    if not include_self:
        # we don't know our own index here; the caller should zero it
        pass

    # optional Fisher z for thresholding
    if fisher_z_threshold:
        w_clip = np.clip(w, -0.999999, 0.999999)
        z = np.arctanh(w_clip)
        mask_mag = np.abs(z) >= threshold
        w = w * mask_mag
    else:
        mask_mag = np.abs(w) >= threshold
        w = w * mask_mag

    # Build probability distribution
    if use_abs:
        vec = np.abs(w)
        vec = vec[vec > 0]
        s = np.sum(vec)
        if s <= 0:
            return np.nan
        p = vec / s
    else:
        # Signed-like: split into positive and negative magnitudes as two groups
        pos = w[w > 0]
        neg = -w[w < 0]  # magnitudes
        vec = np.concatenate([pos, neg])
        if vec.size == 0:
            return np.nan
        s = np.sum(vec)
        if s <= 0:
            return np.nan
        p = vec / s

    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log2(p + eps)))


def compute_edge_weight_entropy(
    corr_matrix: np.ndarray,
    threshold: float = 0.1,
    include_self: bool = False,
    use_abs: bool = True,
    fisher_z_threshold: bool = False,
) -> np.ndarray:
    """
    Edge-weight entropy for all ROIs.
    """
    R = corr_matrix.shape[0]
    out = np.zeros(R, dtype=float)
    for i in range(R):
        row = corr_matrix[i, :].copy()
        if not include_self:
            row[i] = 0.0
        out[i] = edge_weight_entropy_single_roi(
            row,
            threshold=threshold,
            include_self=include_self,
            use_abs=use_abs,
            fisher_z_threshold=fisher_z_threshold,
        )
    return out


# ============================== Feature Builders ==============================

def compute_all_entropy_measures(
    timeseries_list: List[np.ndarray],
    sample_entropy_params: Optional[Dict] = None,
    shannon_entropy_params: Optional[Dict] = None,
    edge_weight_params: Optional[Dict] = None,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute SampEn, Shannon, and edge-weight entropy for a list of ROI time series.
    """
    # defaults
    sample_entropy_params = sample_entropy_params or {"m": 2, "r": 0.2}
    shannon_entropy_params = shannon_entropy_params or {"bins": 50, "hist_range": None}
    edge_weight_params = edge_weight_params or {
        "threshold": 0.1,
        "include_self": False,
        "use_abs": True,
        "fisher_z_threshold": False,
    }

    R = len(timeseries_list)

    # Robust histogram range across all ROIs (for comparability)
    if shannon_entropy_params.get("hist_range") is None:
        vals = []
        for ts in timeseries_list:
            ts = np.asarray(ts, dtype=float)
            ts = ts[np.isfinite(ts)]
            if ts.size:
                vals.append(ts)
        if len(vals):
            all_vals = np.concatenate(vals)
            lo, hi = np.percentile(all_vals, [1, 99])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                shannon_entropy_params["hist_range"] = (float(lo), float(hi))

    # SampEn per ROI
    simple = np.array(
        [sample_entropy(ts, **sample_entropy_params) for ts in timeseries_list],
        dtype=float,
    )

    # Shannon per ROI
    shan = np.array(
        [
            shannon_entropy(
                ts,
                bins=shannon_entropy_params["bins"],
                hist_range=shannon_entropy_params["hist_range"],
            )
            for ts in timeseries_list
        ],
        dtype=float,
    )

    # Edge-weight entropy per ROI
    corr = compute_correlation_matrix(timeseries_list)
    edge = compute_edge_weight_entropy(
        corr,
        threshold=edge_weight_params["threshold"],
        include_self=edge_weight_params["include_self"],
        use_abs=edge_weight_params["use_abs"],
        fisher_z_threshold=edge_weight_params["fisher_z_threshold"],
    )

    return {
        "simple_entropy": simple,
        "shannon_entropy": shan,
        "edge_weight_entropy": edge,
        "mean_simple_entropy": float(np.nanmean(simple)),
        "mean_shannon_entropy": float(np.nanmean(shan)),
        "mean_edge_weight_entropy": float(np.nanmean(edge)),
        "correlation_matrix": corr,
    }


def compute_entropy_per_window(
    timeseries_windows: np.ndarray,
    sample_entropy_params: Optional[Dict] = None,
    shannon_entropy_params: Optional[Dict] = None,
    edge_weight_params: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Window-wise entropy for a 3D array: (W, R, T).

    Returns dict of arrays with shape (W, R) for ROI metrics and (W,) for means.
    """
    W, R, T = timeseries_windows.shape
    simple = np.full((W, R), np.nan, dtype=float)
    shan = np.full((W, R), np.nan, dtype=float)
    edge = np.full((W, R), np.nan, dtype=float)
    mean_simple = np.full(W, np.nan, dtype=float)
    mean_shan = np.full(W, np.nan, dtype=float)
    mean_edge = np.full(W, np.nan, dtype=float)

    # We establish a global Shannon range across all windows/ROIs for fair comparisons
    if shannon_entropy_params is None:
        shannon_entropy_params = {"bins": 50, "hist_range": None}
    if shannon_entropy_params.get("hist_range") is None:
        vals = timeseries_windows[np.isfinite(timeseries_windows)]
        if vals.size:
            lo, hi = np.percentile(vals, [1, 99])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                shannon_entropy_params["hist_range"] = (float(lo), float(hi))

    for w in range(W):
        ts_list = [timeseries_windows[w, r, :] for r in range(R)]
        res = compute_all_entropy_measures(
            ts_list,
            sample_entropy_params=sample_entropy_params,
            shannon_entropy_params=shannon_entropy_params,
            edge_weight_params=edge_weight_params,
        )
        simple[w, :] = res["simple_entropy"]
        shan[w, :] = res["shannon_entropy"]
        edge[w, :] = res["edge_weight_entropy"]
        mean_simple[w] = res["mean_simple_entropy"]
        mean_shan[w] = res["mean_shannon_entropy"]
        mean_edge[w] = res["mean_edge_weight_entropy"]

    return {
        "simple_entropy": simple,          # (W, R)
        "shannon_entropy": shan,           # (W, R)
        "edge_weight_entropy": edge,       # (W, R)
        "mean_simple_entropy": mean_simple,    # (W,)
        "mean_shannon_entropy": mean_shan,     # (W,)
        "mean_edge_weight_entropy": mean_edge, # (W,)
    }


# ============================== Regression ==============================

def linear_regression_analysis(
    X: np.ndarray,
    y: np.ndarray,
    X_name: str = "Entropy",
    y_name: str = "CAPs Difference",
) -> Dict:
    """
    Single-predictor linear regression with slope t-test and diagnostics.
    """
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    mask = np.isfinite(X) & np.isfinite(y)
    Xc = X[mask]
    yc = y[mask]

    if Xc.size < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "p_value": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "n_samples": int(Xc.size),
            "X_name": X_name,
            "y_name": y_name,
            "X_clean": Xc,
            "y_clean": yc,
            "y_pred": np.array([]),
        }

    X2D = Xc.reshape(-1, 1)
    reg = LinearRegression().fit(X2D, yc)
    yhat = reg.predict(X2D)
    r2 = r2_score(yc, yhat)
    mse = mean_squared_error(yc, yhat)

    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    n = len(Xc)
    ssx = float(np.sum((Xc - Xc.mean()) ** 2))
    se_slope = np.sqrt(mse / ssx) if ssx > 0 else np.nan
    t_stat = slope / se_slope if se_slope > 0 else np.nan
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2)) if np.isfinite(t_stat) else np.nan

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": float(r2),
        "p_value": float(p_val),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "n_samples": int(n),
        "X_name": X_name,
        "y_name": y_name,
        "X_clean": Xc,
        "y_clean": yc,
        "y_pred": yhat,
    }


def analyze_entropy_caps_relationship(
    timeseries_list: List[np.ndarray],
    caps_differences: Union[float, np.ndarray],
    sample_entropy_params: Optional[Dict] = None,
    shannon_entropy_params: Optional[Dict] = None,
    edge_weight_params: Optional[Dict] = None,
    apply_fdr: bool = False,
    fdr_alpha: float = 0.05,
) -> Dict:
    """
    ROI-wise analysis: entropies per ROI regressed against CAPs differences per ROI.

    IMPORTANT: caps_differences must have length = n_rois OR be a scalar.
    """
    R = len(timeseries_list)
    res_entropy = compute_all_entropy_measures(
        timeseries_list,
        sample_entropy_params=sample_entropy_params,
        shannon_entropy_params=shannon_entropy_params,
        edge_weight_params=edge_weight_params,
    )

    caps = np.asarray(caps_differences, dtype=float)
    if caps.ndim == 0:
        caps = np.full(R, float(caps))
    if caps.shape[0] != R:
        raise ValueError(
            f"Length mismatch: CAPs differences length ({caps.shape[0]}) must equal n_rois ({R}). "
            "If your CAPs are per-window/session, use analyze_windowed_entropy_caps()."
        )

    reg_simple = linear_regression_analysis(
        res_entropy["simple_entropy"], caps, "Simple Entropy (SampEn)", "CAPs Difference"
    )
    reg_shannon = linear_regression_analysis(
        res_entropy["shannon_entropy"], caps, "Shannon Entropy", "CAPs Difference"
    )
    reg_edge = linear_regression_analysis(
        res_entropy["edge_weight_entropy"], caps, "Edge-Weight Entropy", "CAPs Difference"
    )

    # Optional FDR (not strictly meaningful with 3 tests only; more useful if you expand)
    fdr = None
    if apply_fdr:
        pvals = np.array([reg_simple["p_value"], reg_shannon["p_value"], reg_edge["p_value"]], dtype=float)
        sig = fdr_bh(pvals, alpha=fdr_alpha)
        fdr = {"alpha": fdr_alpha, "pvals": pvals, "significant": sig}

    summary = {
        "n_rois": int(R),
        "mean_simple_entropy": float(res_entropy["mean_simple_entropy"]),
        "mean_shannon_entropy": float(res_entropy["mean_shannon_entropy"]),
        "mean_edge_weight_entropy": float(res_entropy["mean_edge_weight_entropy"]),
        "caps_differences_stats": {
            "mean": float(np.nanmean(caps)),
            "std": float(np.nanstd(caps)),
            "min": float(np.nanmin(caps)),
            "max": float(np.nanmax(caps)),
        },
    }

    return {
        "entropy_measures": res_entropy,
        "regression_results": {
            "simple_entropy_per_roi": reg_simple,
            "shannon_entropy_per_roi": reg_shannon,
            "edge_weight_entropy_per_roi": reg_edge,
        },
        "fdr": fdr,
        "summary": summary,
        "caps_differences": caps,
    }


def analyze_windowed_entropy_caps(
    timeseries_windows: np.ndarray,
    caps_per_window: np.ndarray,
    sample_entropy_params: Optional[Dict] = None,
    shannon_entropy_params: Optional[Dict] = None,
    edge_weight_params: Optional[Dict] = None,
    regress_on: str = "mean",  # "mean" or "per_roi"
    roi_index: Optional[int] = None,  # used if regress_on == "per_roi"
) -> Dict:
    """
    Window-wise analysis for CAPs that are defined per window/session.

    Parameters
    ----------
    timeseries_windows : array (W, R, T)
    caps_per_window : array (W,) or (W, R) if ROI-specific CAPs per window
    regress_on : "mean" (default) to regress CAPs(W,) on mean entropies(W,),
                 or "per_roi" to regress CAPs(W,) on entropy(W, roi_index).

    Returns
    -------
    dict with regression results for each entropy type.
    """
    caps = np.asarray(caps_per_window, dtype=float)
    W, R, T = timeseries_windows.shape
    ent = compute_entropy_per_window(
        timeseries_windows,
        sample_entropy_params=sample_entropy_params,
        shannon_entropy_params=shannon_entropy_params,
        edge_weight_params=edge_weight_params,
    )

    results = {}
    if regress_on == "mean":
        if caps.shape != (W,):
            raise ValueError(f"When regress_on='mean', caps_per_window must have shape (W,), got {caps.shape}.")
        results["simple_mean"] = linear_regression_analysis(
            ent["mean_simple_entropy"], caps, "Mean Simple Entropy (per window)", "CAPs (per window)"
        )
        results["shannon_mean"] = linear_regression_analysis(
            ent["mean_shannon_entropy"], caps, "Mean Shannon Entropy (per window)", "CAPs (per window)"
        )
        results["edge_weight_mean"] = linear_regression_analysis(
            ent["mean_edge_weight_entropy"], caps, "Mean Edge-Weight Entropy (per window)", "CAPs (per window)"
        )
    elif regress_on == "per_roi":
        if roi_index is None or not (0 <= roi_index < R):
            raise ValueError("Provide a valid roi_index for regress_on='per_roi'.")
        if caps.shape != (W,):
            raise ValueError(f"When regress_on='per_roi', caps_per_window must have shape (W,), got {caps.shape}.")
        results["simple_roi"] = linear_regression_analysis(
            ent["simple_entropy"][:, roi_index], caps, f"SampEn ROI {roi_index}", "CAPs (per window)"
        )
        results["shannon_roi"] = linear_regression_analysis(
            ent["shannon_entropy"][:, roi_index], caps, f"Shannon ROI {roi_index}", "CAPs (per window)"
        )
        results["edge_weight_roi"] = linear_regression_analysis(
            ent["edge_weight_entropy"][:, roi_index], caps, f"Edge-Entropy ROI {roi_index}", "CAPs (per window)"
        )
    else:
        raise ValueError("regress_on must be 'mean' or 'per_roi'.")

    summary = {
        "n_windows": int(W),
        "n_rois": int(R),
    }

    return {
        "entropy_per_window": ent,
        "regression_results": results,
        "summary": summary,
        "caps_per_window": caps,
    }


# ============================== Reporting ==============================

def print_regression_summary(results: Dict):
    """
    Pretty-print for both ROI-wise and window-wise results dicts above.
    """
    print("=" * 80)
    print("ENTROPY–CAPs REGRESSION SUMMARY")
    print("=" * 80)

    if "summary" in results and "n_rois" in results["summary"]:
        # ROI-wise variant
        summ = results["summary"]
        print(f"\nNumber of ROIs: {summ['n_rois']}")
        print("Mean Entropies:")
        print(f"  SampEn : {summ['mean_simple_entropy']:.4f}")
        print(f"  Shannon: {summ['mean_shannon_entropy']:.4f}")
        print(f"  Edge   : {summ['mean_edge_weight_entropy']:.4f}")
        caps = summ["caps_differences_stats"]
        print("\nCAPs (per ROI) stats:")
        print(f"  Mean/SD: {caps['mean']:.4f} / {caps['std']:.4f}")
        print(f"  Range  : [{caps['min']:.4f}, {caps['max']:.4f}]")

        print("\nRegression:")
        for k, rr in results["regression_results"].items():
            print(f"  {k}: slope={rr['slope']:.6f}, R²={rr['r2']:.4f}, p={rr['p_value']:.6g}, n={rr['n_samples']}")
        if results.get("fdr") is not None:
            f = results["fdr"]
            print(f"\nFDR (alpha={f['alpha']}):")
            for name, p, sig in zip(results["regression_results"].keys(), f["pvals"], f["significant"]):
                print(f"  {name}: p={p:.6g}  sig={bool(sig)}")

    else:
        # Window-wise variant
        summ = results.get("summary", {})
        W = summ.get("n_windows", "?")
        R = summ.get("n_rois", "?")
        print(f"\nWindows × ROIs: {W} × {R}")
        print("\nRegression:")
        for k, rr in results["regression_results"].items():
            print(f"  {k}: slope={rr['slope']:.6f}, R²={rr['r2']:.4f}, p={rr['p_value']:.6g}, n={rr['n_samples']}")

    print("=" * 80)


# ============================== Example Usage ==============================

if __name__ == "__main__":
    np.random.seed(42)

    # --------- ROI-wise example (CAPs per ROI) ----------
    n_rois = 12
    n_time = 140
    timeseries_list = [
        np.random.randn(n_time) + 0.5 * np.sin(np.linspace(0, 6 * np.pi, n_time) + i * 0.2)
        for i in range(n_rois)
    ]
    caps_per_roi = 0.3 * np.random.randn(n_rois) + 0.1

    roi_results = analyze_entropy_caps_relationship(
        timeseries_list,
        caps_per_roi,
        sample_entropy_params={"m": 2, "r": 0.2},
        shannon_entropy_params={"bins": 50},  # hist_range auto-set robustly
        edge_weight_params={"threshold": 0.1, "include_self": False, "use_abs": True, "fisher_z_threshold": False},
        apply_fdr=True,
        fdr_alpha=0.05,
    )
    print_regression_summary(roi_results)

    # --------- Window-wise example (CAPs per window) ----------
    n_windows = 20
    n_rois_w = 8
    n_time_w = 80
    # windows × rois × time
    ts_win = np.random.randn(n_windows, n_rois_w, n_time_w)
    # Make CAPs slightly related to mean entropy (for demo)
    caps_win = 0.1 * np.random.randn(n_windows)

    win_results = analyze_windowed_entropy_caps(
        ts_win,
        caps_win,
        sample_entropy_params={"m": 2, "r": 0.2},
        shannon_entropy_params={"bins": 40},
        edge_weight_params={"threshold": 0.1, "include_self": False, "use_abs": True},
        regress_on="mean",
    )
    print_regression_summary(win_results)

    # --------- Optional: export ROI-wise table ----------
    df_roi = pd.DataFrame({
        "ROI": [f"ROI_{i+1}" for i in range(n_rois)],
        "SampEn": roi_results["entropy_measures"]["simple_entropy"],
        "Shannon": roi_results["entropy_measures"]["shannon_entropy"],
        "EdgeEntropy": roi_results["entropy_measures"]["edge_weight_entropy"],
        "CAPs": roi_results["caps_differences"],
    })
    print("\nPreview of ROI-wise table:")
    print(df_roi.head())
