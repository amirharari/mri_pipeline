"""
Analyze regional entropy differences in the 30-second animation data
to find which regions have the highest psilocybin - baseline differences.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch
from typing import List, Tuple, Dict


# Load data
BASELINE_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"
PSILOCYBIN_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-3_task-rest_aal_ts.csv"

TR_SEC = 1.0
WINDOW_SEC = 30
STEP_SEC = 5
WINDOW_PTS = int(WINDOW_SEC / TR_SEC)
STEP_PTS = int(STEP_SEC / TR_SEC)


def spectral_entropy(ts: np.ndarray, fs: float, normalize: bool = True) -> float:
    if ts is None or len(ts) < 10:
        return np.nan
    freqs, psd = welch(ts, fs=fs, nperseg=min(256, max(16, len(ts) // 4)))
    psd = np.asarray(psd)
    s = np.sum(psd)
    if s <= 0:
        return np.nan
    psd = psd / s
    psd = psd[psd > 0]
    if psd.size == 0:
        return np.nan
    H = -np.sum(psd * np.log2(psd))
    if normalize and psd.size > 1:
        H /= np.log2(psd.size)
    return float(H)


def sample_entropy(ts: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    x = np.asarray(ts, dtype=float)
    if x.size < m + 2 or np.allclose(np.std(x), 0.0):
        return np.nan
    r = r_frac * np.std(x)
    N = x.size

    def _phi(mm: int) -> float:
        patterns = np.array([x[i:i+mm] for i in range(N-mm)], dtype=float)
        if patterns.size == 0:
            return 0.0
        count = 0
        total = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i == j:
                    continue
                if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                    count += 1
                total += 1
        return (count / total) if total > 0 else 0.0

    Bm = _phi(m)
    Bm1 = _phi(m+1)
    if Bm == 0 or Bm1 == 0:
        return np.nan
    return float(-np.log(Bm1 / Bm))


def sliding_windows(n: int, w: int, s: int) -> List[Tuple[int, int]]:
    idx = []
    for start in range(0, n - w + 1, s):
        idx.append((start, start + w))
    return idx


def window_entropy_series(df: pd.DataFrame, w_pts: int, s_pts: int, fs: float, mode: str) -> List[Dict[str, float]]:
    """Return a list of dicts: per-frame {roi_name: entropy} for each sliding window."""
    frames: List[Dict[str, float]] = []
    T = df.shape[0]
    windows = sliding_windows(T, w_pts, s_pts)
    cols = list(df.columns)
    for (a, b) in windows:
        sub = df.iloc[a:b, :]
        fr: Dict[str, float] = {}
        for c in cols:
            ts = sub[c].dropna().values
            if mode == 'sampen':
                fr[c] = sample_entropy(ts, m=2, r_frac=0.2)
            else:
                fr[c] = spectral_entropy(ts, fs=fs, normalize=True)
        frames.append(fr)
    return frames


def main():
    print("=" * 80)
    print("ANALYZING REGIONAL ENTROPY DIFFERENCES")
    print("=" * 80)
    
    # Load data
    df_A = pd.read_csv(BASELINE_CSV).select_dtypes(include=[np.number])
    df_B = pd.read_csv(PSILOCYBIN_CSV).select_dtypes(include=[np.number])
    
    # Test both entropy modes
    for mode in ['sampen', 'spectral']:
        print(f"\n--- Using {mode} ---")
        
        # Compute entropy for all windows
        frames_A = window_entropy_series(df_A, WINDOW_PTS, STEP_PTS, TR_SEC, mode)
        frames_B = window_entropy_series(df_B, WINDOW_PTS, STEP_PTS, TR_SEC, mode)
        
        # Select top 30 frames with largest positive differences
        n_frames = len(frames_A)
        roi_keys = sorted(set(frames_A[0].keys()) | set(frames_B[0].keys()))
        
        # Compute mean difference per frame
        mean_diffs = []
        for i in range(n_frames):
            diffs = [frames_B[i].get(k, np.nan) - frames_A[i].get(k, np.nan) for k in roi_keys]
            mean_diffs.append(np.nanmean(diffs))
        
        # Select top 30 frames
        top_k = 30
        pos_mask = np.isfinite(mean_diffs) & (np.array(mean_diffs) > 0)
        candidate_idx = np.where(pos_mask)[0]
        if candidate_idx.size == 0:
            select_idx = np.argsort(mean_diffs)[-top_k:]
        else:
            select_idx = candidate_idx[np.argsort(np.array(mean_diffs)[candidate_idx])[-min(top_k, candidate_idx.size):]]
        select_idx = np.sort(select_idx)
        
        # Compute cumulative STD for selected frames
        accum_A = {k: [] for k in roi_keys}
        accum_B = {k: [] for k in roi_keys}
        
        for t in range(len(select_idx)):
            idx = select_idx[t]
            for k in roi_keys:
                accum_A[k].append(frames_A[idx].get(k, np.nan))
                accum_B[k].append(frames_B[idx].get(k, np.nan))
        
        # Compute STD across frames for each ROI
        std_A = {k: np.nanstd(vals) for k, vals in accum_A.items()}
        std_B = {k: np.nanstd(vals) for k, vals in accum_B.items()}
        
        # Compute difference in STD (higher STD in psilocybin = higher variability)
        std_diff = {k: std_B.get(k, 0) - std_A.get(k, 0) for k in roi_keys}
        
        # Sort regions by difference
        sorted_regions = sorted(std_diff.items(), key=lambda x: x[1], reverse=True)
        
        # Print top 20 regions
        print(f"\nTop 20 regions with highest psilocybin - baseline STD difference:")
        print(f"{'Region':<30} {'Baseline STD':>15} {'Psilocybin STD':>15} {'Difference':>15}")
        print("-" * 80)
        for roi, diff_val in sorted_regions[:20]:
            base_std = std_A.get(roi, 0)
            psilo_std = std_B.get(roi, 0)
            print(f"{roi:<30} {base_std:>15.4f} {psilo_std:>15.4f} {diff_val:>15.4f}")
        
        # Also compute mean differences per region
        mean_vals_A = {k: np.nanmean(vals) for k, vals in accum_A.items()}
        mean_vals_B = {k: np.nanmean(vals) for k, vals in accum_B.items()}
        mean_diff = {k: mean_vals_B.get(k, 0) - mean_vals_A.get(k, 0) for k in roi_keys}
        
        # Sort by mean difference
        sorted_means = sorted(mean_diff.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n\nTop 20 regions with highest psilocybin - baseline mean entropy:")
        print(f"{'Region':<30} {'Baseline Mean':>15} {'Psilocybin Mean':>15} {'Difference':>15}")
        print("-" * 80)
        for roi, diff_val in sorted_means[:20]:
            base_mean = mean_vals_A.get(roi, 0)
            psilo_mean = mean_vals_B.get(roi, 0)
            print(f"{roi:<30} {base_mean:>15.4f} {psilo_mean:>15.4f} {diff_val:>15.4f}")


if __name__ == "__main__":
    main()

