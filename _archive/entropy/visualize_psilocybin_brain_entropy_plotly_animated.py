"""
============================================================================
Animated Psychedelic Brain: Side-by-side fsaverage 3D with ROI Entropy over Time
============================================================================

Creates an animated Plotly HTML where LEFT is Baseline entropy, RIGHT is
Psilocybin entropy. Each frame corresponds to a sliding window along the
timeseries. Designed for compelling, social-ready visuals.

Inputs
------
- Two AAL timeseries CSVs (columns=ROI names, rows=timepoints)

Outputs
-------
- HTML with play/pause controls (self-contained)
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, surface
import plotly.graph_objects as go
from scipy.signal import welch


# ========================= USER INPUT =========================
CONDITION_A_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"  # Baseline
CONDITION_B_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-3_task-rest_aal_ts.csv"  # Psilocybin

TITLE_A = "Baseline"
TITLE_B = "Psilocybin"

TR_SEC = 1.0            # seconds
WINDOW_SEC = 30         # sliding window (seconds)
STEP_SEC = 5            # step (seconds)

# Entropy modes to try; script will auto-pick the metric and frames
ENTROPY_MODES = ['sampen', 'spectral']

BASE_COLORSCALE = "Cividis"
PSILO_COLORSCALE = "Turbo"
BACKGROUND_COLOR = "black"
TEXT_COLOR = "white"
OPACITY = 1.0
HEMISPHERE_SHIFT = 55

OUT_HTML = "psilo_vs_base_entropy_animated.html"
# =============================================================


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
    """Sample Entropy (SampEn) with self-matches excluded.
    r = r_frac * std(ts). Returns NaN if insufficient matches.
    """
    x = np.asarray(ts, dtype=float)
    if x.size < m + 2 or np.allclose(np.std(x), 0.0):
        return np.nan
    r = r_frac * np.std(x)
    N = x.size

    def _phi(mm: int) -> float:
        # build embedding vectors
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


def load_numeric(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).select_dtypes(include=[np.number])
    # Drop constant/all-NaN columns
    keep = [c for c in df.columns if df[c].std(ddof=0) > 0 and not df[c].isna().all()]
    return df[keep]


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


def fetch_aal_spm12() -> Tuple[dict, nib.Nifti1Image, np.ndarray, Dict[str, int]]:
    aal = datasets.fetch_atlas_aal(version="SPM12")
    atlas_img = nib.load(aal["maps"])
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)
    labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    if "indices" in aal:
        indices_raw = list(aal["indices"])  # voxel-value per label
    else:
        indices_raw = sorted(int(v) for v in np.unique(atlas) if v > 0)[: len(labels)]
    name_to_index = {labels[i]: int(indices_raw[i]) for i in range(len(labels))}
    return aal, atlas_img, atlas, name_to_index


def fetch_fsaverage_surfaces():
    fsavg = datasets.fetch_surf_fsaverage()
    return fsavg["pial_left"], fsavg["pial_right"], fsavg["white_left"], fsavg["white_right"]


def sample_labels(atlas_img, white_mesh, pial_mesh):
    try:
        lbl = surface.vol_to_surf(
            atlas_img, pial_mesh,
            inner_mesh=white_mesh, kind="line", n_samples=25, interpolation="nearest"
        )
    except TypeError:
        lbl = surface.vol_to_surf(atlas_img, pial_mesh)
    lbl = np.rint(np.asarray(lbl)).astype(np.int32)
    lbl[lbl < 0] = 0
    return lbl


def build_vertex_data(surf_mesh, labels_on_vertices, roi_index_to_val: Dict[int, float], idx_to_name: Dict[int, str]):
    coords, faces = surface.load_surf_mesh(surf_mesh)
    intens = np.zeros(coords.shape[0], dtype=float)
    names = np.empty(coords.shape[0], dtype=object)
    for idx_val, v in roi_index_to_val.items():
        mask = labels_on_vertices == idx_val
        if np.any(mask):
            intens[mask] = 0.0 if v is None or np.isnan(v) else float(v)
            names[mask] = idx_to_name.get(idx_val, "")
    return coords, faces, intens, names


def map_roi_to_indices(roi_to_val: Dict[str, float], name_to_index: Dict[str, int]) -> Tuple[Dict[int, float], Dict[int, str]]:
    val_map: Dict[int, float] = {}
    name_map: Dict[int, str] = {}
    for roi, v in roi_to_val.items():
        idx = name_to_index.get(roi)
        if idx is not None:
            val_map[idx] = v
            name_map[idx] = roi
    return val_map, name_map


def configure_dual_scene_layout(fig: go.Figure, metric_name: str = "Entropy"):
    # Format metric name for display
    metric_display = metric_name.replace("sampen", "Sample").replace("spectral", "Spectral")
    fig.update_layout(
        title=dict(text=f"Regional {metric_display} Entropy Variability (AAL): Baseline vs Psilocybin",
                   font=dict(color=TEXT_COLOR, size=22)),
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=0, r=0, t=50, b=0),
        scene_domain=dict(x=[0.0, 0.49], y=[0.0, 1.0]),
        scene2_domain=dict(x=[0.51, 1.0], y=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.02)
    )


def main():
    print("Building animated fsaverage 3D entropy visuals (Plotly)...")

    # Load series
    A = load_numeric(CONDITION_A_CSV)
    B = load_numeric(CONDITION_B_CSV)
    fs = 1.0 / TR_SEC
    w_pts = max(1, int(round(WINDOW_SEC / TR_SEC)))
    s_pts = max(1, int(round(STEP_SEC / TR_SEC)))

    # Evaluate multiple entropy modes and select the metric + frames with the largest psilo-baseline separation
    mode_to_data = {}
    for mode in ENTROPY_MODES:
        fA = window_entropy_series(A, w_pts, s_pts, fs, mode)
        fB = window_entropy_series(B, w_pts, s_pts, fs, mode)
        n = min(len(fA), len(fB))
        # per-frame mean difference (psilo - baseline)
        diffs = []
        for t in range(n):
            valsA = np.array(list(fA[t].values()), dtype=float)
            valsB = np.array(list(fB[t].values()), dtype=float)
            diffs.append(float(np.nanmean(valsB) - np.nanmean(valsA)))
        mode_to_data[mode] = dict(frames_A=fA, frames_B=fB, diffs=np.array(diffs), n=n)

    # Pick mode with maximum positive separation
    best_mode = None
    best_score = -np.inf
    for mode, d in mode_to_data.items():
        score = np.nanmax(d['diffs']) if d['diffs'].size else -np.inf
        if score > best_score:
            best_score = score
            best_mode = mode
    frames_A = mode_to_data[best_mode]['frames_A']
    frames_B = mode_to_data[best_mode]['frames_B']
    diffs = mode_to_data[best_mode]['diffs']
    n_frames = mode_to_data[best_mode]['n']
    print(f"Frames: {n_frames} (window {w_pts}pts, step {s_pts}pts) | Chosen metric: {best_mode} | Max d={np.nanmax(diffs):.3f}")

    # Select top 30 frames with largest positive psilo-baseline mean difference
    top_k = 30
    pos_mask = np.isfinite(diffs) & (diffs > 0)
    candidate_idx = np.where(pos_mask)[0]
    if candidate_idx.size == 0:
        select_idx = np.argsort(diffs)[-top_k:]
    else:
        select_idx = candidate_idx[np.argsort(diffs[candidate_idx])[-min(top_k, candidate_idx.size):]]
    # Sort selected frames by time for smoother storytelling
    select_idx = np.sort(select_idx)

    # Restrict to selected frames only
    frames_A = [frames_A[i] for i in select_idx]
    frames_B = [frames_B[i] for i in select_idx]
    n_frames = len(frames_A)

    # Build accumulators per ROI for cumulative STD visualization
    roi_keys = sorted(set(frames_A[0].keys()) | set(frames_B[0].keys()))
    accum_A = {k: [] for k in roi_keys}
    accum_B = {k: [] for k in roi_keys}

    # Atlas and surfaces
    _, atlas_img, atlas, name_to_index = fetch_aal_spm12()
    pial_L, pial_R, white_L, white_R = fetch_fsaverage_surfaces()
    lbl_L = sample_labels(atlas_img, white_L, pial_L)
    lbl_R = sample_labels(atlas_img, white_R, pial_R)

    # Build static geometry (coords/faces)
    coords_L, faces_L = surface.load_surf_mesh(pial_L)
    coords_R, faces_R = surface.load_surf_mesh(pial_R)
    coords_L_shift = coords_L.copy(); coords_L_shift[:, 0] -= HEMISPHERE_SHIFT
    coords_R_shift = coords_R.copy(); coords_R_shift[:, 0] += HEMISPHERE_SHIFT

    # Initial intensities (frame 0) as STD so far (single window -> std = 0)
    for k in roi_keys:
        accum_A[k].append(frames_A[0].get(k, np.nan))
        accum_B[k].append(frames_B[0].get(k, np.nan))
    std_A0 = {k: 0.0 for k in roi_keys}
    std_B0 = {k: 0.0 for k in roi_keys}
    A_idx_val0, A_idx_name = map_roi_to_indices(std_A0, name_to_index)
    B_idx_val0, B_idx_name = map_roi_to_indices(std_B0, name_to_index)

    _, _, intens_L_A0, names_L_A = build_vertex_data(pial_L, lbl_L, A_idx_val0, A_idx_name)
    _, _, intens_R_A0, names_R_A = build_vertex_data(pial_R, lbl_R, A_idx_val0, A_idx_name)
    _, _, intens_L_B0, names_L_B = build_vertex_data(pial_L, lbl_L, B_idx_val0, B_idx_name)
    _, _, intens_R_B0, names_R_B = build_vertex_data(pial_R, lbl_R, B_idx_val0, B_idx_name)

    # Color scale (fixed across frames & conditions)
    # Determine color range from final cumulative STD (computed below). Placeholder for now.
    cmin, cmax = 0.0, 1.0

    # Figure with 4 Mesh3d traces (2 scenes)
    fig = go.Figure()
    hover_tmpl = "<b>%{customdata}</b><br>Entropy: %{intensity:.2f}<extra></extra>"
    lighting = dict(ambient=0.35, diffuse=0.7, specular=0.6, roughness=0.4)

    # Baseline L/R (scene1)
    fig.add_trace(go.Mesh3d(x=coords_L_shift[:,0], y=coords_L_shift[:,1], z=coords_L_shift[:,2],
                            i=faces_L[:,0], j=faces_L[:,1], k=faces_L[:,2],
                            intensity=intens_L_A0, cmin=cmin, cmax=cmax, colorscale='Turbo',
                            opacity=OPACITY, lighting=lighting, showscale=True,
                            customdata=names_L_A, hovertemplate=hover_tmpl, scene="scene1"))
    fig.add_trace(go.Mesh3d(x=coords_R_shift[:,0], y=coords_R_shift[:,1], z=coords_R_shift[:,2],
                            i=faces_R[:,0], j=faces_R[:,1], k=faces_R[:,2],
                            intensity=intens_R_A0, cmin=cmin, cmax=cmax, colorscale='Turbo',
                            opacity=OPACITY, lighting=lighting, showscale=False,
                            customdata=names_R_A, hovertemplate=hover_tmpl, scene="scene1"))

    # Psilocybin L/R (scene2)
    fig.add_trace(go.Mesh3d(x=coords_L_shift[:,0], y=coords_L_shift[:,1], z=coords_L_shift[:,2],
                            i=faces_L[:,0], j=faces_L[:,1], k=faces_L[:,2],
                            intensity=intens_L_B0, cmin=cmin, cmax=cmax, colorscale='Turbo',
                            opacity=OPACITY, lighting=lighting, showscale=True,
                            customdata=names_L_B, hovertemplate=hover_tmpl, scene="scene2"))
    fig.add_trace(go.Mesh3d(x=coords_R_shift[:,0], y=coords_R_shift[:,1], z=coords_R_shift[:,2],
                            i=faces_R[:,0], j=faces_R[:,1], k=faces_R[:,2],
                            intensity=intens_R_B0, cmin=cmin, cmax=cmax, colorscale='Turbo',
                            opacity=OPACITY, lighting=lighting, showscale=False,
                            customdata=names_R_B, hovertemplate=hover_tmpl, scene="scene2"))

    # Frames: update the 4 traces' intensity arrays per frame using cumulative STD
    # Precompute frame-level summary (mean entropy STD) for annotation
    def frame_means(idx_val_map):
        vals = np.array(list(idx_val_map.values()), dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else np.nan

    frames = []
    std_min, std_max = np.inf, -np.inf
    std_frames_A_idx = []
    std_frames_B_idx = []
    for t in range(n_frames):
        # Accumulate values
        for k in roi_keys:
            accum_A[k].append(frames_A[t].get(k, np.nan)) if t > 0 else None
            accum_B[k].append(frames_B[t].get(k, np.nan)) if t > 0 else None
        # Compute STD per ROI so far
        std_A = {k: float(np.nanstd(accum_A[k])) if len(accum_A[k]) > 1 else 0.0 for k in roi_keys}
        std_B = {k: float(np.nanstd(accum_B[k])) if len(accum_B[k]) > 1 else 0.0 for k in roi_keys}
        # Track global min/max for color scaling
        arrA = np.array(list(std_A.values()), dtype=float)
        arrB = np.array(list(std_B.values()), dtype=float)
        std_min = min(std_min, np.nanmin(arrA), np.nanmin(arrB))
        std_max = max(std_max, np.nanmax(arrA), np.nanmax(arrB))
        # Map to indices and cache
        A_idx_val, _ = map_roi_to_indices(std_A, name_to_index)
        B_idx_val, _ = map_roi_to_indices(std_B, name_to_index)
        std_frames_A_idx.append(A_idx_val)
        std_frames_B_idx.append(B_idx_val)
    # Set color range from observed std range
    if np.isfinite(std_min) and np.isfinite(std_max) and std_max > std_min:
        cmin, cmax = float(std_min), float(std_max)
    else:
        cmin, cmax = 0.0, 1.0
    # Rebuild initial intensities using std frame 0
    A_idx_val0 = std_frames_A_idx[0]
    B_idx_val0 = std_frames_B_idx[0]
    _, _, intens_L_A0, names_L_A = build_vertex_data(pial_L, lbl_L, A_idx_val0, A_idx_name)
    _, _, intens_R_A0, names_R_A = build_vertex_data(pial_R, lbl_R, A_idx_val0, A_idx_name)
    _, _, intens_L_B0, names_L_B = build_vertex_data(pial_L, lbl_L, B_idx_val0, B_idx_name)
    _, _, intens_R_B0, names_R_B = build_vertex_data(pial_R, lbl_R, B_idx_val0, B_idx_name)

    # Build frames with cumulative STD intensities
    for t in range(n_frames):
        A_idx_val = std_frames_A_idx[t]
        B_idx_val = std_frames_B_idx[t]
        _, _, intens_L_A, _ = build_vertex_data(pial_L, lbl_L, A_idx_val, A_idx_name)
        _, _, intens_R_A, _ = build_vertex_data(pial_R, lbl_R, A_idx_val, A_idx_name)
        _, _, intens_L_B, _ = build_vertex_data(pial_L, lbl_L, B_idx_val, B_idx_name)
        _, _, intens_R_B, _ = build_vertex_data(pial_R, lbl_R, B_idx_val, B_idx_name)
        meanA = frame_means(A_idx_val)
        meanB = frame_means(B_idx_val)
        diff = meanB - meanA if (np.isfinite(meanA) and np.isfinite(meanB)) else np.nan

        frames.append(go.Frame(data=[
            dict(type="mesh3d", intensity=intens_L_A),
            dict(type="mesh3d", intensity=intens_R_A),
            dict(type="mesh3d", intensity=intens_L_B),
            dict(type="mesh3d", intensity=intens_R_B),
        ], name=f"t{t:03d}", layout=dict(
            annotations=[
                dict(x=0.245, y=1.06, xref='paper', yref='paper',
                     text=f"{TITLE_A} mean H: {meanA:.2f}", showarrow=False, font=dict(color=TEXT_COLOR, size=16)),
                dict(x=0.755, y=1.06, xref='paper', yref='paper',
                     text=f"{TITLE_B} mean H: {meanB:.2f} (Δ {diff:+.2f})", showarrow=False, font=dict(color=TEXT_COLOR, size=16))
            ]
        )))

    fig.frames = frames

    # Animation controls
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "y": 1.05,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "bottom",
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 120, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}]
                }
            ]
        }]
    )

    # Configure scenes
    fig.update_layout(**{
        "scene": dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                       aspectmode="data", bgcolor=BACKGROUND_COLOR,
                       camera=dict(eye=dict(x=1.6, y=1.2, z=0.7))),
        "scene2": dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                        aspectmode="data", bgcolor=BACKGROUND_COLOR,
                        camera=dict(eye=dict(x=1.6, y=1.2, z=0.7)))
    })
    configure_dual_scene_layout(fig, best_mode)

    # Initial annotations for frame 0
    fig.update_layout(annotations=[
        dict(x=0.245, y=1.06, xref='paper', yref='paper',
             text=f"{TITLE_A} mean STD(H): {frame_means(std_frames_A_idx[0]):.2f}", showarrow=False, font=dict(color=TEXT_COLOR, size=16)),
        dict(x=0.755, y=1.06, xref='paper', yref='paper',
             text=f"{TITLE_B} mean STD(H): {frame_means(std_frames_B_idx[0]):.2f}", showarrow=False, font=dict(color=TEXT_COLOR, size=16))
    ])

    # Save
    fig.write_html(OUT_HTML, include_plotlyjs="inline", full_html=True, auto_open=False)
    print(f"Wrote: {OUT_HTML}")


if __name__ == "__main__":
    main()


