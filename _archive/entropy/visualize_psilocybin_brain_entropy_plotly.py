"""
============================================================================
Psychedelic Brain Visual: Plotly fsaverage 3D with AAL ROI Entropy
============================================================================

High-impact, social-ready visualization comparing baseline vs psilocybin
regional entropy on an fsaverage surface using Plotly (interactive HTML + PNG).

Data source: AAL time-series CSVs (columns = ROI names, rows = time).

What it does
------------
- Computes per-ROI spectral entropy (normalized to [0,1]) from each CSV
- Aggregates per condition (single file or folder of CSVs)
- Maps entropy to AAL atlas regions sampled onto fsaverage surface
- Renders two side-by-side 3D brain scenes (Baseline vs Psilocybin)

Outputs
-------
- HTML: interactive (rotate/zoom), self-contained
- PNG: high-res static (needs kaleido)
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
# Provide EITHER file paths OR directories with multiple *_aal_ts.csv files
CONDITION_A_PATH = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"  # Baseline
CONDITION_B_PATH = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-3_task-rest_aal_ts.csv"  # Psilocybin

TR_SEC = 1.0  # sampling rate for spectral entropy (fs = 1/TR)

TITLE_A = "Baseline"
TITLE_B = "Psilocybin"

COLORSCALE = "Plasma"  # "Turbo","Magma","Viridis","Plasma","Inferno","Cividis","Amp"
BACKGROUND_COLOR = "black"
TEXT_COLOR = "white"
OPACITY = 1.0
HEMISPHERE_SHIFT = 55  # mm separation between hemispheres

OUT_HTML = "psilo_vs_base_entropy_3d.html"
OUT_PNG = "psilo_vs_base_entropy_3d.png"
# =============================================================

EPS = 1e-16


def spectral_entropy(ts: np.ndarray, fs: float, normalize: bool = True) -> float:
    if ts is None or len(ts) < 10:
        return np.nan
    freqs, psd = welch(ts, fs=fs, nperseg=min(256, max(16, len(ts) // 4)))
    psd = np.asarray(psd)
    if np.sum(psd) <= 0:
        return np.nan
    psd = psd / np.sum(psd)
    psd = psd[psd > 0]
    if psd.size == 0:
        return np.nan
    H = -np.sum(psd * np.log2(psd))
    if normalize and psd.size > 1:
        H /= np.log2(psd.size)
    return float(H)


def load_entropy_from_csv(csv_path: str, fs: float) -> Dict[str, float]:
    df = pd.read_csv(csv_path).select_dtypes(include=[np.number])
    # Drop constant/all-NaN columns
    keep = [c for c in df.columns if df[c].std(ddof=0) > 0 and not df[c].isna().all()]
    df = df[keep]
    out: Dict[str, float] = {}
    for col in df.columns:
        ts = df[col].dropna().values
        out[col] = spectral_entropy(ts, fs=fs, normalize=True)
    return out


def aggregate_entropy(path: str, fs: float) -> Dict[str, float]:
    if os.path.isdir(path):
        vals: List[Dict[str, float]] = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith("_aal_ts.csv"):
                vals.append(load_entropy_from_csv(os.path.join(path, fname), fs))
        if not vals:
            raise FileNotFoundError(f"No _aal_ts.csv files in folder: {path}")
        keys = set().union(*[v.keys() for v in vals])
        return {k: np.nanmean([v.get(k, np.nan) for v in vals]) for k in keys}
    else:
        return load_entropy_from_csv(path, fs)


def fetch_aal_atlas() -> Tuple[dict, nib.Nifti1Image, np.ndarray, Dict[str, int]]:
    print("Fetching AAL (SPM12)...")
    aal = datasets.fetch_atlas_aal(version="SPM12")  # MNI space
    atlas_img = nib.load(aal["maps"])
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)
    labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    if "indices" in aal:
        indices_raw = list(aal["indices"])  # voxel values for each label
    else:
        # Fallback
        indices_raw = sorted(int(v) for v in np.unique(atlas) if v > 0)[: len(labels)]
    name_to_index = {labels[i]: int(indices_raw[i]) for i in range(len(labels))}
    return aal, atlas_img, atlas, name_to_index


def fetch_fsaverage_surfaces():
    print("Fetching fsaverage surfaces...")
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


def build_vertex_data(surf_mesh, labels_on_vertices, roi_index_to_intensity: Dict[int, float],
                      roi_name_map: Dict[int, str]):
    coords, faces = surface.load_surf_mesh(surf_mesh)
    intens = np.zeros(coords.shape[0], dtype=float)
    names = np.empty(coords.shape[0], dtype=object)
    for idx_val, inten in roi_index_to_intensity.items():
        mask = labels_on_vertices == idx_val
        if np.any(mask):
            intens[mask] = 0.0 if inten is None or np.isnan(inten) else float(inten)
            names[mask] = roi_name_map.get(idx_val, "")
    return coords, faces, intens, names


def entropy_map_to_indices(entropy_by_roi: Dict[str, float], name_to_index: Dict[str, int]) -> Tuple[Dict[int, float], Dict[int, str]]:
    out_val: Dict[int, float] = {}
    out_name: Dict[int, str] = {}
    for roi_name, ent in entropy_by_roi.items():
        idx = name_to_index.get(roi_name)
        if idx is not None:
            out_val[idx] = ent
            out_name[idx] = roi_name
    return out_val, out_name


def create_scene(fig: go.Figure, row: int, title: str,
                 coords_L_shift, faces_L, intens_L, names_L,
                 coords_R_shift, faces_R, intens_R, names_R,
                 cmin: float, cmax: float, colorscale: str, scene_id: str):
    hover_tmpl = "<b>%{customdata}</b><br>Entropy: %{intensity:.2f}<extra></extra>"
    lighting = dict(ambient=0.35, diffuse=0.6, specular=0.4, roughness=0.6)

    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:, 0], y=coords_L_shift[:, 1], z=coords_L_shift[:, 2],
        i=faces_L[:, 0], j=faces_L[:, 1], k=faces_L[:, 2],
        intensity=intens_L, cmin=cmin, cmax=cmax, colorscale=colorscale,
        opacity=OPACITY, flatshading=False, lighting=lighting,
        showscale=True if scene_id == "scene1" else False,
        customdata=names_L, hovertemplate=hover_tmpl,
        name=f"{title} L", scene=scene_id
    ))
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:, 0], y=coords_R_shift[:, 1], z=coords_R_shift[:, 2],
        i=faces_R[:, 0], j=faces_R[:, 1], k=faces_R[:, 2],
        intensity=intens_R, cmin=cmin, cmax=cmax, colorscale=colorscale,
        opacity=OPACITY, flatshading=False, lighting=lighting,
        showscale=False, customdata=names_R, hovertemplate=hover_tmpl,
        name=f"{title} R", scene=scene_id
    ))

    fig.update_layout(**{
        scene_id: dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.7)),
            bgcolor=BACKGROUND_COLOR,
        )
    })


def main():
    print("Building fsaverage 3D entropy visuals (Plotly)...")
    fs = 1.0 / TR_SEC

    # Load atlas + fsaverage
    aal, atlas_img, atlas, name_to_index = fetch_aal_atlas()
    pial_L, pial_R, white_L, white_R = fetch_fsaverage_surfaces()

    # Entropy per condition
    ent_A = aggregate_entropy(CONDITION_A_PATH, fs)
    ent_B = aggregate_entropy(CONDITION_B_PATH, fs)

    # Map ROI names to atlas indices
    A_idx_val, A_idx_name = entropy_map_to_indices(ent_A, name_to_index)
    B_idx_val, B_idx_name = entropy_map_to_indices(ent_B, name_to_index)

    # Sample labels to surface
    lbl_L = sample_labels(atlas_img, white_L, pial_L)
    lbl_R = sample_labels(atlas_img, white_R, pial_R)

    # Build vertex data (Baseline)
    coords_L, faces_L, intens_L_A, names_L_A = build_vertex_data(pial_L, lbl_L, A_idx_val, A_idx_name)
    coords_R, faces_R, intens_R_A, names_R_A = build_vertex_data(pial_R, lbl_R, A_idx_val, A_idx_name)
    # Build vertex data (Psilo)
    _, _, intens_L_B, names_L_B = build_vertex_data(pial_L, lbl_L, B_idx_val, B_idx_name)
    _, _, intens_R_B, names_R_B = build_vertex_data(pial_R, lbl_R, B_idx_val, B_idx_name)

    # Shift hemispheres
    coords_L_shift = coords_L.copy(); coords_L_shift[:, 0] -= HEMISPHERE_SHIFT
    coords_R_shift = coords_R.copy(); coords_R_shift[:, 0] += HEMISPHERE_SHIFT

    # Color scale from both conditions for consistency
    all_vals = np.concatenate([
        intens_L_A[~np.isnan(intens_L_A)], intens_R_A[~np.isnan(intens_R_A)],
        intens_L_B[~np.isnan(intens_L_B)], intens_R_B[~np.isnan(intens_R_B)],
    ])
    cmin, cmax = (0.0, 1.0) if all_vals.size == 0 else (float(np.nanmin(all_vals)), float(np.nanmax(all_vals)))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
        cmin, cmax = 0.0, 1.0

    # Create side-by-side scenes
    fig = go.Figure()
    create_scene(fig, 1, TITLE_A, coords_L_shift, faces_L, intens_L_A, names_L_A,
                 coords_R_shift, faces_R, intens_R_A, names_R_A,
                 cmin, cmax, COLORSCALE, scene_id="scene1")
    create_scene(fig, 1, TITLE_B, coords_L_shift, faces_L, intens_L_B, names_L_B,
                 coords_R_shift, faces_R, intens_R_B, names_R_B,
                 cmin, cmax, COLORSCALE, scene_id="scene2")

    # Layout with two scenes
    fig.update_layout(
        title=dict(text="Regional Spectral Entropy (AAL): Baseline vs Psilocybin",
                   font=dict(color=TEXT_COLOR, size=24)),
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=0, r=0, t=50, b=0),
        scene_domain=dict(x=[0.0, 0.49], y=[0.0, 1.0]),
        scene2_domain=dict(x=[0.51, 1.0], y=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.02, font=dict(color=TEXT_COLOR)),
    )

    # Save outputs
    fig.write_html(OUT_HTML, include_plotlyjs="inline", full_html=True, auto_open=False)
    print(f"Wrote: {OUT_HTML}")
    try:
        fig.write_image(OUT_PNG, width=1920, height=1080, scale=2)
        print(f"Wrote: {OUT_PNG}")
    except Exception as e:
        print("PNG export needs 'kaleido' (pip install -U kaleido). Error:", e)


if __name__ == "__main__":
    main()


