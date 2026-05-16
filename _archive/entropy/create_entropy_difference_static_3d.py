"""
Create a static 3D brain visualization showing Sample Entropy differences
between psilocybin and baseline sessions.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, surface
import plotly.graph_objects as go
from scipy.signal import welch
import plotly


# Configuration
BASELINE_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"
PSILOCYBIN_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-3_task-rest_aal_ts.csv"
OUTPUT_HTML = "publication_figures/sample_entropy_differences_static_3d.html"
OUTPUT_PNG = "publication_figures/sample_entropy_differences_static_3d.png"

TR_SEC = 1.0


def sample_entropy(ts: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Sample Entropy with self-matches excluded"""
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


def fetch_aal_spm12():
    """Load AAL atlas"""
    aal = datasets.fetch_atlas_aal(version="SPM12")
    atlas_img = nib.load(aal["maps"])
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)
    labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    if "indices" in aal:
        indices_raw = list(aal["indices"])
    else:
        indices_raw = sorted(int(v) for v in np.unique(atlas) if v > 0)[:len(labels)]
    name_to_index = {labels[i]: int(indices_raw[i]) for i in range(len(labels))}
    return aal, atlas_img, atlas, name_to_index


def fetch_fsaverage_surfaces():
    """Load fsaverage surfaces"""
    fsavg = datasets.fetch_surf_fsaverage()
    return fsavg["pial_left"], fsavg["pial_right"], fsavg["white_left"], fsavg["white_right"]


def sample_labels(atlas_img, white_mesh, pial_mesh):
    """Sample AAL labels onto surface"""
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


def build_vertex_data(surf_mesh, labels_on_vertices, roi_index_to_val: dict, index_to_name: dict = None):
    """Build vertex data for surface rendering"""
    coords, faces = surface.load_surf_mesh(surf_mesh)
    intens = np.zeros(coords.shape[0], dtype=float)
    names = np.empty(coords.shape[0], dtype=object)
    
    for idx_val, v in roi_index_to_val.items():
        mask = labels_on_vertices == idx_val
        if np.any(mask):
            intens[mask] = 0.0 if v is None or np.isnan(v) else float(v)
            if index_to_name is not None:
                names[mask] = index_to_name.get(idx_val, "")
            else:
                names[mask] = ""
    
    return coords, faces, intens, names


def map_roi_to_indices(roi_to_val: dict, name_to_index: dict):
    """Map ROI names to atlas indices"""
    val_map = {}
    for roi, v in roi_to_val.items():
        idx = name_to_index.get(roi)
        if idx is not None:
            val_map[idx] = v
    return val_map


def main():
    print("="*80)
    print("CREATING STATIC 3D SAMPLE ENTROPY DIFFERENCE VISUALIZATION")
    print("="*80)
    
    # Load data
    df_A = pd.read_csv(BASELINE_CSV).select_dtypes(include=[np.number])
    df_B = pd.read_csv(PSILOCYBIN_CSV).select_dtypes(include=[np.number])
    
    # Compute Sample Entropy for each ROI
    print("\nComputing Sample Entropy for each ROI...")
    entropy_A = {}
    entropy_B = {}
    
    for roi in df_A.columns:
        ts_A = df_A[roi].dropna().values
        ts_B = df_B[roi].dropna().values
        entropy_A[roi] = sample_entropy(ts_A, m=2, r_frac=0.2)
        entropy_B[roi] = sample_entropy(ts_B, m=2, r_frac=0.2)
    
    # Compute differences
    diff = {roi: entropy_B[roi] - entropy_A[roi] for roi in entropy_A.keys()}
    
    # Get summary stats
    valid_diffs = [v for v in diff.values() if not np.isnan(v)]
    print(f"\nDifference statistics:")
    print(f"  Mean: {np.mean(valid_diffs):.4f}")
    print(f"  Min: {np.min(valid_diffs):.4f}")
    print(f"  Max: {np.max(valid_diffs):.4f}")
    print(f"  Std: {np.std(valid_diffs):.4f}")
    
    # Load atlas and surfaces
    print("\nLoading atlas and brain surfaces...")
    _, atlas_img, atlas, name_to_index = fetch_aal_spm12()
    pial_L, pial_R, white_L, white_R = fetch_fsaverage_surfaces()
    
    # Sample labels onto surfaces
    lbl_L = sample_labels(atlas_img, white_L, pial_L)
    lbl_R = sample_labels(atlas_img, white_R, pial_R)
    
    # Map differences to atlas indices
    diff_map = map_roi_to_indices(diff, name_to_index)
    
    # Create reverse mapping: index -> name (for hover tooltips)
    index_to_name = {idx: name for name, idx in name_to_index.items()}
    
    # Build vertex data
    coords_L, faces_L, intens_L, names_L = build_vertex_data(pial_L, lbl_L, diff_map, index_to_name)
    coords_R, faces_R, intens_R, names_R = build_vertex_data(pial_R, lbl_R, diff_map, index_to_name)
    
    # Get color scale range (symmetric around zero)
    all_intens = np.concatenate([intens_L[intens_L != 0], intens_R[intens_R != 0]])
    vmax = np.percentile(np.abs(all_intens), 95)
    vmin = -vmax
    
    print(f"\nColor scale range: [{vmin:.3f}, {vmax:.3f}]")
    
    # Create figure
    print("\nCreating 3D visualization...")
    fig = go.Figure()
    
    # Configuration
    HEMISPHERE_SHIFT = 55
    BACKGROUND_COLOR = "black"
    TEXT_COLOR = "white"
    OPACITY = 1.0
    
    # Shift hemispheres
    coords_L_shift = coords_L.copy()
    coords_L_shift[:, 0] -= HEMISPHERE_SHIFT
    coords_R_shift = coords_R.copy()
    coords_R_shift[:, 0] += HEMISPHERE_SHIFT
    
    # Create customdata arrays for hover (region names and entropy values)
    custom_L = np.stack([
        names_L.astype(str),
        intens_L
    ], axis=1)
    custom_R = np.stack([
        names_R.astype(str),
        intens_R
    ], axis=1)
    
    # Hover template showing region name and entropy difference
    hover_tmpl = "<b>%{customdata[0]}</b><br>Entropy Difference: %{customdata[1]:.3f}<extra></extra>"
    lighting = dict(ambient=0.35, diffuse=0.7, specular=0.6, roughness=0.4)
    
    # Left hemisphere
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:, 0], y=coords_L_shift[:, 1], z=coords_L_shift[:, 2],
        i=faces_L[:, 0], j=faces_L[:, 1], k=faces_L[:, 2],
        intensity=intens_L, cmin=vmin, cmax=vmax, colorscale='RdBu_r',
        intensitymode='vertex',
        opacity=OPACITY, lighting=lighting,
        showscale=True,
        customdata=custom_L, hovertemplate=hover_tmpl,
        colorbar=dict(title="Entropy<br>Difference", len=0.6, y=0.5)
    ))
    
    # Right hemisphere
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:, 0], y=coords_R_shift[:, 1], z=coords_R_shift[:, 2],
        i=faces_R[:, 0], j=faces_R[:, 1], k=faces_R[:, 2],
        intensity=intens_R, cmin=vmin, cmax=vmax, colorscale='RdBu_r',
        intensitymode='vertex',
        opacity=OPACITY, lighting=lighting,
        showscale=False,
        customdata=custom_R, hovertemplate=hover_tmpl
    ))
    
    # Legend/description text
    legend_text = (
        "<b>Figure Description:</b><br>"
        "Three-dimensional brain visualization showing regional differences in Sample Entropy "
        "between psilocybin and baseline resting-state sessions. Sample Entropy quantifies the "
        "complexity and irregularity of neural time series, with higher values indicating greater "
        "temporal complexity. The color-coded map displays the difference (Psilocybin - Baseline), "
        "where warm colors (red) indicate regions with increased entropy/complexity during "
        "psilocybin administration, and cool colors (blue) indicate regions with decreased "
        "entropy/complexity. The visualization is rendered on the fsaverage brain surface with "
        "AAL (Automated Anatomical Labeling) atlas regions."
    )
    
    # Update layout
    fig.update_layout(
        title=dict(text="Sample Entropy Differences: Psilocybin - Baseline",
                   font=dict(color=TEXT_COLOR, size=24)),
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            aspectmode="data",
            bgcolor=BACKGROUND_COLOR,
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.7))
        ),
        annotations=[
            dict(
                text=legend_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor="center", yanchor="top",
                align="left",
                showarrow=False,
                font=dict(color=TEXT_COLOR, size=11),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=TEXT_COLOR,
                borderwidth=1,
                borderpad=10
            )
        ]
    )
    
    # Save
    os.makedirs("publication_figures", exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="inline", full_html=True, auto_open=False)
    print(f"Saved HTML: {OUTPUT_HTML}")
    
    # Also save as static PNG
    fig.write_image(OUTPUT_PNG, width=1600, height=1200)
    print(f"Saved PNG: {OUTPUT_PNG}")
    
    print("\nDone!")


if __name__ == "__main__":
    import os
    main()

