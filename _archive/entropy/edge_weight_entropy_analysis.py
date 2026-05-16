"""
Edge Weight Entropy Analysis: Connectivity Entropy for Baseline vs Psilocybin

Calculates Shannon entropy of connectivity profiles for each ROI and visualizes
differences between baseline and psilocybin sessions.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, surface
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from typing import Optional


# Configuration
BASELINE_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"
PSILOCYBIN_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-2_task-rest_aal_ts.csv"
OUTPUT_DIR = "publication_figures"


def edge_weight_entropy(
    corr: np.ndarray,
    roi_names=None,
    use_abs: bool = True,
    split_sign: bool = False,
    threshold: Optional[float] = None,
    include_self: bool = False,
    base: int = 2,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Calculate edge weight entropy (Shannon entropy on connectivity profile of each ROI)
    from correlation matrix (N x N).
    
    Interpretation:
        High H_edge -> ROI distributes connections across many regions (high diversity/integration)
        Low H_edge -> Connections concentrated in few areas (high specificity)
    """
    assert corr.ndim == 2 and corr.shape[0] == corr.shape[1], "corr must be square"
    N = corr.shape[0]

    # Prepare ROI names
    if roi_names is None:
        roi_names = [f"ROI_{i:03d}" for i in range(N)]
    else:
        assert len(roi_names) == N, "roi_names length must match corr size"

    W = corr.copy().astype(float)

    # Handle sign
    if use_abs:
        W = np.abs(W)

    # Threshold
    if threshold is not None:
        if use_abs:
            W[np.abs(W) < threshold] = 0.0
        else:
            W[W < threshold] = 0.0

    # Diagonal
    if not include_self:
        np.fill_diagonal(W, 0.0)

    # Helper: Shannon entropy for probability vector
    def _shannon_prob(p_vec: np.ndarray) -> float:
        p = p_vec[p_vec > 0]
        if p.size == 0:
            return np.nan
        logp = np.log(p + eps) / np.log(base)
        return float(-np.sum(p * logp))

    H_edge = np.full(N, np.nan)
    H_pos = np.full(N, np.nan)
    H_neg = np.full(N, np.nan)
    strength = np.zeros(N)  # Sum of weights
    degree = np.zeros(N)    # Number of non-zero edges
    n_nonzero = np.zeros(N) # Same, for transparency

    for i in range(N):
        row = W[i, :].copy()
        if not include_self:
            row[i] = 0.0

        strength[i] = np.sum(row)
        nonzero_mask = row > 0
        degree[i] = int(np.count_nonzero(nonzero_mask))
        n_nonzero[i] = degree[i]

        if degree[i] > 0:
            p = row / (row.sum() + eps)
            H_edge[i] = _shannon_prob(p)

        if split_sign and not use_abs:
            # Split into positive/negative connections based on original correlation
            row_signed = corr[i, :].copy()
            if not include_self:
                row_signed[i] = 0.0

            pos = row_signed[row_signed > 0]
            neg = -row_signed[row_signed < 0]  # Magnitude of negatives

            if threshold is not None:
                pos = pos[pos >= threshold]
                neg = neg[neg >= threshold]

            if pos.size > 0:
                ppos = pos / (pos.sum() + eps)
                H_pos[i] = _shannon_prob(ppos)
            if neg.size > 0:
                pneg = neg / (neg.sum() + eps)
                H_neg[i] = _shannon_prob(pneg)

    out = {
        "roi": roi_names,
        "H_edge": H_edge,
        "strength": strength,
        "degree": degree.astype(int),
        "n_nonzero": n_nonzero.astype(int),
    }
    if split_sign and not use_abs:
        out["H_edge_pos"] = H_pos
        out["H_edge_neg"] = H_neg

    return pd.DataFrame(out)


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


def build_vertex_data(surf_mesh, labels_on_vertices, roi_index_to_val: dict):
    """Build vertex data for surface rendering"""
    coords, faces = surface.load_surf_mesh(surf_mesh)
    intens = np.zeros(coords.shape[0], dtype=float)
    for idx_val, v in roi_index_to_val.items():
        mask = labels_on_vertices == idx_val
        if np.any(mask):
            intens[mask] = 0.0 if v is None or np.isnan(v) else float(v)
    return coords, faces, intens


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
    print("EDGE WEIGHT ENTROPY ANALYSIS: Baseline vs Psilocybin")
    print("="*80)
    
    # Load data
    print("\nLoading time series data...")
    df_baseline = pd.read_csv(BASELINE_CSV).select_dtypes(include=[np.number])
    df_psilo = pd.read_csv(PSILOCYBIN_CSV).select_dtypes(include=[np.number])
    
    roi_names = list(df_baseline.columns)
    print(f"Number of ROIs: {len(roi_names)}")
    
    # Compute correlation matrices
    print("\nComputing correlation matrices...")
    corr_baseline = df_baseline.corr().values
    corr_psilo = df_psilo.corr().values
    print(f"Correlation matrix shape: {corr_baseline.shape}")
    
    # Calculate edge weight entropy
    print("\nCalculating edge weight entropy...")
    print("  Baseline...")
    entropy_baseline = edge_weight_entropy(
        corr_baseline,
        roi_names=roi_names,
        use_abs=True,
        threshold=0.1,  # Only consider correlations |r| > 0.1
        include_self=False
    )
    
    print("  Psilocybin...")
    entropy_psilo = edge_weight_entropy(
        corr_psilo,
        roi_names=roi_names,
        use_abs=True,
        threshold=0.1,
        include_self=False
    )
    
    # Compute differences
    entropy_baseline = entropy_baseline.set_index('roi')
    entropy_psilo = entropy_psilo.set_index('roi')
    
    diff = entropy_psilo['H_edge'] - entropy_baseline['H_edge']
    
    print(f"\nEdge Weight Entropy Statistics:")
    print(f"  Baseline: mean={entropy_baseline['H_edge'].mean():.4f}, "
          f"std={entropy_baseline['H_edge'].std():.4f}")
    print(f"  Psilocybin: mean={entropy_psilo['H_edge'].mean():.4f}, "
          f"std={entropy_psilo['H_edge'].std():.4f}")
    print(f"  Difference: mean={diff.mean():.4f}, "
          f"std={diff.std():.4f}")
    print(f"  Range: [{diff.min():.4f}, {diff.max():.4f}]")
    
    # Save results
    results_df = pd.DataFrame({
        'H_edge_baseline': entropy_baseline['H_edge'],
        'H_edge_psilocybin': entropy_psilo['H_edge'],
        'H_edge_difference': diff,
        'strength_baseline': entropy_baseline['strength'],
        'strength_psilocybin': entropy_psilo['strength'],
        'degree_baseline': entropy_baseline['degree'],
        'degree_psilocybin': entropy_psilo['degree']
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "edge_weight_entropy_results.csv"))
    print(f"\nSaved results to: {OUTPUT_DIR}/edge_weight_entropy_results.csv")
    
    # Create 3D brain visualization
    print("\nCreating 3D brain visualization...")
    _, atlas_img, atlas, name_to_index = fetch_aal_spm12()
    pial_L, pial_R, white_L, white_R = fetch_fsaverage_surfaces()
    
    lbl_L = sample_labels(atlas_img, white_L, pial_L)
    lbl_R = sample_labels(atlas_img, white_R, pial_R)
    
    # Map differences to atlas indices
    diff_dict = diff.to_dict()
    diff_map = map_roi_to_indices(diff_dict, name_to_index)
    
    # Build vertex data
    coords_L, faces_L, intens_L = build_vertex_data(pial_L, lbl_L, diff_map)
    coords_R, faces_R, intens_R = build_vertex_data(pial_R, lbl_R, diff_map)
    
    # Get color scale range
    all_intens = np.concatenate([intens_L[intens_L != 0], intens_R[intens_R != 0]])
    vmax = np.percentile(np.abs(all_intens), 95)
    vmin = -vmax
    
    # Create Plotly figure
    fig = go.Figure()
    
    HEMISPHERE_SHIFT = 55
    BACKGROUND_COLOR = "black"
    TEXT_COLOR = "white"
    OPACITY = 1.0
    
    coords_L_shift = coords_L.copy()
    coords_L_shift[:, 0] -= HEMISPHERE_SHIFT
    coords_R_shift = coords_R.copy()
    coords_R_shift[:, 0] += HEMISPHERE_SHIFT
    
    hover_tmpl = "<b>%{customdata}</b><br>Edge Weight Entropy Diff: %{intensity:.3f}<extra></extra>"
    lighting = dict(ambient=0.35, diffuse=0.7, specular=0.6, roughness=0.4)
    
    # Left hemisphere
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:, 0], y=coords_L_shift[:, 1], z=coords_L_shift[:, 2],
        i=faces_L[:, 0], j=faces_L[:, 1], k=faces_L[:, 2],
        intensity=intens_L, cmin=vmin, cmax=vmax, colorscale='RdBu_r',
        intensitymode='vertex',
        opacity=OPACITY, lighting=lighting,
        showscale=True,
        customdata=coords_L, hovertemplate=hover_tmpl,
        colorbar=dict(title="Edge Weight<br>Entropy Diff", len=0.6, y=0.5)
    ))
    
    # Right hemisphere
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:, 0], y=coords_R_shift[:, 1], z=coords_R_shift[:, 2],
        i=faces_R[:, 0], j=faces_R[:, 1], k=faces_R[:, 2],
        intensity=intens_R, cmin=vmin, cmax=vmax, colorscale='RdBu_r',
        intensitymode='vertex',
        opacity=OPACITY, lighting=lighting,
        showscale=False,
        customdata=coords_R, hovertemplate=hover_tmpl
    ))
    
    # Legend/description text
    legend_text = (
        "<b>Figure Description:</b><br>"
        "Three-dimensional brain visualization displaying regional differences in Edge Weight Entropy "
        "between psilocybin and baseline resting-state sessions. Edge Weight Entropy quantifies the "
        "diversity and distribution of functional connectivity patterns for each brain region, computed "
        "as the Shannon entropy of each region's connectivity profile (correlation weights). Higher "
        "entropy values indicate that a region distributes its connections across many brain areas "
        "(high integration/diversity), whereas lower values indicate more concentrated, specific "
        "connectivity patterns. The color-coded map shows the difference (Psilocybin - Baseline), "
        "where warm colors (red) represent regions with increased connectivity diversity, and cool "
        "colors (blue) represent regions with decreased diversity. The visualization is rendered on "
        "the fsaverage brain surface with AAL (Automated Anatomical Labeling) atlas regions."
    )
    
    # Update layout
    fig.update_layout(
        title=dict(text="Edge Weight Entropy Differences: Psilocybin - Baseline",
                   font=dict(color=TEXT_COLOR, size=24)),
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=0, r=0, t=50, b=120),
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
                x=0.5, y=-0.08,
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
    html_path = os.path.join(OUTPUT_DIR, "edge_weight_entropy_differences_3d.html")
    png_path = os.path.join(OUTPUT_DIR, "edge_weight_entropy_differences_3d.png")
    
    fig.write_html(html_path, include_plotlyjs="inline", full_html=True, auto_open=False)
    print(f"Saved HTML: {html_path}")
    
    fig.write_image(png_path, width=1600, height=1200)
    print(f"Saved PNG: {png_path}")
    
    # Create bar chart of top/bottom differences
    print("\nCreating bar chart of top differences...")
    sorted_diff = diff.sort_values(ascending=False)
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top 20 increases
    top_20 = sorted_diff.head(20)
    ax1.barh(range(len(top_20)), top_20.values, color='red', alpha=0.7)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20.index, fontsize=8)
    ax1.set_xlabel('Edge Weight Entropy Difference (Psilocybin - Baseline)', fontsize=12)
    ax1.set_title('Top 20 ROIs: Increased Connectivity Diversity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Bottom 20 decreases
    bottom_20 = sorted_diff.tail(20)
    ax2.barh(range(len(bottom_20)), bottom_20.values, color='blue', alpha=0.7)
    ax2.set_yticks(range(len(bottom_20)))
    ax2.set_yticklabels(bottom_20.index, fontsize=8)
    ax2.set_xlabel('Edge Weight Entropy Difference (Psilocybin - Baseline)', fontsize=12)
    ax2.set_title('Bottom 20 ROIs: Decreased Connectivity Diversity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "edge_weight_entropy_top_bottom.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart: {bar_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
