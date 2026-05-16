"""
Entropy Analysis Pipeline
Loads data, calculates entropy metrics, saves results, and creates visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from nilearn import datasets, surface
import nibabel as nib

from simple_entropy import sample_entropy
from edge_weight_entropy import edge_weight_entropy


# Configuration
BASELINE_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"
PSILOCYBIN_CSV = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-2_task-rest_aal_ts.csv"
OUTPUT_DIR = "entropy_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load time series data from CSV"""
    return pd.read_csv(csv_path).select_dtypes(include=[np.number])


def calculate_simple_entropy(timeseries_df: pd.DataFrame) -> pd.Series:
    """Calculate Sample Entropy for each ROI"""
    entropy_dict = {}
    for roi in timeseries_df.columns:
        ts = timeseries_df[roi].dropna().values
        entropy_dict[roi] = sample_entropy(ts, m=2, r=0.2)
    return pd.Series(entropy_dict)


def calculate_edge_weight_entropy(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Edge Weight Entropy from correlation matrix"""
    corr_matrix = timeseries_df.corr().values
    roi_names = list(timeseries_df.columns)
    
    return edge_weight_entropy(
        corr_matrix,
        roi_names=roi_names,
        use_abs=True,
        threshold=0.1,
        include_self=False
    )


def save_results(baseline_simple: pd.Series, psilo_simple: pd.Series,
                 baseline_edge: pd.DataFrame, psilo_edge: pd.DataFrame):
    """Save all entropy results to CSV"""
    
    # Simple entropy results
    simple_df = pd.DataFrame({
        'roi': baseline_simple.index,
        'simple_entropy_baseline': baseline_simple.values,
        'simple_entropy_psilocybin': psilo_simple.values,
        'simple_entropy_difference': psilo_simple.values - baseline_simple.values
    })
    simple_df.to_csv(os.path.join(OUTPUT_DIR, "simple_entropy_results.csv"), index=False)
    print(f"Saved: {OUTPUT_DIR}/simple_entropy_results.csv")
    
    # Edge weight entropy results
    edge_df = pd.DataFrame({
        'roi': baseline_edge['roi'],
        'edge_entropy_baseline': baseline_edge['H_edge'].values,
        'edge_entropy_psilocybin': psilo_edge['H_edge'].values,
        'edge_entropy_difference': psilo_edge['H_edge'].values - baseline_edge['H_edge'].values,
        'strength_baseline': baseline_edge['strength'].values,
        'strength_psilocybin': psilo_edge['strength'].values,
        'degree_baseline': baseline_edge['degree'].values,
        'degree_psilocybin': psilo_edge['degree'].values
    })
    edge_df.to_csv(os.path.join(OUTPUT_DIR, "edge_weight_entropy_results.csv"), index=False)
    print(f"Saved: {OUTPUT_DIR}/edge_weight_entropy_results.csv")


def create_visualizations(baseline_simple: pd.Series, psilo_simple: pd.Series,
                         baseline_edge: pd.DataFrame, psilo_edge: pd.DataFrame):
    """Create visualizations"""
    
    # 1. Simple bar chart comparison
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Simple entropy comparison
    mean_base = baseline_simple.mean()
    mean_psilo = psilo_simple.mean()
    ax1.bar(['Baseline', 'Psilocybin'], [mean_base, mean_psilo], 
            color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Mean Sample Entropy')
    ax1.set_title('Global Sample Entropy Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Edge weight entropy comparison
    mean_edge_base = baseline_edge['H_edge'].mean()
    mean_edge_psilo = psilo_edge['H_edge'].mean()
    ax2.bar(['Baseline', 'Psilocybin'], [mean_edge_base, mean_edge_psilo],
            color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Mean Edge Weight Entropy')
    ax2.set_title('Global Edge Weight Entropy Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "entropy_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/entropy_comparison.png")
    
    # 2. Top differences bar chart
    simple_diff = psilo_simple - baseline_simple
    edge_diff = psilo_edge['H_edge'].values - baseline_edge['H_edge'].values
    edge_diff_series = pd.Series(edge_diff, index=baseline_edge['roi'])
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top 20 simple entropy differences
    top_simple = simple_diff.sort_values(ascending=False).head(20)
    ax1.barh(range(len(top_simple)), top_simple.values, color='red', alpha=0.7)
    ax1.set_yticks(range(len(top_simple)))
    ax1.set_yticklabels(top_simple.index, fontsize=8)
    ax1.set_xlabel('Sample Entropy Difference (Psilocybin - Baseline)')
    ax1.set_title('Top 20 ROIs: Increased Sample Entropy', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Top 20 edge entropy differences
    top_edge = edge_diff_series.sort_values(ascending=False).head(20)
    ax2.barh(range(len(top_edge)), top_edge.values, color='orange', alpha=0.7)
    ax2.set_yticks(range(len(top_edge)))
    ax2.set_yticklabels(top_edge.index, fontsize=8)
    ax2.set_xlabel('Edge Weight Entropy Difference (Psilocybin - Baseline)')
    ax2.set_title('Top 20 ROIs: Increased Edge Weight Entropy', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_differences.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/top_differences.png")
    
    # 3. 3D brain visualizations
    try:
        create_3d_brain_visualization(simple_diff, "simple_entropy")
        print(f"Saved: {OUTPUT_DIR}/simple_entropy_differences_3d.html")
    except Exception as e:
        print(f"Warning: Could not create simple entropy 3D visualization: {e}")
    
    try:
        create_3d_brain_visualization(edge_diff_series, "edge_weight_entropy")
        print(f"Saved: {OUTPUT_DIR}/edge_weight_entropy_differences_3d.html")
    except Exception as e:
        print(f"Warning: Could not create edge weight entropy 3D visualization: {e}")


def create_3d_brain_visualization(diff_series: pd.Series, metric_name: str):
    """Create 3D brain visualization of entropy differences"""
    
    # Load atlas and surfaces
    aal = datasets.fetch_atlas_aal(version="SPM12")
    atlas_img = nib.load(aal["maps"])
    labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    if "indices" in aal:
        indices_raw = list(aal["indices"])
    else:
        atlas_data = np.rint(atlas_img.get_fdata()).astype(np.int32)
        indices_raw = sorted(int(v) for v in np.unique(atlas_data) if v > 0)[:len(labels)]
    
    name_to_index = {labels[i]: int(indices_raw[i]) for i in range(len(labels))}
    
    fsavg = datasets.fetch_surf_fsaverage()
    pial_L, pial_R = fsavg["pial_left"], fsavg["pial_right"]
    white_L, white_R = fsavg["white_left"], fsavg["white_right"]
    
    # Sample labels onto surfaces
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
    
    lbl_L = sample_labels(atlas_img, white_L, pial_L)
    lbl_R = sample_labels(atlas_img, white_R, pial_R)
    
    # Map differences to atlas indices
    diff_dict = diff_series.to_dict()
    diff_map = {}
    for roi, v in diff_dict.items():
        idx = name_to_index.get(roi)
        if idx is not None:
            diff_map[idx] = v
    
    # Create reverse mapping: index -> ROI name
    index_to_name = {idx: name for name, idx in name_to_index.items()}
    
    # Build vertex data
    def build_vertex_data(surf_mesh, labels_on_vertices, roi_index_to_val: dict, index_to_name: dict):
        coords, faces = surface.load_surf_mesh(surf_mesh)
        intens = np.full(coords.shape[0], np.nan, dtype=float)  # Initialize with NaN
        roi_names = np.full(coords.shape[0], "", dtype=object)  # Store ROI names for each vertex
        
        # Find all valid AAL indices that exist in the labels
        valid_indices = set()
        for idx_val, v in roi_index_to_val.items():
            mask = labels_on_vertices == idx_val
            if np.any(mask):
                intens[mask] = 0.0 if v is None or np.isnan(v) else float(v)
                roi_name = index_to_name.get(idx_val, "Unknown")
                roi_names[mask] = roi_name
                valid_indices.add(idx_val)
        
        # Mark vertices that don't belong to any ROI (label=0 or label not in diff_map)
        vertices_with_roi = np.zeros(coords.shape[0], dtype=bool)
        for idx_val in valid_indices:
            vertices_with_roi[labels_on_vertices == idx_val] = True
        
        # Set intensity to NaN for vertices without ROI assignment
        intens[~vertices_with_roi] = np.nan
        
        return coords, faces, intens, roi_names
    
    coords_L, faces_L, intens_L, names_L = build_vertex_data(pial_L, lbl_L, diff_map, index_to_name)
    coords_R, faces_R, intens_R, names_R = build_vertex_data(pial_R, lbl_R, diff_map, index_to_name)
    
    # Get color scale range (only from valid vertices with ROI assignments)
    valid_L = intens_L[~np.isnan(intens_L)]
    valid_R = intens_R[~np.isnan(intens_R)]
    all_intens = np.concatenate([valid_L, valid_R]) if (valid_L.size > 0 or valid_R.size > 0) else np.array([0.1])
    vmax = np.percentile(np.abs(all_intens), 95) if all_intens.size > 0 else 0.1
    vmin = -vmax
    
    # Prepare display name and hover template
    metric_display = metric_name.replace('_', ' ').title()
    
    # Convert names to list for text parameter - only show ROI names for vertices with valid assignments
    names_L_list = []
    for i, n in enumerate(names_L):
        if not np.isnan(intens_L[i]) and n:
            names_L_list.append(str(n))
        else:
            names_L_list.append("")  # Empty string for vertices without ROI
    
    names_R_list = []
    for i, n in enumerate(names_R):
        if not np.isnan(intens_R[i]) and n:
            names_R_list.append(str(n))
        else:
            names_R_list.append("")  # Empty string for vertices without ROI
    
    # Create Plotly figure
    fig = go.Figure()
    
    HEMISPHERE_SHIFT = 55
    coords_L_shift = coords_L.copy()
    coords_L_shift[:, 0] -= HEMISPHERE_SHIFT
    coords_R_shift = coords_R.copy()
    coords_R_shift[:, 0] += HEMISPHERE_SHIFT
    
    lighting = dict(ambient=0.35, diffuse=0.7, specular=0.6, roughness=0.4)
    
    # Use a value well outside the normal range to mark unassigned vertices
    # This ensures they don't interfere with actual entropy differences near 0
    UASSIGNED_VALUE = -999.0
    
    # Replace NaN with unassigned value for plotting
    intens_L_plot = intens_L.copy()
    intens_R_plot = intens_R.copy()
    intens_L_plot[np.isnan(intens_L_plot)] = UASSIGNED_VALUE
    intens_R_plot[np.isnan(intens_R_plot)] = UASSIGNED_VALUE
    
    # Use original vmin/vmax for the colormap (unassigned vertices will be off-scale)
    vmin_plot = vmin
    vmax_plot = vmax
    
    # Hover template: show ROI name and value if available
    has_roi_L = np.array([not np.isnan(val) and name != "" for val, name in zip(intens_L, names_L_list)])
    has_roi_R = np.array([not np.isnan(val) and name != "" for val, name in zip(intens_R, names_R_list)])
    
    # Add base gray brain surfaces (for unassigned regions)
    # Left hemisphere - base
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:, 0], y=coords_L_shift[:, 1], z=coords_L_shift[:, 2],
        i=faces_L[:, 0], j=faces_L[:, 1], k=faces_L[:, 2],
        color='lightgray', opacity=0.3, lighting=lighting,
        showlegend=False, hoverinfo='skip'
    ))
    
    # Right hemisphere - base
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:, 0], y=coords_R_shift[:, 1], z=coords_R_shift[:, 2],
        i=faces_R[:, 0], j=faces_R[:, 1], k=faces_R[:, 2],
        color='lightgray', opacity=0.3, lighting=lighting,
        showlegend=False, hoverinfo='skip'
    ))
    
    # Left hemisphere - ROI data (only vertices with valid ROI)
    # Use the original intens_L (with NaN for unassigned) - Plotly should skip NaN vertices
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:, 0], y=coords_L_shift[:, 1], z=coords_L_shift[:, 2],
        i=faces_L[:, 0], j=faces_L[:, 1], k=faces_L[:, 2],
        intensity=intens_L, cmin=vmin_plot, cmax=vmax_plot, colorscale='RdBu_r',
        intensitymode='vertex',
        opacity=1.0, lighting=lighting,
        showscale=True,
        colorbar=dict(title=f"{metric_display}<br>Difference", len=0.6, y=0.5),
        text=names_L_list,
        customdata=np.column_stack([has_roi_L, np.nan_to_num(intens_L, nan=0.0)]),
        hovertemplate="<b>%{text}</b><br>" + f"{metric_display} Difference: %{{customdata[1]:.4f}}<extra></extra>"
    ))
    
    # Right hemisphere - ROI data
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:, 0], y=coords_R_shift[:, 1], z=coords_R_shift[:, 2],
        i=faces_R[:, 0], j=faces_R[:, 1], k=faces_R[:, 2],
        intensity=intens_R, cmin=vmin_plot, cmax=vmax_plot, colorscale='RdBu_r',
        intensitymode='vertex',
        opacity=1.0, lighting=lighting,
        showscale=False,
        text=names_R_list,
        customdata=np.column_stack([has_roi_R, np.nan_to_num(intens_R, nan=0.0)]),
        hovertemplate="<b>%{text}</b><br>" + f"{metric_display} Difference: %{{customdata[1]:.4f}}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=f"{metric_display} Differences: Psilocybin - Baseline",
                   font=dict(color="white", size=24)),
        paper_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            aspectmode="data",
            bgcolor="black",
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.7))
        )
    )
    
    # Save
    html_path = os.path.join(OUTPUT_DIR, f"{metric_name}_differences_3d.html")
    png_path = os.path.join(OUTPUT_DIR, f"{metric_name}_differences_3d.png")
    
    fig.write_html(html_path, include_plotlyjs="inline", full_html=True, auto_open=False)
    
    try:
        fig.write_image(png_path, width=1600, height=1200)
        print(f"Saved: {png_path}")
    except:
        pass  # PNG export requires kaleido, but HTML is saved


def main():
    print("="*80)
    print("ENTROPY ANALYSIS PIPELINE")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    df_baseline = load_data(BASELINE_CSV)
    df_psilo = load_data(PSILOCYBIN_CSV)
    print(f"   Baseline: {df_baseline.shape[1]} ROIs, {df_baseline.shape[0]} timepoints")
    print(f"   Psilocybin: {df_psilo.shape[1]} ROIs, {df_psilo.shape[0]} timepoints")
    
    # Calculate Simple Entropy
    print("\n2. Calculating Simple Entropy (Sample Entropy)...")
    simple_entropy_baseline = calculate_simple_entropy(df_baseline)
    simple_entropy_psilo = calculate_simple_entropy(df_psilo)
    print(f"   Baseline mean: {simple_entropy_baseline.mean():.4f}")
    print(f"   Psilocybin mean: {simple_entropy_psilo.mean():.4f}")
    
    # Calculate Edge Weight Entropy
    print("\n3. Calculating Edge Weight Entropy...")
    edge_entropy_baseline = calculate_edge_weight_entropy(df_baseline)
    edge_entropy_psilo = calculate_edge_weight_entropy(df_psilo)
    print(f"   Baseline mean: {edge_entropy_baseline['H_edge'].mean():.4f}")
    print(f"   Psilocybin mean: {edge_entropy_psilo['H_edge'].mean():.4f}")
    
    # Save results
    print("\n4. Saving results...")
    save_results(simple_entropy_baseline, simple_entropy_psilo,
                 edge_entropy_baseline, edge_entropy_psilo)
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(simple_entropy_baseline, simple_entropy_psilo,
                         edge_entropy_baseline, edge_entropy_psilo)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
