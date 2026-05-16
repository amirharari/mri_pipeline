"""
============================================================================
PUBLICATION-READY SLIDES FROM ENTROPY AND NETWORK ANALYSES
============================================================================

Creates publication-ready figure panels summarizing:
1. Global entropy metrics and network connectivity results
2. Time-varying entropy dynamics
3. Correlation matrices visualization

Both panels designed for journal submission (high-resolution, readable).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = "publication_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure parameters for publication
DPI = 300
SNS_STYLE = "whitegrid"
COLOR_PALETTE = sns.color_palette("Set2")

# Data input paths
ENTROPY_CSV = "entropy_analysis_results/global_entropy_metrics.csv"
NETWORK_CSV = "network_connectivity_results/network_connectivity_summary.csv"
TIME_VARYING_CSV_BASE = "time_varying_entropy_results/sub-001_ses-1_task-rest_schaefer100_ts_time_varying_entropy.csv"
TIME_VARYING_CSV_PSILO = "time_varying_entropy_results/sub-001_ses-3_task-rest_schaefer100_ts_time_varying_entropy.csv"

# Correlation matrix images
CORR_MAT_DIR = "correlation_matrices_plots"


# =============================================================================
# SLIDE 1: GLOBAL METRICS & NETWORK CONNECTIVITY
# =============================================================================
def create_slide1_global_and_network():
    """
    Create Slide 1: Global Entropy Metrics + Network Connectivity
    """
    # Load data
    ent_df = pd.read_csv(ENTROPY_CSV)
    net_df = pd.read_csv(NETWORK_CSV)

    # Create figure with subpanels - 2x2 LAYOUT with spider plot
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    sessions = ['ses-1', 'ses-2', 'ses-3']
    labels = ['Baseline', 'Psilocybin Session', 'Follow-up']
    
    # Panel A: Sample Entropy Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get Sample Entropy only
    sampen = [ent_df['sample_entropy_mean'].iloc[i] for i in range(3)]
    x_pos = np.arange(len(sessions))
    width = 0.5
    
    ax1.bar(x_pos, sampen, width, alpha=0.8, color=COLOR_PALETTE[0])
    ax1.set_xlabel('Session', fontsize=11)
    ax1.set_ylabel('Sample Entropy', fontsize=11)
    ax1.set_title('A. Global Sample Entropy', fontweight='bold', fontsize=13)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Overall Within-Network Connectivity
    ax2 = fig.add_subplot(gs[0, 1])
    within = net_df['overall_within_network'].values
    x_pos = np.arange(len(sessions))
    width = 0.5
    ax2.bar(x_pos, within, width, color=COLOR_PALETTE[2], alpha=0.8)
    ax2.set_xlabel('Session', fontsize=11)
    ax2.set_ylabel('Mean Correlation (Fisher z)', fontsize=11)
    ax2.set_title('B. Overall Within-Network Connectivity', fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel C: Spider Plot of Network Connectivity
    ax3 = fig.add_subplot(gs[1, 0], projection='polar')
    
    network_cols = [c for c in net_df.columns if c.startswith('within_')]
    network_names = [c.replace('within_', '') for c in network_cols]
    
    # Number of networks
    num_networks = len(network_names)
    angles = np.linspace(0, 2 * np.pi, num_networks, endpoint=False).tolist()
    
    # Close the plot by appending first value
    angles += angles[:1]
    
    for i, session in enumerate(['ses-1', 'ses-2', 'ses-3']):
        values = [net_df[col].iloc[i] for col in network_cols]
        values += values[:1]  # Close the plot
        ax3.plot(angles, values, 'o-', linewidth=2, label=labels[i], alpha=0.8)
        ax3.fill(angles, values, alpha=0.15)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(network_names, fontsize=9)
    ax3.set_ylim(0, max([max(net_df[col]) for col in network_cols]) * 1.1)
    ax3.set_title('C. Network Connectivity Profile (Spider Plot)', fontweight='bold', fontsize=13, pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax3.grid(True)

    # Panel D: Within-Network Connectivity by Network (Grouped Bar Chart)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Grouped bar chart
    width = 0.25
    x = np.arange(len(network_names))
    for i, session in enumerate(['ses-1', 'ses-2', 'ses-3']):
        values = [net_df[col].iloc[i] for col in network_cols]
        ax4.bar(x + (i-1)*width, values, width, label=labels[i], alpha=0.8)
    
    ax4.set_xlabel('Network', fontsize=11)
    ax4.set_ylabel('Mean Connectivity (Fisher z)', fontsize=11)
    ax4.set_title('D. Within-Network Connectivity by Network', fontweight='bold', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels(network_names, fontsize=9, rotation=45, ha='right')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Brain Entropy and Network Connectivity: Baseline → Psilocybin → Follow-up', 
                 fontsize=20, fontweight='bold', y=0.99, family='Arial')
    
    fig.savefig(os.path.join(OUTPUT_DIR, "Slide1_Global_Network_Summary.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Slide 1 saved: Global Entropy & Network Connectivity")


# =============================================================================
# SLIDE 2: TIME-VARYING DYNAMICS
# =============================================================================
def create_slide2_time_varying():
    """
    Create Slide 2: Time-Varying Entropy Dynamics
    """
    # Load data
    base_df = pd.read_csv(TIME_VARYING_CSV_BASE)
    psilo_df = pd.read_csv(TIME_VARYING_CSV_PSILO)

    # Create figure with subpanels
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Correlation Entropy Over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(base_df['window_center'], base_df['correlation_entropy'], 
             label='Baseline (ses-1)', linewidth=2.5, color=COLOR_PALETTE[0])
    ax1.plot(psilo_df['window_center'], psilo_df['correlation_entropy'], 
             label='Psilocybin (ses-3)', linewidth=2.5, color=COLOR_PALETTE[1])
    ax1.set_xlabel('Time (timepoints)', fontsize=11)
    ax1.set_ylabel('Correlation Entropy', fontsize=11)
    ax1.set_title('A. Time-Varying Correlation Entropy', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: Mean Correlation Over Time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(base_df['window_center'], base_df['mean_correlation'], 
             label='Baseline (ses-1)', linewidth=2.5, color=COLOR_PALETTE[0])
    ax2.plot(psilo_df['window_center'], psilo_df['mean_correlation'], 
             label='Psilocybin (ses-3)', linewidth=2.5, color=COLOR_PALETTE[1])
    ax2.set_xlabel('Time (timepoints)', fontsize=11)
    ax2.set_ylabel('Mean Correlation', fontsize=11)
    ax2.set_title('B. Mean Connectivity Strength', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel C: Effective Connectivity Over Time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(base_df['window_center'], base_df['effective_connectivity_pct'], 
             label='Baseline (ses-1)', linewidth=2.5, color=COLOR_PALETTE[0])
    ax3.plot(psilo_df['window_center'], psilo_df['effective_connectivity_pct'], 
             label='Psilocybin (ses-3)', linewidth=2.5, color=COLOR_PALETTE[1])
    ax3.set_xlabel('Time (timepoints)', fontsize=11)
    ax3.set_ylabel('Effective Connectivity (%)', fontsize=11)
    ax3.set_title('C. Strong Connections Over Time', fontweight='bold', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel D: Entropy Variability (Std)
    ax4 = fig.add_subplot(gs[1, 1])
    base_std = base_df['correlation_entropy'].rolling(window=10, center=True).std()
    psilo_std = psilo_df['correlation_entropy'].rolling(window=10, center=True).std()
    ax4.plot(base_df['window_center'], base_std, 
             label='Baseline (ses-1)', linewidth=2.5, color=COLOR_PALETTE[0])
    ax4.plot(psilo_df['window_center'], psilo_std, 
             label='Psilocybin (ses-3)', linewidth=2.5, color=COLOR_PALETTE[1])
    ax4.set_xlabel('Time (timepoints)', fontsize=11)
    ax4.set_ylabel('Rolling Std of Entropy (10-window)', fontsize=11)
    ax4.set_title('D. Entropy Variability', fontweight='bold', fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Time-Varying Brain Dynamics: Baseline vs Psilocybin Session', 
                 fontsize=20, fontweight='bold', y=0.98, family='Arial')
    
    fig.savefig(os.path.join(OUTPUT_DIR, "Slide2_Time_Varying_Dynamics.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Slide 2 saved: Time-Varying Dynamics")


# =============================================================================
# SLIDE 3: CORRELATION MATRICES VISUALIZATION
# =============================================================================
def get_aal_region_categories():
    """Define AAL regions by anatomical category for labeling - ordered as they appear in AAL"""
    # Categories based on exact AAL ordering
    categories = {
        'Frontal': (0, 27),  # Precentral + Frontal + Olfactory + Medial Frontal + Rectus
        'Limbic': (20, 41),  # Olfactory + Medial Frontal + Insula + Cingulum + Hippo/Amygdala
        'Occipital': (42, 55),
        'Parietal': (56, 69),
        'Subcortical': (70, 77),
        'Temporal': (78, 89),
        'Cerebellar': (90, 107),
        'Vermal': (108, 115),  # Last 8 regions
    }
    return categories

def create_slide3_correlation_matrices():
    """
    Create Slide 3: Correlation Matrices for Each Session with colored regions
    """
    # Load correlation matrices as data
    import pandas as pd
    from nilearn.plotting import plot_matrix
    
    corr_matrices = []
    labels = ['Baseline (ses-1)', 'Psilocybin Session (ses-2)', 'Follow-up (ses-3)']
    
    data_folder = r"C:\kpeSoundPath\rest-results-itamar"
    for ses in ['ses-1', 'ses-2', 'ses-3']:
        csv_path = os.path.join(data_folder, f"sub-001_{ses}_task-rest_aal_ts.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            corr_matrix = df.corr()
            corr_matrices.append(corr_matrix)
        else:
            print(f"Warning: {csv_path} not found")

    if len(corr_matrices) == 3:
        # Get AAL region order for annotations
        import pandas as pd
        df = pd.read_csv('sub-001_ses-1_task-rest_aal_ts.csv')
        aal_regions = df.columns.tolist()
        categories = get_aal_region_categories()
        
        # Categories are already in the right format
        category_boundaries = categories
        
        # Define colors and abbreviations for each region category
        cat_colors = {
            'Frontal': '#1f77b4',      # Blue
            'Limbic': '#ff7f0e',       # Orange
            'Occipital': '#2ca02c',    # Green
            'Parietal': '#d62728',     # Red
            'Subcortical': '#9467bd',  # Purple
            'Temporal': '#8c564b',     # Brown
            'Cerebellar': '#e377c2',   # Pink
            'Vermal': '#7f7f7f'        # Gray
        }
        
        cat_abbreviations = {
            'Frontal': 'Fr',
            'Limbic': 'Lim',
            'Occipital': 'Occ',
            'Parietal': 'Par',
            'Subcortical': 'Sub',
            'Temporal': 'Temp',
            'Cerebellar': 'Cer',
            'Vermal': 'Ver'
        }
        
        # Create figure with titles
        fig = plt.figure(figsize=(18, 8))
        gs = GridSpec(5, 3, figure=fig, height_ratios=[0.5, 4, 0.1, 0.3, 0.2], hspace=0.5, wspace=0.3)
        
        panel_labels = ['A. Baseline', 'B. Psilocybin Session', 'C. Follow-up']
        
        # Plot correlation matrices with colored region overlays
        axes = []
        num_regions = len(aal_regions)
        
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            
            # Plot the correlation matrix
            # Use extent to ensure data coordinates align with matrix indices
            im = ax.imshow(corr_matrices[i], cmap='RdBu_r', vmin=-0.8, vmax=0.8, 
                          extent=[-0.5, num_regions-0.5, num_regions-0.5, -0.5], aspect='auto')
            
            # Add colored overlay for each region and separation lines
            for cat_name, (start, end) in category_boundaries.items():
                color = cat_colors.get(cat_name, 'gray')
                # Create a mask for this region
                mask = np.zeros_like(corr_matrices[i].values)
                mask[start:end+1, :] = 0.15  # Light overlay for rows
                mask[:, start:end+1] = 0.15  # Light overlay for columns
                # Apply the colored overlay
                colored_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                # Convert hex color to RGB
                import matplotlib.colors as mcolors
                rgb = mcolors.hex2color(color)
                colored_overlay[:, :, 0] = rgb[0]  # R
                colored_overlay[:, :, 1] = rgb[1]  # G
                colored_overlay[:, :, 2] = rgb[2]  # B
                colored_overlay[:, :, 3] = mask    # Alpha
                ax.imshow(colored_overlay, aspect='auto', interpolation='nearest')
                
                # Add strong separation lines at both ends of each region
                ax.axvline(start - 0.5, color=color, linewidth=3, alpha=0.8)
                ax.axhline(start - 0.5, color=color, linewidth=3, alpha=0.8)
                ax.axvline(end + 0.5, color=color, linewidth=3, alpha=0.8)
                ax.axhline(end + 0.5, color=color, linewidth=3, alpha=0.8)
            
            # Add region labels at the top of the matrix
            for cat_name, (start, end) in category_boundaries.items():
                mid_pos = (start + end) / 2
                abbrev = cat_abbreviations.get(cat_name, cat_name)
                ax.text(mid_pos, -2, abbrev, ha='center', va='top', 
                       fontsize=12, fontweight='bold', color='black', family='Arial',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Add title
            ax.set_title(f'{panel_labels[i]}\n{labels[i]}', fontweight='bold', fontsize=13, pad=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation', fontsize=9)
            
            ax.set_xticks([])
            ax.set_yticks([])
            axes.append(ax)
        
        # Add region category legend below matrices
        ax_label = fig.add_subplot(gs[3, :])
        ax_label.axis('off')
        
        # Create color legend with patches
        legend_elements = []
        for cat_name in sorted(category_boundaries.keys(), key=lambda x: category_boundaries[x][0]):
            color = cat_colors.get(cat_name, 'gray')
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                                edgecolor='black', linewidth=0.5,
                                                label=f'{cat_name} ({category_boundaries[cat_name][0]}-{category_boundaries[cat_name][1]})'))
        
        ax_label.legend(handles=legend_elements, loc='center', ncol=8, 
                       fontsize=9, frameon=True, fancybox=True, shadow=True)
        
        # Add title (moved higher to avoid overlap)
        fig.text(0.5, 0.98, 'Functional Connectivity Patterns Across Sessions', 
                 ha='center', fontsize=20, fontweight='bold', family='Arial')
        
        # Add explanation text
        explanation = (
            "Correlation matrices show pairwise connectivity between 116 AAL brain regions.\n"
            "Colors indicate correlation strength: Red = strong positive, Blue = strong negative.\n"
            "Diagonal = 1.0 (self-correlation). Matrix is symmetric."
        )
        fig.text(0.5, 0.08, explanation, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.savefig(os.path.join(OUTPUT_DIR, "Slide3_Correlation_Matrices.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print("Slide 3 saved: Correlation Matrices")


# =============================================================================
# SLIDE 4: CORRELATION DIFFERENCE MATRICES
# =============================================================================
def create_slide4_correlation_differences():
    """
    Create Slide 4: Correlation difference matrices (Psilocybin - Baseline, Follow-up - Baseline)
    """
    # Load correlation matrices as data
    import pandas as pd
    
    corr_matrices = []
    labels = ['Baseline (ses-1)', 'Psilocybin Session (ses-2)', 'Follow-up (ses-3)']
    
    data_folder = r"C:\kpeSoundPath\rest-results-itamar"
    for ses in ['ses-1', 'ses-2', 'ses-3']:
        csv_path = os.path.join(data_folder, f"sub-001_{ses}_task-rest_aal_ts.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            corr_matrix = df.corr()
            corr_matrices.append(corr_matrix)
        else:
            print(f"Warning: {csv_path} not found")

    if len(corr_matrices) == 3:
        # Get AAL region order for annotations
        df = pd.read_csv('sub-001_ses-1_task-rest_aal_ts.csv')
        aal_regions = df.columns.tolist()
        categories = get_aal_region_categories()
        category_boundaries = categories
        
        # Define colors and abbreviations for each region category
        cat_colors = {
            'Frontal': '#1f77b4',      # Blue
            'Limbic': '#ff7f0e',       # Orange
            'Occipital': '#2ca02c',    # Green
            'Parietal': '#d62728',     # Red
            'Subcortical': '#9467bd',  # Purple
            'Temporal': '#8c564b',     # Brown
            'Cerebellar': '#e377c2',   # Pink
            'Vermal': '#7f7f7f'        # Gray
        }
        
        cat_abbreviations = {
            'Frontal': 'Fr',
            'Limbic': 'Lim',
            'Occipital': 'Occ',
            'Parietal': 'Par',
            'Subcortical': 'Sub',
            'Temporal': 'Temp',
            'Cerebellar': 'Cer',
            'Vermal': 'Ver'
        }
        
        # Calculate difference matrices
        diff_psilo = corr_matrices[1] - corr_matrices[0]  # Psilocybin - Baseline
        diff_follow = corr_matrices[2] - corr_matrices[0]  # Follow-up - Baseline
        
        # Get absolute max for symmetric colormap
        max_diff = max(np.abs(diff_psilo.values.max()), np.abs(diff_psilo.values.min()),
                      np.abs(diff_follow.values.max()), np.abs(diff_follow.values.min()))
        
        # Create figure
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(5, 2, figure=fig, height_ratios=[0.5, 4, 0.1, 0.3, 0.2], hspace=0.5, wspace=0.3)
        
        panel_labels = ['A. Psilocybin - Baseline', 'B. Follow-up - Baseline']
        diff_matrices = [diff_psilo, diff_follow]
        num_regions = len(aal_regions)
        
        # Plot difference matrices with colored region overlays
        for i, (diff_mat, label) in enumerate(zip(diff_matrices, panel_labels)):
            ax = fig.add_subplot(gs[1, i])
            
            # Plot the difference matrix
            im = ax.imshow(diff_mat, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, 
                          extent=[-0.5, num_regions-0.5, num_regions-0.5, -0.5], aspect='auto')
            
            # Add colored horizontal bars at the top to mark region boundaries
            for cat_name, (start, end) in category_boundaries.items():
                color = cat_colors.get(cat_name, 'gray')
                # Draw colored horizontal bar at the top
                ax.add_patch(plt.Rectangle((start - 0.5, -1.5), end - start + 1, 1.5,
                                          facecolor=color, edgecolor='black', linewidth=1,
                                          alpha=0.8, clip_on=False))
            
            # Add region labels at the top
            for cat_name, (start, end) in category_boundaries.items():
                mid_pos = (start + end) / 2
                abbrev = cat_abbreviations.get(cat_name, cat_name)
                ax.text(mid_pos, -0.75, abbrev, ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='black', family='Arial',
                       clip_on=False)
            
            # Add title
            ax.set_title(label, fontweight='bold', fontsize=13, pad=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation Difference', fontsize=9)
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add region category legend below matrices
        ax_label = fig.add_subplot(gs[3, :])
        ax_label.axis('off')
        
        # Create color legend with patches
        legend_elements = []
        for cat_name in sorted(category_boundaries.keys(), key=lambda x: category_boundaries[x][0]):
            color = cat_colors.get(cat_name, 'gray')
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                                edgecolor='black', linewidth=0.5,
                                                label=f'{cat_name} ({category_boundaries[cat_name][0]}-{category_boundaries[cat_name][1]})'))
        
        ax_label.legend(handles=legend_elements, loc='center', ncol=8, 
                       fontsize=9, frameon=True, fancybox=True, shadow=True)
        
        # Add title
        fig.text(0.5, 0.98, 'Functional Connectivity Changes Across Sessions', 
                 ha='center', fontsize=20, fontweight='bold', family='Arial')
        
        # Add explanation text
        explanation = (
            "Difference matrices show connectivity changes: Red = increased connectivity, Blue = decreased connectivity.\n"
            "Comparing: (A) Psilocybin Session vs Baseline, (B) Follow-up vs Baseline."
        )
        fig.text(0.5, 0.08, explanation, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.savefig(os.path.join(OUTPUT_DIR, "Slide4_Correlation_Differences.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print("Slide 4 saved: Correlation Differences")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("CREATING PUBLICATION SLIDES")
    print("="*80)
    
    # Set style
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    sns.set_palette("Set2")

    # Create slides
    create_slide1_global_and_network()
    create_slide2_time_varying()
    create_slide3_correlation_matrices()
    create_slide4_correlation_differences()

    print(f"\nAll slides saved to: {OUTPUT_DIR}/")
    print("Publication-ready figures created!")


if __name__ == "__main__":
    main()

