import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "ventral_striatum_results"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_timeseries(filepath):
    """Load timeseries CSV"""
    try:
        return pd.read_csv(filepath)
    except:
        print(f"Could not load: {filepath}")
        return None

def find_ventral_striatum_columns(df):
    """Find ventral striatum related columns"""
    # Ventral striatum includes: Caudate, Putamen, Nucleus Accumbens, Pallidum
    keywords = ['Caudate', 'Putamen', 'Pallidum', 'Accumbens']
    
    vs_cols = []
    for col in df.columns:
        if any(kw in col for kw in keywords):
            vs_cols.append(col)
    
    return vs_cols

def smooth_timeseries(data, window=5):
    """Smooth timeseries by averaging every 'window' points"""
    n_points = len(data) // window * window
    data_trimmed = data[:n_points]
    reshaped = data_trimmed.reshape(-1, window)
    smoothed = reshaped.mean(axis=1)
    return smoothed

def plot_ventral_striatum_timeline(ses1_df, ses2_df, ses3_df, task_name='GLASS'):
    """Plot BOLD activation timeline across 3 sessions with smoothing"""
    
    # Find ventral striatum columns
    if ses1_df is not None:
        vs_cols = find_ventral_striatum_columns(ses1_df)
    elif ses2_df is not None:
        vs_cols = find_ventral_striatum_columns(ses2_df)
    else:
        print("No data available!")
        return
    
    if not vs_cols:
        print("No ventral striatum regions found!")
        return
    
    print(f"Found {len(vs_cols)} ventral striatum regions:")
    for col in vs_cols:
        print(f"  - {col}")
    
    # Average across ventral striatum regions (raw)
    if ses1_df is not None:
        vs1_avg = ses1_df[vs_cols].mean(axis=1).values
        vs1_smooth = smooth_timeseries(vs1_avg, window=5)
        time1 = np.arange(len(vs1_avg))
        time1_smooth = np.arange(len(vs1_smooth)) * 5 + 2.5  # Center of each bin
    else:
        vs1_avg = None
        vs1_smooth = None
        time1 = None
        time1_smooth = None
    
    if ses2_df is not None:
        vs2_avg = ses2_df[vs_cols].mean(axis=1).values
        vs2_smooth = smooth_timeseries(vs2_avg, window=5)
        time2 = np.arange(len(vs2_avg))
        time2_smooth = np.arange(len(vs2_smooth)) * 5 + 2.5
    else:
        vs2_avg = None
        vs2_smooth = None
        time2 = None
        time2_smooth = None
    
    if ses3_df is not None:
        vs3_avg = ses3_df[vs_cols].mean(axis=1).values
        vs3_smooth = smooth_timeseries(vs3_avg, window=5)
        time3 = np.arange(len(vs3_avg))
        time3_smooth = np.arange(len(vs3_smooth)) * 5 + 2.5
    else:
        vs3_avg = None
        vs3_smooth = None
        time3 = None
        time3_smooth = None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    
    # Plot 1: SMOOTHED sessions overlaid (averaged every 5 TRs)
    ax1 = axes[0]
    
    if vs1_smooth is not None:
        ax1.plot(time1_smooth, vs1_smooth, color='#1f77b4', linewidth=3, alpha=0.9, label='Session 1', marker='o', markersize=4)
    if vs2_smooth is not None:
        ax1.plot(time2_smooth, vs2_smooth, color='#ff7f0e', linewidth=3, alpha=0.9, label='Session 2', marker='s', markersize=4)
    if vs3_smooth is not None:
        ax1.plot(time3_smooth, vs3_smooth, color='#2ca02c', linewidth=3, alpha=0.9, label='Session 3', marker='^', markersize=4)
    
    ax1.set_xlabel('Time (TRs)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('BOLD Signal (z-scored)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Ventral Striatum BOLD Activation Timeline (SMOOTHED - Avg every 5 TRs)\n{task_name.upper()} Task - All Sessions Overlaid', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add text annotation about smoothing
    ax1.text(0.02, 0.98, 'Data smoothed: 5-TR moving average', 
             transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Sessions concatenated (SMOOTHED)
    ax2 = axes[1]
    
    all_signals_smooth = []
    all_times_smooth = []
    session_boundaries = []
    current_time = 0
    
    if vs1_smooth is not None:
        all_signals_smooth.extend(vs1_smooth)
        all_times_smooth.extend(np.arange(current_time, current_time + len(vs1_smooth) * 5, 5))
        current_time += len(vs1_smooth) * 5
        session_boundaries.append((current_time, 'Session 1 end'))
    
    if vs2_smooth is not None:
        all_signals_smooth.extend(vs2_smooth)
        all_times_smooth.extend(np.arange(current_time, current_time + len(vs2_smooth) * 5, 5))
        current_time += len(vs2_smooth) * 5
        session_boundaries.append((current_time, 'Session 2 end'))
    
    if vs3_smooth is not None:
        all_signals_smooth.extend(vs3_smooth)
        all_times_smooth.extend(np.arange(current_time, current_time + len(vs3_smooth) * 5, 5))
        current_time += len(vs3_smooth) * 5
    
    # Determine colors based on session
    colors_smooth = []
    idx = 0
    if vs1_smooth is not None:
        colors_smooth.extend(['#1f77b4'] * len(vs1_smooth))
        idx += len(vs1_smooth)
    if vs2_smooth is not None:
        colors_smooth.extend(['#ff7f0e'] * len(vs2_smooth))
        idx += len(vs2_smooth)
    if vs3_smooth is not None:
        colors_smooth.extend(['#2ca02c'] * len(vs3_smooth))
    
    # Plot as colored line
    for i in range(len(all_times_smooth)-1):
        ax2.plot(all_times_smooth[i:i+2], all_signals_smooth[i:i+2], 
                color=colors_smooth[i], linewidth=3, alpha=0.9, marker='o', markersize=3)
    
    # Add session boundary lines
    for boundary_time, boundary_label in session_boundaries[:-1] if session_boundaries else []:
        ax2.axvline(boundary_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.text(boundary_time, ax2.get_ylim()[1]*0.95, 'Session\nBoundary', 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add session labels
    session_midpoints = []
    current = 0
    if vs1_smooth is not None:
        session_midpoints.append((current, current + len(vs1_smooth)*5, 'Session 1'))
        current += len(vs1_smooth)*5
    if vs2_smooth is not None:
        session_midpoints.append((current, current + len(vs2_smooth)*5, 'Session 2'))
        current += len(vs2_smooth)*5
    if vs3_smooth is not None:
        session_midpoints.append((current, current + len(vs3_smooth)*5, 'Session 3'))
    
    for start, end, label in session_midpoints:
        midpoint = (start + end) / 2
        ax2.text(midpoint, ax2.get_ylim()[0]*0.95, label, 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax2.set_xlabel('Time (TRs) - Continuous', fontsize=14, fontweight='bold')
    ax2.set_ylabel('BOLD Signal (z-scored)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Ventral Striatum BOLD Activation Timeline (SMOOTHED - Avg every 5 TRs)\n{task_name.upper()} Task - Sessions Concatenated', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add smoothing note
    ax2.text(0.02, 0.98, 'Data smoothed: 5-TR moving average', 
             transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'{task_name}_ventral_striatum_timeline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")
    
    # Also create individual region plots
    plot_individual_regions(ses1_df, ses2_df, ses3_df, vs_cols, task_name)

def plot_individual_regions(ses1_df, ses2_df, ses3_df, vs_cols, task_name):
    """Plot each ventral striatum region separately with smoothing"""
    
    n_regions = len(vs_cols)
    n_cols = 2
    n_rows = (n_regions + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_regions > 1 else [axes]
    
    for idx, region in enumerate(vs_cols):
        ax = axes[idx]
        
        if ses1_df is not None and region in ses1_df.columns:
            data1 = ses1_df[region].values
            data1_smooth = smooth_timeseries(data1, window=5)
            time1_smooth = np.arange(len(data1_smooth)) * 5 + 2.5
            ax.plot(time1_smooth, data1_smooth, color='#1f77b4', linewidth=2.5, alpha=0.9, 
                   label='Session 1', marker='o', markersize=3)
        
        if ses2_df is not None and region in ses2_df.columns:
            data2 = ses2_df[region].values
            data2_smooth = smooth_timeseries(data2, window=5)
            time2_smooth = np.arange(len(data2_smooth)) * 5 + 2.5
            ax.plot(time2_smooth, data2_smooth, color='#ff7f0e', linewidth=2.5, alpha=0.9, 
                   label='Session 2', marker='s', markersize=3)
        
        if ses3_df is not None and region in ses3_df.columns:
            data3 = ses3_df[region].values
            data3_smooth = smooth_timeseries(data3, window=5)
            time3_smooth = np.arange(len(data3_smooth)) * 5 + 2.5
            ax.plot(time3_smooth, data3_smooth, color='#2ca02c', linewidth=2.5, alpha=0.9, 
                   label='Session 3', marker='^', markersize=3)
        
        ax.set_xlabel('Time (TRs)', fontsize=10)
        ax.set_ylabel('BOLD Signal (z-scored)', fontsize=10)
        ax.set_title(f'{region}\n(smoothed: 5-TR average)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Individual Ventral Striatum Regions - {task_name.upper()} Task', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'{task_name}_ventral_striatum_individual_regions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("VENTRAL STRIATUM BOLD ACTIVATION TIMELINE")
    print("="*80)
    
    # Load GLASS task data
    print("\nLoading GLASS task data...")
    ses1 = load_timeseries("sub-001_ses-1_task-glass_aal_ts.csv")
    ses2 = load_timeseries("sub-001_ses-2_task-glass_aal_ts.csv")
    ses3 = load_timeseries("sub-001_ses-3_task-glass_aal_ts.csv")
    
    if ses1 is not None:
        print(f"  Session 1: {ses1.shape}")
    if ses2 is not None:
        print(f"  Session 2: {ses2.shape}")
    if ses3 is not None:
        print(f"  Session 3: {ses3.shape}")
    
    # Create timeline plots
    plot_ventral_striatum_timeline(ses1, ses2, ses3, 'GLASS')
    
    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}/")
    print("="*80 + "\n")
