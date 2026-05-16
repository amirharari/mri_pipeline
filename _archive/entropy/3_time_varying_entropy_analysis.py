"""
============================================================================
TIME-VARYING ENTROPY ANALYSIS WITH SLIDING WINDOWS
============================================================================

This script analyzes how brain entropy changes over time during a 10-minute
resting-state scan using a sliding window approach.

METHODOLOGY:
1. Define a sliding window using seconds (converted to points via TR)
2. Slide this window across the scan with configurable step
3. For each window, compute a correlation matrix and derive entropy metrics
4. Track how entropy evolves over the course of the scan

This approach allows us to:
- Detect temporal dynamics in brain connectivity
- Identify periods of high vs low brain complexity
- Study state transitions during rest
- Characterize non-stationarity in resting-state networks

ENTROPY-RELATED MEASURES:
1. Correlation Matrix Entropy (Shannon on r, fixed bin range [-1, 1])
2. Effective Connectivity: Fraction of strong connections (configurable thresholding)
3. Summary distribution stats: mean, std, skewness, kurtosis

Note: Community/modularity metrics are not computed here to keep runtime and
dependencies light. They can be added in a separate module if needed.

USAGE:
    python 3_time_varying_entropy_analysis.py

OUTPUT:
    - CSV files with time-varying entropy for each scan
    - Plots showing entropy trajectories over time
    - Statistics on entropy stability/change

REFERENCES:
- Deco et al. (2013) "Resting-state functional connectivity emerges from
  structurally and dynamically shaped slow linear fluctuations"
- Zalesky et al. (2014) "Time-resolved resting-state brain networks"
- Hansen et al. (2015) "Loose coupling between large-scale brain networks"
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as scipy_entropy

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FOLDER = r"C:\kpeSoundPath\rest-results-itamar"
OUTPUT_FOLDER = "time_varying_entropy_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Sampling/Window parameters
TR_SEC = 1.0                # Repetition time (seconds). Set to your fMRI TR.
WINDOW_SIZE_SEC = 30        # Window length in seconds
STEP_SIZE_SEC = 5           # Step size in seconds

# Derived (points)
WINDOW_SIZE = max(1, int(round(WINDOW_SIZE_SEC / TR_SEC)))
STEP_SIZE = max(1, int(round(STEP_SIZE_SEC / TR_SEC)))

# Entropy/threshold parameters
CORR_BINS = 50               # Number of bins for correlation histogram
HIST_RANGE_R = (-1.0, 1.0)   # Fixed histogram range for r to ensure comparability across windows

# Thresholding strategy for effective connectivity
#   'fixed_r'     -> absolute r > FIXED_R_THRESHOLD
#   'density_top' -> retain top DENSITY_PCT% strongest |r|
#   'zscore'      -> |z| > ZSCORE_THRESHOLD, where z = atanh(r)
THRESHOLD_MODE = 'fixed_r'
FIXED_R_THRESHOLD = 0.3     # For THRESHOLD_MODE = 'fixed_r'
DENSITY_PCT = 10.0          # For THRESHOLD_MODE = 'density_top' (0-100)
ZSCORE_THRESHOLD = 0.5      # For THRESHOLD_MODE = 'zscore'

# Option: summarize distributions in Fisher z (more Gaussian); entropy stays on r in [-1,1]
USE_FISHER_FOR_SUMMARY = False

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# =============================================================================
# SLIDING WINDOW FUNCTIONS
# =============================================================================

def sliding_window_correlation(timeseries, window_size=30, step_size=5):
    """
    Calculate sliding window correlation matrices.
    
    Parameters:
    -----------
    timeseries : pandas DataFrame
        Time series data (rows=timepoints, columns=ROIs)
    window_size : int
        Size of sliding window in timepoints
    step_size : int
        Step size for sliding window
    
    Returns:
    --------
    list : List of correlation matrices for each window
    list : List of window start timepoints
    list : List of window end timepoints
    """
    n_timepoints = timeseries.shape[0]
    n_rois = timeseries.shape[1]
    
    correlation_matrices = []
    window_starts = []
    window_ends = []
    
    # Slide window across time
    for start in range(0, n_timepoints - window_size + 1, step_size):
        end = start + window_size
        
        # Extract window data
        window_data = timeseries.iloc[start:end, :]
        
        # Compute correlation matrix (numeric columns only, drop constant/NaN columns)
        corr_matrix = compute_correlation_matrix(window_data)
        
        correlation_matrices.append(corr_matrix)
        window_starts.append(start)
        window_ends.append(end - 1)
    
    return correlation_matrices, window_starts, window_ends


def shannon_entropy(x, bins=50, hist_range=None):
    """
    Calculate Shannon Entropy of a distribution.
    
    Parameters:
    -----------
    x : array-like
        Data values
    bins : int
        Number of bins for histogram
    
    Returns:
    --------
    float : Shannon Entropy value
    """
    # Create histogram with fixed range if provided (improves comparability across windows)
    if hist_range is not None:
        hist, _ = np.histogram(x, bins=bins, range=hist_range)
    else:
        hist, _ = np.histogram(x, bins=bins)
    
    # Normalize to probabilities
    probs = hist / np.sum(hist)
    
    # Remove zeros
    probs = probs[probs > 0]
    
    # Calculate entropy
    entropy_val = -np.sum(probs * np.log2(probs))
    
    return entropy_val


def fisher_z(values: np.ndarray) -> np.ndarray:
    """Convert correlations to Fisher z with clipping for numerical stability."""
    values = np.asarray(values)
    values = np.clip(values, -0.999999, 0.999999)
    return np.arctanh(values)


def compute_correlation_matrix(window_data: pd.DataFrame) -> np.ndarray:
    """
    Compute a robust correlation matrix:
      - Keep numeric columns only
      - Drop columns that are constant or all-NaN
      - Drop rows with any NaN
    Returns an (n_rois x n_rois) numpy array.
    """
    X = window_data.select_dtypes(include=[np.number]).copy()
    # Drop columns that are all-NaN or constant
    non_constant = X.columns[(X.var(ddof=0) > 0) & (~X.isna().all())]
    X = X[non_constant]
    # Drop rows with any NaN
    X = X.dropna(axis=0, how='any')
    if X.shape[1] == 0:
        # Fallback to zeros if nothing remains
        return np.zeros((0, 0))
    return X.corr().values


def calculate_window_entropy_metrics(corr_matrix):
    """
    Calculate entropy metrics for a single correlation matrix window.
    
    Parameters:
    -----------
    corr_matrix : numpy array
        Correlation matrix for this window
    
    Returns:
    --------
    dict : Dictionary of entropy metrics
    """
    metrics = {}
    
    # Extract upper triangle (excluding diagonal)
    if corr_matrix.size == 0:
        # Empty matrix fallback
        return {
            'correlation_entropy': np.nan,
            'mean_correlation': np.nan,
            'std_correlation': np.nan,
            'effective_connectivity': 0,
            'effective_connectivity_pct': 0.0,
            'positive_corr_fraction': np.nan,
            'negative_corr_fraction': np.nan,
            'correlation_skewness': np.nan,
            'correlation_kurtosis': np.nan,
        }
    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    
    # 1. Correlation Matrix Entropy
    # Entropy on r with fixed histogram range for comparability
    metrics['correlation_entropy'] = shannon_entropy(upper_triangle, bins=CORR_BINS, hist_range=HIST_RANGE_R)
    
    # 2. Mean Correlation Strength
    metrics['mean_correlation'] = np.nanmean(upper_triangle)
    
    # 3. Standard Deviation of Correlations
    metrics['std_correlation'] = np.nanstd(upper_triangle)
    
    # 4. Effective Connectivity (number of strong connections)
    # Effective connectivity using configurable thresholding strategy
    if THRESHOLD_MODE == 'fixed_r':
        strong_mask = np.abs(upper_triangle) > FIXED_R_THRESHOLD
    elif THRESHOLD_MODE == 'density_top':
        k = int(np.ceil((DENSITY_PCT / 100.0) * upper_triangle.size))
        if k <= 0:
            strong_mask = np.zeros_like(upper_triangle, dtype=bool)
        else:
            thresh = np.partition(np.abs(upper_triangle), -k)[-k]
            strong_mask = np.abs(upper_triangle) >= thresh
    elif THRESHOLD_MODE == 'zscore':
        z_vals = fisher_z(upper_triangle)
        strong_mask = np.abs(z_vals) > ZSCORE_THRESHOLD
    else:
        strong_mask = np.abs(upper_triangle) > FIXED_R_THRESHOLD
    strong_connections = int(np.sum(strong_mask))
    metrics['effective_connectivity'] = strong_connections
    metrics['effective_connectivity_pct'] = 100.0 * strong_connections / len(upper_triangle)
    
    # 5. Positive Correlation Fraction
    metrics['positive_corr_fraction'] = np.sum(upper_triangle > 0) / len(upper_triangle)
    
    # 6. Negative Correlation Fraction
    metrics['negative_corr_fraction'] = np.sum(upper_triangle < 0) / len(upper_triangle)
    
    # 7. Skewness of distribution
    from scipy.stats import skew
    if USE_FISHER_FOR_SUMMARY:
        vals_for_summary = fisher_z(upper_triangle)
    else:
        vals_for_summary = upper_triangle
    metrics['correlation_skewness'] = skew(vals_for_summary)
    
    # 8. Kurtosis of distribution
    from scipy.stats import kurtosis
    metrics['correlation_kurtosis'] = kurtosis(vals_for_summary)
    
    return metrics


def analyze_time_varying_entropy(timeseries, filename):
    """
    Main function to analyze time-varying entropy for a scan.
    
    Parameters:
    -----------
    timeseries : pandas DataFrame
        Time series data
    filename : str
        Name of the file
    
    Returns:
    --------
    pd.DataFrame : Results for each window
    """
    print(f"  Computing sliding window correlations...")
    print(f"    Window size: {WINDOW_SIZE} timepoints ({WINDOW_SIZE_SEC}s)")
    print(f"    Step size: {STEP_SIZE} timepoints ({STEP_SIZE_SEC}s)")
    
    # Compute sliding window correlations
    correlation_matrices, window_starts, window_ends = sliding_window_correlation(
        timeseries, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )
    
    n_windows = len(correlation_matrices)
    print(f"  Created {n_windows} windows")
    
    # Calculate metrics for each window
    print(f"  Calculating entropy metrics for each window...")
    
    window_results = []
    
    for i, corr_matrix in enumerate(correlation_matrices):
        metrics = calculate_window_entropy_metrics(corr_matrix)
        
        metrics['window'] = i
        metrics['start_timepoint'] = window_starts[i]
        metrics['end_timepoint'] = window_ends[i]
        metrics['window_center'] = (window_starts[i] + window_ends[i]) / 2
        
        window_results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(window_results)
    
    return results_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_entropy_trajectory(results_df, filename, output_folder):
    """
    Plot time-varying entropy trajectories.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Window-by-window results
    filename : str
        Name of the file
    output_folder : str
        Output directory
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Time-Varying Entropy Metrics\n{filename}', fontsize=14)
    
    # Plot 1: Correlation Entropy
    ax = axes[0, 0]
    ax.plot(results_df['window_center'], results_df['correlation_entropy'], 'b-', linewidth=2)
    ax.set_xlabel('Time (timepoints)')
    ax.set_ylabel('Correlation Entropy')
    ax.set_title('Correlation Matrix Entropy')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean Correlation
    ax = axes[0, 1]
    ax.plot(results_df['window_center'], results_df['mean_correlation'], 'g-', linewidth=2)
    ax.set_xlabel('Time (timepoints)')
    ax.set_ylabel('Mean Correlation')
    ax.set_title('Average Connectivity Strength')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Effective Connectivity
    ax = axes[1, 0]
    ax.plot(results_df['window_center'], results_df['effective_connectivity_pct'], 'r-', linewidth=2)
    ax.set_xlabel('Time (timepoints)')
    ax.set_ylabel('Effective Connectivity (%)')
    ax.set_title('Percentage of Strong Connections')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Positive/Negative Correlations
    ax = axes[1, 1]
    ax.plot(results_df['window_center'], results_df['positive_corr_fraction']*100, 
            'b-', label='Positive', linewidth=2)
    ax.plot(results_df['window_center'], results_df['negative_corr_fraction']*100, 
            'r-', label='Negative', linewidth=2)
    ax.set_xlabel('Time (timepoints)')
    ax.set_ylabel('Fraction (%)')
    ax.set_title('Positive vs Negative Correlations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Correlation Skewness
    ax = axes[2, 0]
    ax.plot(results_df['window_center'], results_df['correlation_skewness'], 'purple', linewidth=2)
    ax.set_xlabel('Time (timepoints)')
    ax.set_ylabel('Skewness')
    ax.set_title('Distribution Skewness')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Correlation Kurtosis
    ax = axes[2, 1]
    ax.plot(results_df['window_center'], results_df['correlation_kurtosis'], 'orange', linewidth=2)
    ax.set_xlabel('Time (timepoints)')
    ax.set_ylabel('Kurtosis')
    ax.set_title('Distribution Kurtosis')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = filename.replace('.csv', '_time_varying_entropy.png')
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {output_filename}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def main():
    print("="*80)
    print("TIME-VARYING ENTROPY ANALYSIS")
    print("="*80)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Window size: {WINDOW_SIZE} timepoints ({WINDOW_SIZE}s)")
    print(f"Step size: {STEP_SIZE} timepoints ({STEP_SIZE}s)\n")
    
    # Find all timeseries files (use Schaefer for computational efficiency)
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_schaefer100_ts.csv')]
    csv_files.sort()
    
    print(f"Found {len(csv_files)} scans to analyze\n")
    
    all_summaries = []
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        
        file_path = os.path.join(DATA_FOLDER, csv_file)
        
        # Read timeseries
        timeseries = pd.read_csv(file_path)
        
        print(f"  Data shape: {timeseries.shape}")
        
        # Analyze time-varying entropy
        results_df = analyze_time_varying_entropy(timeseries, csv_file)
        
        # Calculate summary statistics
        summary = {
            'filename': csv_file,
            'n_windows': len(results_df),
            'entropy_mean': results_df['correlation_entropy'].mean(),
            'entropy_std': results_df['correlation_entropy'].std(),
            'entropy_range': results_df['correlation_entropy'].max() - results_df['correlation_entropy'].min(),
            'entropy_cv': results_df['correlation_entropy'].std() / results_df['correlation_entropy'].mean(),  # Coefficient of variation
            'mean_corr_mean': results_df['mean_correlation'].mean(),
            'mean_corr_std': results_df['mean_correlation'].std(),
            'eff_conn_mean': results_df['effective_connectivity_pct'].mean(),
            'eff_conn_std': results_df['effective_connectivity_pct'].std(),
        }
        all_summaries.append(summary)
        
        # Save detailed results
        output_path = os.path.join(OUTPUT_FOLDER, csv_file.replace('.csv', '_time_varying_entropy.csv'))
        results_df.to_csv(output_path, index=False)
        print(f"  Saved detailed results: {output_path}")
        
        # Create visualization
        plot_entropy_trajectory(results_df, csv_file, OUTPUT_FOLDER)
        
        print(f"  [OK] Completed\n")
    
    # Save summary
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(OUTPUT_FOLDER, "time_varying_entropy_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(f"\n{summary_df.to_string()}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"""
Time-Varying Entropy Analysis reveals:

1. Entropy Mean: Average diversity of connectivity patterns across the scan
   - Higher = more variable connectivity patterns
   - Lower = more stable connectivity patterns

2. Entropy Std: Temporal variability in entropy
   - Higher = entropy fluctuates more during the scan
   - Lower = entropy remains relatively stable

3. Entropy Range: Maximum spread of entropy values
   - Indicates how much the brain "changes state" during rest

4. Entropy CV (Coefficient of Variation): Normalized variability
   - CV = std/mean
   - Higher CV = more relative variability

Differences between sessions:
- Increasing entropy might indicate more exploratory brain states
- Decreasing entropy might indicate more focused/stable states
- High variability might indicate multiple discrete states during rest
    """)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

