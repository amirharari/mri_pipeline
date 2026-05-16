"""
============================================================================
GLOBAL ENTROPY ANALYSIS OF REST STATE FUNCTIONAL CONNECTIVITY
============================================================================

This script calculates several well-known entropy metrics from functional connectivity
correlation matrices. These metrics provide measures of brain complexity, integration,
and information capacity from different perspectives.

ENTROPY METRICS CALCULATED:
1. Sample Entropy (SampEn): Measures regularity/predictability of a time series
   - Lower values indicate more predictable/regular patterns
   - Higher values indicate more irregular/complex patterns
   - Important in neuroimaging for quantifying system complexity

2. Approximate Entropy (ApEn): Similar to SampEn but less robust to outliers
   - Also measures regularity
   - Older method, included for comparison with literature

3. Spectral Entropy: Entropy of the power spectral density
   - Measures frequency content complexity
   - High spectral entropy = broadband frequency content (less predictable)
   - Low spectral entropy = narrowband frequency content (more predictable)

4. Shannon Entropy: Information-theoretic measure of distribution uncertainty
   - Measures how "spread out" probability distributions are
   - Applied to the distribution of correlation values

5. Correlation Matrix Entropy: Shannon entropy of the correlation distribution
   - Uses histogram of correlation values
   - Higher entropy = more diverse connectivity patterns

USAGE:
    python 1_global_entropy_analysis.py

OUTPUT:
    - CSV file with entropy metrics for each scan
    - Text summary printed to console
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FOLDER = r"C:\kpeSoundPath\rest-results-itamar"
OUTPUT_FOLDER = "entropy_analysis_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =============================================================================
# ENTROPY CALCULATION FUNCTIONS
# =============================================================================

def sample_entropy(time_series, m=2, r=0.2):
    """
    Calculate Sample Entropy (SampEn) of a time series.
    
    Sample Entropy is the negative natural logarithm of the conditional probability
    that two sequences of length m that match point-wise (within a tolerance r)
    will also match at the next point.
    
    KEY DIFFERENCE FROM ApEn: SampEn EXCLUDES self-matches (i != j) to make it
    more robust and consistent across different data lengths.
    
    Ref: Richman & Moorman (2000) "Physiological time-series analysis using 
         approximate entropy and sample entropy"
    
    Parameters:
    -----------
    time_series : array-like
        The time series data
    m : int
        Pattern length (default 2)
    r : float
        Tolerance (default 0.2 = 20% of standard deviation)
    
    Returns:
    --------
    sampen : float
        Sample Entropy value
    """
    N = len(time_series)
    r = r * np.std(time_series)
    
    def _maxdist(xi, xj, m):
        """Compute maximum distance between vectors"""
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    # Count matches for patterns of length m
    def _count_matches_m(m_pattern_length):
        """Count matches excluding self-matches"""
        # Create all patterns of length m
        patterns = np.array([time_series[i:i + m_pattern_length] 
                           for i in range(N - m_pattern_length)])
        
        matches = 0
        total_pairs = 0
        
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j:  # EXCLUDE self-matches (key difference from ApEn)
                    total_pairs += 1
                    if _maxdist(patterns[i], patterns[j], m_pattern_length) <= r:
                        matches += 1
        
        if total_pairs == 0:
            return 0.0
        
        return matches / total_pairs
    
    try:
        B_m = _count_matches_m(m)
        B_m1 = _count_matches_m(m + 1)
        
        if B_m == 0 or B_m1 == 0:
            return np.nan
        
        return -np.log(B_m1 / B_m)
    except:
        return np.nan


def approximate_entropy(time_series, m=2, r=0.2):
    """
    Calculate Approximate Entropy (ApEn) of a time series.
    
    ApEn is similar to Sample Entropy but INCLUDES self-matches in the calculation.
    The key difference is that ApEn uses the average of logarithms:
    ApEn(m,r,N) = Phi_m(r) - Phi_m+1(r)
    where Phi_m = (1/(N-m+1)) * sum_i ln(C_i^m(r))
    
    Note: The correct formula should be mean(log(C_i)), not log(mean(C_i))
    
    Parameters:
    -----------
    time_series : array-like
        The time series data
    m : int
        Pattern length (default 2)
    r : float
        Tolerance (default 0.2 = 20% of standard deviation)
    
    Returns:
    --------
    apen : float
        Approximate Entropy value
    """
    N = len(time_series)
    r = r * np.std(time_series)
    
    def _maxdist(xi, xj, m):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi_m(m_length):
        """
        Calculate Phi_m using the correct formula:
        Phi_m = (1/(N-m+1)) * sum_i ln(C_i^m(r))
        where C_i^m is the conditional probability
        """
        # Create all patterns of length m
        patterns = np.array([time_series[i:i + m_length] 
                           for i in range(N - m_length)])
        
        log_sum = 0.0
        valid_count = 0
        
        for i in range(len(patterns)):
            # Count matches INCLUDING self-matches (i == j is allowed)
            matches = sum(1 for j in range(len(patterns)) 
                         if _maxdist(patterns[i], patterns[j], m_length) <= r)
            
            # C_i^m = matches / (N - m + 1)
            C_i_m = matches / (N - m_length + 1)
            
            if C_i_m > 0:
                # Use ln(C_i^m) not log(mean(C))
                log_sum += np.log(C_i_m)
                valid_count += 1
        
        if valid_count == 0:
            return 0.0
        
        # Phi_m = (1/(N-m+1)) * sum_i ln(C_i^m)
        return (1.0 / (N - m_length + 1)) * log_sum
    
    try:
        phi_m = _phi_m(m)
        phi_m1 = _phi_m(m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return np.nan
        
        # ApEn = Phi_m - Phi_m1
        return phi_m - phi_m1
    except:
        return np.nan


def spectral_entropy(time_series, fs=1.0, normalize=True):
    """
    Calculate Spectral Entropy of a time series.
    
    Measures the entropy of the power spectral density (PSD).
    A higher spectral entropy indicates more uniform distribution of power
    across frequencies (i.e., more broadband or "noisy" signal).
    
    Parameters:
    -----------
    time_series : array-like
        The time series data
    fs : float
        Sampling frequency in Hz (default 1.0 for fMRI with TR=1s)
        For TR=2s, use fs=0.5
    normalize : bool
        If True, normalize entropy to [0, 1] range by dividing by log2(P.size)
        This makes entropy comparable across different scan lengths
    
    Returns:
    --------
    spectral_entropy : float
        Spectral Entropy value (in [0, 1] if normalize=True)
    """
    try:
        # Calculate power spectral density using Welch's method
        freqs, psd = welch(time_series, fs=fs, nperseg=min(256, len(time_series) // 4))
        
        # Normalize PSD to make it a probability distribution
        psd_norm = psd / np.sum(psd)
        
        # Remove zero values to avoid log(0)
        psd_norm = psd_norm[psd_norm > 0]
        
        # Calculate Shannon entropy
        spectral_entropy_val = -np.sum(psd_norm * np.log2(psd_norm))
        
        # Normalize to [0, 1] range if requested
        if normalize:
            max_entropy = np.log2(len(psd_norm))
            if max_entropy > 0:
                spectral_entropy_val = spectral_entropy_val / max_entropy
        
        return spectral_entropy_val
    except:
        return np.nan


def shannon_entropy(x, bins=50):
    """
    Calculate Shannon Entropy of a distribution.
    
    Measures the uncertainty/information content in a probability distribution.
    Higher entropy means more "surprise" or less predictability.
    
    H(X) = -Σ p(x) log2(p(x))
    
    Parameters:
    -----------
    x : array-like
        Data values
    bins : int
        Number of bins for histogram
    
    Returns:
    --------
    shannon_entropy : float
        Shannon Entropy value
    """
    try:
        # Create histogram to estimate probability distribution
        hist, _ = np.histogram(x, bins=bins)
        
        # Normalize to probabilities
        probs = hist / np.sum(hist)
        
        # Remove zeros
        probs = probs[probs > 0]
        
        # Calculate Shannon entropy
        shannon_entropy_val = -np.sum(probs * np.log2(probs))
        
        return shannon_entropy_val
    except:
        return np.nan


def correlation_matrix_entropy(corr_matrix, bins=50):
    """
    Calculate entropy of the correlation matrix distribution.
    
    This measures the diversity of connectivity patterns in the brain.
    Higher entropy = more diverse connectivity patterns.
    Lower entropy = more uniform connectivity (clustered around a few values).
    
    Parameters:
    -----------
    corr_matrix : numpy array
        Correlation matrix
    bins : int
        Number of bins for histogram
    
    Returns:
    --------
    corr_entropy : float
        Correlation matrix entropy
    """
    # Extract upper triangle (excluding diagonal)
    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    
    # Calculate Shannon entropy of the distribution
    return shannon_entropy(upper_triangle, bins=bins)


def calculate_all_entropy_metrics(timeseries_data, correlation_matrix):
    """
    Calculate all entropy metrics for a given scan.
    
    Parameters:
    -----------
    timeseries_data : pandas DataFrame
        Time series data (rows = timepoints, columns = ROIs)
    correlation_matrix : numpy array
        Correlation matrix (ROIs x ROIs)
    
    Returns:
    --------
    dict : Dictionary containing all entropy metrics
    """
    results = {}
    
    print("  Computing entropy metrics...")
    
    # Calculate entropy for each ROI time series (for Sample/Approximate Entropy)
    sampen_values = []
    apen_values = []
    spectral_entropy_values = []
    
    # Take first 20 ROIs to avoid too long computation
    n_rois_to_use = min(20, timeseries_data.shape[1])
    
    for col_idx in range(n_rois_to_use):
        ts = timeseries_data.iloc[:, col_idx].values
        
        # Remove NaN values
        ts_clean = ts[~np.isnan(ts)]
        
        if len(ts_clean) > 20:  # Need sufficient data points
            # Sample Entropy
            sampen = sample_entropy(ts_clean, m=2, r=0.2)
            if not np.isnan(sampen):
                sampen_values.append(sampen)
            
            # Approximate Entropy
            apen = approximate_entropy(ts_clean, m=2, r=0.2)
            if not np.isnan(apen):
                apen_values.append(apen)
            
            # Spectral Entropy
            spec_ent = spectral_entropy(ts_clean, fs=1.0)
            if not np.isnan(spec_ent):
                spectral_entropy_values.append(spec_ent)
    
    # Average across ROIs
    results['sample_entropy_mean'] = np.mean(sampen_values) if sampen_values else np.nan
    results['sample_entropy_std'] = np.std(sampen_values) if sampen_values else np.nan
    results['approximate_entropy_mean'] = np.mean(apen_values) if apen_values else np.nan
    results['approximate_entropy_std'] = np.std(apen_values) if apen_values else np.nan
    results['spectral_entropy_mean'] = np.mean(spectral_entropy_values) if spectral_entropy_values else np.nan
    results['spectral_entropy_std'] = np.std(spectral_entropy_values) if spectral_entropy_values else np.nan
    
    # Correlation matrix entropy
    results['correlation_matrix_entropy'] = correlation_matrix_entropy(correlation_matrix, bins=50)
    
    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def main():
    print("="*80)
    print("GLOBAL ENTROPY ANALYSIS")
    print("="*80)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}\n")
    
    # Find all AAL timeseries files
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_aal_ts.csv')]
    csv_files.sort()
    
    print(f"Found {len(csv_files)} scans to analyze\n")
    
    all_results = []
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        
        file_path = os.path.join(DATA_FOLDER, csv_file)
        
        # Read timeseries
        timeseries = pd.read_csv(file_path)
        
        # Compute correlation matrix
        corr_matrix = timeseries.corr().values
        
        print(f"  Data shape: {timeseries.shape}")
        print(f"  Correlation matrix shape: {corr_matrix.shape}")
        
        # Calculate all entropy metrics
        entropy_results = calculate_all_entropy_metrics(timeseries, corr_matrix)
        
        # Add file information
        entropy_results['filename'] = csv_file
        entropy_results['n_timepoints'] = timeseries.shape[0]
        entropy_results['n_rois'] = timeseries.shape[1]
        
        all_results.append(entropy_results)
        
        print(f"  [OK] Completed\n")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_path = os.path.join(OUTPUT_FOLDER, "global_entropy_metrics.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}\n")
    
    # Print summary
    print("="*80)
    print("ENTROPY SUMMARY")
    print("="*80)
    print(f"\n{results_df.to_string()}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Sample Entropy (mean): Lower = more predictable/regular
                      Higher = more irregular/complex
                      
Approximate Entropy (mean): Similar to Sample Entropy
                          Less robust to outliers

Spectral Entropy (mean): Lower = narrowband frequency content
                        Higher = broadband frequency content

Correlation Matrix Entropy: Lower = uniform connectivity patterns
                           Higher = diverse connectivity patterns
    """)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()

