"""
Run Entropy-CAPs Regression Analysis for All Subjects

This script:
1. Reads time series from \\fmri-guy\Shared\connectivity_output_nih\ (timeseries CSV files)
2. Reads CAPs data from NIH_CAPS file (calculates T2 - T1 difference)
3. Matches each subject's timeseries with their CAPs difference
4. Runs entropy_caps_regression analysis for each subject
5. Aggregates and saves results
"""

import numpy as np
import pandas as pd
import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from entropy_caps_regression import compute_all_entropy_measures, linear_regression_analysis


# Configuration
BASE_PATH = r"\\fmri-guy\Shared\connectivity_output_nih"
# Determine NIH_CAPS file path (try .xlsx first, then .csv)
NIH_CAPS_FILE_XLSX = os.path.join(BASE_PATH, "NIH_CAPS.xlsx")
NIH_CAPS_FILE_CSV = os.path.join(BASE_PATH, "NIH_CAPS.csv")
if os.path.exists(NIH_CAPS_FILE_XLSX):
    NIH_CAPS_FILE = NIH_CAPS_FILE_XLSX
elif os.path.exists(NIH_CAPS_FILE_CSV):
    NIH_CAPS_FILE = NIH_CAPS_FILE_CSV
else:
    NIH_CAPS_FILE = NIH_CAPS_FILE_XLSX  # Default, will error if not found
OUTPUT_DIR = "entropy_caps_results"


def extract_subject_id(filename: str) -> str:
    """
    Extract subject ID from filename.
    
    Examples:
        sub-010_ses-MRI1_aal_ts.csv -> '010'
        sub-024_ses-MRI1_aal_ts.csv -> '024'
    """
    basename = os.path.basename(filename)
    # Extract subject ID (format: sub-XXX_ses-MRI1_aal_ts.csv)
    parts = basename.split('_')
    if len(parts) > 0 and parts[0].startswith('sub-'):
        sub_id = parts[0].replace('sub-', '')
        return sub_id
    return None


def load_timeseries_csv(csv_path: str) -> pd.DataFrame:
    """
    Load timeseries CSV file.
    
    Expected format:
        - Columns: ROI names (AAL regions)
        - Rows: Time points
    """
    df = pd.read_csv(csv_path)
    return df


def dataframe_to_timeseries_list(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
    """
    Convert DataFrame to list of time series arrays.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns = ROI names, rows = time points
    
    Returns:
    --------
    tuple[List[np.ndarray], List[str]]
        Tuple of (list of 1D arrays, one per ROI column, list of ROI names)
    """
    timeseries_list = []
    roi_names = []
    
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            ts = df[col].dropna().values
            if len(ts) > 0:
                timeseries_list.append(ts.astype(float))
                roi_names.append(col)
    
    return timeseries_list, roi_names


def load_caps_mapping(caps_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load CAPs data from NIH_CAPS file and calculate T2 - T1 difference.
    
    Parameters:
    -----------
    caps_file : str
        Path to NIH_CAPS.xlsx or NIH_CAPS.csv
    
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary mapping subject ID (as string) to dict with 'caps_difference' (T2 - T1) and 'screening' scores
    """
    # Try to read as Excel first, then CSV
    try:
        if caps_file.endswith('.xlsx'):
            df = pd.read_excel(caps_file)
        else:
            df = pd.read_csv(caps_file)
    except Exception as e:
        raise FileNotFoundError(f"Could not read CAPs file {caps_file}: {e}")
    
    print(f"CAPs file columns: {list(df.columns)}")
    
    # Find T1 and T2 columns (case-insensitive, flexible naming)
    t1_col = None
    t2_col = None
    sub_col = None
    screening_col = None
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        # Check for T1 column (supports T1, T1_TOT_CAPS5, caps_t1, etc.)
        if 't1' in col_lower and t1_col is None:
            t1_col = col
        # Check for T2 column (supports T2, T2_TOT_CAPS5, caps_t2, etc.)
        if 't2' in col_lower and t2_col is None and 't1' not in col_lower:  # Make sure T2 doesn't match T1 columns
            t2_col = col
        # Check for subject ID column
        if col_lower in ['sub', 'subid', 'subject', 'subject_id', 'id']:
            sub_col = col
        # Check for screening column
        if col_lower in ['screening', 'screening_score', 'initial_ptsd', 'ptsdscreening']:
            screening_col = col
    
    if t1_col is None or t2_col is None:
        raise ValueError(f"Could not find T1 and T2 columns in CAPs file. Available columns: {list(df.columns)}")
    
    if sub_col is None:
        raise ValueError(f"Could not find subject ID column in CAPs file. Available columns: {list(df.columns)}")
    
    print(f"Found columns: Subject={sub_col}, T1={t1_col}, T2={t2_col}, Screening={screening_col}")
    
    # Create mapping: subject ID -> {caps_difference, screening}
    mapping = {}
    
    for _, row in df.iterrows():
        # Extract subject ID - handle various formats
        sub_val = row[sub_col]
        if pd.isna(sub_val):
            continue
            
        # Try to normalize subject ID
        sub_str = str(sub_val).strip()
        # Remove 'sub-' prefix if present
        if sub_str.startswith('sub-'):
            sub_str = sub_str.replace('sub-', '')
        # Extract numeric part
        match = re.search(r'\d+', sub_str)
        if match:
            sub_id = match.group(0)
        else:
            sub_id = sub_str
        
        # Calculate T2 - T1 difference
        t1_val = row[t1_col]
        t2_val = row[t2_col]
        
        if pd.isna(t1_val) or pd.isna(t2_val):
            print(f"Warning: Subject {sub_id} has missing T1 or T2 value, skipping")
            continue
        
        # Handle potential Unicode minus sign and convert to float
        t1_str = str(t1_val).strip().replace('\u2212', '-')
        t2_str = str(t2_val).strip().replace('\u2212', '-')
        
        try:
            t1_float = float(t1_str)
            t2_float = float(t2_str)
            caps_diff = t2_float - t1_float  # T2 - T1
        except ValueError as e:
            print(f"Warning: Could not convert T1/T2 to float for subject {sub_id}: {e}")
            continue
        
        # Get screening score if available
        screening = np.nan
        if screening_col is not None and pd.notna(row[screening_col]):
            screening_str = str(row[screening_col]).strip().replace('\u2212', '-')
            try:
                screening = float(screening_str)
            except ValueError:
                screening = np.nan
        
        mapping[sub_id] = {
            'caps_difference': caps_diff,
            'screening': screening
        }
    
    print(f"Loaded {len(mapping)} subjects with CAPs data (T2 - T1)")
    return mapping


def find_all_timeseries_files(directory: str) -> Dict[str, str]:
    """
    Find all timeseries CSV files in the directory.
    Supports various naming patterns: sub-XXX_*.csv, *_aal_ts.csv, etc.
    
    Parameters:
    -----------
    directory : str
        Directory containing the CSV files
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping subject ID to file path
    """
    # Try multiple patterns to find timeseries files
    patterns = [
        os.path.join(directory, "sub-*_aal_ts.csv"),
        os.path.join(directory, "sub-*_ses-*_aal_ts.csv"),
        os.path.join(directory, "sub-*_*.csv"),
        os.path.join(directory, "*_aal_ts.csv"),
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    # Remove duplicates
    all_files = list(set(all_files))
    
    subject_files = {}
    
    for file_path in all_files:
        # Skip NIH_CAPS file if it's in the same directory
        if 'NIH_CAPS' in os.path.basename(file_path):
            continue
            
        sub_id = extract_subject_id(file_path)
        if sub_id is not None:
            # If multiple files for same subject, prefer MRI1/ses-1 if available
            if sub_id not in subject_files:
                subject_files[sub_id] = file_path
            else:
                # Prefer files with ses-MRI1 or ses-1 in name
                current_file = subject_files[sub_id]
                if ('ses-MRI1' in file_path or 'ses-1' in file_path) and \
                   ('ses-MRI1' not in current_file and 'ses-1' not in current_file):
                    subject_files[sub_id] = file_path
    
    return subject_files


def process_single_subject(
    csv_path: str,
    caps_difference: float,
    screening_score: float,
    subject_id: str,
    output_dir: str
) -> Dict:
    """
    Process a single subject: load data and run entropy-CAPs analysis.
    
    Parameters:
    -----------
    csv_path : str
        Path to subject's timeseries CSV file
    caps_difference : float
        CAPs difference for this subject
    screening_score : float
        Initial PTSD (screening) score for this subject
    subject_id : str
        Subject ID
    output_dir : str
        Directory to save individual results
    
    Returns:
    --------
    dict
        Analysis results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Processing Subject: {subject_id}")
    print(f"CAPs Difference: {caps_difference}")
    print(f"Screening Score: {screening_score}")
    print(f"{'='*80}")
    
    try:
        # Load timeseries
        df = load_timeseries_csv(csv_path)
        print(f"Loaded timeseries: {df.shape[0]} time points, {df.shape[1]} ROIs")
        
        # Convert to list of time series
        timeseries_list, roi_names = dataframe_to_timeseries_list(df)
        print(f"Valid ROIs: {len(timeseries_list)}")
        
        if len(timeseries_list) == 0:
            print(f"WARNING: No valid timeseries found for subject {subject_id}")
            return None
        
        # Compute global entropy measures (mean across all ROIs)
        entropy_results = compute_all_entropy_measures(
            timeseries_list,
            sample_entropy_params={'m': 2, 'r': 0.2},
            shannon_entropy_params={'bins': 50, 'hist_range': None},
            edge_weight_params={
                'threshold': 0.1, 
                'include_self': False,
                'use_abs': True,
                'fisher_z_threshold': False
            }
        )
        
        # Create results dictionary with global (mean) entropy measures
        results = {
            'subject_id': subject_id,
            'caps_difference': caps_difference,
            'screening_score': screening_score,
            'csv_path': csv_path,
            'n_timepoints': df.shape[0],
            'n_rois': len(timeseries_list),
            'mean_simple_entropy': entropy_results['mean_simple_entropy'],
            'mean_shannon_entropy': entropy_results['mean_shannon_entropy'],
            'mean_edge_weight_entropy': entropy_results['mean_edge_weight_entropy'],
            # Also keep full arrays for reference
            'simple_entropy_array': entropy_results['simple_entropy'],
            'shannon_entropy_array': entropy_results['shannon_entropy'],
            'edge_weight_entropy_array': entropy_results['edge_weight_entropy'],
            'roi_names': roi_names
        }
        
        print(f"Mean Simple Entropy: {results['mean_simple_entropy']:.4f}")
        print(f"Mean Shannon Entropy: {results['mean_shannon_entropy']:.4f}")
        print(f"Mean Edge Weight Entropy: {results['mean_edge_weight_entropy']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"ERROR processing subject {subject_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_results(all_results: List[Dict]) -> pd.DataFrame:
    """
    Aggregate results from all subjects into a single DataFrame.
    
    Parameters:
    -----------
    all_results : List[Dict]
        List of results dictionaries from each subject
    
    Returns:
    --------
    pd.DataFrame
        Aggregated results with one row per subject
    """
    rows = []
    
    for res in all_results:
        if res is None:
            continue
        
        row = {
            'Subject_ID': res['subject_id'],
            'CAPs_Difference': res['caps_difference'],
            'Screening_Score': res['screening_score'],
            'N_ROIs': res['n_rois'],
            'N_Timepoints': res['n_timepoints'],
            'Mean_Simple_Entropy': res['mean_simple_entropy'],
            'Mean_Shannon_Entropy': res['mean_shannon_entropy'],
            'Mean_Edge_Weight_Entropy': res['mean_edge_weight_entropy'],
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def compute_correlations(df: pd.DataFrame, outcome_var: str = 'CAPs_Difference') -> pd.DataFrame:
    """
    Compute Pearson correlations between outcome variable and entropy measures.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with one row per subject
    outcome_var : str
        Name of outcome variable column ('CAPs_Difference' or 'Screening_Score')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlation results
    """
    from scipy.stats import pearsonr
    
    entropy_measures = ['Mean_Simple_Entropy', 'Mean_Shannon_Entropy', 'Mean_Edge_Weight_Entropy']
    results = []
    
    # Filter out NaN values for outcome variable
    df_clean = df.dropna(subset=[outcome_var])
    
    for entropy_type in entropy_measures:
        # Remove NaN values for both variables
        mask = df_clean[entropy_type].notna() & df_clean[outcome_var].notna()
        x = df_clean.loc[mask, entropy_type].values
        y = df_clean.loc[mask, outcome_var].values
        
        if len(x) > 2:
            corr, p_value = pearsonr(x, y)
            results.append({
                'Entropy_Type': entropy_type.replace('Mean_', '').replace('_', ' '),
                'Correlation': corr,
                'P_Value': p_value,
                'N_Samples': len(x)
            })
        else:
            results.append({
                'Entropy_Type': entropy_type.replace('Mean_', '').replace('_', ' '),
                'Correlation': np.nan,
                'P_Value': np.nan,
                'N_Samples': len(x)
            })
    
    return pd.DataFrame(results)


def perform_cross_subject_regression(df: pd.DataFrame, outcome_var: str = 'CAPs_Difference') -> Dict:
    """
    Perform linear regression across subjects: Entropy measures vs outcome variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with one row per subject, containing entropy measures and outcome
    outcome_var : str
        Name of outcome variable column ('CAPs_Difference' or 'Screening_Score')
    
    Returns:
    --------
    dict
        Dictionary containing regression results for each entropy type
    """
    regression_results = {}
    outcome_name = 'CAPs Difference' if outcome_var == 'CAPs_Difference' else 'Initial PTSD Score (Screening)'
    
    # Regression 1: Mean Simple Entropy vs Outcome
    regression_results['simple_entropy'] = linear_regression_analysis(
        df['Mean_Simple_Entropy'].values,
        df[outcome_var].values,
        X_name='Mean Simple Entropy',
        y_name=outcome_name
    )
    
    # Regression 2: Mean Shannon Entropy vs Outcome
    regression_results['shannon_entropy'] = linear_regression_analysis(
        df['Mean_Shannon_Entropy'].values,
        df[outcome_var].values,
        X_name='Mean Shannon Entropy',
        y_name=outcome_name
    )
    
    # Regression 3: Mean Edge Weight Entropy vs Outcome
    regression_results['edge_weight_entropy'] = linear_regression_analysis(
        df['Mean_Edge_Weight_Entropy'].values,
        df[outcome_var].values,
        X_name='Mean Edge Weight Entropy',
        y_name=outcome_name
    )
    
    return regression_results


def plot_correlations(df: pd.DataFrame, regression_results: Dict, correlation_results: pd.DataFrame, outcome_var: str = 'CAPs_Difference', output_dir: str = OUTPUT_DIR):
    """
    Create scatter plots with regression lines for entropy measures vs outcome variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with one row per subject
    regression_results : Dict
        Dictionary containing regression results for each entropy type
    correlation_results : pd.DataFrame
        DataFrame containing correlation results
    outcome_var : str
        Name of outcome variable column
    output_dir : str
        Directory to save plots
    """
    from scipy.stats import pearsonr
    
    entropy_measures = {
        'Mean_Simple_Entropy': 'Simple Entropy',
        'Mean_Shannon_Entropy': 'Shannon Entropy',
        'Mean_Edge_Weight_Entropy': 'Edge Weight Entropy'
    }
    
    outcome_name = 'CAPs Difference' if outcome_var == 'CAPs_Difference' else 'Initial PTSD Score (Screening)'
    
    # Filter out NaN values
    df_clean = df.dropna(subset=[outcome_var])
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Entropy Measures vs {outcome_name}', fontsize=16, fontweight='bold')
    
    for idx, (entropy_col, entropy_name) in enumerate(entropy_measures.items()):
        ax = axes[idx]
        
        # Get data
        mask = df_clean[entropy_col].notna() & df_clean[outcome_var].notna()
        x = df_clean.loc[mask, entropy_col].values
        y = df_clean.loc[mask, outcome_var].values
        
        # Get regression results
        entropy_key = entropy_col.lower().replace('mean_', '').replace('_', '_')
        if entropy_key == 'simple_entropy':
            reg_key = 'simple_entropy'
        elif entropy_key == 'shannon_entropy':
            reg_key = 'shannon_entropy'
        else:
            reg_key = 'edge_weight_entropy'
        
        reg = regression_results[reg_key]
        
        # Calculate Pearson correlation
        if len(x) > 2:
            r, _ = pearsonr(x, y)
        else:
            r = np.nan
        
        # Plot scatter
        ax.scatter(x, y, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        
        # Plot regression line
        if len(reg['X_clean']) > 0 and not np.isnan(reg['slope']):
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = reg['slope'] * x_line + reg['intercept']
            ax.plot(x_line, y_line, 'r-', linewidth=2, label='Regression line')
        
        # Add statistics to plot
        corr_text = f'r = {r:.3f}' if not np.isnan(r) else 'r = NaN'
        stats_text = f'R² = {reg["r2"]:.3f}\np = {reg["p_value"]:.3f}'
        ax.text(0.05, 0.95, f'{corr_text}\n{stats_text}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        # Labels and title
        ax.set_xlabel(entropy_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(outcome_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{entropy_name} vs {outcome_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if len(reg['X_clean']) > 0 and not np.isnan(reg['slope']):
            ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save figure
    plot_filename = os.path.join(output_dir, f'correlation_plots_{outcome_var.lower()}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved correlation plots to: {plot_filename}")
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("ENTROPY-CAPs REGRESSION ANALYSIS - BATCH PROCESSING")
    print("="*80)
    
    # Load CAPs mapping
    print(f"\nLoading CAPs data from: {NIH_CAPS_FILE}")
    if not os.path.exists(NIH_CAPS_FILE):
        raise FileNotFoundError(f"CAPs file not found: {NIH_CAPS_FILE}")
    
    caps_mapping = load_caps_mapping(NIH_CAPS_FILE)
    print(f"Loaded {len(caps_mapping)} subjects with CAPs differences (T2 - T1)")
    
    # Find all timeseries files
    print(f"\nScanning for timeseries files in: {BASE_PATH}")
    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"Directory not found: {BASE_PATH}")
    
    subject_files = find_all_timeseries_files(BASE_PATH)
    print(f"Found {len(subject_files)} timeseries files")
    
    # Match subjects with CAPs differences and screening scores
    # Handle both zero-padded (e.g., '010') and non-padded (e.g., '10') formats
    matched_subjects = {}
    unmatched_files = []
    unmatched_caps = set(caps_mapping.keys())
    
    for sub_id, file_path in subject_files.items():
        matched = False
        
        # Try exact match first
        if sub_id in caps_mapping:
            matched_subjects[sub_id] = {
                'file_path': file_path,
                'caps_difference': caps_mapping[sub_id]['caps_difference'],
                'screening': caps_mapping[sub_id]['screening']
            }
            matched = True
            if sub_id in unmatched_caps:
                unmatched_caps.remove(sub_id)
        else:
            # Try converting to integer (removes leading zeros) for matching
            try:
                sub_id_int = int(sub_id)
                sub_id_str = str(sub_id_int)
                if sub_id_str in caps_mapping:
                    matched_subjects[sub_id] = {
                        'file_path': file_path,
                        'caps_difference': caps_mapping[sub_id_str]['caps_difference'],
                        'screening': caps_mapping[sub_id_str]['screening']
                    }
                    matched = True
                    if sub_id_str in unmatched_caps:
                        unmatched_caps.remove(sub_id_str)
                else:
                    # Try zero-padded version (e.g., '10' -> '010')
                    sub_id_str_for_pad = str(sub_id_int)
                    sub_id_padded = sub_id_str_for_pad.zfill(3) if len(sub_id_str_for_pad) < 3 else sub_id_str_for_pad
                    if sub_id_padded in caps_mapping:
                        matched_subjects[sub_id] = {
                            'file_path': file_path,
                            'caps_difference': caps_mapping[sub_id_padded]['caps_difference'],
                            'screening': caps_mapping[sub_id_padded]['screening']
                        }
                        matched = True
                        if sub_id_padded in unmatched_caps:
                            unmatched_caps.remove(sub_id_padded)
            except ValueError:
                pass
        
        if not matched:
            unmatched_files.append(sub_id)
    
    # Print matching summary
    if unmatched_files:
        print(f"\nWARNING: {len(unmatched_files)} subjects found in files but not in CAPs mapping:")
        for sub_id in unmatched_files[:10]:  # Show first 10
            print(f"  - Subject {sub_id}")
        if len(unmatched_files) > 10:
            print(f"  ... and {len(unmatched_files) - 10} more")
    
    if unmatched_caps:
        print(f"\nINFO: {len(unmatched_caps)} subjects in CAPs mapping but no matching timeseries files")
        if len(unmatched_caps) <= 20:
            for sub_id in list(unmatched_caps)[:20]:
                print(f"  - Subject {sub_id}")
    
    print(f"\nMatched {len(matched_subjects)} subjects for analysis")
    
    if len(matched_subjects) == 0:
        print("ERROR: No subjects matched. Exiting.")
        return
    
    # Process each subject
    all_results = []
    
    for sub_id, info in matched_subjects.items():
        results = process_single_subject(
            info['file_path'],
            info['caps_difference'],
            info['screening'],
            sub_id,
            OUTPUT_DIR
        )
        
        if results is not None:
            all_results.append(results)
    
    print(f"\n{'='*80}")
    print(f"PROCESSED {len(all_results)} SUBJECTS SUCCESSFULLY")
    print(f"{'='*80}")
    
    # Aggregate results
    if len(all_results) > 0:
        print("\nAggregating results...")
        df_aggregated = aggregate_results(all_results)
        
        # Perform cross-subject regression
        print("\nPerforming cross-subject regression analysis...")
        regression_results = perform_cross_subject_regression(df_aggregated)
        
        # Save aggregated results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        aggregated_file = os.path.join(OUTPUT_DIR, "aggregated_results.csv")
        df_aggregated.to_csv(aggregated_file, index=False)
        print(f"Saved aggregated results to: {aggregated_file}")
        
        # Save regression results
        regression_summary = pd.DataFrame({
            'Entropy_Type': ['Simple Entropy', 'Shannon Entropy', 'Edge Weight Entropy'],
            'Slope': [
                regression_results['simple_entropy']['slope'],
                regression_results['shannon_entropy']['slope'],
                regression_results['edge_weight_entropy']['slope']
            ],
            'Intercept': [
                regression_results['simple_entropy']['intercept'],
                regression_results['shannon_entropy']['intercept'],
                regression_results['edge_weight_entropy']['intercept']
            ],
            'R2': [
                regression_results['simple_entropy']['r2'],
                regression_results['shannon_entropy']['r2'],
                regression_results['edge_weight_entropy']['r2']
            ],
            'P_Value': [
                regression_results['simple_entropy']['p_value'],
                regression_results['shannon_entropy']['p_value'],
                regression_results['edge_weight_entropy']['p_value']
            ],
            'RMSE': [
                regression_results['simple_entropy']['rmse'],
                regression_results['shannon_entropy']['rmse'],
                regression_results['edge_weight_entropy']['rmse']
            ],
            'N_Samples': [
                regression_results['simple_entropy']['n_samples'],
                regression_results['shannon_entropy']['n_samples'],
                regression_results['edge_weight_entropy']['n_samples']
            ]
        })
        
        regression_file = os.path.join(OUTPUT_DIR, "cross_subject_regression_results.csv")
        regression_summary.to_csv(regression_file, index=False)
        print(f"Saved regression results to: {regression_file}")
        
        # Compute correlations between CAPs difference and entropy measures
        print("\nComputing correlations between CAPs Difference and entropy measures...")
        correlation_results = compute_correlations(df_aggregated, outcome_var='CAPs_Difference')
        correlation_file = os.path.join(OUTPUT_DIR, "caps_difference_entropy_correlations.csv")
        correlation_results.to_csv(correlation_file, index=False)
        print(f"Saved correlation results to: {correlation_file}")
        
        # Create correlation plots
        print("\nCreating correlation plots...")
        plot_correlations(df_aggregated, regression_results, correlation_results, outcome_var='CAPs_Difference', output_dir=OUTPUT_DIR)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("AGGREGATED SUMMARY STATISTICS")
        print("="*80)
        print(f"\nNumber of subjects: {len(df_aggregated)}")
        print(f"\nCAPs Differences:")
        print(f"  Mean: {df_aggregated['CAPs_Difference'].mean():.2f}")
        print(f"  Std: {df_aggregated['CAPs_Difference'].std():.2f}")
        print(f"  Range: [{df_aggregated['CAPs_Difference'].min():.2f}, "
              f"{df_aggregated['CAPs_Difference'].max():.2f}]")
        
        print(f"\nMean Entropy Measures (across subjects):")
        print(f"  Simple Entropy: {df_aggregated['Mean_Simple_Entropy'].mean():.4f} "
              f"(±{df_aggregated['Mean_Simple_Entropy'].std():.4f})")
        print(f"  Shannon Entropy: {df_aggregated['Mean_Shannon_Entropy'].mean():.4f} "
              f"(±{df_aggregated['Mean_Shannon_Entropy'].std():.4f})")
        print(f"  Edge Weight Entropy: {df_aggregated['Mean_Edge_Weight_Entropy'].mean():.4f} "
              f"(±{df_aggregated['Mean_Edge_Weight_Entropy'].std():.4f})")
        
        # Print cross-subject regression results
        print("\n" + "="*80)
        print("CROSS-SUBJECT REGRESSION RESULTS")
        print("="*80)
        for entropy_type, reg_res in regression_results.items():
            print(f"\n{entropy_type.replace('_', ' ').title()}:")
            print(f"  X Variable: {reg_res['X_name']}")
            print(f"  Y Variable: {reg_res['y_name']}")
            print(f"  Slope: {reg_res['slope']:.6f}")
            print(f"  Intercept: {reg_res['intercept']:.6f}")
            print(f"  R²: {reg_res['r2']:.4f}")
            print(f"  RMSE: {reg_res['rmse']:.6f}")
            print(f"  P-value: {reg_res['p_value']:.6f}")
            print(f"  N samples: {reg_res['n_samples']}")
        
        # Print correlation results
        print("\n" + "="*80)
        print("CORRELATION: CAPs DIFFERENCE vs ENTROPY MEASURES")
        print("="*80)
        for _, row in correlation_results.iterrows():
            print(f"\n{row['Entropy_Type']}:")
            print(f"  Pearson r: {row['Correlation']:.4f}")
            print(f"  P-value: {row['P_Value']:.6f}")
            print(f"  N samples: {int(row['N_Samples'])}")
        
        print("\n" + "="*80)
        
        # Display first few rows
        print("\nFirst 5 rows of aggregated results:")
        print(df_aggregated.head().to_string())
        
        print(f"\nAll results saved to: {OUTPUT_DIR}/")
    else:
        print("\nWARNING: No successful results to aggregate.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
