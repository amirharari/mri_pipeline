"""
Edge Weight Entropy Calculation
Takes a connectivity/correlation matrix and returns Edge Weight Entropy for each ROI
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List


def edge_weight_entropy(
    corr_matrix: np.ndarray,
    roi_names: Optional[List[str]] = None,
    use_abs: bool = True,
    threshold: Optional[float] = None,
    include_self: bool = False,
    base: int = 2,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Calculate Edge Weight Entropy (Shannon entropy of connectivity profile) for each ROI.
    
    Parameters:
    -----------
    corr_matrix : np.ndarray
        Correlation/connectivity matrix (N x N)
    roi_names : list[str], optional
        Names of ROIs. If None, will use 'ROI_001', 'ROI_002', etc.
    use_abs : bool
        Use absolute values of correlations (default: True)
    threshold : float, optional
        Only consider correlations above threshold (default: None = use all)
    include_self : bool
        Include self-connections (diagonal) (default: False)
    base : int
        Logarithm base for entropy (default: 2 = bits)
    eps : float
        Small epsilon to avoid log(0) (default: 1e-12)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: 'roi', 'H_edge', 'strength', 'degree'
    """
    assert corr_matrix.ndim == 2, "corr_matrix must be 2D"
    assert corr_matrix.shape[0] == corr_matrix.shape[1], "corr_matrix must be square"
    
    N = corr_matrix.shape[0]
    
    # Prepare ROI names
    if roi_names is None:
        roi_names = [f"ROI_{i:03d}" for i in range(N)]
    else:
        assert len(roi_names) == N, "roi_names length must match matrix size"
    
    W = corr_matrix.copy().astype(float)
    
    # Handle sign
    if use_abs:
        W = np.abs(W)
    
    # Apply threshold
    if threshold is not None:
        W[W < threshold] = 0.0
    
    # Remove diagonal if needed
    if not include_self:
        np.fill_diagonal(W, 0.0)
    
    # Helper: Shannon entropy for probability vector
    def _shannon_entropy(p_vec: np.ndarray) -> float:
        """Calculate Shannon entropy H = -Σ(p_i * log(p_i))"""
        p = p_vec[p_vec > 0]
        if p.size == 0:
            return np.nan
        logp = np.log(p + eps) / np.log(base)
        return float(-np.sum(p * logp))
    
    # Calculate for each ROI
    H_edge = np.full(N, np.nan)
    strength = np.zeros(N)
    degree = np.zeros(N, dtype=int)
    
    for i in range(N):
        row = W[i, :].copy()
        if not include_self:
            row[i] = 0.0
        
        strength[i] = np.sum(row)
        degree[i] = int(np.count_nonzero(row > 0))
        
        if degree[i] > 0:
            # Normalize to probabilities
            p = row / (row.sum() + eps)
            H_edge[i] = _shannon_entropy(p)
    
    return pd.DataFrame({
        'roi': roi_names,
        'H_edge': H_edge,
        'strength': strength,
        'degree': degree
    })
