"""
Simple Entropy Calculation
Takes a time series and returns Sample Entropy
"""

import numpy as np
from typing import Union


def sample_entropy(time_series: Union[np.ndarray, list], m: int = 2, r: float = 0.2) -> float:
    """
    Calculate Sample Entropy of a time series.
    
    Parameters:
    -----------
    time_series : array-like
        The time series data (1D array)
    m : int
        Pattern length (default: 2)
    r : float
        Tolerance as fraction of standard deviation (default: 0.2)
    
    Returns:
    --------
    float
        Sample Entropy value. Returns np.nan if calculation fails.
    """
    x = np.asarray(time_series, dtype=float)
    N = len(x)
    
    # Check for valid input
    if N < m + 2:
        return np.nan
    
    std_x = np.std(x)
    if np.allclose(std_x, 0.0):
        return np.nan
    
    r_abs = r * std_x
    
    def _count_matches(pattern_length: int) -> float:
        """Count matches excluding self-matches"""
        patterns = np.array([x[i:i + pattern_length] 
                           for i in range(N - pattern_length + 1)])
        
        if patterns.size == 0:
            return 0.0
        
        matches = 0
        total_pairs = 0
        
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j:  # Exclude self-matches
                    total_pairs += 1
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r_abs:
                        matches += 1
        
        return matches / total_pairs if total_pairs > 0 else 0.0
    
    Bm = _count_matches(m)
    Bm1 = _count_matches(m + 1)
    
    if Bm == 0 or Bm1 == 0:
        return np.nan
    
    return float(-np.log(Bm1 / Bm))
