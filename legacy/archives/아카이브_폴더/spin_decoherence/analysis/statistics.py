"""
Statistical analysis utilities for simulation results.

This module provides functions for computing statistics and confidence intervals
on simulation data.
"""

import numpy as np
from typing import Tuple, Optional


def compute_statistics(data: np.ndarray, axis: Optional[int] = None) -> dict:
    """
    Compute basic statistics on data.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    axis : int, optional
        Axis along which to compute statistics
        
    Returns
    -------
    stats : dict
        Dictionary containing mean, std, min, max, median
    """
    return {
        'mean': np.mean(data, axis=axis),
        'std': np.std(data, axis=axis, ddof=1),
        'min': np.min(data, axis=axis),
        'max': np.max(data, axis=axis),
        'median': np.median(data, axis=axis),
    }


def compute_confidence_intervals(data: np.ndarray, confidence: float = 0.95,
                                 axis: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals using percentile method.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    confidence : float
        Confidence level (default: 0.95 for 95% CI)
    axis : int, optional
        Axis along which to compute intervals
        
    Returns
    -------
    lower : ndarray
        Lower confidence bound
    upper : ndarray
        Upper confidence bound
    """
    alpha = 1.0 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower = np.percentile(data, lower_percentile, axis=axis)
    upper = np.percentile(data, upper_percentile, axis=axis)
    
    return lower, upper

