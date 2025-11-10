"""
Input validation utilities for simulation parameters.

This module provides functions for validating simulation inputs.
"""

from typing import Tuple
import warnings


def validate_simulation_params(B_rms: float, dt: float, T_max: float, M: int) -> bool:
    """
    Validate basic simulation parameters.
    
    Parameters
    ----------
    B_rms : float
        RMS noise amplitude (Tesla)
    dt : float
        Time step (seconds)
    T_max : float
        Maximum simulation time (seconds)
    M : int
        Number of realizations
        
    Returns
    -------
    valid : bool
        True if all parameters are valid
    """
    if B_rms <= 0:
        raise ValueError(f"B_rms must be positive, got {B_rms}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if T_max <= 0:
        raise ValueError(f"T_max must be positive, got {T_max}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if not isinstance(M, int):
        raise TypeError(f"M must be an integer, got {type(M)}")
    
    # Check that T_max is reasonable compared to dt
    if T_max < 10 * dt:
        warnings.warn(
            f"T_max={T_max:.2e}s is very small compared to dt={dt:.2e}s. "
            f"Recommend T_max >= 10*dt for meaningful results.",
            UserWarning
        )
    
    return True


def validate_tau_c_range(tau_c_range: Tuple[float, float], tau_c_num: int) -> bool:
    """
    Validate tau_c range and number of points.
    
    Parameters
    ----------
    tau_c_range : tuple
        (min, max) correlation time range (seconds)
    tau_c_num : int
        Number of tau_c values
        
    Returns
    -------
    valid : bool
        True if range is valid
    """
    if len(tau_c_range) != 2:
        raise ValueError(f"tau_c_range must be a tuple of (min, max), got {tau_c_range}")
    
    tau_c_min, tau_c_max = tau_c_range
    
    if tau_c_min <= 0:
        raise ValueError(f"tau_c_range min must be positive, got {tau_c_min}")
    if tau_c_max <= tau_c_min:
        raise ValueError(f"tau_c_range max must be > min, got max={tau_c_max}, min={tau_c_min}")
    if tau_c_num <= 0:
        raise ValueError(f"tau_c_num must be positive, got {tau_c_num}")
    if not isinstance(tau_c_num, int):
        raise TypeError(f"tau_c_num must be an integer, got {type(tau_c_num)}")
    
    return True

