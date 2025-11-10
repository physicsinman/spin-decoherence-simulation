"""
Bootstrap resampling for confidence interval estimation.

This module provides bootstrap methods for estimating confidence intervals
on fitted parameters, particularly T_2 values.
"""

import numpy as np
from typing import Tuple, Optional
from spin_decoherence.analysis.fitting import (
    fit_coherence_decay_with_offset,
    select_fit_window,
)


def bootstrap_T2(t, E_abs_all, E_se=None, B=500, rng=None, verbose=False,
                 tau_c=None, gamma_e=None, B_rms=None):
    """
    Bootstrap resampling to estimate T_2 confidence intervals.
    
    CRITICAL FIX: Use fixed fitting window and scalar T2 values only.
    - Each bootstrap sample uses the same fit_window_idx to ensure consistency
    - Only scalar T2 values are stored (not arrays)
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_abs_all : ndarray
        Array of |E| trajectories, shape (M, N_steps)
    E_se : ndarray, optional
        Standard error (used for fitting window selection)
    B : int
        Number of bootstrap samples (default: 500)
    rng : numpy.random.Generator, optional
        Random number generator
    verbose : bool
        Whether to print diagnostic information
    tau_c : float, optional
        Correlation time (for regime-aware window selection)
    gamma_e : float, optional
        Electron gyromagnetic ratio (for regime-aware window selection)
    B_rms : float, optional
        RMS noise amplitude (for regime-aware window selection)
        
    Returns
    -------
    T2_mean : float
        Mean T_2 from bootstrap samples
    T2_ci : tuple
        (lower, upper) 95% confidence interval
    T2_samples : ndarray
        All bootstrap T_2 values (scalar array)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    M = E_abs_all.shape[0]
    
    # CRITICAL FIX: For static regime, use per-sample fitting window to avoid degenerate CI
    # In static regime, fixed window causes all bootstrap samples to produce identical T2
    # Solution: Allow each bootstrap sample to select its own fitting window
    use_per_sample_window = False
    if tau_c is not None and gamma_e is not None and B_rms is not None:
        Delta_omega = gamma_e * B_rms
        xi = Delta_omega * tau_c
        # Static regime: use per-sample window
        if xi > 2.0:
            use_per_sample_window = True
            if verbose:
                print(f"  Static regime detected (ξ = {xi:.3f} > 2.0): using per-sample fitting window")
    
    # Select fitting window for original data (if not using per-sample window)
    fit_window_idx = None
    if not use_per_sample_window:
        t_fit, E_fit = select_fit_window(
            t, np.mean(E_abs_all, axis=0), E_se=E_se,
            tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
        )
        # Find indices of fitting window
        fit_window_idx = np.searchsorted(t, t_fit)
        fit_window_idx = fit_window_idx[fit_window_idx < len(t)]
        if len(fit_window_idx) == 0:
            if verbose:
                print("  Warning: No valid fitting window found")
            return None, None, np.array([])
    
    # Bootstrap resampling
    vals = np.empty(B, dtype=np.float64)
    failed_fits = 0
    
    # Show progress for bootstrap if verbose
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(range(B), desc="Bootstrap", disable=False)
    else:
        iterator = range(B)
    
    for b in iterator:
        # Resample trajectories with replacement
        idx = rng.integers(0, M, size=M)
        
        # Bootstrap mean |E|
        E_boot = np.mean(E_abs_all[idx], axis=0)
        
        # CRITICAL FIX: For static regime, select fitting window per sample
        if use_per_sample_window:
            # Select fitting window for this bootstrap sample
            E_se_boot = np.std(E_abs_all[idx], axis=0, ddof=1) / np.sqrt(M) if E_se is None else E_se
            t_fit_boot, E_fit_boot = select_fit_window(
                t, E_boot, E_se=E_se_boot,
                tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
            )
            E_se_fit_boot = None  # Don't use SE for per-sample window
        else:
            # Fit using FIXED window indices
            t_fit_boot = t[fit_window_idx]
            E_fit_boot = E_boot[fit_window_idx]
            E_se_fit_boot = E_se[fit_window_idx] if E_se is not None else None
        
        # Fit to get T_2 (use fitting with offset)
        fit_result = fit_coherence_decay_with_offset(
            t_fit_boot, E_fit_boot, E_se=E_se_fit_boot, model='auto',
            tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms, M=M
        )
        
        if fit_result is not None:
            vals[b] = fit_result['T2']
        else:
            failed_fits += 1
            vals[b] = np.nan
    
    # Remove failed fits
    vals = vals[~np.isnan(vals)]
    
    if len(vals) == 0:
        if verbose:
            print(f"  Warning: All {B} bootstrap fits failed")
        return None, None, np.array([])
    
    if failed_fits > 0 and verbose:
        print(f"  Warning: {failed_fits}/{B} bootstrap fits failed")
    
    # Compute confidence interval (95%)
    T2_mean = np.mean(vals)
    T2_ci = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    
    return T2_mean, T2_ci, vals

