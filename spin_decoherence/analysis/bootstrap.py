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


def bootstrap_T2(t, E_abs_all, E_se=None, B=800, rng=None, verbose=False,  # 물리학적 정확도와 시간 절약의 균형
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
    # IMPROVED: Use per-sample window for better bootstrap variance, especially in static/crossover regimes
    use_per_sample_window = False
    if tau_c is not None and gamma_e is not None and B_rms is not None:
        Delta_omega = gamma_e * B_rms
        xi = Delta_omega * tau_c
        # Use per-sample window for static and crossover regimes (ξ > 0.5)
        # This ensures better bootstrap variance and avoids degenerate CI
        if xi > 0.5:  # Changed from 2.0 to 0.5 to include crossover regime
            use_per_sample_window = True
            if verbose:
                print(f"  Regime detected (ξ = {xi:.3f} > 0.5): using per-sample fitting window for better bootstrap variance")
    
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
    T2_std = np.std(vals, ddof=1)
    
    # Check for degenerate CI (all samples produce same value)
    # CRITICAL FIX: Relax degenerate condition to allow more bootstrap samples through
    # Use relative std instead of absolute std to account for different T2 scales
    T2_std_relative = T2_std / T2_mean if T2_mean > 0 else 0
    
    if T2_std == 0 or (T2_std_relative < 1e-6 and len(np.unique(vals)) < 3):
        if verbose:
            print(f"  Warning: Degenerate bootstrap CI (std = {T2_std:.2e}, std_rel = {T2_std_relative:.2e}, unique values = {len(np.unique(vals))})")
            print(f"  This can happen in static regime or when fitting window is too restrictive")
        # Return None to indicate degenerate CI
        return T2_mean, None, vals
    
    # Compute percentile-based CI
    T2_ci = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    
    # Check if CI is too narrow (likely degenerate)
    ci_width = T2_ci[1] - T2_ci[0]
    ci_width_relative = ci_width / T2_mean if T2_mean > 0 else 0
    
    # If CI width is less than 0.01% of mean, treat as degenerate
    if ci_width_relative < 1e-4:
        if verbose:
            print(f"  Warning: CI width too narrow ({ci_width_relative*100:.4f}%), treating as degenerate")
        return T2_mean, None, vals
    
    return T2_mean, T2_ci, vals

