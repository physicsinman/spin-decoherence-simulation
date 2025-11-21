#!/usr/bin/env python3
"""
OU Noise Validation Script
Validates OU noise generation by checking:
1. Autocorrelation: ρ(Δt) = e^(-Δt/τc)
2. Variance: σ² = B_rms²
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def exponential_decay(t, tau, A=1.0):
    """Exponential decay model: A * exp(-t/tau)"""
    return A * np.exp(-t / tau)

def validate_ou_noise(trajectory_file, tau_c, B_rms, dt, verbose=True):
    """
    Validate OU noise from trajectory file.
    
    Parameters
    ----------
    trajectory_file : str or Path
        Path to noise trajectory CSV file
    tau_c : float
        Expected correlation time (seconds)
    B_rms : float
        Expected RMS amplitude (Tesla)
    dt : float
        Time step (seconds)
    verbose : bool
        Whether to print results
        
    Returns
    -------
    results : dict
        Validation results
    """
    # Load trajectory
    df = pd.read_csv(trajectory_file)
    
    if 'time' in df.columns and 'B_z' in df.columns:
        t = df['time'].values
        B_z = df['B_z'].values
    elif 't' in df.columns and 'delta_B' in df.columns:
        t = df['t'].values
        B_z = df['delta_B'].values
    else:
        # Assume first column is time, second is B_z
        t = df.iloc[:, 0].values
        B_z = df.iloc[:, 1].values
    
    N = len(B_z)
    
    # 1. Variance check: σ² = B_rms²
    var_empirical = np.var(B_z)
    var_theoretical = B_rms**2
    var_ratio = var_empirical / var_theoretical
    
    # 2. Autocorrelation check
    # Compute autocorrelation function
    max_lag = min(1000, N // 10)  # Limit lag to avoid edge effects
    lags = np.arange(0, max_lag)
    autocorr = np.zeros(max_lag)
    
    mean_B = np.mean(B_z)
    B_centered = B_z - mean_B
    
    for lag in lags:
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            autocorr[lag] = np.corrcoef(B_centered[:-lag], B_centered[lag:])[0, 1]
    
    # Fit exponential decay to autocorrelation
    # Skip lag=0 (always 1.0)
    # Use up to 3×τc for fitting to improve accuracy (but at least 10 points)
    # This ensures we capture the exponential decay properly
    fit_range_theoretical = int(3 * tau_c / dt) if dt > 0 else max_lag // 3
    fit_range = min(fit_range_theoretical, max_lag // 2, 500)  # Limit to 500 for speed
    lags_fit = lags[1:max(10, fit_range)]
    autocorr_fit = autocorr[1:max(10, fit_range)]
    
    try:
        # Fit: ρ(Δt) = A * exp(-Δt/τc)
        popt, pcov = curve_fit(
            exponential_decay,
            lags_fit * dt,  # Convert lag index to time
            autocorr_fit,
            p0=[tau_c, 1.0],
            bounds=([tau_c * 0.1, 0.5], [tau_c * 10, 1.5])
        )
        tau_c_fitted = popt[0]
        A_fitted = popt[1]
        tau_c_error = np.sqrt(pcov[0, 0])
        
        # Compute R²
        autocorr_pred = exponential_decay(lags_fit * dt, tau_c_fitted, A_fitted)
        ss_res = np.sum((autocorr_fit - autocorr_pred)**2)
        ss_tot = np.sum((autocorr_fit - np.mean(autocorr_fit))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Autocorrelation fit failed: {e}")
        tau_c_fitted = np.nan
        A_fitted = np.nan
        tau_c_error = np.nan
        r2 = np.nan
    
    results = {
        'var_empirical': var_empirical,
        'var_theoretical': var_theoretical,
        'var_ratio': var_ratio,
        'std_empirical': np.sqrt(var_empirical),
        'std_theoretical': B_rms,
        'tau_c_expected': tau_c,
        'tau_c_fitted': tau_c_fitted,
        'tau_c_error': tau_c_error,
        'A_fitted': A_fitted,
        'r2_autocorr': r2,
        'N_points': N,
        'dt': dt,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"OU Noise Validation: {Path(trajectory_file).name}")
        print(f"{'='*70}")
        
        print(f"\n1️⃣ Variance Check:")
        print(f"   Empirical: σ² = {var_empirical:.6e} T²")
        print(f"   Theoretical: σ² = {var_theoretical:.6e} T²")
        print(f"   Ratio: {var_ratio:.4f}")
        if abs(var_ratio - 1.0) < 0.1:
            print(f"   ✅ Pass (within 10%)")
        elif abs(var_ratio - 1.0) < 0.2:
            print(f"   ⚠️  Warning (10-20% deviation)")
        else:
            print(f"   ❌ Fail (>20% deviation)")
        
        print(f"\n2️⃣ Autocorrelation Check:")
        print(f"   Expected: ρ(Δt) = exp(-Δt/τc) with τc = {tau_c:.2e} s")
        if not np.isnan(tau_c_fitted):
            print(f"   Fitted: τc = {tau_c_fitted:.2e} ± {tau_c_error:.2e} s")
            print(f"   A = {A_fitted:.4f}")
            print(f"   R² = {r2:.4f}")
            tau_ratio = tau_c_fitted / tau_c
            if abs(tau_ratio - 1.0) < 0.2:
                print(f"   ✅ Pass (τc within 20%)")
            elif abs(tau_ratio - 1.0) < 0.5:
                print(f"   ⚠️  Warning (τc 20-50% deviation)")
            else:
                print(f"   ❌ Fail (τc >50% deviation)")
        else:
            print(f"   ❌ Fit failed")
    
    return results

def main():
    """Validate noise trajectories from generate_noise_examples.py"""
    print("="*70)
    print("OU Noise Validation")
    print("="*70)
    
    # Parameters
    gamma_e = 1.76e11  # rad/(s·T)
    B_rms = 0.05e-3    # T (50 μT)
    
    # Fast noise (MN regime)
    tau_c_fast = 1e-8  # 10 ns
    dt_fast = tau_c_fast / 100
    
    # Slow noise (QS regime) - FIXED: match generate_noise_examples.py
    tau_c_slow = 1e-5  # 10 μs (was incorrectly 1e-4 = 100 μs)
    dt_slow = tau_c_slow / 100
    
    results = {}
    
    # Validate fast noise
    fast_file = Path("results_comparison/noise_trajectory_fast.csv")
    if fast_file.exists():
        print(f"\n📊 Validating fast noise (τc = {tau_c_fast*1e9:.1f} ns)...")
        results['fast'] = validate_ou_noise(
            fast_file, tau_c_fast, B_rms, dt_fast, verbose=True
        )
    else:
        print(f"\n❌ {fast_file} not found!")
    
    # Validate slow noise
    slow_file = Path("results_comparison/noise_trajectory_slow.csv")
    if slow_file.exists():
        print(f"\n📊 Validating slow noise (τc = {tau_c_slow*1e6:.1f} μs)...")
        results['slow'] = validate_ou_noise(
            slow_file, tau_c_slow, B_rms, dt_slow, verbose=True
        )
    else:
        print(f"\n❌ {slow_file} not found!")
    
    # Save validation results
    if results:
        output_file = Path("results_comparison/ou_noise_validation.txt")
        with open(output_file, 'w') as f:
            f.write("OU Noise Validation Results\n")
            f.write("="*70 + "\n\n")
            
            for name, res in results.items():
                f.write(f"{name.upper()} Noise:\n")
                f.write(f"  Variance ratio: {res['var_ratio']:.4f}\n")
                f.write(f"  τc fitted: {res['tau_c_fitted']:.2e} ± {res['tau_c_error']:.2e} s\n")
                f.write(f"  τc expected: {res['tau_c_expected']:.2e} s\n")
                f.write(f"  R² (autocorr): {res['r2_autocorr']:.4f}\n")
                f.write("\n")
        
        print(f"\n✅ Validation results saved to: {output_file}")
    
    return results

if __name__ == '__main__':
    results = main()

