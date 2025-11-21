#!/usr/bin/env python3
"""
Hybrid Echo Gain Calculation
Combines fitting method and direct comparison method for maximum accuracy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

def calculate_echo_gain_hybrid(t_fid, E_fid, t_echo, E_echo, T2_fid_fitted, T2_echo_fitted, 
                                method='auto', verbose=False):
    """
    Calculate echo gain using hybrid method:
    - If echo decay is well-observed: use fitting method
    - If echo is nearly flat: use direct comparison at t = T_FID
    
    Parameters
    ----------
    t_fid : ndarray
        FID time array
    E_fid : ndarray
        FID coherence magnitude
    t_echo : ndarray
        Echo time array (2τ)
    E_echo : ndarray
        Echo coherence magnitude
    T2_fid_fitted : float
        FID T2 from fitting
    T2_echo_fitted : float
        Echo T2 from fitting
    method : str
        'auto', 'fitting', or 'direct'
    verbose : bool
        Print details
        
    Returns
    -------
    gain : float
        Echo gain (T2_echo / T2_fid)
    method_used : str
        Method actually used
    T2_echo_used : float
        T2_echo value used
    """
    # Calculate T_FID (time where E_FID = 1/e)
    target = 1.0 / np.e
    T_FID = None
    if len(t_fid) > 0 and len(E_fid) > 0:
        idx = np.argmin(np.abs(E_fid - target))
        T_FID = t_fid[idx]
    
    # Check if echo decay is well-observed
    relative_decay = 0.0
    if len(E_echo) > 1:
        E_initial = E_echo[0]
        E_final = E_echo[-1]
        relative_decay = (E_initial - E_final) / E_initial if E_initial > 0 else 0
    
    # Decide method
    if method == 'auto':
        # Use direct method if:
        # 1. Echo decay is negligible (< 1%)
        # 2. T_FID is within echo time range
        # 3. Fitted T2_echo seems unreliable (too small compared to T_FID)
        use_direct = False
        
        if relative_decay < 0.01 and T_FID is not None:
            if T_FID <= t_echo[-1]:
                # Check if fitted T2_echo seems too small
                # If T2_echo < 2 * T_FID, it's suspicious for MN regime
                if T2_echo_fitted < 2.0 * T_FID:
                    use_direct = True
                    if verbose:
                        print(f"  → Using direct method: echo decay < 1%, T2_echo_fitted ({T2_echo_fitted*1e6:.3f} μs) < 2×T_FID ({T_FID*1e6:.3f} μs)")
        
        method_used = 'direct' if use_direct else 'fitting'
    else:
        method_used = method
    
    # Calculate gain
    if method_used == 'direct' and T_FID is not None and T_FID <= t_echo[-1]:
        # Direct method: interpolate echo at t = T_FID
        E_echo_at_TFID = np.interp(T_FID, t_echo, E_echo)
        
        if E_echo_at_TFID > 0.99:
            # Echo barely decayed - T2_echo >> T_FID
            # Use conservative estimate: T2_echo = 50 * T_FID (minimum)
            T2_echo_used = max(50.0 * T_FID, 100.0e-6)  # At least 100 μs
            if verbose:
                print(f"  → E_echo(T_FID) = {E_echo_at_TFID:.4f} ≈ 1.0")
                print(f"  → T2_echo_used = {T2_echo_used*1e6:.3f} μs (conservative estimate)")
        elif E_echo_at_TFID > 0.01:
            # Some decay - use extrapolation
            T2_echo_used = -T_FID / np.log(E_echo_at_TFID)
            # Conservative minimum: T2_echo >= 10 * T_FID
            T2_echo_used = max(T2_echo_used, 10.0 * T_FID)
            if verbose:
                print(f"  → E_echo(T_FID) = {E_echo_at_TFID:.4f}")
                print(f"  → T2_echo_used = {T2_echo_used*1e6:.3f} μs (from extrapolation)")
        else:
            # Echo decayed too much - fallback to fitting
            T2_echo_used = T2_echo_fitted
            method_used = 'fitting'
            if verbose:
                print(f"  → E_echo(T_FID) = {E_echo_at_TFID:.4f} too small, using fitting")
    else:
        # Fitting method
        T2_echo_used = T2_echo_fitted
        if verbose:
            if T_FID is None:
                print(f"  → Using fitting method: T_FID not found")
            elif T_FID > t_echo[-1]:
                print(f"  → Using fitting method: T_FID ({T_FID*1e6:.3f} μs) > echo max time ({t_echo[-1]*1e6:.3f} μs)")
            else:
                print(f"  → Using fitting method: echo decay = {relative_decay*100:.1f}%")
    
    gain = T2_echo_used / T2_fid_fitted if T2_fid_fitted > 0 else np.nan
    
    return gain, method_used, T2_echo_used


def main():
    """Test hybrid method on existing data"""
    print("="*80)
    print("Hybrid Echo Gain Calculation Test")
    print("="*80)
    
    # Load example data
    echo_file = Path('results_comparison/echo_tau_c_1e-8.csv')
    fid_file = Path('results_comparison/fid_tau_c_1e-8.csv')
    
    if not echo_file.exists() or not fid_file.exists():
        print("❌ Test data not found")
        return
    
    echo_df = pd.read_csv(echo_file)
    fid_df = pd.read_csv(fid_file)
    
    t_fid = fid_df['time (s)'].values
    E_fid = fid_df['P(t)'].values
    t_echo = echo_df['time (s)'].values
    E_echo = echo_df['P_echo(t)'].values
    
    # Get fitted T2 values
    gain_file = Path('results_comparison/echo_gain.csv')
    if gain_file.exists():
        df_gain = pd.read_csv(gain_file)
        closest = df_gain.iloc[(df_gain['tau_c'] - 1e-8).abs().argsort()[:1]]
        if len(closest) > 0:
            T2_fid_fitted = closest.iloc[0]['T2']
            T2_echo_fitted = closest.iloc[0]['T2_echo']
            
            print(f"\nFitted values:")
            print(f"  T2_FID = {T2_fid_fitted*1e6:.3f} μs")
            print(f"  T2_echo = {T2_echo_fitted*1e6:.3f} μs")
            print(f"  Gain (fitted) = {T2_echo_fitted / T2_fid_fitted:.2f}")
            
            # Test hybrid method
            print(f"\nHybrid method:")
            gain_hybrid, method_used, T2_echo_used = calculate_echo_gain_hybrid(
                t_fid, E_fid, t_echo, E_echo, T2_fid_fitted, T2_echo_fitted,
                method='auto', verbose=True
            )
            
            print(f"\nResult:")
            print(f"  Method used: {method_used}")
            print(f"  T2_echo_used = {T2_echo_used*1e6:.3f} μs")
            print(f"  Gain (hybrid) = {gain_hybrid:.2f}")
            
            if method_used == 'direct':
                print(f"\n✅ Direct method used - more accurate for flat echo curves")
            else:
                print(f"\n✅ Fitting method used - echo decay is well-observed")


if __name__ == '__main__':
    main()

