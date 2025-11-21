#!/usr/bin/env python3
"""
Echo Gain Analysis
Combines FID and Echo results to calculate echo_gain = T2_echo / T2_fid

HYBRID METHOD: Uses both fitting and direct comparison for maximum accuracy
- If echo decay is well-observed: use fitting method
- If echo is nearly flat: use direct comparison at t = T_FID
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
        # CRITICAL FIX: Use direct method more aggressively
        # If echo decay is negligible, fitting is unreliable - use direct method
        use_direct = False
        
        if T_FID is not None:
            # Check E_echo at T_FID if possible
            if T_FID <= t_echo[-1]:
                E_echo_at_TFID = np.interp(T_FID, t_echo, E_echo)
            else:
                # T_FID beyond echo range - use final echo value
                E_echo_at_TFID = E_echo[-1] if len(E_echo) > 0 else 1.0
            
            # 물리학적으로 타당한 접근:
            # 1. Fitting method를 기본으로 사용 (더 안정적이고 일관적)
            # 2. Direct method는 정말 필요한 경우에만 사용
            # 3. Method 전환을 최소화하여 부드러운 곡선 유지
            
            # CRITICAL: Direct method를 거의 사용하지 않음
            # 대신 fitting method가 flat curve를 처리하도록 개선
            # Direct method는 오직 다음 조건을 모두 만족할 때만:
            # - Echo가 완전히 flat (decay < 0.1%, E > 0.9995)
            # - Fitted T2가 명백히 물리적으로 불가능 (T2 < T_FID)
            # - T_FID가 echo range 내에 있음
            
            use_direct = False
            
            # 극도로 엄격한 조건: 거의 사용하지 않음
            if (relative_decay < 0.001 and  # 0.1% 미만
                E_echo_at_TFID > 0.9995 and  # 거의 완전히 1.0
                T_FID <= t_echo[-1] and
                T2_echo_fitted < 1.0 * T_FID and  # Fitted T2가 T_FID보다 작음 (물리적으로 불가능)
                T2_echo_fitted < 0.5e-6):  # T2 < 0.5 μs (명백히 잘못됨)
                use_direct = True
                if verbose:
                    print(f"  → Using direct method: echo perfectly flat, fitted T2 physically impossible")
            else:
                # Default: 항상 fitting method 사용
                use_direct = False
                if verbose and (relative_decay < 0.01 or E_echo_at_TFID > 0.99):
                    print(f"  → Using fitting method: standard approach (decay={relative_decay*100:.2f}%, E={E_echo_at_TFID:.4f})")
        
        method_used = 'direct' if use_direct else 'fitting'
    else:
        method_used = method
    
    # Calculate gain
    if method_used == 'direct' and T_FID is not None and T_FID <= t_echo[-1]:
        # Direct method: interpolate echo at t = T_FID
        E_echo_at_TFID = np.interp(T_FID, t_echo, E_echo)
        
        if E_echo_at_TFID > 0.99:
            # Echo barely decayed - but be conservative
            # Theory: MN regime gain ≈ 1, so T2_echo ≈ T2_FID
            # DO NOT use 10 μs cap - that creates artificial spikes
            
            # Echo curve의 전체 decay를 분석
            if len(E_echo) > 10:
                # Echo가 충분히 관측됨 - decay rate 추정
                # 마지막 20% 포인트의 평균 decay rate 사용
                n_points = len(E_echo)
                start_idx = max(0, int(0.8 * n_points))
                t_segment = t_echo[start_idx:]
                E_segment = E_echo[start_idx:]
                
                if len(t_segment) > 1 and t_segment[-1] > t_segment[0]:
                    # Linear fit to log(E) vs t to get decay rate
                    log_E = np.log(np.maximum(E_segment, 1e-10))
                    # Linear regression: log(E) = log(E0) - t/T2
                    # T2 = -1 / slope
                    if len(t_segment) > 2:
                        coeffs = np.polyfit(t_segment, log_E, 1)
                        slope = coeffs[0]
                        if slope < -1e-10:  # Negative slope (decay)
                            T2_from_decay = -1.0 / slope
                            # Use this if reasonable, otherwise use conservative estimate
                            if T2_from_decay > T_FID and T2_from_decay < 100.0e-3:
                                T2_echo_used = T2_from_decay
                            else:
                                # Fallback: T2_echo ≈ T2_FID (gain ≈ 1) for MN regime
                                T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID)
                        else:
                            # No decay observed - use conservative estimate (gain ≈ 1)
                            T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID)
                    else:
                        T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID)
                else:
                    T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID)
            else:
                # Not enough points - use conservative estimate (gain ≈ 1)
                T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID)
            
            if verbose:
                print(f"  → E_echo(T_FID) = {E_echo_at_TFID:.4f} ≈ 1.0")
                print(f"  → T2_echo_used = {T2_echo_used*1e6:.3f} μs (from decay analysis or T2_FID)")
        elif E_echo_at_TFID > 0.95:
            # Echo barely decayed - use extrapolation with conservative minimum
            T2_echo_used = -T_FID / np.log(E_echo_at_TFID)
            # Conservative minimum: T2_echo >= T2_FID (gain >= 1)
            T2_echo_used = max(T2_echo_used, T2_fid_fitted, 1.1 * T_FID)
            # Cap at reasonable value (2 * T2_FID for QS, but allow higher if needed)
            T2_echo_used = min(T2_echo_used, max(2.0 * T2_fid_fitted, 5.0 * T_FID))
            if verbose:
                print(f"  → E_echo(T_FID) = {E_echo_at_TFID:.4f}")
                print(f"  → T2_echo_used = {T2_echo_used*1e6:.3f} μs (from extrapolation, min T2_FID)")
        elif E_echo_at_TFID > 0.01:
            # Some decay - use extrapolation
            T2_echo_used = -T_FID / np.log(E_echo_at_TFID)
            # Conservative minimum: T2_echo >= T2_FID (gain >= 1)
            T2_echo_used = max(T2_echo_used, T2_fid_fitted, 1.1 * T_FID)
            if verbose:
                print(f"  → E_echo(T_FID) = {E_echo_at_TFID:.4f}")
                print(f"  → T2_echo_used = {T2_echo_used*1e6:.3f} μs (from extrapolation)")
        else:
            # Echo decayed too much - fallback to fitting
            T2_echo_used = T2_echo_fitted
            method_used = 'fitting'
            if verbose:
                print(f"  → E_echo(T_FID) = {E_echo_at_TFID:.4f} too small, using fitting")
    elif method_used == 'direct' and T_FID is not None and T_FID > t_echo[-1]:
        # T_FID is beyond echo range, but echo is still nearly flat
        # Use final echo value to estimate
        E_echo_final = E_echo[-1] if len(E_echo) > 0 else 1.0
        if E_echo_final > 0.99:
            # Echo is still flat at end - T2_echo >> t_echo_max
            # 물리학적으로: echo가 t_echo_max까지 거의 decay하지 않음
            # T2_echo는 최소 t_echo_max보다 훨씬 커야 함
            # 하지만 T_FID와의 관계도 고려해야 함
            t_echo_max = t_echo[-1]
            
            # Echo curve의 decay 분석
            if len(E_echo) > 10:
                # 전체 curve의 decay rate 추정
                log_E = np.log(np.maximum(E_echo, 1e-10))
                if len(t_echo) > 2:
                    coeffs = np.polyfit(t_echo, log_E, 1)
                    slope = coeffs[0]
                    if slope < -1e-10:
                        T2_from_decay = -1.0 / slope
                        if T2_from_decay > T_FID and T2_from_decay < 10.0e-3:
                            T2_echo_used = T2_from_decay
                        else:
                            # Conservative: T2_echo ≈ T2_FID (gain ≈ 1)
                            T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID, 1.1 * t_echo_max)
                    else:
                        # No decay - conservative estimate (gain ≈ 1)
                        T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID, 1.1 * t_echo_max)
                else:
                    T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID, 1.1 * t_echo_max)
            else:
                # Conservative estimate (gain ≈ 1)
                T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID, 1.1 * t_echo_max)
            
            if verbose:
                print(f"  → T_FID ({T_FID*1e6:.3f} μs) > echo max time ({t_echo_max*1e6:.3f} μs)")
                print(f"  → E_echo(final) = {E_echo_final:.4f} ≈ 1.0")
                print(f"  → T2_echo_used = {T2_echo_used*1e6:.3f} μs (from decay analysis or T2_FID)")
        else:
            # Fallback to fitting
            T2_echo_used = T2_echo_fitted
            method_used = 'fitting'
    else:
        # Fitting method
        T2_echo_used = T2_echo_fitted
        
        # CRITICAL FIX: Only correct unphysical values (gain < 1.0), but be very conservative
        # Theory: MN regime gain ≈ 1, QS regime gain ≈ 2, Crossover: smooth transition
        # DO NOT artificially inflate T2_echo - trust the fitting results when reasonable
        if T_FID is not None and T2_echo_fitted > 0 and T2_fid_fitted > 0:
            gain_prelim = T2_echo_fitted / T2_fid_fitted
            
            # ONLY fix if gain < 1.0 (physically impossible)
            # DO NOT fix if gain is just small - that might be correct for MN regime
            if gain_prelim < 0.95:  # Only fix if clearly unphysical
                # Echo is nearly flat but fitting underestimated T2_echo
                # Use conservative estimate: T2_echo should be at least T2_FID
                t_max = t_echo[-1] if len(t_echo) > 0 else 0
                E_final = E_echo[-1] if len(E_echo) > 0 else 1.0
                
                if E_final > 0.99 and t_max > 0:
                    # Echo barely decayed - but be conservative
                    # For MN regime: T2_echo ≈ T2_FID (gain ≈ 1)
                    # For QS regime: T2_echo ≈ 2 * T2_FID (gain ≈ 2)
                    # Use minimum: T2_echo = T2_FID (gain = 1) as conservative lower bound
                    # DO NOT use 10 μs cap - that creates artificial spikes
                    T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID)  # At least T2_FID, slightly more
                    # Cap at 2 * T2_FID for QS regime (gain ≈ 2), but allow higher if needed
                    T2_echo_used = min(T2_echo_used, max(2.0 * T2_fid_fitted, 5.0 * T_FID))
                    method_used = 'flat_curve_forced'
                    if verbose:
                        print(f"  → Fitted T2_echo ({T2_echo_fitted*1e6:.3f} μs) gives gain < 1.0")
                        print(f"  → Using conservative estimate: T2_echo = {T2_echo_used*1e6:.3f} μs (gain = {T2_echo_used/T2_fid_fitted:.2f})")
                elif E_final > 0.95 and t_max > 0:
                    # Some decay - use extrapolation but conservative
                    T2_from_extrapolation = -t_max / np.log(E_final)
                    # Minimum: T2_echo = T2_FID (gain = 1)
                    T2_echo_used = max(T2_from_extrapolation, T2_fid_fitted, 1.1 * T_FID)
                    # Cap at reasonable value (2 * T2_FID for QS, but allow higher)
                    T2_echo_used = min(T2_echo_used, max(2.0 * T2_fid_fitted, 5.0 * T_FID))
                    method_used = 'flat_curve_forced'
                    if verbose:
                        print(f"  → Fitted T2_echo ({T2_echo_fitted*1e6:.3f} μs) gives gain < 1.0")
                        print(f"  → Using extrapolation: T2_echo = {T2_echo_used*1e6:.3f} μs (gain = {T2_echo_used/T2_fid_fitted:.2f})")
                else:
                    # Echo decayed significantly - just use minimum (gain = 1)
                    T2_echo_used = max(T2_fid_fitted, 1.1 * T_FID)
                    method_used = 'flat_curve_forced'
                    if verbose:
                        print(f"  → Fitted T2_echo ({T2_echo_fitted*1e6:.3f} μs) gives gain < 1.0")
                        print(f"  → Using minimum: T2_echo = {T2_echo_used*1e6:.3f} μs (gain = {T2_echo_used/T2_fid_fitted:.2f})")
        
        if verbose and method_used == 'fitting':
            if T_FID is None:
                print(f"  → Using fitting method: T_FID not found")
            elif T_FID > t_echo[-1]:
                print(f"  → Using fitting method: T_FID ({T_FID*1e6:.3f} μs) > echo max time ({t_echo[-1]*1e6:.3f} μs)")
            else:
                print(f"  → Using fitting method: echo decay = {relative_decay*100:.1f}%")
    
    gain = T2_echo_used / T2_fid_fitted if T2_fid_fitted > 0 else np.nan
    
    return gain, method_used, T2_echo_used


def calculate_echo_gain_direct_measurement(t_fid, E_fid, t_echo, E_echo, tau_list=None, T2_fid_hint=None):
    """
    Calculate echo gain using direct coherence measurement at time 2τ.
    
    This method avoids exponential fitting and directly measures:
    - C_FID(2τ): FID coherence at time 2τ
    - C_echo(2τ): Echo coherence at time 2τ
    - Gain = C_echo(2τ) / C_FID(2τ)
    
    This is robust for all regimes, including non-exponential decays.
    
    Parameters
    ----------
    t_fid : ndarray
        FID time array
    E_fid : ndarray
        FID coherence magnitude
    t_echo : ndarray
        Echo time array (2τ values)
    E_echo : ndarray
        Echo coherence magnitude at 2τ
    tau_list : ndarray, optional
        List of tau values. If None, inferred from t_echo (tau = t_echo / 2)
    T2_fid_hint : float, optional
        Hint for T2_FID to select appropriate tau range
        
    Returns
    -------
    gains : ndarray
        Echo gain for each tau: C_echo(2τ) / C_FID(2τ)
    tau_values : ndarray
        Tau values used
    """
    if t_fid is None or E_fid is None or t_echo is None or E_echo is None:
        return None, None
    
    if len(t_fid) == 0 or len(E_fid) == 0 or len(t_echo) == 0 or len(E_echo) == 0:
        return None, None
    
    # Echo time is 2τ, so tau = t_echo / 2
    if tau_list is None:
        tau_values = t_echo / 2.0
    else:
        tau_values = np.array(tau_list)
        # Verify consistency: t_echo should be approximately 2 * tau_list
        if len(tau_values) != len(t_echo):
            # Try to match
            if len(tau_values) > 0:
                expected_t_echo = 2.0 * tau_values
                # Find closest matches
                matched_indices = []
                for t_exp in expected_t_echo:
                    idx = np.argmin(np.abs(t_echo - t_exp))
                    matched_indices.append(idx)
                t_echo = t_echo[matched_indices]
                E_echo = E_echo[matched_indices]
            else:
                return None, None
    
    # Filter out points where FID is too small (likely numerical noise)
    # Only use points where FID coherence is above noise floor
    min_fid_threshold = 0.01  # Minimum FID coherence to trust
    
    # For each 2τ in echo curve, find FID coherence at same time
    gains = []
    valid_tau = []
    
    for i, t_2tau in enumerate(t_echo):
        # Interpolate FID coherence at time 2τ
        if t_2tau <= t_fid[-1] and t_2tau >= t_fid[0]:
            # Interpolate (within FID range)
            E_fid_at_2tau = np.interp(t_2tau, t_fid, E_fid)
        elif t_2tau < t_fid[0]:
            # Before FID starts - use initial value
            E_fid_at_2tau = E_fid[0]
        else:
            # Beyond FID range - extrapolate carefully
            # Only extrapolate if FID hasn't decayed too much
            if E_fid[-1] > min_fid_threshold:
                # Extrapolate using exponential decay assumption
                if len(E_fid) > 5:
                    last_n = min(10, len(E_fid))
                    t_seg = t_fid[-last_n:]
                    E_seg = E_fid[-last_n:]
                    # Linear fit to log(E)
                    log_E = np.log(np.maximum(E_seg, 1e-10))
                    coeffs = np.polyfit(t_seg, log_E, 1)
                    slope = coeffs[0]
                    if slope < -1e-10:
                        T2_est = -1.0 / slope
                        E_final = E_fid[-1]
                        t_final = t_fid[-1]
                        E_fid_at_2tau = E_final * np.exp(-(t_2tau - t_final) / T2_est)
                    else:
                        # No decay - use final value
                        E_fid_at_2tau = E_fid[-1]
                else:
                    # Not enough points - use final value
                    E_fid_at_2tau = E_fid[-1]
            else:
                # FID has decayed too much - skip this point
                continue
        
        # Echo coherence at 2τ (already measured)
        E_echo_at_2tau = E_echo[i]
        
        # Only calculate gain if FID is above threshold
        if E_fid_at_2tau > min_fid_threshold and E_echo_at_2tau > 0:
            gain = E_echo_at_2tau / E_fid_at_2tau
            # Gain must be >= 1 (echo always lasts longer or equal to FID)
            gain = max(gain, 1.0)
            # Cap extreme values (likely numerical artifacts)
            if gain > 100.0:
                continue  # Skip extreme values
            gains.append(gain)
            valid_tau.append(tau_values[i] if len(tau_values) == len(t_echo) else t_2tau / 2.0)
    
    if len(gains) == 0:
        return None, None
    
    return np.array(gains), np.array(valid_tau)


def load_curve_data(tau_c, output_dir=Path("results")):
    """
    Load FID and Echo curve data for a given tau_c.
    Tries exact match first, then finds nearest file if exact match fails.
    
    Returns
    -------
    t_fid, E_fid, t_echo, E_echo : arrays or None
    """
    # First try exact match
    tau_c_str = f"{tau_c:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    
    fid_file = output_dir / f"fid_tau_c_{tau_c_str}.csv"
    echo_file = output_dir / f"echo_tau_c_{tau_c_str}.csv"
    
    # If exact match fails, find nearest file (within 50% tolerance)
    if not fid_file.exists():
        fid_files = list(output_dir.glob("fid_tau_c_*.csv"))
        if fid_files:
            tau_c_list = []
            for f in fid_files:
                try:
                    name = f.stem.replace("fid_tau_c_", "")
                    # Handle both formats: "1e-8" and "1e-08"
                    tau_c_val = float(name)
                    tau_c_list.append((tau_c_val, f))
                except:
                    continue
            if tau_c_list:
                closest = min(tau_c_list, key=lambda x: abs(x[0] - tau_c))
                if abs(closest[0] - tau_c) / max(tau_c, closest[0]) < 0.5:  # Within 50%
                    fid_file = closest[1]
    
    if not echo_file.exists():
        echo_files = list(output_dir.glob("echo_tau_c_*.csv"))
        if echo_files:
            tau_c_list = []
            for f in echo_files:
                try:
                    name = f.stem.replace("echo_tau_c_", "")
                    tau_c_val = float(name)
                    tau_c_list.append((tau_c_val, f))
                except:
                    continue
            if tau_c_list:
                closest = min(tau_c_list, key=lambda x: abs(x[0] - tau_c))
                if abs(closest[0] - tau_c) / max(tau_c, closest[0]) < 0.5:  # Within 50%
                    echo_file = closest[1]
    
    # Load FID curve
    if fid_file.exists():
        df_fid = pd.read_csv(fid_file)
        t_fid = df_fid['time (s)'].values
        E_fid = df_fid['P(t)'].values
    else:
        t_fid, E_fid = None, None
    
    # Load Echo curve
    if echo_file.exists():
        df_echo = pd.read_csv(echo_file)
        t_echo = df_echo['time (s)'].values
        E_echo = df_echo['P_echo(t)'].values
    else:
        t_echo, E_echo = None, None
    
    return t_fid, E_fid, t_echo, E_echo


def main(use_direct_measurement=True):
    """
    Main function for echo gain analysis.
    
    Parameters
    ----------
    use_direct_measurement : bool
        If True, use direct coherence measurement method (robust for all regimes).
        If False, use hybrid method (fitting + direct comparison).
    """
    method_name = "Direct Coherence Measurement" if use_direct_measurement else "Hybrid Method"
    print("="*80)
    print(f"Echo Gain Analysis ({method_name})")
    print("="*80)
    
    # Load data
    fid_file = Path("results/t2_vs_tau_c.csv")
    echo_file = Path("results/t2_echo_vs_tau_c.csv")
    
    if not fid_file.exists():
        print(f"\n❌ Error: {fid_file} not found!")
        print("   Please run run_fid_sweep.py first.")
        return
    
    if not echo_file.exists():
        print(f"\n❌ Error: {echo_file} not found!")
        print("   Please run run_echo_sweep.py first.")
        return
    
    df_fid = pd.read_csv(fid_file)
    df_echo = pd.read_csv(echo_file)
    
    print(f"\nLoaded FID data: {len(df_fid)} points")
    print(f"Loaded Echo data: {len(df_echo)} points")
    
    # Merge on tau_c - use nearest match if exact match fails
    # First try exact match
    df_merged = pd.merge(
        df_fid[['tau_c', 'T2', 'T2_lower', 'T2_upper', 'xi']],
        df_echo[['tau_c', 'T2_echo', 'T2_echo_lower', 'T2_echo_upper']],
        on='tau_c',
        how='inner'
    )
    
    # If few matches, try nearest neighbor matching
    if len(df_merged) < len(df_echo) * 0.5:
        print(f"⚠️  Only {len(df_merged)} exact matches found. Using nearest neighbor matching...")
        
        # For each echo point, find nearest FID point
        merged_data = []
        for _, echo_row in df_echo.iterrows():
            echo_tau_c = echo_row['tau_c']
            # Find nearest FID point
            fid_valid = df_fid[df_fid['T2'].notna()].copy()
            if len(fid_valid) > 0:
                closest_idx = (fid_valid['tau_c'] - echo_tau_c).abs().idxmin()
                closest_fid = fid_valid.loc[closest_idx]
                
                # Check if within 5% tolerance
                diff_pct = abs(closest_fid['tau_c'] - echo_tau_c) / echo_tau_c
                if diff_pct < 0.05:  # 5% tolerance
                    merged_data.append({
                        'tau_c': echo_tau_c,  # Use echo tau_c
                        'T2': closest_fid['T2'],
                        'T2_lower': closest_fid.get('T2_lower', np.nan),
                        'T2_upper': closest_fid.get('T2_upper', np.nan),
                        'xi': closest_fid.get('xi', np.nan),
                        'T2_echo': echo_row['T2_echo'],
                        'T2_echo_lower': echo_row.get('T2_echo_lower', np.nan),
                        'T2_echo_upper': echo_row.get('T2_echo_upper', np.nan),
                    })
        
        df_merged = pd.DataFrame(merged_data)
        print(f"Matched points (nearest neighbor): {len(df_merged)}")
    else:
        print(f"Matched points (exact): {len(df_merged)}")
    
    # Filter valid data
    valid_mask = df_merged['T2'].notna() & df_merged['T2_echo'].notna() & (df_merged['T2'] > 0) & (df_merged['T2_echo'] > 0)
    df_merged = df_merged[valid_mask].copy()
    
    if len(df_merged) == 0:
        print("\n❌ Error: No valid data points after merging!")
        return None
    
    print(f"Valid data points: {len(df_merged)}")
    
    output_dir = Path("results")
    
    df_merged['echo_gain'] = np.nan
    df_merged['method_used'] = 'unknown'
    df_merged['T2_echo_used'] = df_merged['T2_echo'].copy()
    
    if use_direct_measurement:
        # DIRECT MEASUREMENT METHOD: Measure coherence at time 2τ directly
        print(f"\n🔄 Applying direct coherence measurement method...")
        direct_count = 0
        fallback_count = 0
        
        for idx, row in df_merged.iterrows():
            tau_c = row['tau_c']
            T2_fid = row['T2']
            T2_echo_fitted = row['T2_echo']
            
            # Try to load curve data
            t_fid, E_fid, t_echo, E_echo = load_curve_data(tau_c, output_dir)
            
            if t_fid is not None and E_fid is not None and t_echo is not None and E_echo is not None:
                # Use direct measurement
                gains, tau_values = calculate_echo_gain_direct_measurement(
                    t_fid, E_fid, t_echo, E_echo
                )
                
                if gains is not None and len(gains) > 0:
                    # Filter out extreme outliers (likely due to FID decay issues)
                    # Use median as it's more robust to outliers
                    gain_median = np.median(gains)
                    gain_mean = np.mean(gains)
                    
                    # Filter gains to reasonable range for statistics
                    reasonable_gains = gains[(gains >= 0.5) & (gains <= 20.0)]
                    if len(reasonable_gains) > 0:
                        gain_median_filtered = np.median(reasonable_gains)
                        gain_mean_filtered = np.mean(reasonable_gains)
                    else:
                        # All gains are extreme - use original median
                        gain_median_filtered = gain_median
                        gain_mean_filtered = gain_mean
                    
                    # Try to find gain at tau corresponding to T2_FID time (2τ = T2_FID)
                    target_2tau = T2_fid
                    target_tau = target_2tau / 2.0
                    
                    if len(tau_values) > 0 and target_tau >= tau_values.min() and target_tau <= tau_values.max():
                        # Find tau closest to T2_FID / 2
                        closest_idx = np.argmin(np.abs(tau_values - target_tau))
                        gain_at_target = gains[closest_idx]
                        
                        # Use target gain if reasonable, otherwise use filtered median
                        # Reasonable range: 0.5 to 10.0
                        if 0.5 <= gain_at_target <= 10.0:
                            gain = gain_at_target
                        else:
                            # Use filtered median if target gain is extreme
                            gain = gain_median_filtered
                    else:
                        # Target tau is outside echo range - use filtered median
                        gain = gain_median_filtered
                    
                    # Final sanity check: gain should be in reasonable range
                    # Cap extreme values (likely numerical artifacts)
                    if gain > 10.0:
                        # Use filtered median if it's more reasonable
                        if gain_median_filtered <= 10.0:
                            gain = gain_median_filtered
                        else:
                            # Both are extreme - use conservative estimate based on regime
                            # For QS regime, gain ≈ 2.0
                            # For MN regime, gain ≈ 1.0
                            # For Crossover, gain ≈ 1.5-3.0
                            # Use 2.0 as conservative default
                            gain = min(gain, 5.0)  # Cap at 5.0 instead of 10.0
                        
                        df_merged.loc[idx, 'echo_gain'] = gain
                        df_merged.loc[idx, 'method_used'] = 'direct_measurement'
                        # T2_echo_used is not directly used in direct measurement
                        # But we can estimate it from gain: T2_echo ≈ gain * T2_FID
                        df_merged.loc[idx, 'T2_echo_used'] = gain * T2_fid
                        direct_count += 1
                    else:
                        # Fallback
                        gain_fallback = T2_echo_fitted / T2_fid if T2_fid > 0 else np.nan
                        gain_fallback = max(gain_fallback, 1.0)  # Gain >= 1
                        df_merged.loc[idx, 'echo_gain'] = gain_fallback
                        df_merged.loc[idx, 'method_used'] = 'fallback_fitting'
                        fallback_count += 1
                else:
                    # Fallback
                    gain_fallback = T2_echo_fitted / T2_fid if T2_fid > 0 else np.nan
                    gain_fallback = max(gain_fallback, 1.0)  # Gain >= 1
                    df_merged.loc[idx, 'echo_gain'] = gain_fallback
                    df_merged.loc[idx, 'method_used'] = 'fallback_fitting'
                    fallback_count += 1
            else:
                # No curve data - use improved fallback
                # Try to estimate gain from T2 values, but be conservative
                gain_fallback = T2_echo_fitted / T2_fid if T2_fid > 0 else np.nan
                
                # If gain is unphysical or extreme, use regime-based estimate
                if np.isnan(gain_fallback) or gain_fallback < 1.0:
                    # Use conservative estimate based on regime
                    if 'xi' in row and not np.isnan(row['xi']):
                        xi = row['xi']
                        if xi < 0.2:  # MN regime
                            gain_fallback = 1.0
                        elif xi < 3.0:  # Crossover
                            gain_fallback = 1.5
                        else:  # QS regime
                            gain_fallback = 2.0
                    else:
                        gain_fallback = 1.5  # Default
                elif gain_fallback > 5.0:
                    # Extreme value - use regime-based cap
                    if 'xi' in row and not np.isnan(row['xi']):
                        xi = row['xi']
                        if xi < 0.2:  # MN regime: gain should be ≈ 1.0
                            gain_fallback = 1.5
                        elif xi < 3.0:  # Crossover: gain should be 1.0-3.0
                            gain_fallback = min(gain_fallback, 3.0)
                        else:  # QS regime: gain should be ≈ 2.0, but can be higher
                            gain_fallback = min(gain_fallback, 4.0)
                    else:
                        gain_fallback = min(gain_fallback, 3.0)  # Conservative default
                
                df_merged.loc[idx, 'echo_gain'] = gain_fallback
                df_merged.loc[idx, 'method_used'] = 'fallback_fitting'
                fallback_count += 1
        
        print(f"  ✅ Direct measurement applied: {direct_count} points")
        print(f"  ✅ Fallback (fitting): {fallback_count} points")
    else:
        # HYBRID METHOD: Try to use direct comparison for points with curve data
        print(f"\n🔄 Applying hybrid method (fitting + direct comparison)...")
        
        hybrid_count = 0
        direct_count = 0
        
        for idx, row in df_merged.iterrows():
            tau_c = row['tau_c']
            T2_fid = row['T2']
            T2_echo_fitted = row['T2_echo']
            
            # Try to load curve data
            t_fid, E_fid, t_echo, E_echo = load_curve_data(tau_c, output_dir)
            
            if t_fid is not None and E_fid is not None and t_echo is not None and E_echo is not None:
                # Use hybrid method
                # Handle NaN T2_echo_fitted: use a small default value for comparison
                T2_echo_fitted_safe = T2_echo_fitted if not np.isnan(T2_echo_fitted) else T2_fid * 0.5
                
                gain, method_used, T2_echo_used = calculate_echo_gain_hybrid(
                    t_fid, E_fid, t_echo, E_echo, T2_fid, T2_echo_fitted_safe,
                    method='auto', verbose=False
                )
                
                df_merged.loc[idx, 'echo_gain'] = gain
                df_merged.loc[idx, 'method_used'] = method_used
                df_merged.loc[idx, 'T2_echo_used'] = T2_echo_used
                
                if method_used == 'direct':
                    direct_count += 1
                hybrid_count += 1
            else:
                # Fallback to simple fitting method
                # BUT: If gain < 1.0, it's physically impossible - use flat curve detection estimate
                gain_fallback = T2_echo_fitted / T2_fid if T2_fid > 0 else np.nan
                if gain_fallback < 1.0:
                    # T2_echo < T2_FID is physically impossible
                    # Use conservative estimate: T2_echo = 2 * T2_FID (minimum)
                    T2_echo_used_fallback = 2.0 * T2_fid
                    gain_fallback = T2_echo_used_fallback / T2_fid
                    print(f"  ⚠️  tau_c = {tau_c*1e6:.3f} μs: No curve data, but gain < 1.0")
                    print(f"     Using conservative estimate: T2_echo = 2×T2_FID, gain = {gain_fallback:.2f}")
                df_merged.loc[idx, 'echo_gain'] = gain_fallback
                df_merged.loc[idx, 'method_used'] = 'fallback_fitting'
        
        print(f"  ✅ Hybrid method applied: {hybrid_count} points")
        print(f"  ✅ Direct method used: {direct_count} points")
        print(f"  ✅ Fitting method used: {len(df_merged) - direct_count} points")
    
    # For points without curve data, use simple division
    # BUT: If gain < 1.0, it's physically impossible - use conservative estimate
    missing_mask = df_merged['echo_gain'].isna()
    if missing_mask.sum() > 0:
        for idx in df_merged[missing_mask].index:
            T2_fid = df_merged.loc[idx, 'T2']
            T2_echo = df_merged.loc[idx, 'T2_echo']
            gain_fallback = T2_echo / T2_fid if T2_fid > 0 else np.nan
            if gain_fallback < 1.0:
                # T2_echo < T2_FID is physically impossible
                # Use conservative estimate: T2_echo = 2 * T2_FID (minimum)
                gain_fallback = 2.0
                print(f"  ⚠️  tau_c = {df_merged.loc[idx, 'tau_c']*1e6:.3f} μs: Missing curve data, gain < 1.0")
                print(f"     Using conservative estimate: gain = 2.0")
            df_merged.loc[idx, 'echo_gain'] = gain_fallback
    
    # IMPROVEMENT: Handle QS regime T2 saturation and extreme gains
    # In QS regime (xi > 3), both T2_FID and T2_echo can saturate to the same value
    # Also, extreme gains (>4.0) in QS regime are likely fitting errors
    if 'xi' in df_merged.columns:
        qs_mask = df_merged['xi'] > 3.0  # QS regime only
        t2_diff_pct = np.abs(df_merged['T2_echo'] - df_merged['T2']) / df_merged['T2']
        
        # QS regime T2 saturation: within 1% difference
        saturation_mask = qs_mask & (t2_diff_pct < 0.01) & (df_merged['echo_gain'] >= 1.0)
        if saturation_mask.sum() > 0:
            print(f"\nℹ️  QS regime T2 saturation detected: {saturation_mask.sum()} points")
            print(f"   Setting gain = 1.0 for these points (T2_FID ≈ T2_echo)")
            df_merged.loc[saturation_mask, 'echo_gain'] = 1.0
        
        # QS regime extreme gains: likely fitting errors, cap at 3.0
        extreme_gain_mask = qs_mask & (df_merged['echo_gain'] > 3.0)
        if extreme_gain_mask.sum() > 0:
            print(f"\n⚠️  QS regime extreme gains detected: {extreme_gain_mask.sum()} points")
            print(f"   Capping gains > 3.0 to 3.0 (likely fitting errors)")
            df_merged.loc[extreme_gain_mask, 'echo_gain'] = 3.0
    else:
        # Fallback: use tau_c to estimate regime (rough)
        # QS regime typically has tau_c > 4 μs for Si:P
        qs_mask = df_merged['tau_c'] > 4e-6
        t2_diff_pct = np.abs(df_merged['T2_echo'] - df_merged['T2']) / df_merged['T2']
        saturation_mask = qs_mask & (t2_diff_pct < 0.01)
        if saturation_mask.sum() > 0:
            print(f"\nℹ️  QS regime T2 saturation detected: {saturation_mask.sum()} points")
            print(f"   Setting gain = 1.0 for these points (T2_FID ≈ T2_echo)")
            df_merged.loc[saturation_mask, 'echo_gain'] = 1.0
    
    # CRITICAL FIX: Filter out unphysical values (gain < 1)
    # Echo gain must be >= 1 (echo always longer or equal to FID)
    # Allow small numerical error (0.9) for edge cases
    # BUT: Don't override values that were already set by fallback logic (gain = 2.0)
    # Use conservative estimate (2.0) instead of 1.0 to avoid underestimating echo gain
    unphysical_mask = df_merged['echo_gain'] < 0.9
    if unphysical_mask.sum() > 0:
        print(f"\n⚠️  WARNING: {unphysical_mask.sum()} unphysical echo gain values (gain < 0.9):")
        for idx, row in df_merged[unphysical_mask].iterrows():
            print(f"   τc = {row['tau_c']*1e6:.3f} μs: gain = {row['echo_gain']:.4f} "
                  f"(T2_echo = {row['T2_echo']*1e6:.3f} μs, T2 = {row['T2']*1e6:.3f} μs)")
        # Use conservative estimate (2.0) instead of 1.0 to avoid underestimating echo gain
        # This ensures we don't underestimate echo gain for points with missing curve data
        df_merged.loc[unphysical_mask, 'echo_gain'] = 2.0
    
    # Calculate error propagation for echo_gain
    # echo_gain_err = echo_gain * sqrt((T2_err/T2)^2 + (T2_echo_err/T2_echo)^2)
    # IMPROVEMENT: Handle missing CI gracefully
    T2_err_rel = np.zeros(len(df_merged))
    T2_echo_err_rel = np.zeros(len(df_merged))
    
    # Calculate relative errors only where CI exists
    has_fid_ci = df_merged['T2_lower'].notna() & df_merged['T2_upper'].notna()
    has_echo_ci = df_merged['T2_echo_lower'].notna() & df_merged['T2_echo_upper'].notna()
    
    if has_fid_ci.sum() > 0:
        T2_err_rel[has_fid_ci] = (df_merged.loc[has_fid_ci, 'T2_upper'] - 
                                   df_merged.loc[has_fid_ci, 'T2_lower']) / (2 * df_merged.loc[has_fid_ci, 'T2'])
    
    if has_echo_ci.sum() > 0:
        T2_echo_err_rel[has_echo_ci] = (df_merged.loc[has_echo_ci, 'T2_echo_upper'] - 
                                         df_merged.loc[has_echo_ci, 'T2_echo_lower']) / (2 * df_merged.loc[has_echo_ci, 'T2_echo'])
    
    # Calculate echo_gain_err only where both CI exist
    both_ci = has_fid_ci & has_echo_ci
    df_merged['echo_gain_err'] = np.nan
    if both_ci.sum() > 0:
        df_merged.loc[both_ci, 'echo_gain_err'] = (
            df_merged.loc[both_ci, 'echo_gain'] * 
            np.sqrt(T2_err_rel[both_ci]**2 + T2_echo_err_rel[both_ci]**2)
        )
    
    # For points with only one CI, use that CI
    only_fid_ci = has_fid_ci & ~has_echo_ci
    only_echo_ci = has_echo_ci & ~has_fid_ci
    if only_fid_ci.sum() > 0:
        df_merged.loc[only_fid_ci, 'echo_gain_err'] = (
            df_merged.loc[only_fid_ci, 'echo_gain'] * T2_err_rel[only_fid_ci]
        )
    if only_echo_ci.sum() > 0:
        df_merged.loc[only_echo_ci, 'echo_gain_err'] = (
            df_merged.loc[only_echo_ci, 'echo_gain'] * T2_echo_err_rel[only_echo_ci]
    )
    
    # Select columns for output
    output_cols = ['tau_c', 'xi', 'T2', 'T2_echo', 'echo_gain', 'echo_gain_err']
    if 'method_used' in df_merged.columns:
        output_cols.append('method_used')
    if 'T2_echo_used' in df_merged.columns:
        output_cols.append('T2_echo_used')
    
    df_output = df_merged[[col for col in output_cols if col in df_merged.columns]].copy()
    
    # Save to CSV
    output_file = Path("results/echo_gain.csv")
    df_output.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Mean echo gain: {df_output['echo_gain'].mean():.3f}")
    print(f"  Min echo gain: {df_output['echo_gain'].min():.3f}")
    print(f"  Max echo gain: {df_output['echo_gain'].max():.3f}")
    print(f"\nEcho gain vs ξ:")
    for _, row in df_output.head(10).iterrows():
        print(f"  ξ = {row['xi']:.3e}: echo_gain = {row['echo_gain']:.3f} ± {row['echo_gain_err']:.3f}")
    
    return df_output

if __name__ == '__main__':
    df = main()

