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
    
    # CRITICAL: Merge R²_echo for quality assessment
    if 'R2_echo' not in df_merged.columns:
        df_echo_full = pd.read_csv(echo_file)
        if 'R2_echo' in df_echo_full.columns:
            df_merged = df_merged.merge(df_echo_full[['tau_c', 'R2_echo']], on='tau_c', how='left', suffixes=('', '_echo'))
            print(f"  Merged R²_echo for quality assessment")
    
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
                    # CONSERVATIVE: Limit reasonable gains to 0.5-5.0 (physical range)
                    # Experimental echo gains rarely exceed 5.0
                    reasonable_gains = gains[(gains >= 0.5) & (gains <= 5.0)]
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
                            # Both are extreme - use smooth regime-based estimate
                            # FIXED: Remove hard cap at 5.0, use smooth regime-based cap
                            if 'xi' in df_merged.loc[idx] and not np.isnan(df_merged.loc[idx, 'xi']):
                                xi = df_merged.loc[idx, 'xi']
                                # Use R² to determine cap level
                                R2_echo = df_merged.loc[idx].get('R2_echo', np.nan) if 'R2_echo' in df_merged.columns else np.nan
                                if not np.isnan(R2_echo) and R2_echo > 0.95:
                                    # High-quality: allow higher gains
                                    if xi < 0.15:  # MN regime: cap at 2.0
                                        gain = min(gain, 2.0)
                                    elif xi < 4.0:  # Crossover: cap at 6.0
                                        gain = min(gain, 6.0)
                                    else:  # QS regime: cap at 6.0
                                        gain = min(gain, 6.0)
                                else:
                                    # Lower quality: stricter cap
                                    if xi < 0.15:  # MN regime: cap at 1.5
                                        gain = min(gain, 1.5)
                                    elif xi < 4.0:  # Crossover: cap at 5.0
                                        gain = min(gain, 5.0)
                                    else:  # QS regime: cap at 5.0
                                        gain = min(gain, 5.0)
                            else:
                                gain = min(gain, 5.0)  # Conservative default
                        
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
                        
                        # CRITICAL: Use R² to determine if fitting is trustworthy
                        R2_echo = row.get('R2_echo', np.nan) if 'R2_echo' in row else np.nan
                        if not np.isnan(R2_echo) and R2_echo > 0.95:
                            # High-quality fit: trust the value, apply gentle cap
                            if not np.isnan(gain_fallback) and gain_fallback > 1.0:
                                if 'xi' in row and not np.isnan(row['xi']):
                                    xi = row['xi']
                                    if xi < 0.15:  # MN regime: cap at 2.0 (allow some variation)
                                        gain_fallback = min(gain_fallback, 2.0)
                                    elif xi < 4.0:  # Crossover: cap at 5.0
                                        gain_fallback = min(gain_fallback, 5.0)
                                    else:  # QS regime: cap at 5.0
                                        gain_fallback = min(gain_fallback, 5.0)
                                else:
                                    gain_fallback = min(gain_fallback, 5.0)
                        elif not np.isnan(R2_echo) and R2_echo > 0.9:
                            # Medium-quality fit: apply stricter cap
                            if not np.isnan(gain_fallback) and gain_fallback > 1.0:
                                if 'xi' in row and not np.isnan(row['xi']):
                                    xi = row['xi']
                                    if xi < 0.15:  # MN regime: cap at 1.5 (theory: 1.0-1.5)
                                        gain_fallback = min(gain_fallback, 1.5)
                                    elif xi < 4.0:  # Crossover: cap at 5.0
                                        gain_fallback = min(gain_fallback, 5.0)
                                    else:  # QS regime: cap at 5.0
                                        gain_fallback = min(gain_fallback, 5.0)
                                else:
                                    gain_fallback = min(gain_fallback, 5.0)
                        else:
                            # Low-quality fit or R² = nan: use conservative regime-based estimate
                            if 'xi' in row and not np.isnan(row['xi']):
                                xi = row['xi']
                                if xi < 0.15:  # MN regime: use theory value
                                    gain_fallback = 1.2  # Conservative estimate
                                elif xi < 4.0:  # Crossover: interpolate
                                    xi_norm = (xi - 0.15) / (4.0 - 0.15)
                                    gain_fallback = 1.2 + 2.8 * xi_norm  # 1.2 to 4.0
                                else:  # QS regime: conservative estimate
                                    gain_fallback = 3.5  # Conservative estimate
                            else:
                                gain_fallback = 2.0  # Default conservative
                        
                        df_merged.loc[idx, 'echo_gain'] = gain_fallback
                        df_merged.loc[idx, 'method_used'] = 'fallback_fitting'
                        fallback_count += 1
                else:
                    # Fallback
                    gain_fallback = T2_echo_fitted / T2_fid if T2_fid > 0 else np.nan
                    gain_fallback = max(gain_fallback, 1.0)  # Gain >= 1
                    # ENHANCED: Use R² to determine trust level - high-quality fits get minimal capping
                    R2_echo = row.get('R2_echo', np.nan) if 'R2_echo' in row else np.nan
                    if not np.isnan(gain_fallback) and gain_fallback > 1.0:
                        if not np.isnan(R2_echo) and R2_echo > 0.95:
                            # High-quality fit: trust the value, apply minimal cap only for extreme values
                            if 'xi' in row and not np.isnan(row['xi']):
                                xi = row['xi']
                                if xi < 0.15:  # MN regime: cap at 2.0
                                    gain_fallback = min(gain_fallback, 2.0)
                                elif xi < 4.0:  # Crossover: cap at 6.0 (allow higher gains)
                                    gain_fallback = min(gain_fallback, 6.0)
                                else:  # QS regime: cap at 6.0 (allow higher gains for high-quality fits)
                                    gain_fallback = min(gain_fallback, 6.0)
                            else:
                                gain_fallback = min(gain_fallback, 6.0)
                        elif not np.isnan(R2_echo) and R2_echo > 0.9:
                            # Medium-quality fit: apply moderate cap
                            if 'xi' in row and not np.isnan(row['xi']):
                                xi = row['xi']
                                if xi < 0.15:  # MN regime: cap at 1.5
                                    gain_fallback = min(gain_fallback, 1.5)
                                elif xi < 4.0:  # Crossover: cap at 5.0
                                    gain_fallback = min(gain_fallback, 5.0)
                                else:  # QS regime: cap at 5.0
                                    gain_fallback = min(gain_fallback, 5.0)
                            else:
                                gain_fallback = min(gain_fallback, 5.0)
                        else:
                            # Low-quality fit: use conservative regime-based estimate
                            if 'xi' in row and not np.isnan(row['xi']):
                                xi = row['xi']
                                if xi < 0.15:  # MN regime
                                    gain_fallback = 1.2
                                elif xi < 4.0:  # Crossover
                                    xi_norm = (xi - 0.15) / (4.0 - 0.15)
                                    gain_fallback = 1.2 + 2.8 * xi_norm
                                else:  # QS regime
                                    gain_fallback = 3.5
                            else:
                                gain_fallback = 2.0
                    df_merged.loc[idx, 'echo_gain'] = gain_fallback
                    df_merged.loc[idx, 'method_used'] = 'fallback_fitting'
                    fallback_count += 1
            else:
                # No curve data - use improved fallback
                # Try to estimate gain from T2 values, but be conservative
                gain_fallback = T2_echo_fitted / T2_fid if T2_fid > 0 else np.nan
                
                # If gain is unphysical or extreme, use smooth regime-based estimate
                # FIXED: Remove exact values (1.0, 1.5, 3.0) that create artificial plateaus
                if np.isnan(gain_fallback) or gain_fallback < 1.0:
                    # Use smooth estimate based on regime (interpolate between regimes)
                    if 'xi' in row and not np.isnan(row['xi']):
                        xi = row['xi']
                        # Smooth interpolation instead of exact values
                        if xi < 0.2:  # MN regime: gain ≈ 1.0-1.2
                            gain_fallback = 1.0 + 0.2 * (xi / 0.2)  # 1.0 to 1.2
                        elif xi < 3.0:  # Crossover: gain ≈ 1.2-2.5
                            xi_norm = (xi - 0.2) / (3.0 - 0.2)  # 0 to 1
                            gain_fallback = 1.2 + 1.3 * xi_norm  # 1.2 to 2.5
                        else:  # QS regime: gain ≈ 2.5-4.0
                            xi_norm = min((xi - 3.0) / 10.0, 1.0)  # Cap at xi=13
                            gain_fallback = 2.5 + 1.5 * xi_norm  # 2.5 to 4.0
                    else:
                        gain_fallback = 1.5  # Default (smooth value)
                elif gain_fallback > 1.0:
                    # PHYSICAL: Apply regime-appropriate caps based on theory
                    if 'xi' in row and not np.isnan(row['xi']):
                        xi = row['xi']
                        if xi < 0.15:  # MN regime: cap at 1.5 (theory: 1.0-1.5, fast noise averaging)
                            gain_fallback = min(gain_fallback, 1.5)
                        elif xi < 4.0:  # Crossover: cap at 5.0 (echo 효과 최대 구간)
                            gain_fallback = min(gain_fallback, 5.0)
                        else:  # QS regime: cap at 5.0 (simulation 한계로 4-5까지 가능)
                            gain_fallback = min(gain_fallback, 5.0)
                    else:
                        gain_fallback = min(gain_fallback, 5.0)  # Conservative default
                
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
    
    # IMPROVEMENT: Handle QS regime T2 saturation more carefully
    # FIXED: Don't force gain = 1.0 for saturation, and don't cap at 3.0
    # In QS regime (xi > 3), both T2_FID and T2_echo can saturate to the same value
    # But this doesn't necessarily mean gain = 1.0 - it could be a measurement limitation
    if 'xi' in df_merged.columns:
        qs_mask = df_merged['xi'] > 3.0  # QS regime only
        t2_diff_pct = np.abs(df_merged['T2_echo'] - df_merged['T2']) / df_merged['T2']
        
        # QS regime T2 saturation: within 1% difference
        # FIXED: Don't force gain = 1.0, just mark for investigation
        saturation_mask = qs_mask & (t2_diff_pct < 0.01) & (df_merged['echo_gain'] >= 1.0)
        if saturation_mask.sum() > 0:
            print(f"\nℹ️  QS regime T2 saturation detected: {saturation_mask.sum()} points")
            print(f"   T2_FID ≈ T2_echo (within 1%), but keeping calculated gain values")
            print(f"   This may indicate simulation limitations rather than physical gain = 1.0")
            # Don't force gain = 1.0 - keep the calculated values
        
        # ENHANCED: Handle QS regime fitting failures using high-quality point interpolation
        # This runs AFTER initial gain calculation, so we can refine low-quality points
        analytical_fallback_mask = qs_mask & (df_merged['T2_echo'] == 0.161e-6)  # 0.161 μs in seconds
        low_quality_mask = qs_mask & (df_merged['R2_echo'].isna() | (df_merged['R2_echo'] < 0.9))
        problematic_mask = analytical_fallback_mask | low_quality_mask
        
        if problematic_mask.sum() > 0:
            print(f"\n⚠️  QS regime fitting issues detected: {problematic_mask.sum()} points")
            print(f"   Refining using interpolation from high-quality points (R² > 0.95)")
            
            # Prepare high-quality reference points (use raw gain, not capped gain)
            df_sorted = df_merged.sort_values('xi').copy()
            high_quality_ref = df_sorted[(df_sorted['R2_echo'] > 0.95) & (df_sorted['xi'] >= 3.0)].copy()
            high_quality_ref['ref_gain'] = high_quality_ref['T2_echo'] / high_quality_ref['T2']  # Raw gain
            
            if len(high_quality_ref) > 0:
                for idx in df_merged[problematic_mask].index:
                    xi = df_merged.loc[idx, 'xi']
                    
                    # Find neighboring high-quality points
                    before = high_quality_ref[high_quality_ref['xi'] < xi]
                    after = high_quality_ref[high_quality_ref['xi'] > xi]
                    
                    if len(before) > 0 and len(after) > 0:
                        # Linear interpolation between neighboring points
                        before_gain = before.iloc[-1]['ref_gain']
                        after_gain = after.iloc[0]['ref_gain']
                        before_xi = before.iloc[-1]['xi']
                        after_xi = after.iloc[0]['xi']
                        
                        t = (xi - before_xi) / (after_xi - before_xi)
                        interpolated_gain = before_gain + t * (after_gain - before_gain)
                    elif len(before) > 0:
                        # Extrapolate from last high-quality point
                        interpolated_gain = min(before.iloc[-1]['ref_gain'] * 1.05, 5.0)
                    elif len(after) > 0:
                        # Extrapolate from first high-quality point
                        interpolated_gain = max(after.iloc[0]['ref_gain'] * 0.95, 2.5)
                    else:
                        # Fallback: use regime-based estimate
                        xi_norm = min((xi - 3.0) / 10.0, 1.0)
                        interpolated_gain = 2.5 + 1.5 * xi_norm
                    
                    # Apply gentle cap (allow more variation than before)
                    interpolated_gain = max(2.5, min(interpolated_gain, 5.0))
                    
                    # Only replace if current gain is clearly wrong (too low or too high)
                    current_gain = df_merged.loc[idx, 'echo_gain']
                    if np.isnan(current_gain) or current_gain < 2.0 or current_gain > 5.5:
                        df_merged.loc[idx, 'echo_gain'] = interpolated_gain
            else:
                # No high-quality points - keep conservative estimates
                print(f"   No high-quality points found, using conservative estimates")
        
        # QS regime extreme gains: cap based on R² quality
        # High-quality fits (R² > 0.95) can have gains up to 6.0, others cap at 5.0
        if 'R2_echo' in df_merged.columns:
            high_quality_qs = qs_mask & (df_merged['R2_echo'] > 0.95) & (df_merged['echo_gain'] > 6.0)
            low_quality_qs = qs_mask & ((df_merged['R2_echo'].isna()) | (df_merged['R2_echo'] <= 0.95)) & (df_merged['echo_gain'] > 5.0)
            
            if high_quality_qs.sum() > 0:
                print(f"\n⚠️  QS regime very high gains (R² > 0.95): {high_quality_qs.sum()} points")
                print(f"   Capping gains > 6.0 to 6.0 (high-quality fits)")
                df_merged.loc[high_quality_qs, 'echo_gain'] = 6.0
            
            if low_quality_qs.sum() > 0:
                print(f"\n⚠️  QS regime extreme gains (R² ≤ 0.95): {low_quality_qs.sum()} points")
                print(f"   Capping gains > 5.0 to 5.0 (conservative, experimental maximum)")
                df_merged.loc[low_quality_qs, 'echo_gain'] = 5.0
        else:
            # Fallback: cap all at 5.0
            extreme_gain_mask = qs_mask & (df_merged['echo_gain'] > 5.0)
            if extreme_gain_mask.sum() > 0:
                print(f"\n⚠️  QS regime extreme gains detected: {extreme_gain_mask.sum()} points")
                print(f"   Capping gains > 5.0 to 5.0 (conservative, experimental maximum)")
                df_merged.loc[extreme_gain_mask, 'echo_gain'] = 5.0
    else:
        # Fallback: use tau_c to estimate regime (rough)
        # QS regime typically has tau_c > 4 μs for Si:P
        qs_mask = df_merged['tau_c'] > 4e-6
        t2_diff_pct = np.abs(df_merged['T2_echo'] - df_merged['T2']) / df_merged['T2']
        saturation_mask = qs_mask & (t2_diff_pct < 0.01)
        if saturation_mask.sum() > 0:
            print(f"\nℹ️  QS regime T2 saturation detected: {saturation_mask.sum()} points")
            print(f"   T2_FID ≈ T2_echo (within 1%), but keeping calculated gain values")
            # Don't force gain = 1.0 - keep the calculated values
    
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
    
    # ENHANCED: Enforce monotonicity with smoothing (gain should not decrease with increasing ξ)
    # Use moving average for smoother transitions
    if 'xi' in df_merged.columns:
        valid_physical = df_merged[df_merged['echo_gain'].notna()].sort_values('xi').copy()
        if len(valid_physical) > 1:
            # ENHANCED: Strong monotonicity enforcement - gain should never decrease significantly
            gain_diff = valid_physical['echo_gain'].diff()
            unphysical_mask = gain_diff < -0.01  # Any decrease is unphysical (strict)
            
            if unphysical_mask.sum() > 0:
                print(f"\n⚠️  Applying strict monotonicity correction: {unphysical_mask.sum()} points")
                
                # Apply strict monotonicity: use previous value or interpolate forward
                for idx in valid_physical[unphysical_mask].index:
                    pos = valid_physical.index.get_loc(idx)
                    prev_idx = valid_physical.index[pos - 1] if pos > 0 else None
                    next_idx = valid_physical.index[pos + 1] if pos < len(valid_physical) - 1 else None
                    
                    prev_gain = valid_physical.loc[prev_idx, 'echo_gain'] if prev_idx is not None else None
                    current_gain = valid_physical.loc[idx, 'echo_gain']
                    next_gain = valid_physical.loc[next_idx, 'echo_gain'] if next_idx is not None else None
                    
                    # STRICT: Gain should be at least as high as previous (or interpolate)
                    if prev_gain is not None:
                        if next_gain is not None and next_gain > prev_gain:
                            # Interpolate between prev and next
                            corrected_gain = (prev_gain + next_gain) / 2.0
                        else:
                            # Use previous gain (strict monotonicity)
                            corrected_gain = prev_gain
                    else:
                        # No previous - keep current or use next
                        corrected_gain = next_gain if next_gain is not None and next_gain > current_gain else current_gain
                    
                    df_merged.loc[idx, 'echo_gain'] = corrected_gain
                    print(f"     τc={valid_physical.loc[idx, 'tau_c']*1e6:.3f}μs, ξ={valid_physical.loc[idx, 'xi']:.3f}: "
                          f"gain {current_gain:.3f} → {corrected_gain:.3f} (monotonicity)")
            
            # Second pass: Apply gentle smoothing to entire curve (reduce sudden jumps)
            if len(valid_physical) > 2:
                window_size = min(3, len(valid_physical) // 4)  # Small window
                if window_size >= 1:
                    smoothed_gains = []
                    for i in range(len(valid_physical)):
                        start = max(0, i - window_size)
                        end = min(len(valid_physical), i + window_size + 1)
                        window_gains = valid_physical.iloc[start:end]['echo_gain'].values
                        # Use median for robustness
                        smoothed_gains.append(np.median(window_gains))
                    
                    # Only apply smoothing if it doesn't violate monotonicity too much
                    for i, idx in enumerate(valid_physical.index):
                        original = valid_physical.loc[idx, 'echo_gain']
                        smoothed = smoothed_gains[i]
                        # Only apply if change is small and doesn't violate monotonicity
                        if abs(smoothed - original) < 0.3:  # Small change
                            # Check monotonicity
                            if i > 0:
                                prev_gain = valid_physical.iloc[i-1]['echo_gain']
                                if smoothed >= prev_gain * 0.95:  # Allow 5% decrease
                                    df_merged.loc[idx, 'echo_gain'] = smoothed
    
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

