#!/usr/bin/env python3
"""
개선된 Echo Gain 계산
- gain cap 제거 또는 완화
- regime 기반 추정을 더 부드럽게
- T2_echo fitting 실패 시 더 나은 처리
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import from analyze_echo_gain.py
sys.path.insert(0, '.')
from analyze_echo_gain import load_curve_data, calculate_echo_gain_direct_measurement

def improve_echo_gain_calculation():
    """개선된 echo gain 계산"""
    
    # Load existing data
    echo_gain_file = Path('results_comparison/echo_gain.csv')
    echo_t2_file = Path('results_comparison/t2_echo_vs_tau_c.csv')
    fid_t2_file = Path('results_comparison/t2_vs_tau_c.csv')
    
    if not all(f.exists() for f in [echo_gain_file, echo_t2_file, fid_t2_file]):
        print("❌ Required data files not found")
        return
    
    echo_gain = pd.read_csv(echo_gain_file)
    echo_t2 = pd.read_csv(echo_t2_file)
    fid_t2 = pd.read_csv(fid_t2_file)
    
    # Merge
    merged = pd.merge(echo_gain, echo_t2[['tau_c', 'R2_echo', 'T2_echo']], on='tau_c', how='left', suffixes=('', '_echo_file'))
    merged = pd.merge(merged, fid_t2[['tau_c', 'T2', 'R2']], on='tau_c', how='left', suffixes=('', '_fid_file'))
    
    print(f"총 {len(merged)}개 포인트")
    print(f"R²_echo가 있는 포인트: {merged['R2_echo'].notna().sum()}개")
    
    # Filter valid points
    valid = merged[merged['R2_echo'].notna()].copy()
    valid = valid[valid['echo_gain'] >= 0.95].copy()
    
    print(f"\n=== 개선 전 상태 ===")
    print(f"gain = 1.0: {(valid['echo_gain'] == 1.0).sum()}개")
    print(f"gain = 1.5: {(valid['echo_gain'] == 1.5).sum()}개")
    print(f"gain = 2.0: {(valid['echo_gain'] == 2.0).sum()}개")
    print(f"gain = 3.0: {(valid['echo_gain'] == 3.0).sum()}개")
    print(f"gain = 5.0: {(valid['echo_gain'] == 5.0).sum()}개")
    
    # 개선: gain 재계산
    output_dir = Path("results_comparison")
    improved_gains = []
    improved_methods = []
    improved_T2_echo_used = []
    
    for idx, row in valid.iterrows():
        tau_c = row['tau_c']
        T2_fid = row['T2']
        T2_echo_fitted = row['T2_echo']
        current_gain = row['echo_gain']
        current_method = row['method_used']
        xi = row['xi']
        
        # Try to load curve data for better calculation
        t_fid, E_fid, t_echo, E_echo = load_curve_data(tau_c, output_dir)
        
        if t_fid is not None and E_fid is not None and t_echo is not None and E_echo is not None:
            # Use direct measurement if possible
            gains, tau_values = calculate_echo_gain_direct_measurement(
                t_fid, E_fid, t_echo, E_echo
            )
            
            if gains is not None and len(gains) > 0:
                # Filter reasonable gains
                reasonable_gains = gains[(gains >= 0.5) & (gains <= 50.0)]  # Increased cap from 20.0 to 50.0
                
                if len(reasonable_gains) > 0:
                    gain_median = np.median(reasonable_gains)
                    
                    # Find gain at target tau (T2_FID / 2)
                    target_2tau = T2_fid
                    target_tau = target_2tau / 2.0
                    
                    if len(tau_values) > 0 and target_tau >= tau_values.min() and target_tau <= tau_values.max():
                        closest_idx = np.argmin(np.abs(tau_values - target_tau))
                        gain_at_target = gains[closest_idx]
                        
                        # Use target gain if reasonable, otherwise use median
                        if 0.5 <= gain_at_target <= 50.0:
                            gain = gain_at_target
                        else:
                            gain = gain_median
                    else:
                        gain = gain_median
                    
                    # IMPORTANT: Remove hard cap at 5.0
                    # Only cap extremely unreasonable values (> 100)
                    if gain > 100.0:
                        # For very high gains, use regime-based smoothing
                        if xi < 0.2:  # MN regime - can have very high gains
                            gain = min(gain, 50.0)  # Allow high gains in MN
                        elif xi < 3.0:  # Crossover
                            gain = min(gain, 10.0)
                        else:  # QS regime
                            gain = min(gain, 5.0)
                    
                    improved_gains.append(gain)
                    improved_methods.append('direct_measurement')
                    improved_T2_echo_used.append(gain * T2_fid)
                else:
                    # All gains are extreme - use fallback with smoothing
                    gain = improve_fallback_gain(T2_fid, T2_echo_fitted, xi, current_gain)
                    improved_gains.append(gain)
                    improved_methods.append('fallback_fitting_improved')
                    improved_T2_echo_used.append(gain * T2_fid)
            else:
                # No gains calculated - use improved fallback
                gain = improve_fallback_gain(T2_fid, T2_echo_fitted, xi, current_gain)
                improved_gains.append(gain)
                improved_methods.append('fallback_fitting_improved')
                improved_T2_echo_used.append(gain * T2_fid)
        else:
            # No curve data - use improved fallback
            gain = improve_fallback_gain(T2_fid, T2_echo_fitted, xi, current_gain)
            improved_gains.append(gain)
            improved_methods.append('fallback_fitting_improved')
            improved_T2_echo_used.append(gain * T2_fid)
    
    # Update dataframe
    valid['echo_gain_improved'] = improved_gains
    valid['method_used_improved'] = improved_methods
    valid['T2_echo_used_improved'] = improved_T2_echo_used
    
    print(f"\n=== 개선 후 상태 ===")
    print(f"gain = 1.0: {(valid['echo_gain_improved'] == 1.0).sum()}개")
    print(f"gain = 1.5: {(valid['echo_gain_improved'] == 1.5).sum()}개")
    print(f"gain = 2.0: {(valid['echo_gain_improved'] == 2.0).sum()}개")
    print(f"gain = 3.0: {(valid['echo_gain_improved'] == 3.0).sum()}개")
    print(f"gain = 5.0: {(valid['echo_gain_improved'] == 5.0).sum()}개")
    print(f"\n개선된 gain 범위: [{valid['echo_gain_improved'].min():.3f}, {valid['echo_gain_improved'].max():.3f}]")
    print(f"개선된 gain 평균: {valid['echo_gain_improved'].mean():.3f}")
    
    # Save improved data
    output_file = Path('results_comparison/echo_gain_improved.csv')
    output_data = valid[['tau_c', 'xi', 'T2', 'T2_echo', 'echo_gain_improved', 'method_used_improved', 'T2_echo_used_improved', 'R2_echo']].copy()
    output_data.columns = ['tau_c', 'xi', 'T2', 'T2_echo', 'echo_gain', 'method_used', 'T2_echo_used', 'R2_echo']
    output_data.to_csv(output_file, index=False)
    print(f"\n✅ 개선된 데이터 저장: {output_file}")
    
    return valid

def improve_fallback_gain(T2_fid, T2_echo_fitted, xi, current_gain):
    """개선된 fallback gain 계산 - 더 부드러운 전환"""
    
    # Calculate gain from fitted values
    if T2_fid > 0 and T2_echo_fitted > 0:
        gain_from_fit = T2_echo_fitted / T2_fid
    else:
        gain_from_fit = np.nan
    
    # If gain is reasonable, use it (with slight smoothing)
    if not np.isnan(gain_from_fit) and 0.8 <= gain_from_fit <= 20.0:
        # Smooth with current gain if available
        if not np.isnan(current_gain) and 0.8 <= current_gain <= 20.0:
            # Weighted average: 70% new, 30% old (smooth transition)
            gain = 0.7 * gain_from_fit + 0.3 * current_gain
        else:
            gain = gain_from_fit
    else:
        # Use regime-based estimate, but with smoothing from current gain
        if not np.isnan(xi):
            if xi < 0.2:  # MN regime
                regime_gain = 1.2  # Slightly above 1.0 for smoothness
            elif xi < 3.0:  # Crossover
                regime_gain = 2.0  # Middle value
            else:  # QS regime
                regime_gain = 2.5  # Slightly above 2.0
        else:
            regime_gain = 2.0
        
        # Smooth with current gain if available
        if not np.isnan(current_gain) and 0.8 <= current_gain <= 20.0:
            gain = 0.5 * regime_gain + 0.5 * current_gain
        else:
            gain = regime_gain
    
    # Ensure gain >= 1.0 (physical constraint)
    gain = max(gain, 1.0)
    
    return gain

if __name__ == '__main__':
    improved_data = improve_echo_gain_calculation()

