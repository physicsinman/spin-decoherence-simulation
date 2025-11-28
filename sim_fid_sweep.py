#!/usr/bin/env python3
"""
FID Full Sweep Simulation
Generates t2_vs_tau_c.csv with 20 points covering tau_c from 1e-8 to 1e-3 s
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single
from spin_decoherence.analysis.bootstrap import bootstrap_T2
from spin_decoherence.analysis.fitting import fit_coherence_decay_with_offset
import json
from datetime import datetime

# ============================================
# SI:P SIMULATION PARAMETERS (FINAL)
# ============================================
gamma_e = 1.76e11          # rad/(s·T) - electron gyromagnetic ratio
B_rms = 0.57e-6            # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration
tau_c_min = 3e-9           # s (start of MN regime)
tau_c_max = 1e-3           # s (extends well into QS regime)
# NOTE: We now build a custom tau_c grid with higher density in the crossover region
# instead of using a single logspace with tau_c_npoints points.
tau_c_npoints = None
N_traj = 2000              # Monte Carlo trajectories per point (빠른 실행: 시간 절약)

# Generate tau_c sweep - 물리학적 정확도와 결과 품질 최우선
def build_tau_c_sweep():
    """
    Create a tau_c grid optimized for physical accuracy and result quality
    
    물리학적 최적화:
    - Crossover regime에 더 많은 포인트 (중요한 transition region)
    - 각 regime에서 충분한 해상도로 정확한 곡선 생성
    
    - MN regime (3e-9 to 3e-8 s): 12 points (증가: 더 나은 slope fitting)
    - Crossover regime (3e-8 to 3e-6 s): 30 points (증가: 더 정확한 transition)
    - QS regime (3e-6 to 1e-3 s): 25 points (증가: 더 부드러운 곡선)
    Total: 67 points (물리학적 정확도와 결과 품질 최우선, fig1/fig2 향상)
    """
    mn = np.logspace(np.log10(3e-9), np.log10(3e-8), 12, endpoint=False)
    crossover = np.logspace(np.log10(3e-8), np.log10(3e-6), 30, endpoint=False)
    qs = np.logspace(np.log10(3e-6), np.log10(1e-3), 25)
    
    tau_vals = np.unique(np.concatenate([mn, crossover, qs]))
    return tau_vals

tau_c_sweep = build_tau_c_sweep()
tau_c_npoints = len(tau_c_sweep)

# Adaptive parameters
def get_dt(tau_c, T_max=None, max_memory_gb=8.0):
    """
    Adaptive timestep selection with memory limit.
    CRITICAL: dt must satisfy dt < tau_c/5 for numerical stability.
    
    Parameters
    ----------
    tau_c : float
        Correlation time
    T_max : float, optional
        Maximum simulation time (for memory calculation)
    max_memory_gb : float
        Maximum memory in GB (default: 8 GB)
    
    Returns
    -------
    dt : float
        Time step (guaranteed to satisfy dt < tau_c/5)
    """
    # CRITICAL: Numerical stability constraint - must be satisfied first
    # Use tau_c/6 to ensure dt < tau_c/5 (strict inequality with safety margin)
    dt_max_stable = tau_c / 6.0  # Maximum dt for numerical stability (safety margin)
    
    # Start with target: 100 steps per tau_c (balanced precision/memory)
    dt_target = tau_c / 100
    
    # If T_max is provided, check memory usage
    if T_max is not None:
        N_traj = 2000  # Fixed ensemble size (빠른 실행: 시간 절약)
        N_steps = int(T_max / dt_target) + 1
        memory_gb = (N_steps * N_traj * 8) / (1024**3)
        
        # If memory exceeds limit, increase dt (but respect stability constraint)
        if memory_gb > max_memory_gb:
            # Calculate required dt to meet memory limit
            max_N_steps = int((max_memory_gb * 1024**3) / (N_traj * 8))
            dt_required = T_max / max_N_steps if max_N_steps > 0 else dt_target
            
            # Ensure minimum precision: at least 50 steps per tau_c
            dt_min = tau_c / 50
            
            # CRITICAL: dt must be <= dt_max_stable (tau_c/5)
            dt = max(dt_required, dt_min)
            dt = min(dt, dt_max_stable)  # Enforce stability constraint
            
            # If dt is still too large for memory, we need to reduce T_max
            # This will be handled by the caller
            return dt
    
    # CRITICAL: Always enforce stability constraint
    return min(dt_target, dt_max_stable)

def get_tmax(tau_c, B_rms, gamma_e):
    """Calculate appropriate simulation duration - FURTHER IMPROVED"""
    # Estimate T2 from motional narrowing theory
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.5:  # MN regime (그래프 기준: ξ < 0.5, 기존: ξ < 0.1)
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        T_max_from_T2 = 10 * T2_est
        # CRITICAL: Cap T_max to prevent memory issues when B_rms is very small
        # For physical B_rms (0.57 μT), T2 can be very long, so we need a reasonable cap
        # Reduced from 100 ms to 10 ms to improve simulation speed
        return min(T_max_from_T2, 10e-3)  # Cap at 10 ms
    elif xi >= 2.0:  # QS regime (그래프 기준: ξ >= 2.0, 기존: ξ > 10)
        # CRITICAL FIX: QS regime T2 = sqrt(2) / (gamma_e * B_rms)
        # This comes from Gaussian decay: E(t) = exp(-(Δω·t)²/2)
        # At t = T2, E(T2) = 1/e, so (Δω·T2)²/2 = 1, giving T2 = sqrt(2)/Δω
        T2_est = np.sqrt(2.0) / (gamma_e * B_rms)
        # 추가 개선: 80-150배 → 100-200배로 증가 (QS regime 정확도 향상)
        # R² 개선 및 RMS 편차 감소를 위해 더 긴 시뮬레이션 시간 필요
        if xi < 10:
            multiplier = 100  # 초기 QS regime: 80 → 100
        elif xi < 50:
            multiplier = 150  # 중간 QS regime: 120 → 150
        else:
            multiplier = 200  # 깊은 QS regime: 150 → 200
        T_max_from_T2 = multiplier * T2_est
        
        # OU noise burn-in도 고려
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        
        # 메모리 제한 (10 ms for speed)
        return min(T_max_final, 10e-3)
    else:  # Crossover (0.5 <= xi < 2.0) - FURTHER IMPROVED
        # Crossover regime: Use intermediate approach
        # For xi close to 2.0, T2 approaches QS value
        T2_QS = np.sqrt(2.0) / (gamma_e * B_rms)
        T2_MN = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        # Interpolate between MN and QS estimates
        weight = (xi - 0.5) / (2.0 - 0.5)  # 0 at xi=0.5, 1 at xi=2.0
        T2_est = (1 - weight) * T2_MN + weight * T2_QS
        # 추가 개선: 15배 → 20배 (더 완전한 decay 포착)
        T_max_from_T2 = 20 * T2_est
        # CRITICAL: Cap T_max to prevent memory issues
        return min(T_max_from_T2, 10e-3)  # Cap at 10 ms for speed

def main():
    print("="*80)
    print("FID Full Sweep Simulation (IMPROVED)")
    print("="*80)
    print(f"\nParameters:")
    print(f"  gamma_e = {gamma_e:.3e} rad/(s·T)")
    print(f"  B_rms = {B_rms*1e6:.2f} μT (0.57 μT for 800 ppm ²⁹Si)")
    print(f"  tau_c range: {tau_c_min*1e6:.2f} to {tau_c_max*1e3:.2f} ms")
    print(f"  N_traj = {N_traj}")
    print(f"\n추가 개선 사항:")
    print(f"  - QS regime: T_max = 80-150×T2 (기존: 50-100×)")
    print(f"  - Crossover regime: 포인트 증가 (30 → 35)")
    print(f"  - QS regime: 포인트 증가 (25 → 30)")
    print(f"  - Crossover: T_max = 20×T2 (기존: 15×)")
    print(f"  - 메모리 제한: 8 GB (자동 dt 조정)")
    print(f"\nExpected time: ~1-2 hours (Bootstrap B=200으로 최적화)")
    print("\nStarting simulation...\n")
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Storage for results
    results_data = []
    
    for i, tau_c in enumerate(tau_c_sweep):
        print(f"\n[{i+1}/{len(tau_c_sweep)}] Processing τ_c = {tau_c*1e6:.3f} μs")
        
        # Calculate adaptive parameters
        # CRITICAL: dt must satisfy dt < tau_c/5 for numerical stability
        # Strategy: First calculate T_max, then dt, then adjust T_max if needed
        T_max = get_tmax(tau_c, B_rms, gamma_e)
        dt = get_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        xi = gamma_e * B_rms * tau_c
        
        # CRITICAL: Enforce dt < tau_c/5 constraint
        dt_max_stable = tau_c / 6.0  # Use 6.0 for safety margin
        if dt > dt_max_stable:
            # dt is too large, reduce it to satisfy stability constraint
            dt = dt_max_stable
            # Recalculate T_max to fit memory with this dt
            max_memory_gb = 8.0
            max_N_steps = int((max_memory_gb * 1024**3) / (N_traj * 8))
            T_max_adjusted = dt * max_N_steps
            if T_max_adjusted < T_max:
                T_max = T_max_adjusted
                print(f"  ⚠️  T_max reduced to {T_max*1e6:.2f} μs to satisfy dt < tau_c/5 constraint")
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.3e}")
        
        # Verify dt constraint (dt must be strictly less than tau_c/5)
        if dt >= tau_c / 5.0:
            print(f"  ❌ ERROR: dt ({dt*1e9:.2f} ns) >= tau_c/5 ({tau_c/5.0*1e9:.2f} ns)")
            print(f"  Skipping this tau_c value")
            continue
        
        # Memory check
        N_steps = int(T_max / dt) + 1
        memory_gb = (N_steps * N_traj * 8) / (1024**3)  # float64 = 8 bytes
        if memory_gb > 10:
            print(f"  ⚠️  Memory warning: {memory_gb:.1f} GB estimated")
        
        # Setup parameters
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),  # Single value
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj,
            'seed': 42 + i,
            'output_dir': str(output_dir),
            'compute_bootstrap': True,
            'B_bootstrap': 800,  # Increased from 200 to 800 for better error bar accuracy
            'save_delta_B_sample': False,
        }
        
        # Run simulation with error handling
        try:
            result = run_simulation_single(tau_c, params=params, verbose=False)
        except Exception as e:
            print(f"  ❌ Simulation failed: {e}")
            results_data.append({
                'tau_c': tau_c,
                'T2': np.nan,
                'T2_lower': np.nan,
                'T2_upper': np.nan,
                'R2': np.nan,
                'xi': xi
            })
            continue
        
        # Extract T2 and fit quality
        fit_result = result['fit_result']
        if fit_result is not None:
            T2 = fit_result.get('T2', np.nan)
            T2_ci = result.get('T2_ci', None)
            T2_samples = result.get('T2_samples', None)  # Extract bootstrap samples if available
            R2 = fit_result.get('R2', np.nan)
            
            # Extract CI bounds with fallback
            if T2_ci is not None:
                T2_lower = T2_ci[0]
                T2_upper = T2_ci[1]
                # Check if CI is degenerate (width = 0)
                ci_width = T2_upper - T2_lower
                if ci_width == 0 or ci_width / T2 < 1e-6:
                    # Fallback to analytical error
                    T2_error = fit_result.get('T2_error', np.nan)
                    if not np.isnan(T2_error) and T2_error > 0:
                        T2_lower = T2 - 1.96 * T2_error
                        T2_upper = T2 + 1.96 * T2_error
                        print(f"  ⚠️  Bootstrap CI degenerate, using analytical error")
                    else:
                        # Last resort: use bootstrap std if available, otherwise analytical error
                        if T2_samples is not None and len(T2_samples) > 0:
                            T2_std = np.std(T2_samples, ddof=1)
                            if T2_std > 0:
                                T2_lower = T2 - 1.96 * T2_std
                                T2_upper = T2 + 1.96 * T2_std
                                print(f"  ⚠️  Bootstrap CI degenerate, using bootstrap std")
                            else:
                                # Use 5% of T2 as uncertainty
                                T2_lower = T2 * 0.95
                                T2_upper = T2 * 1.05
                                print(f"  ⚠️  Bootstrap CI degenerate, using 5% uncertainty")
                        else:
                            # Use 5% of T2 as uncertainty
                            T2_lower = T2 * 0.95
                            T2_upper = T2 * 1.05
                            print(f"  ⚠️  Bootstrap CI degenerate, using 5% uncertainty")
            else:
                # Bootstrap CI is None, use analytical error
                T2_error = fit_result.get('T2_error', np.nan)
                if not np.isnan(T2_error) and T2_error > 0:
                    T2_lower = T2 - 1.96 * T2_error
                    T2_upper = T2 + 1.96 * T2_error
                    print(f"  ⚠️  Bootstrap CI is None, using analytical error")
                elif T2_samples is not None and len(T2_samples) > 0:
                    # Use std of bootstrap samples even if CI is degenerate
                    T2_std = np.std(T2_samples, ddof=1)
                    if T2_std > 0:
                        T2_lower = T2 - 1.96 * T2_std
                        T2_upper = T2 + 1.96 * T2_std
                        print(f"  ⚠️  Bootstrap CI is None, using bootstrap std")
                    else:
                        # Last resort: use 5% of T2 as uncertainty
                        T2_lower = T2 * 0.95
                        T2_upper = T2 * 1.05
                        print(f"  ⚠️  Bootstrap CI is None, using 5% uncertainty")
                else:
                    # Last resort: use 5% of T2 as uncertainty
                    T2_lower = T2 * 0.95
                    T2_upper = T2 * 1.05
                    print(f"  ⚠️  Bootstrap CI is None, using 5% uncertainty")
            
            results_data.append({
                'tau_c': tau_c,
                'T2': T2,
                'T2_lower': T2_lower,
                'T2_upper': T2_upper,
                'R2': R2,
                'xi': xi
            })
            
            print(f"  T2 = {T2*1e6:.3f} μs (R² = {R2:.4f})")
            if T2_ci is not None:
                print(f"  95% CI: [{T2_lower*1e6:.3f}, {T2_upper*1e6:.3f}] μs")
        else:
            print(f"  ⚠️  Fit failed!")
            results_data.append({
                'tau_c': tau_c,
                'T2': np.nan,
                'T2_lower': np.nan,
                'T2_upper': np.nan,
                'R2': np.nan,
                'xi': xi
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(results_data)
    output_file = output_dir / "t2_vs_tau_c.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Total points: {len(results_data)}")
    print(f"  Successful fits: {df['T2'].notna().sum()}")
    if df['R2'].notna().sum() > 0:
        print(f"  Mean R²: {df['R2'].mean():.4f}")
    
    # Validation
    if df['T2'].notna().sum() < len(df) * 0.8:
        print(f"\n⚠️  Warning: Only {df['T2'].notna().sum()}/{len(df)} fits successful (<80%)")
    
    return df

if __name__ == '__main__':
    df = main()

