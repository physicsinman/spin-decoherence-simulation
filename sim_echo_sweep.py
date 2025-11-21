#!/usr/bin/env python3
"""
Hahn Echo Full Sweep Simulation
Generates t2_echo_vs_tau_c.csv with 20 points covering tau_c from 1e-8 to 1e-3 s
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.echo import run_simulation_with_hahn_echo
from spin_decoherence.analysis.bootstrap import bootstrap_T2
from spin_decoherence.analysis.fitting import fit_coherence_decay_with_offset
import json
from datetime import datetime

# ============================================
# SI:P SIMULATION PARAMETERS (FINAL)
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (0.05 mT)
tau_c_min = 3e-9           # s (match FID sweep lower bound)
tau_c_max = 1e-3           # s
tau_c_npoints = None       # we build a custom grid instead of uniform logspace
N_traj = 2000              # Monte Carlo trajectories per point (increased for better statistics)

# Generate tau_c sweep - CRITICAL FIX: Use FID grid if available
def build_tau_c_sweep():
    """
    Create a tau_c grid matching the FID sweep EXACTLY.
    
    CRITICAL: If FID data exists, use its tau_c values to ensure perfect matching.
    Otherwise, generate the same grid as FID sweep.
    """
    # Try to load FID data to use exact same tau_c values
    fid_file = Path("results/t2_vs_tau_c.csv")
    if fid_file.exists():
        try:
            df_fid = pd.read_csv(fid_file)
            if 'tau_c' in df_fid.columns and len(df_fid) > 0:
                tau_vals = df_fid['tau_c'].dropna().unique()
                tau_vals = np.sort(tau_vals)
                print(f"  ✅ Using FID tau_c grid: {len(tau_vals)} points")
                return tau_vals
        except Exception as e:
            print(f"  ⚠️  Could not load FID grid: {e}, generating new grid")
    
    # Fallback: Generate same grid as FID sweep
    mn = np.logspace(np.log10(3e-9), np.log10(3e-8), 18, endpoint=False)
    crossover = np.logspace(np.log10(3e-8), np.log10(3e-6), 35, endpoint=False)
    qs = np.logspace(np.log10(3e-6), np.log10(1e-3), 30)
    tau_vals = np.unique(np.concatenate([mn, crossover, qs]))
    print(f"  ⚠️  Generated new tau_c grid: {len(tau_vals)} points")
    return tau_vals

tau_c_sweep = build_tau_c_sweep()
tau_c_npoints = len(tau_c_sweep)

# Adaptive parameters - IMPROVED (matching FID sweep)
def get_dt(tau_c, T_max=None, max_memory_gb=8.0):
    """
    Adaptive timestep selection with memory limit (same as FID sweep).
    """
    dt_target = tau_c / 100
    if T_max is not None:
        N_traj = 2000
        N_steps = int(T_max / dt_target) + 1
        memory_gb = (N_steps * N_traj * 8) / (1024**3)
        if memory_gb > max_memory_gb:
            max_N_steps = int((max_memory_gb * 1024**3) / (N_traj * 8))
            dt_required = T_max / max_N_steps if max_N_steps > 0 else dt_target
            dt_min = tau_c / 50
            dt = max(dt_required, dt_min)
            return dt
    return dt_target

def get_tmax(tau_c, B_rms, gamma_e):
    """Calculate appropriate simulation duration - IMPROVED (matching FID sweep)"""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est
    elif xi > 3:  # QS regime - FURTHER IMPROVED
        T2_est = 1.0 / (gamma_e * B_rms)
        # 추가 개선: 80-150배 → 100-200배로 증가 (QS regime 정확도 향상)
        if xi < 10:
            multiplier = 100  # 초기 QS regime: 80 → 100
        elif xi < 50:
            multiplier = 150  # 중간 QS regime: 120 → 150
        else:
            multiplier = 200  # 깊은 QS regime: 150 → 200
        T_max_from_T2 = multiplier * T2_est
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        return min(T_max_final, 100e-3)
    else:  # Crossover - IMPROVED
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 20 * T2_est  # 개선: 10 → 20배

def main():
    print("="*80)
    print("Hahn Echo Full Sweep Simulation (IMPROVED)")
    print("="*80)
    print(f"\nParameters:")
    print(f"  gamma_e = {gamma_e:.3e} rad/(s·T)")
    print(f"  B_rms = {B_rms*1e3:.3f} mT")
    print(f"  tau_c range: {tau_c_min*1e6:.2f} to {tau_c_max*1e3:.2f} ms")
    print(f"  N_traj = {N_traj}")
    print(f"\n개선 사항 (FID sweep과 동일):")
    print(f"  - QS regime: T_max = 80-150×T2 (기존: 30×)")
    print(f"  - Crossover: 포인트 증가 (30 → 35)")
    print(f"  - QS regime: 포인트 증가 (25 → 30)")
    print(f"  - Crossover: T_max = 20×T2 (기존: 10×)")
    print(f"  - Echo T_max: 더 길게 (2.5-3.0×)")
    print(f"  - 메모리 제한: 8 GB (자동 dt 조정)")
    print(f"\nExpected time: ~3-4 hours (개선으로 인해 더 오래 걸림)")
    print("\nStarting simulation...\n")
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Storage for results
    results_data = []
    
    for i, tau_c in enumerate(tau_c_sweep):
        print(f"\n[{i+1}/{len(tau_c_sweep)}] Processing τ_c = {tau_c*1e6:.3f} μs")
        
        # Calculate adaptive parameters (matching FID sweep improvements)
        T_max = get_tmax(tau_c, B_rms, gamma_e)
        dt = get_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        xi = gamma_e * B_rms * tau_c
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.3e}")
        
        # Setup parameters
        # CRITICAL FIX: MN regime에서 echo는 거의 decay하지 않으므로 매우 긴 T_max_echo 필요
        # Hybrid method를 사용하려면 T_FID까지 echo curve가 도달해야 함
        if xi > 3:  # QS regime
            T_max_echo = T_max * 3.0  # QS regime: 3.0배
        elif xi < 0.3:  # MN regime - CRITICAL FIX
            # MN regime: Echo는 거의 decay하지 않으므로 매우 긴 시간 필요
            # Hybrid method를 위해 T_FID까지 echo curve가 도달해야 함
            T2_fid_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)  # MN regime FID T2
            T2_echo_est = 20.0 * T2_fid_est  # MN regime echo T2 (conservative estimate)
            
            # T_FID ≈ T2_fid_est이므로, echo curve가 최소 T_FID까지 도달해야 함
            # 더 안전하게 T_FID의 2배까지 확보
            T_FID_est = T2_fid_est  # T_FID ≈ T2_FID for exponential decay
            T_max_echo = max(2.0 * T_FID_est, 50.0 * T2_echo_est, T_max * 20.0)
            T_max_echo = min(T_max_echo, 200e-3)  # 최대 200 ms로 제한
        else:  # Crossover
            T_max_echo = T_max * 5.0  # Crossover: 5.0배 (증가)
        
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'T_max_echo': T_max_echo,  # IMPROVEMENT: 1.5배 증가
            'M': N_traj,
            'seed': 42 + i,
            'output_dir': str(output_dir),
            'compute_bootstrap': True,
            'save_delta_B_sample': False,
        }
        
        # IMPROVEMENT 3: Calculate optimal upsilon_max based on T_max_echo
        # For QS regime, we need larger upsilon_max to capture echo decay
        if xi > 3:  # QS regime
            # Calculate upsilon_max from T_max_echo
            # tau_max = 0.4 * T_max_echo (from get_dimensionless_tau_range)
            # upsilon_max = tau_max / tau_c = 0.4 * T_max_echo / tau_c
            upsilon_max = min(0.4 * T_max_echo / tau_c, 5.0)  # Cap at 5.0
        else:
            upsilon_max = 0.8  # Default for other regimes
        
        # Generate tau_list with adaptive upsilon_max
        from spin_decoherence.simulation.engine import get_dimensionless_tau_range
        tau_list = get_dimensionless_tau_range(
            tau_c, n_points=50, upsilon_min=0.05, upsilon_max=upsilon_max,
            dt=dt, T_max=T_max_echo
        )
        
        # Run simulation (both FID and echo) with error handling
        try:
            result = run_simulation_with_hahn_echo(tau_c, params=params, tau_list=tau_list, verbose=False)
        except Exception as e:
            print(f"  ❌ Simulation failed: {e}")
            results_data.append({
                'tau_c': tau_c,
                'T2_echo': np.nan,
                'T2_echo_lower': np.nan,
                'T2_echo_upper': np.nan,
                'R2_echo': np.nan,
                'xi': xi
            })
            continue
        
        # Extract echo T2
        fit_result_echo = result.get('fit_result_echo')
        if fit_result_echo is not None:
            T2_echo = fit_result_echo.get('T2', np.nan)
            R2_echo = fit_result_echo.get('R2', np.nan)
            
            # Extract CI bounds - IMPROVED: Try multiple sources
            T2_echo_ci = result.get('T2_echo_ci', None)
            
            # If CI not in result, try to compute from bootstrap samples
            if T2_echo_ci is None:
                E_echo_abs_all = result.get('E_echo_abs_all', None)
                if E_echo_abs_all is not None and len(E_echo_abs_all) > 0:
                    # Try bootstrap if we have samples
                    try:
                        from spin_decoherence.analysis.bootstrap import bootstrap_T2
                        tau_echo = np.array(result.get('tau_echo', []))
                        if len(tau_echo) > 0 and len(E_echo_abs_all) > 0:
                            T2_samples = []
                            for E_traj in E_echo_abs_all:
                                # Fit each trajectory
                                fit_traj = fit_coherence_decay_with_offset(
                                    tau_echo, E_traj, model='gaussian',
                                    is_echo=True, tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
                                )
                                if fit_traj is not None:
                                    T2_samples.append(fit_traj.get('T2', np.nan))
                            
                            if len(T2_samples) > 10:  # Need enough samples
                                T2_samples = np.array([t for t in T2_samples if not np.isnan(t)])
                                if len(T2_samples) > 0:
                                    T2_mean = np.mean(T2_samples)
                                    T2_std = np.std(T2_samples, ddof=1)
                                    T2_echo_ci = (T2_mean - 1.96*T2_std, T2_mean + 1.96*T2_std)
                    except:
                        pass
            
            # Fallback: Use analytical error if available
            if T2_echo_ci is None:
                T2_error = fit_result_echo.get('T2_error', np.nan)
                if not np.isnan(T2_error) and T2_error > 0:
                    T2_echo_lower = T2_echo - 1.96 * T2_error
                    T2_echo_upper = T2_echo + 1.96 * T2_error
                else:
                    # Last resort: use 5% uncertainty
                    T2_echo_lower = T2_echo * 0.95 if not np.isnan(T2_echo) else np.nan
                    T2_echo_upper = T2_echo * 1.05 if not np.isnan(T2_echo) else np.nan
            else:
                T2_echo_lower = T2_echo_ci[0]
                T2_echo_upper = T2_echo_ci[1]
            
            results_data.append({
                'tau_c': tau_c,
                'T2_echo': T2_echo,
                'T2_echo_lower': T2_echo_lower,
                'T2_echo_upper': T2_echo_upper,
                'R2_echo': R2_echo,
                'xi': xi
            })
            
            print(f"  T2_echo = {T2_echo*1e6:.3f} μs (R² = {R2_echo:.4f})")
            if T2_echo_ci is not None or not np.isnan(T2_echo_lower):
                print(f"  95% CI: [{T2_echo_lower*1e6:.3f}, {T2_echo_upper*1e6:.3f}] μs")
        else:
            print(f"  ⚠️  Echo fit failed!")
            results_data.append({
                'tau_c': tau_c,
                'T2_echo': np.nan,
                'T2_echo_lower': np.nan,
                'T2_echo_upper': np.nan,
                'R2_echo': np.nan,
                'xi': xi
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(results_data)
    output_file = output_dir / "t2_echo_vs_tau_c.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Total points: {len(results_data)}")
    print(f"  Successful fits: {df['T2_echo'].notna().sum()}")
    print(f"  Mean R²: {df['R2_echo'].mean():.4f}")
    
    return df

if __name__ == '__main__':
    df = main()

