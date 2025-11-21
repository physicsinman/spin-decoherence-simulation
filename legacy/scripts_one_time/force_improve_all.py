#!/usr/bin/env python3
"""
Force Improve All: 더 엄격한 기준으로 모든 포인트 개선
R² < 0.98인 포인트도 재시뮬레이션 (최고 품질 목표)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single
import time

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11
B_rms = 0.05e-3
N_traj_improved = 10000  # Increased from 5000 for better R²

def get_adaptive_dt(tau_c, T_max=None, max_memory_gb=8.0):
    """Adaptive timestep with memory limit."""
    dt_target = tau_c / 100
    if T_max is not None:
        N_steps = int(T_max / dt_target) + 1
        memory_gb = (N_steps * N_traj_improved * 8) / (1024**3)
        if memory_gb > max_memory_gb:
            max_N_steps = int((max_memory_gb * 1024**3) / (N_traj_improved * 8))
            dt_required = T_max / max_N_steps if max_N_steps > 0 else dt_target
            dt_min = tau_c / 50
            dt = max(dt_required, dt_min)
            return dt
    return dt_target

def get_tmax_maximum(tau_c, B_rms, gamma_e):
    """Calculate simulation duration - MAXIMUM quality (enhanced for better R²)."""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 20 * T2_est  # Increased from 15
    elif xi > 3:  # QS regime - MAXIMUM
        T2_est = 1.0 / (gamma_e * B_rms)
        if xi < 10:
            multiplier = 300  # Increased from 250
        elif xi < 50:
            multiplier = 350  # Increased from 300
        else:
            multiplier = 400  # Increased from 350
        T_max_from_T2 = multiplier * T2_est
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        return min(T_max_final, 100e-3)
    else:  # Crossover - MAXIMUM (most problematic)
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 50 * T2_est  # Increased from 35 for better R²

def main():
    print("="*80)
    print("Force Improve All: 최고 품질 목표")
    print("="*80)
    print("\n기준: R² < 0.98인 모든 포인트 재시뮬레이션")
    print(f"N_traj = {N_traj_improved}")
    print("T_max = Maximum (최고 품질)")
    print("\n⚠️  This will take MANY hours!")
    print()
    
    fid_file = Path("results_comparison/t2_vs_tau_c.csv")
    df_fid = pd.read_csv(fid_file)
    
    # More strict criteria: R² < 0.98
    poor_fit = df_fid[df_fid['R2'] < 0.98].sort_values('tau_c')
    
    print(f"Found {len(poor_fit)} points with R² < 0.98")
    print(f"Estimated time: ~{len(poor_fit) * 50 / 60:.1f} hours (increased T_max and N_traj)\n")
    
    if len(poor_fit) == 0:
        print("✅ All points already have R² ≥ 0.98!")
        return
    
    print("Starting re-simulation...\n")
    start_total = time.time()
    
    for i, (idx, row) in enumerate(poor_fit.iterrows()):
        tau_c = row['tau_c']
        xi = row['xi']
        old_r2 = row['R2']
        
        print(f"\n[{i+1}/{len(poor_fit)}] τc = {tau_c*1e6:.3f} μs (ξ = {xi:.3f})")
        print(f"  Previous R² = {old_r2:.4f}")
        
        T_max = get_tmax_maximum(tau_c, B_rms, gamma_e)
        dt = get_adaptive_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, N_traj = {N_traj_improved}")
        print(f"  ⏱️  Estimated: ~40-60 minutes (enhanced parameters)...")
        
        params = {
            'gamma_e': gamma_e,
            'B_rms': B_rms,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj_improved,
            'seed': 42 + i + 40000,
            'compute_bootstrap': True,
        }
        
        try:
            start = time.time()
            result = run_simulation_single(tau_c, params=params, verbose=True)
            elapsed = time.time() - start
            
            fit_result = result.get('fit_result', {})
            T2 = fit_result.get('T2', np.nan)
            R2 = fit_result.get('R2', np.nan)
            T2_ci = result.get('T2_ci', None)
            
            if T2_ci is not None:
                T2_lower, T2_upper = T2_ci
            else:
                T2_lower = T2_upper = np.nan
            
            df_fid.loc[idx, 'T2'] = T2
            df_fid.loc[idx, 'T2_error'] = fit_result.get('T2_error', np.nan)
            df_fid.loc[idx, 'T2_lower'] = T2_lower
            df_fid.loc[idx, 'T2_upper'] = T2_upper
            df_fid.loc[idx, 'R2'] = R2
            
            improvement = R2 - old_r2
            print(f"  ✅ R² = {R2:.4f} (improvement: {improvement:+.4f}, time: {elapsed/60:.1f} min)")
            
            # Save after each point
            df_fid.to_csv(fid_file, index=False)
            
            # Progress estimate
            elapsed_total = time.time() - start_total
            avg_time = elapsed_total / (i + 1)
            remaining = avg_time * (len(poor_fit) - i - 1)
            print(f"  📊 Progress: {i+1}/{len(poor_fit)}, Remaining: ~{remaining/3600:.1f} hours")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed_total = time.time() - start_total
    hours = int(elapsed_total // 3600)
    minutes = int((elapsed_total % 3600) // 60)
    
    print(f"\n{'='*80}")
    print(f"✅ Force improvement completed!")
    print(f"Total time: {hours}h {minutes}m")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

