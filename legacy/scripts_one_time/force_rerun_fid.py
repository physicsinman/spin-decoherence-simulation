#!/usr/bin/env python3
"""
Force re-simulation for points with R² < 0.95 (more strict criteria)
실제로 시뮬레이션이 실행되는지 확인하기 위한 강제 재시뮬레이션
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (0.05 mT)
N_traj_improved = 5000     # Increased from 2000

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

def get_tmax_improved(tau_c, B_rms, gamma_e):
    """Calculate simulation duration - IMPROVED."""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est
    elif xi > 3:  # QS regime
        T2_est = 1.0 / (gamma_e * B_rms)
        if xi < 10:
            multiplier = 150
        elif xi < 50:
            multiplier = 200
        else:
            multiplier = 250
        T_max_from_T2 = multiplier * T2_est
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        return min(T_max_final, 100e-3)
    else:  # Crossover
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 25 * T2_est

def main():
    print("="*80)
    print("Force Re-simulation for R² < 0.95 (Strict Criteria)")
    print("="*80)
    print(f"\nThis will ACTUALLY run simulations (not skip)")
    print(f"N_traj = {N_traj_improved}")
    print(f"\n⚠️  This will take time! (~15-30 min per point)")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    # Load existing data
    fid_file = Path("results_comparison/t2_vs_tau_c.csv")
    if fid_file.exists():
        df_fid = pd.read_csv(fid_file)
    else:
        print("❌ FID data file not found!")
        return
    
    # Find points with R² < 0.95 (more strict)
    poor_fit = df_fid[df_fid['R2'] < 0.95].sort_values('tau_c')
    
    print(f"\nFound {len(poor_fit)} points with R² < 0.95")
    
    if len(poor_fit) == 0:
        print("✅ All points have R² ≥ 0.95!")
        return
    
    print(f"\n⚠️  This will take approximately {len(poor_fit) * 20 / 60:.1f} hours")
    print(f"Starting re-simulation...\n")
    
    # Re-simulate each point
    for i, (idx, row) in enumerate(poor_fit.iterrows()):
        tau_c = row['tau_c']
        xi = row['xi']
        old_r2 = row['R2']
        
        print(f"\n[{i+1}/{len(poor_fit)}] Re-simulating τc = {tau_c*1e6:.3f} μs (ξ = {xi:.3f})")
        print(f"  Previous R² = {old_r2:.4f}")
        
        # Calculate improved parameters
        T_max = get_tmax_improved(tau_c, B_rms, gamma_e)
        dt = get_adaptive_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, N_traj = {N_traj_improved}")
        print(f"  ⏱️  This will take ~15-30 minutes...")
        
        params = {
            'gamma_e': gamma_e,
            'B_rms': B_rms,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj_improved,
            'seed': 42 + i + 10000,  # Different seed
            'compute_bootstrap': True,
        }
        
        try:
            import time
            start_time = time.time()
            
            result = run_simulation_single(tau_c, params=params, verbose=True)
            
            elapsed = time.time() - start_time
            print(f"  ⏱️  Elapsed time: {elapsed/60:.1f} minutes")
            
            fit_result = result.get('fit_result', {})
            T2 = fit_result.get('T2', np.nan)
            T2_error = fit_result.get('T2_error', np.nan)
            T2_ci = result.get('T2_ci', None)
            R2 = fit_result.get('R2', np.nan)
            
            if T2_ci is not None:
                T2_lower = T2_ci[0]
                T2_upper = T2_ci[1]
            else:
                T2_lower = np.nan
                T2_upper = np.nan
            
            # Update dataframe
            df_fid.loc[idx, 'T2'] = T2
            df_fid.loc[idx, 'T2_error'] = T2_error
            df_fid.loc[idx, 'T2_lower'] = T2_lower
            df_fid.loc[idx, 'T2_upper'] = T2_upper
            df_fid.loc[idx, 'R2'] = R2
            
            improvement = R2 - old_r2
            print(f"  ✅ Updated: T2 = {T2*1e6:.3f} μs, R² = {R2:.4f} (improvement: {improvement:+.4f})")
            
            # Save after each point (in case of interruption)
            df_fid.to_csv(fid_file, index=False)
            
        except Exception as e:
            print(f"  ❌ Simulation failed: {e}")
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✅ Re-simulation completed!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

