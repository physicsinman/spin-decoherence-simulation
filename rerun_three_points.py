#!/usr/bin/env python3
"""
Re-run simulation for three problematic points that had incorrect T2 values
due to regime mismatch (xi between 2.0 and 10.0).

These points were incorrectly treated as Crossover regime in simulation
but should be QS regime (xi >= 2.0).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single
from datetime import datetime

# ============================================
# SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T) - electron gyromagnetic ratio
B_rms = 0.57e-6            # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration
N_traj = 2000              # Monte Carlo trajectories per point

# Three problematic points to re-run
problematic_tau_cs = [
    5.477225575051662e-05,  # 54.77 μs, xi = 5.49
    6.977190408210102e-05,  # 69.77 μs, xi = 7.00
    8.887927898050809e-05,  # 88.88 μs, xi = 8.92
]

# Adaptive parameters (same as sim_fid_sweep.py)
def get_dt(tau_c, T_max=None, max_memory_gb=8.0):
    """Adaptive timestep selection with memory limit."""
    dt_max_stable = tau_c / 6.0  # Maximum dt for numerical stability
    dt_target = tau_c / 100
    
    if T_max is not None:
        N_steps = int(T_max / dt_target) + 1
        memory_gb = (N_steps * N_traj * 8) / (1024**3)
        
        if memory_gb > max_memory_gb:
            max_N_steps = int((max_memory_gb * 1024**3) / (N_traj * 8))
            dt_required = T_max / max_N_steps if max_N_steps > 0 else dt_target
            dt_min = tau_c / 50
            dt = max(dt_required, dt_min)
            dt = min(dt, dt_max_stable)
            return dt
    
    return min(dt_target, dt_max_stable)

def get_tmax(tau_c, B_rms, gamma_e):
    """Calculate appropriate simulation duration - FIXED for QS regime."""
    xi = gamma_e * B_rms * tau_c
    
    # These points are now correctly identified as QS regime (xi >= 2.0)
    if xi >= 2.0:  # QS regime
        T2_est = np.sqrt(2.0) / (gamma_e * B_rms)
        if xi < 10:
            multiplier = 100
        elif xi < 50:
            multiplier = 150
        else:
            multiplier = 200
        T_max_from_T2 = multiplier * T2_est
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        return min(T_max_final, 10e-3)
    else:
        # Should not happen for these points, but keep for safety
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        T_max_from_T2 = 20 * T2_est
        return min(T_max_from_T2, 10e-3)

def main():
    print("="*80)
    print("Re-run Simulation for Three Problematic Points")
    print("="*80)
    print(f"\nParameters:")
    print(f"  gamma_e = {gamma_e:.3e} rad/(s·T)")
    print(f"  B_rms = {B_rms*1e6:.2f} μT")
    print(f"  N_traj = {N_traj}")
    print(f"\nPoints to re-run:")
    for i, tau_c in enumerate(problematic_tau_cs):
        xi = gamma_e * B_rms * tau_c
        print(f"  [{i+1}] τc = {tau_c*1e6:.2f} μs, ξ = {xi:.2f} (QS regime)")
    print("\nStarting simulation...\n")
    
    # Load existing data
    data_file = Path("results/t2_vs_tau_c.csv")
    if not data_file.exists():
        print(f"❌ Error: {data_file} not found!")
        return
    
    df = pd.read_csv(data_file)
    print(f"✅ Loaded existing data: {len(df)} points")
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Storage for new results
    new_results = []
    
    for i, tau_c in enumerate(problematic_tau_cs):
        xi = gamma_e * B_rms * tau_c
        print(f"\n[{i+1}/{len(problematic_tau_cs)}] Processing τ_c = {tau_c*1e6:.3f} μs (ξ = {xi:.2f})")
        
        # Calculate adaptive parameters
        T_max = get_tmax(tau_c, B_rms, gamma_e)
        dt = get_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        
        # Enforce dt < tau_c/5 constraint
        dt_max_stable = tau_c / 6.0
        if dt > dt_max_stable:
            dt = dt_max_stable
            max_memory_gb = 8.0
            max_N_steps = int((max_memory_gb * 1024**3) / (N_traj * 8))
            T_max_adjusted = dt * max_N_steps
            if T_max_adjusted < T_max:
                T_max = T_max_adjusted
                print(f"  ⚠️  T_max reduced to {T_max*1e6:.2f} μs to satisfy dt < tau_c/5 constraint")
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs")
        
        # Verify dt constraint
        if dt >= tau_c / 5.0:
            print(f"  ❌ ERROR: dt ({dt*1e9:.2f} ns) >= tau_c/5 ({tau_c/5.0*1e9:.2f} ns)")
            print(f"  Skipping this tau_c value")
            continue
        
        # Memory check
        N_steps = int(T_max / dt) + 1
        memory_gb = (N_steps * N_traj * 8) / (1024**3)
        if memory_gb > 10:
            print(f"  ⚠️  Memory warning: {memory_gb:.1f} GB estimated")
        
        # Setup parameters
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj,
            'seed': 999 + i,  # Different seed from original
            'output_dir': str(output_dir),
            'compute_bootstrap': True,
            'B_bootstrap': 800,
            'save_delta_B_sample': False,
        }
        
        # Run simulation
        try:
            result = run_simulation_single(tau_c, params=params, verbose=True)
        except Exception as e:
            print(f"  ❌ Simulation failed: {e}")
            continue
        
        # Extract results
        fit_result = result['fit_result']
        if fit_result is not None:
            T2 = fit_result.get('T2', np.nan)
            T2_ci = result.get('T2_ci', None)
            R2 = fit_result.get('R2', np.nan)
            
            T2_lower = T2_ci[0] if T2_ci is not None else np.nan
            T2_upper = T2_ci[1] if T2_ci is not None else np.nan
            
            print(f"  ✅ T2 = {T2*1e6:.3f} μs (expected ~14.1 μs for QS regime)")
            print(f"     R² = {R2:.4f}")
            if T2_ci is not None:
                print(f"     95% CI: [{T2_lower*1e6:.3f}, {T2_upper*1e6:.3f}] μs")
            
            new_results.append({
                'tau_c': tau_c,
                'T2': T2,
                'T2_lower': T2_lower,
                'T2_upper': T2_upper,
                'R2': R2,
                'xi': xi
            })
        else:
            print(f"  ❌ Fitting failed")
    
    # Update the dataframe with new results
    if len(new_results) > 0:
        print(f"\n{'='*80}")
        print("Updating t2_vs_tau_c.csv with new results...")
        
        for new_result in new_results:
            tau_c = new_result['tau_c']
            # Find matching row (use tolerance for floating point comparison)
            mask = np.isclose(df['tau_c'], tau_c, rtol=1e-10)
            if mask.sum() > 0:
                idx = df[mask].index[0]
                print(f"  Updating τc = {tau_c*1e6:.2f} μs:")
                print(f"    Old T2: {df.loc[idx, 'T2']*1e6:.3f} μs, R² = {df.loc[idx, 'R2']:.4f}")
                print(f"    New T2: {new_result['T2']*1e6:.3f} μs, R² = {new_result['R2']:.4f}")
                
                df.loc[idx, 'T2'] = new_result['T2']
                df.loc[idx, 'T2_lower'] = new_result['T2_lower']
                df.loc[idx, 'T2_upper'] = new_result['T2_upper']
                df.loc[idx, 'R2'] = new_result['R2']
            else:
                print(f"  ⚠️  Could not find matching row for τc = {tau_c*1e6:.2f} μs")
        
        # Save updated data
        df = df.sort_values('tau_c')  # Ensure sorted
        df.to_csv(data_file, index=False)
        print(f"\n✅ Updated {data_file}")
        print(f"   {len(new_results)} points updated")
    else:
        print(f"\n❌ No results to update")
    
    print(f"\n{'='*80}")
    print("✅ Done!")
    print("="*80)

if __name__ == '__main__':
    main()

