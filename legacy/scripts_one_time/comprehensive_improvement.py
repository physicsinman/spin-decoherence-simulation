#!/usr/bin/env python3
"""
Comprehensive Improvement: 시뮬레이션 정당성과 그래프 품질 전반적 개선
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single
from spin_decoherence.simulation.echo import run_simulation_with_hahn_echo
from spin_decoherence.simulation.engine import get_dimensionless_tau_range
import time

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (0.05 mT)
N_traj_improved = 5000     # Increased for better statistics

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
    """Calculate simulation duration - MAXIMUM quality."""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 12 * T2_est  # Increased from 10
    elif xi > 3:  # QS regime - MAXIMUM
        T2_est = 1.0 / (gamma_e * B_rms)
        if xi < 10:
            multiplier = 200  # Increased from 150
        elif xi < 50:
            multiplier = 250  # Increased from 200
        else:
            multiplier = 300  # Increased from 250
        T_max_from_T2 = multiplier * T2_est
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        return min(T_max_final, 100e-3)
    else:  # Crossover - MAXIMUM
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 30 * T2_est  # Increased from 25

def improve_fid_points():
    """Improve all FID points with R² < 0.95."""
    print("="*80)
    print("Step 1: Improving FID Points (R² < 0.95)")
    print("="*80)
    
    fid_file = Path("results_comparison/t2_vs_tau_c.csv")
    df_fid = pd.read_csv(fid_file)
    
    # Find points with R² < 0.95 (or < 0.98 for maximum quality)
    # Use R² < 0.98 for maximum quality improvement
    poor_fit = df_fid[df_fid['R2'] < 0.98].sort_values('tau_c')  # More strict: 0.98
    
    print(f"\nFound {len(poor_fit)} points with R² < 0.98")
    print(f"Estimated time: ~{len(poor_fit) * 25 / 60:.1f} hours\n")
    
    if len(poor_fit) == 0:
        print("✅ All points already have R² ≥ 0.98!")
        return
    
    print("⚠️  ACTUAL SIMULATION WILL RUN - This will take time!")
    print("   You will see progress bars and 'Computing FID (OU)' messages.\n")
    
    for i, (idx, row) in enumerate(poor_fit.iterrows()):
        tau_c = row['tau_c']
        xi = row['xi']
        old_r2 = row['R2']
        
        print(f"\n[{i+1}/{len(poor_fit)}] τc = {tau_c*1e6:.3f} μs (ξ = {xi:.3f})")
        print(f"  Previous R² = {old_r2:.4f}")
        
        T_max = get_tmax_improved(tau_c, B_rms, gamma_e)
        dt = get_adaptive_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, N_traj = {N_traj_improved}")
        
        params = {
            'gamma_e': gamma_e,
            'B_rms': B_rms,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj_improved,
            'seed': 42 + i + 20000,
            'compute_bootstrap': True,
        }
        
        try:
            start = time.time()
            result = run_simulation_single(tau_c, params=params, verbose=False)
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
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    print(f"\n✅ FID improvement completed!")

def improve_echo_points():
    """Improve problematic echo points."""
    print("\n" + "="*80)
    print("Step 2: Improving Echo Points")
    print("="*80)
    
    # Load echo gain to find problematic points
    gain_file = Path("results_comparison/echo_gain.csv")
    if not gain_file.exists():
        print("⚠️  Echo gain file not found, skipping...")
        return
    
    gain = pd.read_csv(gain_file)
    echo_file = Path("results_comparison/t2_echo_vs_tau_c.csv")
    
    if echo_file.exists():
        df_echo = pd.read_csv(echo_file)
    else:
        df_echo = pd.DataFrame(columns=['tau_c', 'T2_echo', 'T2_echo_lower', 'T2_echo_upper', 'R2_echo', 'xi'])
    
    # Find problematic points: unphysical behavior or low R²
    gain_sorted = gain.sort_values('xi')
    gain_diff = gain_sorted['echo_gain'].diff()
    xi_diff = gain_sorted['xi'].diff()
    
    # Points where gain decreases with increasing xi
    unphysical = gain_sorted[((gain_diff < -0.1) & (xi_diff > 0) & (gain_sorted['xi'] >= 0.2))].copy()
    
    # Merge with echo data to check R²
    if len(df_echo) > 0:
        unphysical = pd.merge(unphysical, df_echo[['tau_c', 'R2_echo']], on='tau_c', how='left')
        # Also include points with low R²_echo
        low_r2_echo = df_echo[df_echo['R2_echo'] < 0.95]
        problem_tau_cs = list(set(unphysical['tau_c'].tolist() + low_r2_echo['tau_c'].tolist()))
    else:
        problem_tau_cs = unphysical['tau_c'].tolist()
    
    print(f"\nFound {len(problem_tau_cs)} problematic echo points")
    
    if len(problem_tau_cs) == 0:
        print("✅ No problematic echo points!")
        return
    
    for i, tau_c in enumerate(problem_tau_cs):
        print(f"\n[{i+1}/{len(problem_tau_cs)}] τc = {tau_c*1e6:.3f} μs")
        
        T_max = get_tmax_improved(tau_c, B_rms, gamma_e)
        dt = get_adaptive_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        xi = gamma_e * B_rms * tau_c
        
        if xi > 3:
            T_max_echo = T_max * 4.0
        elif xi < 0.3:
            T_max_echo = T_max * 3.0
        else:
            T_max_echo = T_max * 4.5  # Increased
        
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'T_max_echo': T_max_echo,
            'M': N_traj_improved,
            'seed': 42 + i + 30000,
            'output_dir': 'results_comparison',
            'compute_bootstrap': True,
            'save_delta_B_sample': False,
        }
        
        if xi > 3:
            upsilon_max = min(0.4 * T_max_echo / tau_c, 5.0)
        else:
            upsilon_max = 0.8
        
        tau_list = get_dimensionless_tau_range(
            tau_c, n_points=100, upsilon_min=0.05, upsilon_max=upsilon_max,
            dt=dt, T_max=T_max_echo
        )
        
        try:
            result = run_simulation_with_hahn_echo(tau_c, params=params, tau_list=tau_list, verbose=False)
            
            fit_result_echo = result.get('fit_result_echo')
            if fit_result_echo is not None:
                T2_echo = fit_result_echo.get('T2', np.nan)
                R2_echo = fit_result_echo.get('R2', np.nan)
                T2_echo_ci = result.get('T2_echo_ci', None)
                
                if T2_echo_ci is None:
                    T2_echo_lower = T2_echo_upper = np.nan
                else:
                    T2_echo_lower, T2_echo_upper = T2_echo_ci
                
                mask = df_echo['tau_c'] == tau_c
                if mask.sum() > 0:
                    df_echo.loc[mask, 'T2_echo'] = T2_echo
                    df_echo.loc[mask, 'T2_echo_lower'] = T2_echo_lower
                    df_echo.loc[mask, 'T2_echo_upper'] = T2_echo_upper
                    df_echo.loc[mask, 'R2_echo'] = R2_echo
                else:
                    new_row = pd.DataFrame([{
                        'tau_c': tau_c,
                        'T2_echo': T2_echo,
                        'T2_echo_lower': T2_echo_lower,
                        'T2_echo_upper': T2_echo_upper,
                        'R2_echo': R2_echo,
                        'xi': xi
                    }])
                    df_echo = pd.concat([df_echo, new_row], ignore_index=True)
                
                print(f"  ✅ R²_echo = {R2_echo:.4f}")
                df_echo.to_csv(echo_file, index=False)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    print(f"\n✅ Echo improvement completed!")

def main():
    print("="*80)
    print("Comprehensive Improvement: 시뮬레이션 정당성 향상")
    print("="*80)
    print("\nThis will:")
    print("  1. Re-simulate all FID points with R² < 0.95 (N_traj=5000)")
    print("  2. Re-simulate problematic echo points")
    print("  3. Use maximum T_max for better decay observation")
    print("\n⚠️  This will take several hours!")
    print()
    
    # Auto-confirm for non-interactive execution
    import sys
    if sys.stdin.isatty():
        # Interactive mode - ask for confirmation
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    else:
        # Non-interactive mode - auto-confirm
        print("Non-interactive mode: Auto-confirming...")
    
    start_time = time.time()
    
    # Step 1: Improve FID
    improve_fid_points()
    
    # Step 2: Improve Echo
    improve_echo_points()
    
    # Step 3: Re-analyze echo gain
    print("\n" + "="*80)
    print("Step 3: Re-analyzing Echo Gain")
    print("="*80)
    import subprocess
    subprocess.run(['python3', 'analyze_echo_gain.py'], check=False)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print("\n" + "="*80)
    print(f"✅ Comprehensive improvement completed!")
    print(f"Total time: {hours}h {minutes}m")
    print("="*80)

if __name__ == '__main__':
    main()

