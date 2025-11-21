#!/usr/bin/env python3
"""
Re-run problematic echo points for publication-quality results
Focuses on crossover regime points with physical inconsistencies
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.echo import run_simulation_with_hahn_echo
from spin_decoherence.analysis.bootstrap import bootstrap_T2
from spin_decoherence.analysis.fitting import fit_coherence_decay_with_offset
from spin_decoherence.simulation.engine import get_dimensionless_tau_range

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (0.05 mT)
N_traj = 2000              # Monte Carlo trajectories

# Problematic tau_c values (crossover regime)
problem_tau_cs = [0.257e-6, 0.300e-6]  # s (from analysis)

def get_adaptive_dt(tau_c, T_max=None, max_memory_gb=8.0):
    """Adaptive timestep with memory limit."""
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
    """Calculate simulation duration - IMPROVED for crossover."""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est
    elif xi > 3:  # QS regime
        T2_est = 1.0 / (gamma_e * B_rms)
        if xi < 10:
            multiplier = 100
        elif xi < 50:
            multiplier = 150
        else:
            multiplier = 200
        T_max_from_T2 = multiplier * T2_est
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        return min(T_max_final, 100e-3)
    else:  # Crossover - IMPROVED
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 25 * T2_est  # Increased from 20 to 25 for better accuracy

def main():
    print("="*80)
    print("Re-run Problematic Echo Points (Publication Quality)")
    print("="*80)
    print(f"\nProblematic tau_c values: {[tau_c*1e6 for tau_c in problem_tau_cs]} μs")
    print(f"N_traj = {N_traj}")
    print(f"\nImprovements:")
    print(f"  - T_max increased for crossover regime (20× → 25×)")
    print(f"  - T_max_echo increased (2.5× → 3.5×)")
    print(f"  - More conservative fitting parameters")
    print("\nStarting simulation...\n")
    
    output_dir = Path("results_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Load existing echo data
    echo_file = output_dir / "t2_echo_vs_tau_c.csv"
    if echo_file.exists():
        df_echo = pd.read_csv(echo_file)
    else:
        df_echo = pd.DataFrame(columns=['tau_c', 'T2_echo', 'T2_echo_lower', 'T2_echo_upper', 'R2_echo', 'xi'])
    
    results_data = []
    
    for i, tau_c in enumerate(problem_tau_cs):
        print(f"\n[{i+1}/{len(problem_tau_cs)}] Processing τ_c = {tau_c*1e6:.3f} μs")
        
        # Calculate adaptive parameters
        T_max = get_tmax(tau_c, B_rms, gamma_e)
        dt = get_adaptive_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        xi = gamma_e * B_rms * tau_c
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.3f}")
        
        # IMPROVED: T_max_echo for crossover regime
        if xi > 3:  # QS regime
            T_max_echo = T_max * 3.0
        elif xi < 0.3:  # MN regime
            T_max_echo = T_max * 2.5
        else:  # Crossover - IMPROVED
            T_max_echo = T_max * 3.5  # Increased from 2.5 to 3.5
        
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'T_max_echo': T_max_echo,
            'M': N_traj,
            'seed': 42 + i + 1000,  # Different seed
            'output_dir': str(output_dir),
            'compute_bootstrap': True,
            'save_delta_B_sample': False,
        }
        
        # Generate tau_list with adaptive upsilon_max
        if xi > 3:  # QS regime
            upsilon_max = min(0.4 * T_max_echo / tau_c, 5.0)
        else:
            upsilon_max = 0.8
        
        tau_list = get_dimensionless_tau_range(
            tau_c, n_points=60, upsilon_min=0.05, upsilon_max=upsilon_max,  # More points
            dt=dt, T_max=T_max_echo
        )
        
        # Run simulation
        try:
            result = run_simulation_with_hahn_echo(tau_c, params=params, tau_list=tau_list, verbose=True)
        except Exception as e:
            print(f"  ❌ Simulation failed: {e}")
            continue
        
        # Extract echo T2
        fit_result_echo = result.get('fit_result_echo')
        if fit_result_echo is not None:
            T2_echo = fit_result_echo.get('T2', np.nan)
            R2_echo = fit_result_echo.get('R2', np.nan)
            
            # Extract CI
            T2_echo_ci = result.get('T2_echo_ci', None)
            if T2_echo_ci is None:
                # Try to compute from bootstrap samples
                E_echo_abs_all = result.get('E_echo_abs_all', None)
                if E_echo_abs_all is not None and len(E_echo_abs_all) > 0:
                    try:
                        tau_echo = np.array(result.get('tau_echo', []))
                        if len(tau_echo) > 0:
                            T2_mean, T2_echo_ci, _ = bootstrap_T2(
                                tau_echo, E_echo_abs_all, E_se=result.get('E_echo_se'),
                                B=500, verbose=False,
                                tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
                            )
                    except:
                        pass
            
            T2_echo_lower = T2_echo_ci[0] if T2_echo_ci is not None else np.nan
            T2_echo_upper = T2_echo_ci[1] if T2_echo_ci is not None else np.nan
            
            # Update or add to dataframe
            mask = df_echo['tau_c'] == tau_c
            if mask.sum() > 0:
                # Update existing
                df_echo.loc[mask, 'T2_echo'] = T2_echo
                df_echo.loc[mask, 'T2_echo_lower'] = T2_echo_lower
                df_echo.loc[mask, 'T2_echo_upper'] = T2_echo_upper
                df_echo.loc[mask, 'R2_echo'] = R2_echo
                print(f"  ✅ Updated: T2_echo = {T2_echo*1e6:.3f} μs (R² = {R2_echo:.4f})")
            else:
                # Add new
                new_row = pd.DataFrame([{
                    'tau_c': tau_c,
                    'T2_echo': T2_echo,
                    'T2_echo_lower': T2_echo_lower,
                    'T2_echo_upper': T2_echo_upper,
                    'R2_echo': R2_echo,
                    'xi': xi
                }])
                df_echo = pd.concat([df_echo, new_row], ignore_index=True)
                print(f"  ✅ Added: T2_echo = {T2_echo*1e6:.3f} μs (R² = {R2_echo:.4f})")
            
            if T2_echo_ci is not None:
                print(f"  95% CI: [{T2_echo_lower*1e6:.3f}, {T2_echo_upper*1e6:.3f}] μs")
        else:
            print(f"  ⚠️  Echo fit failed!")
    
    # Save updated dataframe
    df_echo = df_echo.sort_values('tau_c')
    df_echo.to_csv(echo_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {echo_file}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

