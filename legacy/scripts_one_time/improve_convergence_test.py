#!/usr/bin/env python3
"""
Improve Convergence Test: Add N=5000 and use analytical error for CI width
Post-meeting improvements
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11  # rad/(s·T)
B_rms = 0.05e-3    # T

# Load material parameters
try:
    import yaml
    with open('profiles.yaml', 'r') as f:
        profiles = yaml.safe_load(f)
    if 'Si_P' in profiles:
        si_p_profile = profiles['Si_P']
    elif 'Si:P' in profiles:
        si_p_profile = profiles['Si:P']
    else:
        si_p_profile = None
    
    if si_p_profile is not None:
        B_rms = si_p_profile['OU']['B_rms']
except:
    pass

# Representative tau_c values
tau_c_test = [1e-8, 1e-7]  # 10 ns (MN), 100 ns (crossover)

# IMPROVED: Add N=5000
N_traj_list = [500, 1000, 2000, 5000]

def calculate_analytical_error(T2, N_traj, tau_c, gamma_e, B_rms):
    """
    Calculate analytical error estimate for T2.
    For bootstrap CI that is degenerate, use analytical estimate.
    """
    # Statistical error: σ_T2 ≈ T2 / √N_traj (rough estimate)
    sigma_stat = T2 / np.sqrt(N_traj)
    
    # Additional uncertainty from fitting (assume 1% for good fits)
    sigma_fit = T2 * 0.01
    
    # Combined error
    sigma_total = np.sqrt(sigma_stat**2 + sigma_fit**2)
    
    # 95% CI (assuming normal distribution)
    ci_lower = T2 - 1.96 * sigma_total
    ci_upper = T2 + 1.96 * sigma_total
    
    return (ci_lower, ci_upper), sigma_total

def run_convergence_test_N_traj_improved(tau_c, N_traj_list=[500, 1000, 2000, 5000], verbose=True):
    """Improved convergence test with N=5000 and analytical error fallback."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing N_traj convergence for τc = {tau_c*1e9:.1f} ns")
        print(f"{'='*70}")
    
    results = []
    
    for N_traj in N_traj_list:
        if verbose:
            print(f"\n[N_traj = {N_traj}] Running simulation...")
        
        # Estimate T2 for adaptive parameters
        from spin_decoherence.simulation.engine import estimate_characteristic_T2
        T2_est = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
        
        # Adaptive dt and T_max
        xi = gamma_e * B_rms * tau_c
        
        if xi < 0.1:  # MN regime
            dt = tau_c / 100
            T_max = max(5 * T2_est, 10 * tau_c)
        elif xi < 3:  # Crossover
            dt = tau_c / 50
            T_max = max(10 * T2_est, 20 * tau_c)
        else:  # QS regime
            dt = tau_c / 50
            T_max = 30 * T2_est
        
        params = {
            'gamma_e': gamma_e,
            'B_rms': B_rms,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj,
            'seed': 42,
            'compute_bootstrap': True,
        }
        
        try:
            result = run_simulation_single(tau_c, params=params, verbose=False)
            
            fit_result = result.get('fit_result', {})
            T2 = fit_result.get('T2', np.nan)
            T2_error = fit_result.get('T2_error', np.nan)
            T2_ci = result.get('T2_ci', None)
            R2 = fit_result.get('R2', np.nan)
            
            # IMPROVED: Use analytical error if bootstrap CI is degenerate
            if T2_ci is not None:
                ci_width = (T2_ci[1] - T2_ci[0]) / T2 * 100 if T2 > 0 else np.nan
                # Check if CI is degenerate (too narrow)
                if ci_width < 0.01:  # Less than 0.01% width
                    if verbose:
                        print(f"  Bootstrap CI degenerate (width={ci_width:.6f}%), using analytical error")
                    T2_ci, sigma = calculate_analytical_error(T2, N_traj, tau_c, gamma_e, B_rms)
                    ci_width = (T2_ci[1] - T2_ci[0]) / T2 * 100 if T2 > 0 else np.nan
            else:
                # No bootstrap CI, use analytical
                T2_ci, sigma = calculate_analytical_error(T2, N_traj, tau_c, gamma_e, B_rms)
                ci_width = (T2_ci[1] - T2_ci[0]) / T2 * 100 if T2 > 0 else np.nan
            
            T2_lower = T2_ci[0]
            T2_upper = T2_ci[1]
            
            results.append({
                'N_traj': N_traj,
                'T2': T2,
                'T2_error': T2_error,
                'T2_lower': T2_lower,
                'T2_upper': T2_upper,
                'ci_width_pct': ci_width,
                'R2': R2,
            })
            
            if verbose:
                print(f"  T₂ = {T2*1e6:.3f} ± {T2_error*1e6:.3f} μs" if not np.isnan(T2_error) else f"  T₂ = {T2*1e6:.3f} μs")
                print(f"  CI: [{T2_lower*1e6:.3f}, {T2_upper*1e6:.3f}] μs (width: {ci_width:.2f}%)")
                print(f"  R² = {R2:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Simulation failed: {e}")
            results.append({
                'N_traj': N_traj,
                'T2': np.nan,
                'T2_error': np.nan,
                'T2_lower': np.nan,
                'T2_upper': np.nan,
                'ci_width_pct': np.nan,
                'R2': np.nan,
            })
    
    return pd.DataFrame(results)

def main():
    print("="*80)
    print("Improved Convergence Tests (Post-Meeting)")
    print("="*80)
    print("\nImprovements:")
    print(f"  - Added N_traj = 5000")
    print(f"  - Analytical error fallback for degenerate CI")
    print("="*80)
    
    output_dir = Path("results_comparison")
    output_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    for tau_c in tau_c_test:
        df = run_convergence_test_N_traj_improved(tau_c, N_traj_list=N_traj_list, verbose=True)
        
        tau_c_str = f"{tau_c:.0e}".replace('+', '')
        key = f'tau_c_{tau_c_str}'
        all_results[key] = df
        
        # Save
        filename = output_dir / f"convergence_N_traj_{tau_c_str}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Saved: {filename}")
    
    print("\n" + "="*80)
    print("✅ Improved convergence tests completed!")
    print("="*80)

if __name__ == '__main__':
    main()

