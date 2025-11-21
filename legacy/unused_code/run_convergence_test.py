#!/usr/bin/env python3
"""
Convergence Tests for Simulation Parameters

Tests:
1. N_traj convergence (500, 1000, 2000)
2. dt convergence (optional)
3. T_sim adequacy (optional)

This addresses the critical question: "How do you know N=2000 is enough?"
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from spin_decoherence.simulation.fid import run_simulation_single
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11  # rad/(s·T) - Electron gyromagnetic ratio for Si:P

# Load material parameters
try:
    with open('profiles.yaml', 'r') as f:
        profiles = yaml.safe_load(f)
    # Try different possible key names
    if 'Si_P' in profiles:
        si_p_profile = profiles['Si_P']
    elif 'Si:P' in profiles:
        si_p_profile = profiles['Si:P']
    else:
        # Use default value
        si_p_profile = None
        B_rms = 0.05e-3  # T (default for Si:P)
    
    if si_p_profile is not None:
        B_rms = si_p_profile['OU']['B_rms']  # T
except:
    # Default values if profiles.yaml not available
    B_rms = 0.05e-3  # T (default for Si:P)

# Representative tau_c values for testing
# Use one in each regime: MN, Crossover, QS
tau_c_test = [1e-8, 1e-7, 1e-5]  # 10 ns (MN), 100 ns (Crossover), 10 μs (QS)

def run_convergence_test_N_traj(tau_c, N_traj_list=[500, 1000, 2000], verbose=True):
    """
    Test N_traj convergence for a given tau_c.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    N_traj_list : list
        List of N_traj values to test
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : dict
        Dictionary with N_traj, T2, T2_error, T2_ci for each value
    """
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
            
            if T2_ci is not None:
                T2_lower = T2_ci[0]
                T2_upper = T2_ci[1]
                ci_width = (T2_upper - T2_lower) / T2 * 100 if T2 > 0 else np.nan
            else:
                T2_lower = np.nan
                T2_upper = np.nan
                ci_width = np.nan
            
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
                if T2_ci is not None:
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

def run_convergence_test_dt(tau_c, N_traj=2000, dt_factors=[0.5, 1.0, 2.0], verbose=True):
    """
    Test dt convergence for a given tau_c.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    N_traj : int
        Number of trajectories (fixed)
    dt_factors : list
        Factors to multiply base dt by
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : DataFrame
        Results for each dt value
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing dt convergence for τc = {tau_c*1e9:.1f} ns")
        print(f"{'='*70}")
    
    # Base dt
    xi = gamma_e * B_rms * tau_c
    if xi < 0.1:
        dt_base = tau_c / 100
    else:
        dt_base = tau_c / 50
    
    from spin_decoherence.simulation.engine import estimate_characteristic_T2
    T2_est = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
    
    if xi < 0.1:
        T_max = max(5 * T2_est, 10 * tau_c)
    elif xi < 3:
        T_max = max(10 * T2_est, 20 * tau_c)
    else:
        T_max = 30 * T2_est
    
    results = []
    
    for dt_factor in dt_factors:
        dt = dt_base * dt_factor
        
        if verbose:
            print(f"\n[dt = {dt*1e9:.3f} ns (factor: {dt_factor}x)] Running simulation...")
        
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
            R2 = fit_result.get('R2', np.nan)
            
            results.append({
                'dt_factor': dt_factor,
                'dt': dt,
                'T2': T2,
                'T2_error': T2_error,
                'R2': R2,
            })
            
            if verbose:
                print(f"  T₂ = {T2*1e6:.3f} ± {T2_error*1e6:.3f} μs" if not np.isnan(T2_error) else f"  T₂ = {T2*1e6:.3f} μs")
                print(f"  R² = {R2:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Simulation failed: {e}")
            results.append({
                'dt_factor': dt_factor,
                'dt': dt,
                'T2': np.nan,
                'T2_error': np.nan,
                'R2': np.nan,
            })
    
    return pd.DataFrame(results)

def analyze_convergence(df, param_name='N_traj'):
    """
    Analyze convergence results.
    
    Parameters
    ----------
    df : DataFrame
        Results dataframe
    param_name : str
        Name of parameter being tested
        
    Returns
    -------
    analysis : dict
        Convergence analysis results
    """
    valid_df = df[df['T2'].notna()].copy()
    
    if len(valid_df) < 2:
        return {'converged': False, 'reason': 'Insufficient data'}
    
    # Check if T2 values are within uncertainty
    T2_values = valid_df['T2'].values
    T2_errors = valid_df['T2_error'].values
    
    # Compare consecutive values
    converged = True
    max_deviation = 0.0
    
    for i in range(len(T2_values) - 1):
        T2_1 = T2_values[i]
        T2_2 = T2_values[i + 1]
        err_1 = T2_errors[i] if not np.isnan(T2_errors[i]) else T2_1 * 0.05
        err_2 = T2_errors[i + 1] if not np.isnan(T2_errors[i + 1]) else T2_2 * 0.05
        
        # Combined uncertainty
        combined_err = np.sqrt(err_1**2 + err_2**2)
        deviation = abs(T2_2 - T2_1) / T2_1 * 100 if T2_1 > 0 else 0
        
        if deviation > max_deviation:
            max_deviation = deviation
        
        # Check if within 2σ
        if abs(T2_2 - T2_1) > 2 * combined_err:
            converged = False
    
    # Check if CI width is decreasing (for N_traj test)
    if param_name == 'N_traj' and 'ci_width_pct' in valid_df.columns:
        ci_widths = valid_df['ci_width_pct'].values
        ci_widths_valid = ci_widths[~np.isnan(ci_widths)]
        if len(ci_widths_valid) > 1:
            ci_decreasing = np.all(np.diff(ci_widths_valid) <= 0) or ci_widths_valid[-1] < ci_widths_valid[0] * 0.8
        else:
            ci_decreasing = False
    else:
        ci_decreasing = None
    
    return {
        'converged': converged,
        'max_deviation_pct': max_deviation,
        'final_T2': T2_values[-1] if len(T2_values) > 0 else np.nan,
        'final_error': T2_errors[-1] if len(T2_errors) > 0 else np.nan,
        'ci_decreasing': ci_decreasing,
    }

def main():
    print("="*80)
    print("Convergence Tests for Simulation Parameters")
    print("="*80)
    print("\nThis addresses the critical question:")
    print('"How do you know N=2000 is enough?"')
    print("="*80)
    
    output_dir = Path("results_comparison")
    output_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    # 1. N_traj convergence test (CRITICAL)
    print("\n" + "="*80)
    print("1️⃣ N_traj Convergence Test (CRITICAL)")
    print("="*80)
    
    N_traj_results = {}
    
    for tau_c in tau_c_test:
        df = run_convergence_test_N_traj(tau_c, N_traj_list=[500, 1000, 2000, 5000, 10000], verbose=True)
        analysis = analyze_convergence(df, param_name='N_traj')
        
        N_traj_results[f'tau_c_{tau_c:.0e}'] = {
            'data': df,
            'analysis': analysis,
        }
        
        print(f"\n📊 Convergence Analysis (τc = {tau_c*1e9:.1f} ns):")
        print(f"   Converged: {analysis['converged']}")
        print(f"   Max deviation: {analysis['max_deviation_pct']:.2f}%")
        if analysis['converged']:
            print(f"   ✅ N=2000 is sufficient (converged within uncertainty)")
        else:
            print(f"   ⚠️  May need more trajectories")
        if analysis.get('ci_decreasing') is not None:
            if analysis['ci_decreasing']:
                print(f"   ✅ CI width decreasing (statistical error improving)")
            else:
                print(f"   ⚠️  CI width not clearly decreasing")
    
    all_results['N_traj'] = N_traj_results
    
    # Save N_traj results
    for key, value in N_traj_results.items():
        tau_c_str = key.replace('tau_c_', '').replace('+', '')
        filename = output_dir / f"convergence_N_traj_{tau_c_str}.csv"
        value['data'].to_csv(filename, index=False)
        print(f"\n✅ Saved: {filename}")
    
    # 2. dt convergence test (OPTIONAL - if time permits)
    print("\n" + "="*80)
    print("2️⃣ dt Convergence Test (OPTIONAL)")
    print("="*80)
    print("Skipping for now (can run later if time permits)")
    
    # Save summary
    summary = {
        'N_traj_convergence': {
            key: {
                'converged': val['analysis']['converged'],
                'max_deviation_pct': val['analysis']['max_deviation_pct'],
                'final_T2_us': val['analysis']['final_T2'] * 1e6 if not np.isnan(val['analysis']['final_T2']) else np.nan,
            }
            for key, val in N_traj_results.items()
        }
    }
    
    summary_file = output_dir / "convergence_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Convergence Test Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. N_traj Convergence Test\n")
        f.write("-"*80 + "\n")
        for key, val in N_traj_results.items():
            tau_c_val = float(key.replace('tau_c_', '').replace('e', 'e'))
            f.write(f"\nτc = {tau_c_val*1e9:.1f} ns:\n")
            f.write(f"  Converged: {val['analysis']['converged']}\n")
            f.write(f"  Max deviation: {val['analysis']['max_deviation_pct']:.2f}%\n")
            f.write(f"  Final T₂: {val['analysis']['final_T2']*1e6:.3f} μs\n")
            if val['analysis']['converged']:
                f.write(f"  ✅ Conclusion: N=2000 is sufficient\n")
            else:
                f.write(f"  ⚠️  Conclusion: May need more trajectories\n")
    
    print(f"\n✅ Summary saved: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ Convergence tests completed!")
    print("="*80)
    print("\nKey findings:")
    for key, val in N_traj_results.items():
        tau_c_val = float(key.replace('tau_c_', '').replace('e', 'e'))
        if val['analysis']['converged']:
            print(f"  ✅ τc = {tau_c_val*1e9:.1f} ns: N=2000 is sufficient")
        else:
            print(f"  ⚠️  τc = {tau_c_val*1e9:.1f} ns: May need more trajectories")
    print("="*80)

if __name__ == '__main__':
    main()
