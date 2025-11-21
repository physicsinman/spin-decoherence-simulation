#!/usr/bin/env python3
"""
Improved Convergence Tests with Enhanced Simulation Parameters

Improvements:
1. Reasonable N_traj values: [500, 1000, 2000, 3000, 5000] (optimized for memory)
2. Different seeds for each N_traj (avoid identical results)
3. Adaptive T_max with memory constraints
4. Numerical stability checks (dt < tau_c/5)
5. Selective bootstrap (disabled for large memory cases)
6. Analytical error fallback for degenerate CI
7. Smaller tau_c values for better convergence visibility
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
    # Handle nested structure: profiles['materials']['Si_P']
    if 'materials' in profiles and 'Si_P' in profiles['materials']:
        si_p_profile = profiles['materials']['Si_P']
        B_rms = si_p_profile['OU']['B_rms']  # T
    elif 'Si_P' in profiles:
        si_p_profile = profiles['Si_P']
        B_rms = si_p_profile['OU']['B_rms']  # T
    elif 'Si:P' in profiles:
        si_p_profile = profiles['Si:P']
        B_rms = si_p_profile['OU']['B_rms']  # T
    else:
        B_rms = 0.05e-3  # T (default for Si:P)
except:
    B_rms = 0.05e-3  # T (default for Si:P)

# Representative tau_c values for testing
# FINAL PARAMETERS: Si-28:P with B_rms = 4.0e-9 T
# Delta_omega = gamma_e * B_rms = 1.76e11 * 4.0e-9 = 704 rad/s
# CRITICAL: For convergence test, use larger tau_c values in QS regime
# QS regime: T2 ≈ sqrt(2)/Delta_omega ≈ 2000 μs (independent of tau_c)
# Using larger tau_c values ensures:
#   1. Faster simulation (fewer steps needed)
#   2. More stable results (QS regime is well-behaved)
#   3. T_max = 10 ms is sufficient (T_max/T2 ≈ 5)
tau_c_test = [5e-6, 10e-6, 20e-6]  # 5 μs, 10 μs, 20 μs (all in QS regime)
# All values give T2 ≈ 2000 μs, T_max = 10 ms provides good decay coverage

# IMPROVED: Reasonable N_traj values (reduced to avoid memory issues)
# Removed 10000 to prevent excessive memory usage
N_traj_list = [500, 1000, 2000, 3000, 5000]

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

def run_convergence_test_N_traj_improved(tau_c, N_traj_list=N_traj_list, verbose=True):
    """
    IMPROVED convergence test with:
    - More N_traj values
    - Different seeds for each N_traj
    - Longer T_max for better fitting
    - Analytical error fallback
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing N_traj convergence for τc = {tau_c*1e9:.1f} ns")
        print(f"{'='*70}")
    
    results = []
    
    for idx, N_traj in enumerate(N_traj_list):
        if verbose:
            print(f"\n[N_traj = {N_traj}] Running simulation...")
        
        # Estimate T2 for adaptive parameters
        from spin_decoherence.simulation.engine import estimate_characteristic_T2
        T2_est = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
        
        # IMPROVED: Adaptive dt and T_max with better resolution
        # CRITICAL: QS regime needs special handling
        xi = gamma_e * B_rms * tau_c
        
        # CRITICAL: Use Si:P parameters from profiles.yaml
        # Si:P has T_max = 5 ms from profiles.yaml, use this as reference
        # But scale appropriately for different tau_c values
        
        # CRITICAL: Use Si:P profile T_max = 10 ms as maximum for all regimes
        max_T_max_sip = 10.0e-3  # 10 ms from Si:P profile
        
        if xi < 0.1:  # MN regime
            dt = tau_c / 150  # IMPROVED: Finer resolution (following Si:P profile: dt = 30 ps)
            # CRITICAL FIX: In MN regime, T2_est can be extremely large (hundreds of seconds)
            # For convergence test, we need to see decay, so use Si:P profile T_max = 10 ms
            # This may not capture full decay, but is sufficient for convergence testing
            # For actual simulations, use longer T_max if needed
            T_max = max_T_max_sip  # Use full Si:P profile T_max = 10 ms
            # Ensure minimum T_max for numerical stability
            T_max = max(T_max, 10.0 * tau_c, 1.0e-6)  # At least 10×tau_c or 1 μs
        elif xi < 3:  # Crossover
            dt = tau_c / 80  # IMPROVED: Finer resolution
            # CRITICAL FIX: In Crossover regime, T2_est can still be very large
            # Use Si:P profile T_max = 10 ms for convergence test
            T_max = max_T_max_sip  # Use full Si:P profile T_max = 10 ms
            # Ensure minimum T_max
            T_max = max(T_max, 20.0 * tau_c, 5.0e-6)  # At least 20×tau_c or 5 μs
        else:  # QS regime - CRITICAL FIX
            # QS regime: Use Si:P profile T_max = 5 ms as reference
            # T2 ≈ sqrt(2)/Delta_omega (independent of tau_c)
            Delta_omega = gamma_e * B_rms
            T2_th = np.sqrt(2) / Delta_omega  # Theoretical T2 for QS regime
            
            # Si:P profile uses T_max = 5 ms, use this as base
            # But ensure we have burn-in time: ~5*tau_c
            burnin_time = 5.0 * tau_c
            T_max_from_burnin = max(10.0 * T2_th, burnin_time)
            
            # Use Si:P profile T_max (5 ms) as maximum, but ensure minimum requirements
            T_max_sip = 5.0e-3  # 5 ms from Si:P profile
            T_max = max(T_max_from_burnin, 1.0e-6)  # At least 1 μs
            T_max = min(T_max, T_max_sip)  # Cap at Si:P profile T_max (5 ms)
            
            # Finer dt for QS regime (following Si:P profile: dt = 30 ps)
            dt = tau_c / 100  # IMPROVED: Finer resolution for QS
            # But don't go below Si:P profile dt
            dt_sip = 3.0e-11  # 30 ps from Si:P profile
            dt = max(dt, dt_sip)  # At least 30 ps
            
            if verbose:
                print(f"  [QS regime] T_max = {T_max*1e6:.2f} μs (Si:P profile: {T_max_sip*1e6:.2f} μs)")
                print(f"    (T2_th = {T2_th*1e6:.2f} μs, burn-in = {burnin_time*1e6:.2f} μs)")
        
        # CRITICAL: Memory check for all regimes - cap T_max to prevent memory errors
        # For bootstrap, we need to store all trajectories: M * N_steps * 8 bytes (float64)
        # Use Si:P profile parameters as reference
        # Si:P profile: T_max = 10 ms, dt = 30 ps
        # T_max is already set above based on regime (T2_est for MN/Crossover, T2_th for QS)
        # Just ensure it doesn't exceed Si:P profile max (already done above)
        
        # CRITICAL: dt must be < tau_c/5 for numerical stability
        dt_max_stable = tau_c / 5.0  # Maximum dt for numerical stability
        
        # IMPROVED: More conservative memory limit to ensure stability
        max_memory_gb = 4.0  # Reduced from 8.0 to be more conservative
        max_N_steps = int(max_memory_gb * 1024**3 / (N_traj * 8))  # Maximum steps for given memory
        
        N_steps_est = int(T_max / dt)
        if N_steps_est > max_N_steps:
            # Calculate required dt to fit memory
            dt_required = T_max / max_N_steps
            
            # Check if required dt violates stability constraint
            if dt_required > dt_max_stable:
                # Cannot increase dt, must reduce T_max
                # Calculate minimum reasonable T_max based on regime
                if xi < 0.1:  # MN regime
                    min_T_max_reasonable = max(10.0 * tau_c, 1.0e-6)
                elif xi < 3:  # Crossover
                    min_T_max_reasonable = max(20.0 * tau_c, 5.0e-6)
                else:  # QS regime
                    min_T_max_reasonable = max(5.0 * T2_th, 1.0e-6)
                
                T_max_new = max(min_T_max_reasonable, max_N_steps * dt_max_stable)
                if T_max_new < T_max:
                    T_max = T_max_new
                    if verbose:
                        print(f"  [MEMORY] T_max reduced to {T_max*1e6:.2f} μs (memory limit: {max_memory_gb} GB, dt stability: {dt_max_stable*1e9:.2f} ns)")
                else:
                    # Cannot reduce T_max further - skip bootstrap for large N_traj
                    if verbose:
                        print(f"  [WARNING] Memory limit reached, T_max = {T_max*1e6:.2f} μs (min: {min_T_max_reasonable*1e6:.2f} μs)")
                        print(f"  [WARNING] Using dt = {dt_max_stable*1e9:.2f} ns (stability limit), bootstrap disabled for N_traj={N_traj}")
                    dt = dt_max_stable
                    # Will disable bootstrap below
            else:
                # Can safely increase dt
                dt = dt_required
                if verbose:
                    print(f"  [MEMORY] dt increased to {dt*1e9:.2f} ns to fit memory (limit: {max_memory_gb} GB)")
        
        # Ensure dt doesn't exceed stability limit
        if dt > dt_max_stable:
            dt = dt_max_stable
            if verbose:
                print(f"  [STABILITY] dt capped at {dt*1e9:.2f} ns (tau_c/5 = {dt_max_stable*1e9:.2f} ns)")
        
        # T_max is already set appropriately above based on regime
        # Just ensure it doesn't exceed Si:P profile maximum (already done above)
        
        # Additional cap: Use Si:P profile T_max (10 ms) as maximum
        # This is already done above, but ensure it's applied
        T_max = min(T_max, max_T_max_sip)
        
        # Check if we need to disable bootstrap due to memory
        N_steps_final = int(T_max / dt)
        memory_gb_final = (N_steps_final * N_traj * 8) / (1024**3)
        use_bootstrap = memory_gb_final <= 4.0  # Only use bootstrap if memory < 4 GB
        
        # IMPROVED: Different seed for each N_traj to avoid identical results
        seed = 42 + idx * 1000 + int(N_traj)
        
        # Memory check is already done above for all regimes
        
        params = {
            'gamma_e': gamma_e,
            'B_rms': B_rms,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj,
            'seed': seed,  # IMPROVED: Different seed
            'compute_bootstrap': use_bootstrap,  # Disable bootstrap if memory too large
        }
        
        if verbose and not use_bootstrap:
            print(f"  [INFO] Bootstrap disabled (memory estimate: {memory_gb_final:.1f} GB)")
        
        try:
            result = run_simulation_single(tau_c, params=params, verbose=verbose)
            
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

def analyze_convergence(df, param_name='N_traj'):
    """
    Analyze convergence results.
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
    print("IMPROVED Convergence Tests for Simulation Parameters")
    print("="*80)
    print("\nImprovements:")
    print("  - Reasonable N_traj values: [500, 1000, 2000, 3000, 5000] (optimized for memory)")
    print("  - Different seeds for each N_traj (avoid identical results)")
    print("  - Adaptive T_max with memory constraints")
    print("  - Numerical stability checks (dt < tau_c/5)")
    print("  - Selective bootstrap (disabled for large memory cases)")
    print("  - Analytical error fallback for degenerate CI")
    print("="*80)
    
    output_dir = Path("results_comparison")
    output_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    # 1. N_traj convergence test (CRITICAL)
    print("\n" + "="*80)
    print("1️⃣ N_traj Convergence Test (IMPROVED)")
    print("="*80)
    
    N_traj_results = {}
    
    for tau_c in tau_c_test:
        df = run_convergence_test_N_traj_improved(tau_c, N_traj_list=N_traj_list, verbose=True)
        analysis = analyze_convergence(df, param_name='N_traj')
        
        N_traj_results[f'tau_c_{tau_c:.0e}'] = {
            'data': df,
            'analysis': analysis,
        }
        
        print(f"\n📊 Convergence Analysis (τc = {tau_c*1e9:.1f} ns):")
        if 'converged' in analysis:
            print(f"   Converged: {analysis['converged']}")
            if 'max_deviation_pct' in analysis:
                print(f"   Max deviation: {analysis['max_deviation_pct']:.2f}%")
            if analysis['converged']:
                print(f"   ✅ N=2000 is sufficient (converged within uncertainty)")
            else:
                print(f"   ⚠️  May need more trajectories")
        else:
            print(f"   ⚠️  {analysis.get('reason', 'Insufficient data')}")
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
    
    # Save summary
    summary_file = output_dir / "convergence_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("IMPROVED Convergence Test Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. N_traj Convergence Test\n")
        f.write("-"*80 + "\n")
        for key, val in N_traj_results.items():
            tau_c_val = float(key.replace('tau_c_', '').replace('e', 'e'))
        f.write(f"\nτc = {tau_c_val*1e9:.1f} ns:\n")
        f.write(f"  Converged: {val['analysis']['converged']}\n")
        if 'max_deviation_pct' in val['analysis']:
            f.write(f"  Max deviation: {val['analysis']['max_deviation_pct']:.2f}%\n")
        if 'final_T2' in val['analysis'] and not np.isnan(val['analysis']['final_T2']):
            f.write(f"  Final T₂: {val['analysis']['final_T2']*1e6:.3f} μs\n")
            if val['analysis']['converged']:
                f.write(f"  ✅ Conclusion: N=2000 is sufficient\n")
            else:
                f.write(f"  ⚠️  Conclusion: May need more trajectories\n")
    
    print(f"\n✅ Summary saved: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ IMPROVED Convergence tests completed!")
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

