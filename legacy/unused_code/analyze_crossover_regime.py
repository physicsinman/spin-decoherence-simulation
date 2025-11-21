#!/usr/bin/env python3
"""
Crossover Regime Analysis
Analyzes the crossover regime (0.1 < ξ < 3) where no analytical formula exists.
Highlights the slope ≈ -0.49 behavior.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * (x ** b)

def main():
    print("="*70)
    print("Crossover Regime Analysis")
    print("="*70)
    
    # Load data
    input_file = Path("results_comparison/t2_vs_tau_c.csv")
    if not input_file.exists():
        print(f"\n❌ Error: {input_file} not found!")
        print("   Please run run_fid_sweep.py first.")
        return
    
    df = pd.read_csv(input_file)
    print(f"\nLoaded {len(df)} data points from {input_file}")
    
    # Filter valid data
    valid_mask = df['T2'].notna() & (df['T2'] > 0) & (df['R2'] > 0.9)
    df_valid = df[valid_mask].copy()
    print(f"Valid data points: {df_valid.shape[0]}")
    
    # Select Crossover regime (0.1 < ξ < 3)
    crossover_mask = (df_valid['xi'] > 0.1) & (df_valid['xi'] < 3)
    df_crossover = df_valid[crossover_mask].copy()
    
    if len(df_crossover) < 3:
        print(f"\n❌ Error: Insufficient points in crossover regime ({len(df_crossover)} < 3)")
        return
    
    print(f"\nCrossover regime points: {len(df_crossover)}")
    print(f"  τc range: {df_crossover['tau_c'].min():.2e} to {df_crossover['tau_c'].max():.2e} s")
    print(f"  ξ range: {df_crossover['xi'].min():.3f} to {df_crossover['xi'].max():.3f}")
    
    # Linear regression in log-log space
    log_tau_c = np.log10(df_crossover['tau_c'].values)
    log_T2 = np.log10(df_crossover['T2'].values)
    
    # Fit with scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau_c, log_T2)
    R2 = r_value**2
    
    # Power law fit in linear space
    try:
        popt, pcov = curve_fit(
            power_law,
            df_crossover['tau_c'].values,
            df_crossover['T2'].values,
            p0=[1e-6, -0.5],
            maxfev=10000
        )
        power_law_slope = popt[1]
        power_law_prefactor = popt[0]
        power_law_slope_err = np.sqrt(pcov[1, 1])
    except:
        power_law_slope = np.nan
        power_law_prefactor = np.nan
        power_law_slope_err = np.nan
    
    # Generate report
    report = f"""Crossover Regime Analysis Results
========================================

Data Selection:
  Total points: {len(df_valid)}
  Crossover regime points (0.1 < ξ < 3): {len(df_crossover)}
  τc range: {df_crossover['tau_c'].min():.2e} to {df_crossover['tau_c'].max():.2e} s
  ξ range: {df_crossover['xi'].min():.3f} to {df_crossover['xi'].max():.3f}

Linear Fit (log-log space):
  Slope: {slope:.4f} ± {std_err:.4f}
  Intercept: {intercept:.4f}
  R²: {R2:.4f}
  p-value: {p_value:.2e}

Power Law Fit (linear space):
  T₂ = {power_law_prefactor:.4e} × τc^{power_law_slope:.4f} ± {power_law_slope_err:.4f}
  
  or equivalently:
  log₁₀(T₂) = {slope:.4f} × log₁₀(τc) + {intercept:.4f}

Physical Interpretation:
  - Crossover regime: No analytical formula exists
  - Numerical result: T₂ ∝ τc^{slope:.4f}
  - Expected behavior: Intermediate between MN (slope = -1) and QS (slope = 0)
  - Literature: Slope ≈ -0.49 in some models

Comparison:
  MN regime (ξ < 0.1): T₂ ∝ τc^-1.0 (analytical)
  Crossover (0.1 < ξ < 3): T₂ ∝ τc^{slope:.4f} (numerical, no analytical formula)
  QS regime (ξ > 3): T₂ ≈ constant (analytical)
"""
    
    print("\n" + report)
    
    # Save report
    output_file = Path("results_comparison/crossover_regime_analysis.txt")
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {output_file}")
    
    return {
        'slope': slope,
        'slope_err': std_err,
        'intercept': intercept,
        'R2': R2,
        'p_value': p_value,
        'n_points': len(df_crossover),
        'power_law_slope': power_law_slope,
        'power_law_slope_err': power_law_slope_err,
    }

if __name__ == '__main__':
    result = main()

