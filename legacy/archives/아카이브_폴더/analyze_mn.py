#!/usr/bin/env python3
"""
Motional Narrowing Regime Linear Fit Analysis
Analyzes t2_vs_tau_c.csv and extracts MN regime slope
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

def linear_log_log(x, slope, intercept):
    """Linear model in log-log space: log(y) = slope * log(x) + intercept"""
    return slope * np.log10(x) + intercept

def main():
    print("="*80)
    print("Motional Narrowing Regime Linear Fit Analysis")
    print("="*80)
    
    # Load data
    input_file = Path("results/t2_vs_tau_c.csv")
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
    
    # Select MN regime (xi < 0.2) - stricter criterion to avoid crossover regime
    # Motional narrowing theory T2 ∝ τc^-1 is valid only for ξ << 1
    # Using ξ < 0.2 ensures we're well within the MN regime
    mn_mask = df_valid['xi'] < 0.2
    df_mn = df_valid[mn_mask].copy()
    
    # If insufficient points, try slightly relaxed criterion
    if len(df_mn) < 3:
        print(f"\n⚠️  Warning: Only {len(df_mn)} points in strict MN regime (xi < 0.2)")
        print("   Using xi < 0.25 instead (still avoiding crossover)...")
        mn_mask = df_valid['xi'] < 0.25
        df_mn = df_valid[mn_mask].copy()
        
        # If still insufficient, use original criterion but warn
        if len(df_mn) < 3:
            print(f"\n⚠️  Warning: Only {len(df_mn)} points even with relaxed criterion (xi < 0.25)")
            print("   Using xi < 0.3 (original criterion) but note: third point may be in crossover regime...")
            mn_mask = df_valid['xi'] < 0.3
            df_mn = df_valid[mn_mask].copy()
    
    print(f"\nMotional Narrowing regime points: {len(df_mn)}")
    print(f"  τc range: {df_mn['tau_c'].min():.2e} to {df_mn['tau_c'].max():.2e} s")
    print(f"  ξ range: {df_mn['xi'].min():.3e} to {df_mn['xi'].max():.3e}")
    
    if len(df_mn) < 3:
        print("\n❌ Error: Insufficient points in MN regime for fitting!")
        print("   Recommendation: Increase tau_c_npoints or adjust tau_c_min to get more MN points")
        return
    
    # Linear regression in log-log space
    log_tau_c = np.log10(df_mn['tau_c'].values)
    log_T2 = np.log10(df_mn['T2'].values)
    
    # Fit with scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau_c, log_T2)
    
    # Calculate R²
    R2 = r_value**2
    
    # Calculate slope error (standard error of slope)
    slope_err = std_err
    
    # Theoretical prediction
    theoretical_slope = -1.0
    deviation = abs(slope - theoretical_slope) / abs(theoretical_slope) * 100
    
    # Generate fit report
    report = f"""Motional Narrowing Regime Fit Results
========================================

Data Selection:
  Total points: {len(df_valid)}
  MN regime points (xi < 0.2, strict criterion): {len(df_mn)}
  τc range: {df_mn['tau_c'].min():.2e} to {df_mn['tau_c'].max():.2e} s
  ξ range: {df_mn['xi'].min():.3e} to {df_mn['xi'].max():.3e}

Linear Fit (log-log space):
  Slope: {slope:.4f} ± {slope_err:.4f}
  Intercept: {intercept:.4f}
  R²: {R2:.4f}
  p-value: {p_value:.2e}

Comparison with Theory:
  Theoretical slope: {theoretical_slope:.4f}
  Deviation: {deviation:.2f}%

Fit Equation:
  log₁₀(T₂) = {slope:.4f} × log₁₀(τc) + {intercept:.4f}
  
  or equivalently:
  T₂ ∝ τc^{slope:.4f}
"""
    
    print("\n" + report)
    
    # Save report
    output_file = Path("results/motional_narrowing_fit.txt")
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {output_file}")
    
    return {
        'slope': slope,
        'slope_err': slope_err,
        'intercept': intercept,
        'R2': R2,
        'p_value': p_value,
        'n_points': len(df_mn),
        'tau_c_range': (df_mn['tau_c'].min(), df_mn['tau_c'].max()),
        'xi_range': (df_mn['xi'].min(), df_mn['xi'].max()),
        'deviation': deviation
    }

if __name__ == '__main__':
    result = main()

