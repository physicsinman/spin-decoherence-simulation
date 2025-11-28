#!/usr/bin/env python3
"""
Check Motional Narrowing Slope Consistency

This script verifies the slope value from the latest simulation results
and compares it with the value mentioned in the paper.
"""

import pandas as pd
from pathlib import Path
from scipy import stats
import numpy as np

def main():
    print("="*80)
    print("Motional Narrowing Slope Consistency Check")
    print("="*80)
    
    # Load data
    input_file = Path("results/t2_vs_tau_c.csv")
    if not input_file.exists():
        print(f"\n❌ Error: {input_file} not found!")
        print("   Please run run_fid_sweep.py first.")
        return
    
    df = pd.read_csv(input_file)
    print(f"\n📊 Loaded {len(df)} data points from {input_file}")
    
    # Physics parameters
    gamma_e = 1.76e11  # rad/(s·T)
    B_rms = 0.57e-6    # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration
    
    # Filter valid data
    valid_mask = df['T2'].notna() & (df['T2'] > 0) & (df['R2'] > 0.9)
    df_valid = df[valid_mask].copy()
    print(f"✅ Valid data points: {df_valid.shape[0]}")
    
    # Calculate xi
    df_valid['xi'] = gamma_e * B_rms * df_valid['tau_c']
    
    # Select MN regime (xi < 0.2)
    mn_mask = df_valid['xi'] < 0.2
    df_mn = df_valid[mn_mask].copy()
    
    if len(df_mn) < 3:
        print(f"\n❌ Error: Only {len(df_mn)} points in MN regime (xi < 0.2)")
        print("   Need at least 3 points for fitting.")
        return
    
    print(f"\n📈 Motional Narrowing Regime:")
    print(f"   Points: {len(df_mn)}")
    print(f"   τc range: {df_mn['tau_c'].min():.2e} to {df_mn['tau_c'].max():.2e} s")
    print(f"   ξ range: {df_mn['xi'].min():.3e} to {df_mn['xi'].max():.3e}")
    
    # Linear regression in log-log space
    log_tau_c = np.log10(df_mn['tau_c'].values)
    log_T2 = np.log10(df_mn['T2'].values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau_c, log_T2)
    R2 = r_value**2
    slope_err = std_err
    
    # Theoretical prediction
    theoretical_slope = -1.0
    deviation = abs(slope - theoretical_slope) / abs(theoretical_slope) * 100
    
    print(f"\n📊 Fit Results:")
    print(f"   Slope: {slope:.4f} ± {slope_err:.4f}")
    print(f"   Intercept: {intercept:.4f}")
    print(f"   R²: {R2:.4f}")
    print(f"   p-value: {p_value:.2e}")
    
    print(f"\n🔬 Comparison with Theory:")
    print(f"   Theoretical slope: {theoretical_slope:.4f}")
    print(f"   Deviation: {deviation:.2f}%")
    
    # Check against paper value
    paper_slope = -1.043
    paper_slope_err = 0.006
    paper_deviation = 4.3
    
    print(f"\n📄 Paper Value (from abstract):")
    print(f"   Slope: {paper_slope:.3f} ± {paper_slope_err:.3f}")
    print(f"   Deviation: {paper_deviation:.1f}%")
    
    # Comparison
    print(f"\n🔍 Comparison:")
    slope_diff = abs(slope - paper_slope)
    if slope_diff < 0.1:
        print(f"   ✅ Current result ({slope:.4f}) is close to paper value ({paper_slope:.3f})")
        print(f"      Difference: {slope_diff:.4f}")
    else:
        print(f"   ⚠️  Current result ({slope:.4f}) differs from paper value ({paper_slope:.3f})")
        print(f"      Difference: {slope_diff:.4f}")
        print(f"   💡 Recommendation: Update paper with current result or re-run simulation")
    
    # Check which is closer to theory
    current_error = abs(slope - theoretical_slope)
    paper_error = abs(paper_slope - theoretical_slope)
    
    print(f"\n🎯 Closer to Theory:")
    if current_error < paper_error:
        print(f"   ✅ Current result is closer to theory")
        print(f"      Current error: {current_error:.4f}")
        print(f"      Paper error: {paper_error:.4f}")
        print(f"   💡 Recommendation: Use current result in paper")
    else:
        print(f"   ⚠️  Paper value is closer to theory")
        print(f"      Current error: {current_error:.4f}")
        print(f"      Paper error: {paper_error:.4f}")
        print(f"   💡 Recommendation: Investigate why current result differs")
    
    # Save report
    report = f"""Motional Narrowing Slope Consistency Report
================================================

Current Simulation Results:
  Slope: {slope:.4f} ± {slope_err:.4f}
  Deviation from theory: {deviation:.2f}%
  R²: {R2:.4f}
  Points used: {len(df_mn)}
  ξ range: {df_mn['xi'].min():.3e} to {df_mn['xi'].max():.3e}

Paper Value (from abstract):
  Slope: {paper_slope:.3f} ± {paper_slope_err:.3f}
  Deviation from theory: {paper_deviation:.1f}%

Recommendation:
"""
    
    if current_error < paper_error:
        report += f"  ✅ Use current result ({slope:.4f} ± {slope_err:.4f}) in paper\n"
        report += f"     Current result is closer to theory.\n"
    elif slope_diff < 0.1:
        report += f"  ✅ Current result matches paper value (within 0.1)\n"
    else:
        report += f"  ⚠️  Investigate discrepancy between current result and paper value\n"
        report += f"     Current: {slope:.4f} ± {slope_err:.4f}\n"
        report += f"     Paper: {paper_slope:.3f} ± {paper_slope_err:.3f}\n"
    
    output_file = Path("results/slope_consistency_report.txt")
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()

