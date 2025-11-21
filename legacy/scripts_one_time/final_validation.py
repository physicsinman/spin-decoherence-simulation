#!/usr/bin/env python3
"""
Final validation of all results after post-meeting improvements
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("Final Validation - Post-Meeting Improvements")
print("="*80)

# 1. FID R² quality check
print("\n1. FID Fitting Quality:")
print("-"*80)
fid_file = Path("results_comparison/t2_vs_tau_c.csv")
if fid_file.exists():
    df_fid = pd.read_csv(fid_file)
    
    r2_excellent = (df_fid['R2'] >= 0.95).sum()
    r2_good = ((df_fid['R2'] >= 0.9) & (df_fid['R2'] < 0.95)).sum()
    r2_fair = ((df_fid['R2'] >= 0.8) & (df_fid['R2'] < 0.9)).sum()
    r2_poor = (df_fid['R2'] < 0.8).sum()
    
    print(f"  R² ≥ 0.95 (excellent): {r2_excellent} points")
    print(f"  0.9 ≤ R² < 0.95 (good): {r2_good} points")
    print(f"  0.8 ≤ R² < 0.9 (fair): {r2_fair} points")
    print(f"  R² < 0.8 (poor): {r2_poor} points")
    
    if r2_poor > 0:
        print(f"\n  ⚠️  {r2_poor} points still have R² < 0.8:")
        poor = df_fid[df_fid['R2'] < 0.8].sort_values('tau_c')
        for idx, row in poor.head(5).iterrows():
            print(f"     τc={row['tau_c']*1e6:.3f}μs: R²={row['R2']:.4f}")
    else:
        print(f"  ✅ All points have R² ≥ 0.8!")
else:
    print("  ❌ FID data file not found!")

# 2. Echo gain check
print("\n2. Echo Gain Quality:")
print("-"*80)
gain_file = Path("results_comparison/echo_gain.csv")
if gain_file.exists():
    df_gain = pd.read_csv(gain_file)
    
    # Check for unphysical values
    unphysical = df_gain[df_gain['echo_gain'] < 0.95]
    if len(unphysical) > 0:
        print(f"  ⚠️  {len(unphysical)} unphysical points (gain < 0.95)")
    else:
        print(f"  ✅ No unphysical echo gain values")
    
    # Check for large fluctuations
    df_gain_sorted = df_gain.sort_values('tau_c')
    gain_diff = df_gain_sorted['echo_gain'].diff().abs()
    large_changes = gain_diff > 1.0
    if large_changes.sum() > 0:
        print(f"  ⚠️  {large_changes.sum()} points with large gain changes (|diff| > 1.0)")
        for idx in df_gain_sorted[large_changes].index[:3]:
            row = df_gain_sorted.loc[idx]
            prev_idx = df_gain_sorted.index[list(df_gain_sorted.index).index(idx) - 1]
            prev_row = df_gain_sorted.loc[prev_idx]
            print(f"     τc={prev_row['tau_c']*1e6:.3f}→{row['tau_c']*1e6:.3f}μs: "
                  f"gain={prev_row['echo_gain']:.3f}→{row['echo_gain']:.3f}")
    else:
        print(f"  ✅ Echo gain changes are smooth")
else:
    print("  ⚠️  Echo gain file not found (run analyze_echo_gain.py)")

# 3. Convergence test check
print("\n3. Convergence Test:")
print("-"*80)
conv_files = list(Path("results_comparison").glob("convergence_N_traj_*.csv"))
if len(conv_files) > 0:
    for conv_file in conv_files:
        df = pd.read_csv(conv_file)
        valid = df[df['T2'].notna()].sort_values('N_traj')
        
        if len(valid) > 1:
            # Check if T2 converges
            t2_values = valid['T2'].values
            t2_changes = np.abs(np.diff(t2_values)) / t2_values[:-1] * 100
            max_change = t2_changes.max()
            
            # Check CI width trend
            ci_widths = valid['ci_width_pct'].values
            ci_widths_valid = ci_widths[~np.isnan(ci_widths)]
            
            print(f"\n  {conv_file.name}:")
            print(f"    T₂ max change: {max_change:.2f}%")
            
            if len(ci_widths_valid) > 1:
                ci_decreasing = np.all(np.diff(ci_widths_valid) <= 0) or ci_widths_valid[-1] < ci_widths_valid[0] * 0.9
                if ci_decreasing:
                    print(f"    ✅ CI width decreasing")
                else:
                    print(f"    ⚠️  CI width not clearly decreasing")
                    print(f"       CI widths: {ci_widths_valid}")
else:
    print("  ⚠️  Convergence test files not found")

# 4. Summary
print("\n" + "="*80)
print("Summary:")
print("-"*80)

if fid_file.exists():
    total_points = len(df_fid)
    good_points = (df_fid['R2'] >= 0.9).sum()
    print(f"FID quality: {good_points}/{total_points} points have R² ≥ 0.9 ({good_points/total_points*100:.1f}%)")

print("="*80)

