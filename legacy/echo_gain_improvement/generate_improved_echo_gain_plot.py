#!/usr/bin/env python3
"""Generate improved echo gain plot"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Publication-quality settings
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

COLORS = {
    'echo': '#F97316',     # Vibrant orange
    'fid': '#1E40AF',      # Deep blue
    'crossover': '#9333EA', # Purple
    'qs': '#B45309',       # Dark orange/brown
}

def plot_improved_echo_gain():
    """Plot improved echo gain"""
    
    # Load improved data
    improved_file = Path('results_comparison/echo_gain_improved.csv')
    if not improved_file.exists():
        print("❌ Improved data not found. Run improve_echo_gain_calculation.py first.")
        return
    
    df = pd.read_csv(improved_file)
    
    # Filter valid points
    valid = df[df['echo_gain'].notna()].copy()
    valid = valid[valid['echo_gain'] >= 0.95].copy()
    
    # Load echo data to check R²
    echo_file = Path('results_comparison/t2_echo_vs_tau_c.csv')
    if echo_file.exists():
        df_echo = pd.read_csv(echo_file)
        if 'R2_echo' in df_echo.columns:
            valid = pd.merge(valid, df_echo[['tau_c', 'R2_echo']], on='tau_c', how='left')
            if 'R2_echo' in valid.columns:
                valid = valid[valid['R2_echo'].notna()].copy()
    
    if len(valid) == 0:
        print("❌ No valid data")
        return
    
    # Sort by tau_c
    valid = valid.sort_values('tau_c')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot with error bars if available
    if 'echo_gain_err' in valid.columns:
        yerr = valid['echo_gain_err'].values
        yerr = np.where(np.isnan(yerr), 0, yerr)
        ax.errorbar(valid['tau_c'] * 1e6, valid['echo_gain'],
                   yerr=yerr, fmt='o-', color=COLORS['echo'],
                   markersize=8, linewidth=2.0, capsize=3, capthick=1.5,
                   elinewidth=1.5, label='Echo gain (improved)', zorder=3, alpha=0.9)
    else:
        ax.plot(valid['tau_c'] * 1e6, valid['echo_gain'],
               'o-', color=COLORS['echo'], markersize=8, linewidth=2.0,
               label='Echo gain (improved)', zorder=3, alpha=0.9)
    
    # Reference line at gain = 1
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5,
              alpha=0.6, label='No gain (gain = 1)', zorder=1)
    
    # Regime boundaries
    if 'xi' in valid.columns:
        xi_valid = valid['xi'].values
        mn_mask = xi_valid < 0.2
        crossover_mask = (xi_valid >= 0.2) & (xi_valid < 3)
        qs_mask = xi_valid >= 3
        
        if mn_mask.sum() > 0:
            ax.scatter(valid[mn_mask]['tau_c'] * 1e6,
                      valid[mn_mask]['echo_gain'],
                      color=COLORS['fid'], s=40, alpha=0.3, zorder=4, label='MN regime')
        if crossover_mask.sum() > 0:
            ax.scatter(valid[crossover_mask]['tau_c'] * 1e6,
                      valid[crossover_mask]['echo_gain'],
                      color=COLORS['crossover'], s=40, alpha=0.3, zorder=4, label='Crossover')
        if qs_mask.sum() > 0:
            ax.scatter(valid[qs_mask]['tau_c'] * 1e6,
                      valid[qs_mask]['echo_gain'],
                      color=COLORS['qs'], s=40, alpha=0.3, zorder=4, label='QS regime')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau_c$ (μs)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Echo Gain (T$_{2,echo}$ / T$_{2,FID}$)', fontsize=14, fontweight='bold')
    ax.set_title('Hahn Echo Gain vs Correlation Time (Improved)', fontsize=15, fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Set y-axis limits
    y_min = max(0.8, valid['echo_gain'].min() * 0.9)
    y_max = min(50.0, valid['echo_gain'].max() * 1.1)  # Cap at 50 for visibility
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    output_path = Path('results_comparison/figures/fig3_echo_gain_improved.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print(f"\n=== 개선된 그래프 통계 ===")
    print(f"포인트 수: {len(valid)}")
    print(f"Gain 범위: [{valid['echo_gain'].min():.3f}, {valid['echo_gain'].max():.3f}]")
    print(f"Gain 평균: {valid['echo_gain'].mean():.3f}")
    print(f"Gain 중앙값: {valid['echo_gain'].median():.3f}")
    print(f"\nMethod 분포:")
    print(valid['method_used'].value_counts())

if __name__ == '__main__':
    plot_improved_echo_gain()

