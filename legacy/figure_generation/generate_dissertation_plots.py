#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Dissertation

Creates all essential figures for the dissertation:
1. T2 vs tau_c (FID) - Main result
2. T2_echo vs tau_c (Hahn Echo)
3. Echo gain vs tau_c
4. Motional Narrowing regime (log-log with slope)
5. Representative coherence decay curves
6. Convergence test results
7. Dimensionless collapse (optional)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import re

# Publication-quality settings - MAXIMUM QUALITY
matplotlib.rcParams['font.size'] = 13  # Increased from 12
matplotlib.rcParams['font.family'] = 'serif'
# Use DejaVu Serif as primary (has better Unicode support) with Times as fallback
matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times', 'serif']
matplotlib.rcParams['axes.linewidth'] = 1.5  # Increased from 1.2
matplotlib.rcParams['axes.labelsize'] = 14  # Increased from 13
matplotlib.rcParams['axes.titlesize'] = 15  # Increased from 14
matplotlib.rcParams['xtick.labelsize'] = 12  # Increased from 11
matplotlib.rcParams['ytick.labelsize'] = 12  # Increased from 11
matplotlib.rcParams['legend.fontsize'] = 12  # Increased from 11
matplotlib.rcParams['legend.framealpha'] = 0.98  # More opaque
matplotlib.rcParams['legend.fancybox'] = True
matplotlib.rcParams['legend.shadow'] = True
matplotlib.rcParams['legend.borderpad'] = 0.5
matplotlib.rcParams['legend.columnspacing'] = 1.0
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.15  # Increased padding
matplotlib.rcParams['lines.linewidth'] = 2.5  # Increased from 2.0
matplotlib.rcParams['lines.markersize'] = 8  # Increased from 7
matplotlib.rcParams['grid.alpha'] = 0.25  # Slightly more subtle
matplotlib.rcParams['grid.linewidth'] = 0.8
matplotlib.rcParams['axes.spines.top'] = False  # Cleaner look
matplotlib.rcParams['axes.spines.right'] = False

# Colors for publication - ENHANCED (more distinct and professional)
COLORS = {
    'fid': '#1E40AF',      # Deep blue (more professional)
    'echo': '#F97316',     # Vibrant orange
    'theory': '#059669',    # Deep green (more visible)
    'mn_fit': '#DC2626',   # Bright red (more visible)
    'crossover': '#9333EA', # Purple (distinct from others)
    'qs': '#B45309',       # Dark orange/brown
    'background': '#F9FAFB', # Light gray for background
}

# Physics parameters
gamma_e = 1.76e11  # rad/(s·T)
B_rms = 0.05e-3  # T

def load_data():
    """Load all data files."""
    data_dir = Path('results_comparison')
    
    data = {}
    
    # Main results
    if (data_dir / 't2_vs_tau_c.csv').exists():
        data['fid'] = pd.read_csv(data_dir / 't2_vs_tau_c.csv')
        data['fid']['xi'] = gamma_e * B_rms * data['fid']['tau_c']
    
    if (data_dir / 't2_echo_vs_tau_c.csv').exists():
        data['echo'] = pd.read_csv(data_dir / 't2_echo_vs_tau_c.csv')
        data['echo']['xi'] = gamma_e * B_rms * data['echo']['tau_c']
    
    if (data_dir / 'echo_gain.csv').exists():
        data['gain'] = pd.read_csv(data_dir / 'echo_gain.csv')
    
    # Convergence test - only use the latest 3 files (most recent tau_c values)
    convergence_files = sorted(data_dir.glob('convergence_N_traj_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)
    if convergence_files:
        data['convergence'] = {}
        # Only use the 3 most recent files (for clean 3-panel figure)
        for f in convergence_files[:3]:
            tau_c_str = f.stem.replace('convergence_N_traj_', '')
            data['convergence'][tau_c_str] = pd.read_csv(f)
            print(f"  Loaded convergence data: {f.name}")
    
    # Representative curves
    fid_curves = list(data_dir.glob('fid_tau_c_*.csv'))
    echo_curves = list(data_dir.glob('echo_tau_c_*.csv'))
    if fid_curves:
        data['fid_curves'] = {}
        for f in fid_curves:
            tau_c_str = f.stem.replace('fid_tau_c_', '')
            data['fid_curves'][tau_c_str] = pd.read_csv(f)
    if echo_curves:
        data['echo_curves'] = {}
        for f in echo_curves:
            tau_c_str = f.stem.replace('echo_tau_c_', '')
            data['echo_curves'][tau_c_str] = pd.read_csv(f)
    
    return data

def plot_T2_vs_tau_c(data, output_dir):
    """Figure 1: T2 vs tau_c (FID) - Main result."""
    if 'fid' not in data:
        print("⚠️  FID data not found, skipping Figure 1")
        return
    
    df = data['fid']
    valid = df[df['T2'].notna()].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Regime boundaries (calculate first for color coding)
    xi_valid = valid['xi'].values
    mn_mask = xi_valid < 0.2
    crossover_mask = (xi_valid >= 0.2) & (xi_valid < 3)
    qs_mask = xi_valid >= 3
    
    # Plot data with error bars, colored by regime
    if 'T2_lower' in valid.columns and 'T2_upper' in valid.columns:
        yerr_lower = valid['T2'] - valid['T2_lower']
        yerr_upper = valid['T2_upper'] - valid['T2']
        yerr = np.array([yerr_lower.values, yerr_upper.values])
        
        # Ensure minimum error bar visibility (even if CI is very small)
        # For very small CI, use a minimum of 0.5% of T2 value for visibility
        ci_width_pct = (yerr[0] + yerr[1]) / valid['T2'].values * 100
        min_visible_error = valid['T2'].values * 0.005  # 0.5% of T2 value
        
        # Apply minimum error for visibility
        yerr_adj = yerr.copy()
        for i in range(len(valid)):
            if yerr[0, i] < min_visible_error[i]:
                yerr_adj[0, i] = min_visible_error[i]
            if yerr[1, i] < min_visible_error[i]:
                yerr_adj[1, i] = min_visible_error[i]
        
        # Plot by regime with different colors (same marker shape)
        # Show error bars for ALL points (important for publication)
        if mn_mask.sum() > 0:
            mn_valid = valid[mn_mask].copy()
            mn_yerr = yerr_adj[:, mn_mask]
            ax.errorbar(mn_valid['tau_c'] * 1e6, 
                       mn_valid['T2'] * 1e6,
                       yerr=mn_yerr, fmt='o', color=COLORS['fid'],
                       markersize=8, capsize=3, capthick=1.5, elinewidth=1.5,  # Enhanced
                       label='FID (MN)', alpha=0.85, zorder=3, markeredgewidth=0.5)
        if crossover_mask.sum() > 0:
            cross_valid = valid[crossover_mask].copy()
            cross_yerr = yerr_adj[:, crossover_mask]
            ax.errorbar(cross_valid['tau_c'] * 1e6, 
                       cross_valid['T2'] * 1e6,
                       yerr=cross_yerr, fmt='o', color=COLORS['crossover'],
                       markersize=8, capsize=3, capthick=1.5, elinewidth=1.5,  # Enhanced
                       label='FID (Crossover)', alpha=0.85, zorder=3, markeredgewidth=0.5)
        if qs_mask.sum() > 0:
            qs_valid = valid[qs_mask].copy()
            qs_yerr = yerr_adj[:, qs_mask]
            ax.errorbar(qs_valid['tau_c'] * 1e6, 
                       qs_valid['T2'] * 1e6,
                       yerr=qs_yerr, fmt='o', color=COLORS['qs'],
                       markersize=8, capsize=3, capthick=1.5, elinewidth=1.5,  # Enhanced
                       label='FID (QS)', alpha=0.85, zorder=3, markeredgewidth=0.5)
    else:
        # Plot without error bars (all use same marker shape, different colors)
        if mn_mask.sum() > 0:
            ax.plot(valid[mn_mask]['tau_c'] * 1e6, valid[mn_mask]['T2'] * 1e6,
                   'o', color=COLORS['fid'], markersize=6,
                   label='FID (MN)', alpha=0.8, zorder=3)
        if crossover_mask.sum() > 0:
            ax.plot(valid[crossover_mask]['tau_c'] * 1e6, valid[crossover_mask]['T2'] * 1e6,
                   'o', color=COLORS['crossover'], markersize=6,
                   label='FID (Crossover)', alpha=0.8, zorder=3)
        if qs_mask.sum() > 0:
            ax.plot(valid[qs_mask]['tau_c'] * 1e6, valid[qs_mask]['T2'] * 1e6,
                   'o', color=COLORS['qs'], markersize=6,
                   label='FID (QS)', alpha=0.8, zorder=3)
    
    # Theoretical curve (Motional Narrowing)
    # Use data range to determine theory plot range
    tau_c_min = valid['tau_c'].min()
    tau_c_max = valid['tau_c'].max()
    tau_c_theory = np.logspace(np.log10(tau_c_min), np.log10(tau_c_max), 1000)
    Delta_omega = gamma_e * B_rms
    T2_MN = 1.0 / (Delta_omega**2 * tau_c_theory)
    
    # Plot theory curve only in MN regime (where it's valid)
    xi_theory = Delta_omega * tau_c_theory
    mask_MN = xi_theory < 0.2  # MN regime (where theory is valid)
    
    # Plot theory curve only in MN regime
    # Note: Theory T₂ = 1/(Δω²τc) is only valid in MN regime
    # In QS regime, T₂ ≈ constant (independent of τc)
    if mask_MN.sum() > 0:
        ax.plot(tau_c_theory[mask_MN] * 1e6, T2_MN[mask_MN] * 1e6,
               '--', color=COLORS['theory'], linewidth=2.5, 
               label='Theory (MN, valid for ξ < 0.2)', alpha=0.9, zorder=2)
    
    # Add QS regime theoretical value (horizontal line, only in QS regime)
    T2_QS_theory = 1.0 / Delta_omega
    mask_QS = xi_theory >= 3  # QS regime (where theory is valid)
    
    # Only plot QS theory line in the QS regime range
    if mask_QS.sum() > 0:
        # Get tau_c range for QS regime
        tau_c_QS_min = tau_c_theory[mask_QS].min()
        tau_c_QS_max = tau_c_theory[mask_QS].max()
        
        # Plot horizontal line only in QS regime range
        ax.hlines(T2_QS_theory * 1e6, tau_c_QS_min * 1e6, tau_c_QS_max * 1e6,
                 color=COLORS['theory'], linestyle=':', linewidth=2.0, alpha=0.7,
                 label=f'Theory (QS, T$_2$ ≈ {T2_QS_theory*1e6:.3f} μs, valid for ξ ≥ 3)', zorder=1)
    
    # Add regime boundary lines (ξ = 0.2 and ξ = 3.0)
    tau_c_min = valid['tau_c'].min()
    tau_c_max = valid['tau_c'].max()
    # Calculate tau_c values for ξ = 0.2 and ξ = 3.0
    tau_c_boundary_02 = 0.2 / Delta_omega
    tau_c_boundary_30 = 3.0 / Delta_omega
    
    # Only show boundaries if they're within the data range
    if tau_c_min <= tau_c_boundary_02 <= tau_c_max:
        ax.axvline(tau_c_boundary_02 * 1e6, color='gray', linestyle=':', 
                  linewidth=1.5, alpha=0.5, zorder=1)
        # Add text annotation
        ax.text(tau_c_boundary_02 * 1e6, ax.get_ylim()[1] * 0.7, 
               r'$\xi = 0.2$', rotation=90, fontsize=9, 
               verticalalignment='bottom', alpha=0.7)
    
    if tau_c_min <= tau_c_boundary_30 <= tau_c_max:
        ax.axvline(tau_c_boundary_30 * 1e6, color='gray', linestyle=':', 
                  linewidth=1.5, alpha=0.5, zorder=1)
        # Add text annotation
        ax.text(tau_c_boundary_30 * 1e6, ax.get_ylim()[1] * 0.7, 
               r'$\xi = 3.0$', rotation=90, fontsize=9, 
               verticalalignment='bottom', alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau_c$ (μs)', fontsize=12)
    ax.set_ylabel(r'$T_2$ (μs)', fontsize=12)
    ax.set_title('FID Coherence Time vs Correlation Time', fontsize=13)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'fig1_T2_vs_tau_c.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_MN_regime_slope(data, output_dir):
    """Figure 2: Motional Narrowing regime (log-log with slope fit)."""
    if 'fid' not in data:
        print("⚠️  FID data not found, skipping Figure 2")
        return
    
    df = data['fid']
    valid = df[df['T2'].notna()].copy()
    
    # MN regime (xi < 0.2)
    mn_data = valid[valid['xi'] < 0.2].copy()
    
    if len(mn_data) < 3:
        print("⚠️  Insufficient MN regime points, skipping Figure 2")
        return
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot data
    if 'T2_lower' in mn_data.columns and 'T2_upper' in mn_data.columns:
        yerr_lower = mn_data['T2'] - mn_data['T2_lower']
        yerr_upper = mn_data['T2_upper'] - mn_data['T2']
        yerr = np.array([yerr_lower.values, yerr_upper.values])
        ax.errorbar(mn_data['tau_c'] * 1e6, mn_data['T2'] * 1e6,
                   yerr=yerr, fmt='o', color=COLORS['fid'],
                   markersize=6, capsize=3, capthick=1.5,
                   label='Numerical data', zorder=3)
    else:
        ax.plot(mn_data['tau_c'] * 1e6, mn_data['T2'] * 1e6,
               'o', color=COLORS['fid'], markersize=6,
               label='Numerical data', zorder=3)
    
    # Linear fit (log-log) with R² calculation
    log_tau = np.log(mn_data['tau_c'])
    log_T2 = np.log(mn_data['T2'])
    coeffs = np.polyfit(log_tau, log_T2, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Calculate R²
    log_T2_pred = slope * log_tau + intercept
    ss_res = np.sum((log_T2 - log_T2_pred)**2)
    ss_tot = np.sum((log_T2 - np.mean(log_T2))**2)
    R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Plot fit
    tau_fit = np.logspace(np.log10(mn_data['tau_c'].min()),
                         np.log10(mn_data['tau_c'].max()), 100)
    T2_fit = np.exp(slope * np.log(tau_fit) + intercept)
    # MAXIMUM QUALITY: Enhanced line styles
    ax.plot(tau_fit * 1e6, T2_fit * 1e6,
           '--', color=COLORS['mn_fit'], linewidth=3.0,  # Increased
           label=f'Fit: slope = {slope:.3f}, R² = {R2:.4f}', zorder=2, alpha=0.9)
    
    # Theoretical line (slope = -1) - Enhanced
    T2_theory = 1.0 / (gamma_e * B_rms)**2 / tau_fit
    ax.plot(tau_fit * 1e6, T2_theory * 1e6,
           ':', color=COLORS['theory'], linewidth=2.5,  # Increased
           label='Theory: slope = -1.000', zorder=1, alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau_c$ (μs)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$T_2$ (μs)', fontsize=13, fontweight='bold')
    ax.set_title('Motional Narrowing Regime (ξ < 0.2)', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    output_path = output_dir / 'fig2_MN_regime_slope.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_echo_gain(data, output_dir):
    """Figure 3: Echo gain vs tau_c."""
    if 'gain' not in data:
        print("⚠️  Echo gain data not found, skipping Figure 3")
        return
    
    df = data['gain']
    valid = df[df['echo_gain'].notna()].copy()
    
    if len(valid) == 0:
        print("⚠️  No valid echo gain data, skipping Figure 3")
        return
    
    # CRITICAL FIX: Filter out problematic points
    # 1. Unphysical values (gain < 1)
    # 2. Points where T2_echo R² is NaN (fitting failed, using analytical estimate)
    # 3. Points with suspiciously low gain (likely fitting errors)
    
    # Load echo data to check R²
    echo_file = Path('results_comparison/t2_echo_vs_tau_c.csv')
    if echo_file.exists():
        df_echo = pd.read_csv(echo_file)
        # Merge to get R²_echo
        valid = pd.merge(valid, df_echo[['tau_c', 'R2_echo']], on='tau_c', how='left')
        # Filter out points where R²_echo is NaN (fitting failed)
        valid_with_r2 = valid[valid['R2_echo'].notna()].copy()
        if len(valid_with_r2) < len(valid):
            n_filtered_r2 = len(valid) - len(valid_with_r2)
            print(f"⚠️  Filtered {n_filtered_r2} points with R²_echo = NaN (fitting failed)")
        valid = valid_with_r2
    
    # Filter out unphysical values (gain < 1)
    valid_physical = valid[valid['echo_gain'] >= 0.95].copy()  # Allow small numerical error
    
    if len(valid_physical) < len(valid):
        n_filtered = len(valid) - len(valid_physical)
        print(f"⚠️  Filtered {n_filtered} unphysical points (gain < 0.95)")
    
    if len(valid_physical) == 0:
        print("❌ No valid physical echo gain data after filtering!")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort by tau_c for clean line plot
    valid_physical = valid_physical.sort_values('tau_c')
    
    # NOTE: We do NOT filter out physically inconsistent points automatically
    # Instead, we mark them for investigation. Data should be fixed at the source.
    # If there are issues, they should be addressed by re-running simulations,
    # not by hiding the data.
    
    # Check for physical inconsistencies (for warning only, not filtering)
    if len(valid_physical) > 1:
        valid_physical_xi = valid_physical.sort_values('xi').copy()
        gain_diff = valid_physical_xi['echo_gain'].diff()
        xi_diff = valid_physical_xi['xi'].diff()
        
        # In crossover/QS regime (xi >= 0.2), gain should increase with xi
        crossover_qs_mask = valid_physical_xi['xi'] >= 0.2
        if crossover_qs_mask.sum() > 1:
            unphysical_mask = (gain_diff < -0.3) & (xi_diff > 0) & crossover_qs_mask
            if unphysical_mask.sum() > 0:
                print(f"⚠️  WARNING: {unphysical_mask.sum()} points show unphysical behavior (gain decreases with increasing ξ)")
                print(f"   These points should be re-simulated, not filtered out!")
                for idx in valid_physical_xi[unphysical_mask].index[:3]:
                    row = valid_physical_xi.loc[idx]
                    prev_idx = valid_physical_xi.index[valid_physical_xi.index.get_loc(idx) - 1]
                    prev_row = valid_physical_xi.loc[prev_idx]
                    print(f"     τc={prev_row['tau_c']*1e6:.3f}→{row['tau_c']*1e6:.3f}μs, ξ={prev_row['xi']:.3f}→{row['xi']:.3f}, gain={prev_row['echo_gain']:.3f}→{row['echo_gain']:.3f}")
        
        # Check for sudden large changes (warning only)
        valid_physical = valid_physical.sort_values('tau_c')
        gain_diff_abs = valid_physical['echo_gain'].diff().abs()
        large_changes = gain_diff_abs > 1.5
        if large_changes.sum() > 0:
            print(f"⚠️  WARNING: {large_changes.sum()} points show sudden large changes (|diff| > 1.5)")
            print(f"   These may indicate fitting errors and should be investigated")
    
    # PUBLICATION QUALITY: Enhanced plotting
    # Plot with error bars if available
    if 'echo_gain_err' in valid_physical.columns:
        yerr = valid_physical['echo_gain_err'].values
        yerr = np.where(np.isnan(yerr), 0, yerr)
        ax.errorbar(valid_physical['tau_c'] * 1e6, valid_physical['echo_gain'],
                   yerr=yerr, fmt='o-', color=COLORS['echo'],
                   markersize=9, linewidth=2.5, capsize=4, capthick=2.0,  # Enhanced
                   elinewidth=2.0, label='Echo gain', zorder=3, alpha=0.9, markeredgewidth=0.5)
    else:
        ax.plot(valid_physical['tau_c'] * 1e6, valid_physical['echo_gain'],
               'o-', color=COLORS['echo'], markersize=9, linewidth=2.5,  # Enhanced
               label='Echo gain', zorder=3, alpha=0.9, markeredgewidth=0.5)
    
    # Reference line at gain = 1 (cleaner styling)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5,
              alpha=0.6, label='No gain (gain = 1)', zorder=1)
    
    # Regime boundaries (use valid_physical)
    if 'xi' in valid_physical.columns:
        xi_valid = valid_physical['xi'].values
        mn_mask = xi_valid < 0.2
        crossover_mask = (xi_valid >= 0.2) & (xi_valid < 3)
        qs_mask = xi_valid >= 3
        
        if mn_mask.sum() > 0:
            ax.scatter(valid_physical[mn_mask]['tau_c'] * 1e6,
                      valid_physical[mn_mask]['echo_gain'],
                      color=COLORS['fid'], s=50, alpha=0.3, zorder=4)
        if crossover_mask.sum() > 0:
            ax.scatter(valid_physical[crossover_mask]['tau_c'] * 1e6,
                      valid_physical[crossover_mask]['echo_gain'],
                      color=COLORS['crossover'], s=50, alpha=0.3, zorder=4)
        if qs_mask.sum() > 0:
            ax.scatter(valid_physical[qs_mask]['tau_c'] * 1e6,
                      valid_physical[qs_mask]['echo_gain'],
                      color=COLORS['qs'], s=50, alpha=0.3, zorder=4)
    
    # Add minimum gain line (gain = 1)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
              alpha=0.7, label='Minimum (gain = 1)', zorder=1)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau_c$ (μs)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Echo Gain (T$_{2,echo}$ / T$_{2,FID}$)', fontsize=13, fontweight='bold')
    ax.set_title('Hahn Echo Gain vs Correlation Time', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # PUBLICATION QUALITY: Set y-axis limits for better presentation
    # Allow higher y_max to show high gains in MN regime, but cap at reasonable value
    y_min = max(0.8, valid_physical['echo_gain'].min() * 0.9)
    # For MN regime, gains can be very high (>10), so use adaptive y_max
    max_gain = valid_physical['echo_gain'].max()
    if max_gain > 10:
        # High gains in MN regime - use log scale or cap at reasonable value
        y_max = min(100.0, max_gain * 1.2)  # Cap at 100 for visibility
    else:
        y_max = min(6.0, max_gain * 1.1)
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    output_path = output_dir / 'fig3_echo_gain.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_representative_curves(data, output_dir):
    """Figure 4: Representative coherence decay curves."""
    if 'fid_curves' not in data or len(data['fid_curves']) == 0:
        print("⚠️  FID curves not found, skipping Figure 4")
        return
    
    # Select representative tau_c values - IMPROVED
    # Choose specific tau_c values that represent each regime well
    tau_c_keys = sorted(data['fid_curves'].keys(), 
                       key=lambda x: float(x.replace('e-', 'e-').replace('e+', 'e+').replace('tau_c_', '')))
    
    # Parse tau_c values to select representative ones
    tau_c_values = []
    for key in tau_c_keys:
        try:
            tau_c_str = key.replace('tau_c_', '').replace('e-', 'e-').replace('e+', 'e+')
            if 'e-' in tau_c_str:
                parts = tau_c_str.split('e-')
                if len(parts) == 2:
                    tau_c_val = float(parts[0]) * 10**(-float(parts[1]))
                else:
                    tau_c_val = float(tau_c_str)
            elif 'e+' in tau_c_str:
                parts = tau_c_str.split('e+')
                if len(parts) == 2:
                    tau_c_val = float(parts[0]) * 10**(float(parts[1]))
                else:
                    tau_c_val = float(tau_c_str)
            else:
                tau_c_val = float(tau_c_str)
            tau_c_values.append((key, tau_c_val))
        except:
            continue
    
    # Select 4 representative curves: one from each regime
    # MN: ~1e-8, Crossover: ~1e-7, QS: ~1e-6, ~1e-5
    target_tau_cs = [1e-8, 1e-7, 1e-6, 1e-5]
    selected_keys = []
    for target in target_tau_cs:
        # Find closest tau_c
        closest = min(tau_c_values, key=lambda x: abs(x[1] - target))
        if closest[0] not in selected_keys:
            selected_keys.append(closest[0])
    
    # If we don't have enough, fill with evenly spaced ones
    if len(selected_keys) < 4:
        n_curves = min(4, len(tau_c_keys))
        indices = np.linspace(0, len(tau_c_keys) - 1, n_curves, dtype=int)
        for i in indices:
            if tau_c_keys[i] not in selected_keys:
                selected_keys.append(tau_c_keys[i])
        selected_keys = selected_keys[:4]
    
    # MAXIMUM QUALITY: Larger figure size
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # Increased from (12, 10)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for idx, tau_c_key in enumerate(selected_keys):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        fid_df = data['fid_curves'][tau_c_key]
        
        # Parse tau_c value - improved parsing
        try:
            # Handle scientific notation properly
            tau_c_str = tau_c_key.replace('e-', 'e-').replace('e+', 'e+')
            # Handle cases like "1e-08" or "1e-8"
            if 'e-' in tau_c_str:
                parts = tau_c_str.split('e-')
                if len(parts) == 2:
                    tau_c_val = float(parts[0]) * 10**(-float(parts[1]))
                else:
                    tau_c_val = float(tau_c_str)
            elif 'e+' in tau_c_str:
                parts = tau_c_str.split('e+')
                if len(parts) == 2:
                    tau_c_val = float(parts[0]) * 10**(float(parts[1]))
                else:
                    tau_c_val = float(tau_c_str)
            else:
                tau_c_val = float(tau_c_str)
        except:
            print(f"⚠️  Could not parse tau_c_key: {tau_c_key}, using default")
            tau_c_val = 1e-8
        
        # Plot FID
        # Try different possible column names
        E_col = None
        if '|E|' in fid_df.columns:
            E_col = '|E|'
        elif 'P(t)' in fid_df.columns:
            E_col = 'P(t)'
        elif 'E' in fid_df.columns:
            E_col = 'E'
        
        # Plot FID
        if 'time (s)' in fid_df.columns and E_col is not None:
            t = fid_df['time (s)'].values
            E = fid_df[E_col].values
            ax.plot(t * 1e6, E, '-', color=COLORS['fid'], linewidth=2,
                   label='FID', alpha=0.8)
        
        # Plot Echo if available
        if 'echo_curves' in data and tau_c_key in data['echo_curves']:
            echo_df = data['echo_curves'][tau_c_key]
            E_echo_col = None
            if '|E|' in echo_df.columns:
                E_echo_col = '|E|'
            elif 'P_echo(t)' in echo_df.columns:
                E_echo_col = 'P_echo(t)'
            elif 'E_echo' in echo_df.columns:
                E_echo_col = 'E_echo'
            
            # Try different time column names
            time_col = None
            for col in ['time (s)', 't', 'time', 'tau']:
                if col in echo_df.columns:
                    time_col = col
                    break
            
            if time_col is not None and E_echo_col is not None:
                t_echo = echo_df[time_col].values
                E_echo = echo_df[E_echo_col].values
                # Filter out NaN values
                valid_mask = ~(np.isnan(t_echo) | np.isnan(E_echo))
                if valid_mask.sum() > 0:
                    t_echo_valid = t_echo[valid_mask]
                    E_echo_valid = E_echo[valid_mask]
                    
                    # Plot echo with thicker line and higher zorder to make it visible
                    ax.plot(t_echo_valid * 1e6, E_echo_valid, 
                           '--', color=COLORS['echo'],
                           linewidth=3.0, label='Hahn Echo', alpha=0.9, zorder=5)
                    
                    # For very short echo ranges, add an inset to show echo clearly
                    if t_echo_valid.max() * 1e6 < 1.0:  # If echo max < 1 μs
                        # Create inset axes in upper right corner
                        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                        axins = inset_axes(ax, width="30%", height="25%", loc='upper right',
                                         bbox_to_anchor=(0.02, 0.02, 1, 1), bbox_transform=ax.transAxes)
                        axins.plot(t_echo_valid * 1e6, E_echo_valid, 
                                 '--', color=COLORS['echo'], linewidth=2.5, alpha=0.9)
                        axins.set_xlim([t_echo_valid.min() * 1e6 * 0.9, t_echo_valid.max() * 1e6 * 1.1])
                        axins.set_ylim([E_echo_valid.min() * 0.99, E_echo_valid.max() * 1.01])
                        axins.tick_params(labelsize=8)
                        axins.set_xlabel('Time (μs)', fontsize=8)
                        axins.set_ylabel('|E|', fontsize=8)
                        axins.grid(True, alpha=0.3)
        
        # Set xlim based on FID (don't truncate FID to show echo)
        if 'time (s)' in fid_df.columns:
            t = fid_df['time (s)'].values
            # Show FID up to where it decays significantly
            if E_col is not None:
                E = fid_df[E_col].values
                decay_mask = E > 0.01
                if decay_mask.sum() > 0:
                    t_max = t[decay_mask].max() * 1e6
                else:
                    t_max = t.max() * 1e6
            else:
                t_max = t.max() * 1e6
            ax.set_xlim([0, t_max * 1.05])
        
        ax.set_xlabel('Time (μs)', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$|E(t)|$', fontsize=12, fontweight='bold')
        ax.set_title(f'τc = {tau_c_val*1e6:.2f} μs', fontsize=13, fontweight='bold', pad=8)
        ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = output_dir / 'fig4_representative_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_convergence_test(data, output_dir):
    """Figure 5: Convergence test results (N_traj)."""
    if 'convergence' not in data or len(data['convergence']) == 0:
        print("⚠️  Convergence test data not found, skipping Figure 5")
        return
    
    n_tests = len(data['convergence'])
    # MAXIMUM QUALITY: Larger figure size
    fig, axes = plt.subplots(1, n_tests, figsize=(7*n_tests, 6))  # Increased
    fig.patch.set_facecolor('white')
    
    if n_tests == 1:
        axes = [axes]
    
    for idx, (tau_c_key, df) in enumerate(data['convergence'].items()):
        ax = axes[idx]
        
        valid = df[df['T2'].notna()].copy()
        
        if len(valid) == 0:
            continue
        
        # PUBLICATION QUALITY: Enhanced T2 plotting
        # Sort by N_traj for clean line
        valid = valid.sort_values('N_traj')
        
        # Check if T2 values are suspiciously constant or have poor convergence
        T2_values = valid['T2'].values
        convergence_warning = None
        if len(T2_values) > 1:
            T2_std = np.std(T2_values)
            T2_mean = np.mean(T2_values)
            T2_cv = T2_std / T2_mean if T2_mean > 0 else 0
            T2_range_pct = (np.max(T2_values) - np.min(T2_values)) / T2_mean * 100 if T2_mean > 0 else 0
            
            # Check if T2 is essentially constant (simulation issue)
            if T2_cv < 1e-10:
                convergence_warning = '⚠️ T₂ constant\n(simulation issue)'
                print(f"❌ CRITICAL: T₂ values are constant for {tau_c_key} - simulation problem!")
            # Check if T2 variation is too large (poor convergence)
            elif T2_range_pct > 15:
                convergence_warning = f'⚠️ Poor convergence\n(T₂ varies {T2_range_pct:.1f}%)'
                print(f"⚠️  WARNING: T₂ variation is {T2_range_pct:.1f}% for {tau_c_key} - may need more trajectories")
            elif T2_range_pct > 10:
                convergence_warning = f'⚠️ Large variation\n(T₂ varies {T2_range_pct:.1f}%)'
                print(f"⚠️  WARNING: T₂ variation is {T2_range_pct:.1f}% for {tau_c_key}")
            
            if convergence_warning:
                ax.text(0.5, 0.95, convergence_warning,
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Plot T2 vs N_traj
        ax.plot(valid['N_traj'], valid['T2'] * 1e6, 'o-',
               color=COLORS['fid'], markersize=8, linewidth=2.5,
               label='T₂', zorder=3, alpha=0.9)
        
        # Error bars if available
        if 'T2_error' in valid.columns:
            yerr = valid['T2_error'].values * 1e6
            yerr = np.where(np.isnan(yerr), 0, yerr)
            # Only show meaningful error bars
            meaningful_err = yerr > valid['T2'].values * 1e6 * 0.01  # > 1% of value
            if meaningful_err.sum() > 0:
                ax.errorbar(valid['N_traj'], valid['T2'] * 1e6,
                           yerr=yerr, fmt='none', color=COLORS['fid'], capsize=4,
                           capthick=1.5, elinewidth=1.5, alpha=0.7, zorder=2)
        
        # CI width calculation - use analytical error if bootstrap CI is degenerate
            ax2 = ax.twinx()
        ci_widths_to_plot = []
        N_traj_ci_to_plot = []
        use_analytical = False
        
        if 'ci_width_pct' in valid.columns:
            ci_widths = valid['ci_width_pct'].values
            # Check if CI widths are degenerate (< 0.01%)
            valid_ci_mask = ~np.isnan(ci_widths) & (ci_widths >= 0.01)
            
            if valid_ci_mask.sum() > 0:
                # Use bootstrap CI widths
                N_traj_ci = valid['N_traj'].values[valid_ci_mask]
                ci_widths_clean = ci_widths[valid_ci_mask]
                sort_idx = np.argsort(N_traj_ci)
                N_traj_ci_to_plot = N_traj_ci[sort_idx]
                ci_widths_to_plot = ci_widths_clean[sort_idx]
            else:
                # All CI widths are degenerate - calculate analytical error
                print(f"⚠️  All CI widths degenerate (< 0.01%) for {tau_c_key} - using analytical error")
                use_analytical = True
        else:
            # No CI width data - calculate analytical error
            use_analytical = True
        
        if use_analytical:
            # Calculate analytical CI width: σ_T2 ≈ T2 / √N_traj
            # Parse tau_c for calculation
            try:
                tau_c_str = tau_c_key.replace('tau_c_', '').replace('e-', 'e-').replace('e+', 'e+')
                if 'e-' in tau_c_str:
                    parts = tau_c_str.split('e-')
                    if len(parts) == 2:
                        tau_c_val = float(parts[0]) * 10**(-float(parts[1]))
                    else:
                        tau_c_val = float(tau_c_str)
                elif 'e+' in tau_c_str:
                    parts = tau_c_str.split('e+')
                    if len(parts) == 2:
                        tau_c_val = float(parts[0]) * 10**(float(parts[1]))
                    else:
                        tau_c_val = float(tau_c_str)
                else:
                    tau_c_val = float(tau_c_str)
            except:
                tau_c_val = None
            
            # Calculate analytical error for each N_traj
            for _, row in valid.iterrows():
                N_traj = row['N_traj']
                T2 = row['T2']
                if not np.isnan(T2) and T2 > 0:
                    # Statistical error: σ_T2 ≈ T2 / √N_traj
                    sigma_stat = T2 / np.sqrt(N_traj)
                    # Additional uncertainty from fitting (assume 1% for good fits)
                    sigma_fit = T2 * 0.01
                    # Combined error
                    sigma_total = np.sqrt(sigma_stat**2 + sigma_fit**2)
                    # CI width as percentage
                    ci_width_pct = (2 * 1.96 * sigma_total) / T2 * 100  # 95% CI
                    ci_widths_to_plot.append(ci_width_pct)
                    N_traj_ci_to_plot.append(N_traj)
            
            if len(ci_widths_to_plot) > 0:
                ci_widths_to_plot = np.array(ci_widths_to_plot)
                N_traj_ci_to_plot = np.array(N_traj_ci_to_plot)
                sort_idx = np.argsort(N_traj_ci_to_plot)
                N_traj_ci_to_plot = N_traj_ci_to_plot[sort_idx]
                ci_widths_to_plot = ci_widths_to_plot[sort_idx]
        
        # Plot CI width on secondary axis
        if len(ci_widths_to_plot) > 0:
            # Check if decreasing (should be for convergence test)
            if len(ci_widths_to_plot) > 1:
                is_decreasing = np.all(np.diff(ci_widths_to_plot) <= 0) or ci_widths_to_plot[-1] < ci_widths_to_plot[0] * 0.9
                if not is_decreasing and not use_analytical:
                    print(f"⚠️  CI width not clearly decreasing for {tau_c_key}")
            
            ax2.plot(N_traj_ci_to_plot, ci_widths_to_plot, 's--',
                     color=COLORS['echo'], markersize=7, linewidth=2.0,
                     label='CI width (%)' + (' (analytical)' if use_analytical else ''),
                     alpha=0.8, zorder=2)
            ax2.set_ylabel('CI Width (%)', fontsize=12, fontweight='bold', color=COLORS['echo'])
            ax2.tick_params(axis='y', labelcolor=COLORS['echo'], labelsize=11)
        
        # Parse tau_c for title - improved parsing
        try:
            tau_c_str = tau_c_key.replace('tau_c_', '').replace('e-', 'e-').replace('e+', 'e+')
            # Handle scientific notation
            if 'e-' in tau_c_str:
                parts = tau_c_str.split('e-')
                if len(parts) == 2:
                    tau_c_val = float(parts[0]) * 10**(-float(parts[1]))
                else:
                    tau_c_val = float(tau_c_str)
            elif 'e+' in tau_c_str:
                parts = tau_c_str.split('e+')
                if len(parts) == 2:
                    tau_c_val = float(parts[0]) * 10**(float(parts[1]))
                else:
                    tau_c_val = float(tau_c_str)
            else:
                tau_c_val = float(tau_c_str)
            
            # CRITICAL FIX: Validate and format tau_c correctly
            if tau_c_val <= 0:
                print(f"❌ ERROR: Invalid tau_c = {tau_c_val} for key {tau_c_key}")
                ax.set_title(f'Convergence Test (invalid τc)', fontsize=13, fontweight='bold', pad=8)
            elif tau_c_val < 1e-9:
                # Very small value (< 1 ns), show in ps
                ax.set_title(f'τc = {tau_c_val*1e12:.1f} ps', fontsize=13, fontweight='bold', pad=8)
            elif tau_c_val < 1e-6:
                # Show in ns (most common case)
                ax.set_title(f'τc = {tau_c_val*1e9:.1f} ns', fontsize=13, fontweight='bold', pad=8)
            else:
                # Show in μs
                ax.set_title(f'τc = {tau_c_val*1e6:.2f} μs', fontsize=13, fontweight='bold', pad=8)
        except Exception as e:
            ax.set_title(f'Convergence Test', fontsize=13, fontweight='bold', pad=8)
            print(f"⚠️  Could not parse tau_c_key: {tau_c_key}, error: {e}")
        
        # Add R² as text annotation if available
        if 'R2' in valid.columns:
            R2_values = valid['R2'].values
            R2_valid = R2_values[~np.isnan(R2_values)]
            if len(R2_valid) > 0:
                R2_mean = np.mean(R2_valid)
                R2_min = np.min(R2_valid)
                # Add R² info as text (small, non-intrusive)
                ax.text(0.98, 0.02, f'R²: {R2_mean:.3f} (min: {R2_min:.3f})',
                       transform=ax.transAxes, fontsize=9,
                       horizontalalignment='right', verticalalignment='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
        
        ax.set_xlabel('N_traj', fontsize=12, fontweight='bold')
        ax.set_ylabel('T₂ (μs)', fontsize=12, fontweight='bold', color=COLORS['fid'])
        ax.tick_params(axis='y', labelcolor=COLORS['fid'], labelsize=11)
        ax.tick_params(axis='x', labelsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = output_dir / 'fig5_convergence_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    print("="*80)
    print("Generate Publication-Quality Figures for Dissertation")
    print("="*80)
    
    output_dir = Path('results_comparison/figures')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    data = load_data()
    
    # Generate figures
    print("\n📈 Generating figures...")
    
    # Figure 1: T2 vs tau_c (Main result)
    print("\n[1/5] Figure 1: T2 vs tau_c (FID)")
    plot_T2_vs_tau_c(data, output_dir)
    
    # Figure 2: MN regime slope
    print("\n[2/5] Figure 2: Motional Narrowing regime")
    plot_MN_regime_slope(data, output_dir)
    
    # Figure 3: Echo gain
    print("\n[3/5] Figure 3: Echo gain")
    plot_echo_gain(data, output_dir)
    
    # Figure 4: Representative curves
    print("\n[4/5] Figure 4: Representative coherence curves")
    plot_representative_curves(data, output_dir)
    
    # Figure 5: Convergence test
    print("\n[5/5] Figure 5: Convergence test")
    plot_convergence_test(data, output_dir)
    
    print("\n" + "="*80)
    print("✅ All figures generated!")
    print(f"📁 Output directory: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()

