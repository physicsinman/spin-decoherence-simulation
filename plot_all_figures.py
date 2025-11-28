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
    'crossover': '#DC143C', # Red/Crimson (distinct from others)
    'qs': '#B45309',       # Dark orange/brown
    'background': '#F9FAFB', # Light gray for background
}

# Physics parameters
# NOTE: These must match the actual simulation parameters used in sim_fid_sweep.py
gamma_e = 1.76e11  # rad/(s·T), electron gyromagnetic ratio
B_rms = 0.57e-6  # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration

def load_data():
    """Load all data files."""
    data_dir = Path('results')
    
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
    # IMPROVEMENT: Narrower crossover range (ξ = 0.5-2.0) for clearer regime distinction
    xi_valid = valid['xi'].values
    mn_mask = xi_valid < 0.5
    crossover_mask = (xi_valid >= 0.5) & (xi_valid < 2.0)
    qs_mask = xi_valid >= 2.0
    
    # Plot data with error bars, colored by regime
    if 'T2_lower' in valid.columns and 'T2_upper' in valid.columns:
        yerr_lower = valid['T2'] - valid['T2_lower']
        yerr_upper = valid['T2_upper'] - valid['T2']
        yerr = np.array([yerr_lower.values, yerr_upper.values])
        
        # Ensure minimum error bar visibility (even if CI is very small)
        # For log scale, use a minimum of 5% of T2 value for visibility
        ci_width_pct = (yerr[0] + yerr[1]) / valid['T2'].values * 100
        min_visible_error = valid['T2'].values * 0.05  # 5% of T2 value (increased for better visibility)
        
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
                       markersize=8, capsize=5, capthick=2.5, elinewidth=2.5,  # Enhanced for better visibility
                       label='FID (MN)', alpha=0.85, zorder=3, markeredgewidth=0.5)
        if crossover_mask.sum() > 0:
            cross_valid = valid[crossover_mask].copy()
            cross_yerr = yerr_adj[:, crossover_mask]
            ax.errorbar(cross_valid['tau_c'] * 1e6, 
                       cross_valid['T2'] * 1e6,
                       yerr=cross_yerr, fmt='o', color=COLORS['crossover'],
                       markersize=8, capsize=5, capthick=2.5, elinewidth=2.5,  # Enhanced for better visibility
                       label='FID (Crossover)', alpha=0.85, zorder=3, markeredgewidth=0.5)
        if qs_mask.sum() > 0:
            qs_valid = valid[qs_mask].copy()
            qs_yerr = yerr_adj[:, qs_mask]
            ax.errorbar(qs_valid['tau_c'] * 1e6, 
                       qs_valid['T2'] * 1e6,
                       yerr=qs_yerr, fmt='o', color=COLORS['qs'],
                       markersize=8, capsize=5, capthick=2.5, elinewidth=2.5,  # Enhanced for better visibility
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
    mask_MN = xi_theory < 0.5  # MN regime (where theory is valid) - updated to match new boundary
    
    # Plot theory curve only in MN regime
    # Note: Theory T₂ = 1/(Δω²τc) is only valid in MN regime
    # In QS regime, T₂ ≈ constant (independent of τc)
    # Crossover regime has no analytical solution (requires numerical methods)
    if mask_MN.sum() > 0:
        ax.plot(tau_c_theory[mask_MN] * 1e6, T2_MN[mask_MN] * 1e6,
               '--', color=COLORS['theory'], linewidth=2.5, 
               label='Theory (MN, valid for ξ < 0.5)', alpha=0.9, zorder=2)
    
    # Add QS regime theoretical value (horizontal line, only in QS regime)
    # CORRECTED: QS regime theory is T₂* = √2 / Δω (not 1/Δω)
    # This comes from the static limit of the coherence function: E(t) = exp(-(Δω·t)²/2)
    # At t = T₂*, E(T₂*) = 1/e, so (Δω·T₂*)²/2 = 1, giving T₂* = √2/Δω
    T2_QS_theory = np.sqrt(2.0) / Delta_omega
    mask_QS = xi_theory >= 2.0  # QS regime (where theory is valid) - updated to match new boundary
    
    # Only plot QS theory line in the QS regime range
    if mask_QS.sum() > 0:
        # Get tau_c range for QS regime
        tau_c_QS_min = tau_c_theory[mask_QS].min()
        tau_c_QS_max = tau_c_theory[mask_QS].max()
        
        # Plot horizontal line only in QS regime range
        ax.hlines(T2_QS_theory * 1e6, tau_c_QS_min * 1e6, tau_c_QS_max * 1e6,
                 color=COLORS['theory'], linestyle=':', linewidth=2.0, alpha=0.7,
                 label=f'Theory (QS, T$_2^*$ = √2/Δω ≈ {T2_QS_theory*1e6:.3f} μs, valid for ξ ≥ 2.0)', zorder=1)
    
    # Detect and highlight QS regime saturation (where T2 becomes constant)
    # This indicates simulation limitations in deep QS regime
    if qs_mask.sum() > 5:  # Need enough points to detect saturation
        qs_tau_c = valid[qs_mask]['tau_c'].values
        qs_T2 = valid[qs_mask]['T2'].values
        # Check if T2 is approximately constant in QS regime
        # Use coefficient of variation (CV) to detect saturation
        if len(qs_T2) > 5:
            qs_T2_mean = np.mean(qs_T2)
            qs_T2_std = np.std(qs_T2)
            qs_T2_cv = qs_T2_std / qs_T2_mean if qs_T2_mean > 0 else 0
            
            # If CV < 5%, T2 is essentially constant (saturation detected)
            if qs_T2_cv < 0.05:
                # Find where saturation starts (where T2 becomes constant)
                # Look for the point where T2 stabilizes
                saturation_start_idx = None
                for i in range(len(qs_T2) - 4):
                    window_T2 = qs_T2[i:i+5]
                    window_cv = np.std(window_T2) / np.mean(window_T2) if np.mean(window_T2) > 0 else 1
                    if window_cv < 0.05:
                        saturation_start_idx = i
                        break
                
                if saturation_start_idx is not None:
                    saturation_tau_c = qs_tau_c[saturation_start_idx]
                    saturation_T2 = qs_T2[saturation_start_idx]
                    
                    # Add vertical line to mark saturation start
                    ax.axvline(saturation_tau_c * 1e6, color='orange', linestyle='--', 
                             linewidth=1.5, alpha=0.5, zorder=1)
                    # IMPROVEMENT: Move annotation to left side to avoid covering data
                    ax.text(0.02, 0.15, 'Saturation\n(limitation)', 
                           transform=ax.transAxes, fontsize=7,
                           verticalalignment='bottom', horizontalalignment='left',
                           alpha=0.7, color='orange',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='orange'))
    
    # Add regime boundary lines (ξ = 0.5 and ξ = 2.0) - IMPROVED boundaries
    tau_c_min = valid['tau_c'].min()
    tau_c_max = valid['tau_c'].max()
    # Calculate tau_c values for ξ = 0.5 and ξ = 2.0
    tau_c_boundary_05 = 0.5 / Delta_omega
    tau_c_boundary_20 = 2.0 / Delta_omega
    
    # Only show boundaries if they're within the data range
    if tau_c_min <= tau_c_boundary_05 <= tau_c_max:
        ax.axvline(tau_c_boundary_05 * 1e6, color='gray', linestyle=':', 
                  linewidth=1.5, alpha=0.5, zorder=1)
        # Add text annotation
        ax.text(tau_c_boundary_05 * 1e6, ax.get_ylim()[1] * 0.7, 
               r'$\xi = 0.5$', rotation=90, fontsize=9, 
               verticalalignment='bottom', alpha=0.7)
    
    if tau_c_min <= tau_c_boundary_20 <= tau_c_max:
        ax.axvline(tau_c_boundary_20 * 1e6, color='gray', linestyle=':', 
                  linewidth=1.5, alpha=0.5, zorder=1)
        # Add text annotation
        ax.text(tau_c_boundary_20 * 1e6, ax.get_ylim()[1] * 0.7, 
               r'$\xi = 2.0$', rotation=90, fontsize=9, 
               verticalalignment='bottom', alpha=0.7)
    
    # Add information about parameters in the plot
    # Calculate actual parameter values for display
    Delta_omega_display = Delta_omega / 1e6  # Convert to MHz for readability
    param_text = f'γ$_e$ = {gamma_e/1e11:.2f}×10¹¹ rad/(s·T), B$_{{rms}}$ = {B_rms*1e6:.2f} μT\n'
    param_text += f'Δω = {Delta_omega_display:.2f} MHz, T$_2^*$ (QS) = {T2_QS_theory*1e6:.3f} μs'
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau_c$ (μs)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$T_2$ (μs)', fontsize=12, fontweight='bold')
    ax.set_title('FID Coherence Time vs Correlation Time', fontsize=13, fontweight='bold', pad=10)
    # IMPROVEMENT: Legend moved to center right to avoid covering boundaries
    ax.legend(loc='center right', frameon=True, fancybox=True, shadow=True, fontsize=10,  # Slightly reduced
             bbox_to_anchor=(0.995, 0.58))  # Moved up slightly
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # IMPROVEMENT: Parameter info moved to right middle, slightly larger and below legend to avoid overlap
    ax.text(0.995, 0.42, param_text, transform=ax.transAxes, fontsize=10,  # Increased for publication, moved up
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'),  # Increased padding
           family='monospace')
    
    # IMPROVEMENT: Add note about crossover regime (no analytical solution)
    # Place in lower left corner to avoid covering data points
    crossover_note = 'Note: Crossover regime (0.5 ≤ ξ < 2.0)\nhas no analytical solution'
    ax.text(0.02, 0.02, crossover_note, transform=ax.transAxes, fontsize=11,  # Increased for publication
           verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='purple'),
           color='purple', style='italic')
    
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
    
    # MN regime (xi < 0.2) - matches figure title
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
        
        # In crossover/QS regime (xi >= 0.5), gain should increase with xi
        crossover_qs_mask = valid_physical_xi['xi'] >= 0.5
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
    
    # Regime boundaries (use valid_physical) - IMPROVED: Match Figure 1 boundaries
    if 'xi' in valid_physical.columns:
        xi_valid = valid_physical['xi'].values
        mn_mask = xi_valid < 0.5
        crossover_mask = (xi_valid >= 0.5) & (xi_valid < 2.0)
        qs_mask = xi_valid >= 2.0
        
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
    
    # Add annotation for red points (gain = 1.0)
    red_points = valid_physical[valid_physical['echo_gain'] == 1.0]
    if len(red_points) > 0:
        # Mark red points with different style
        ax.scatter(red_points['tau_c'] * 1e6, red_points['echo_gain'],
                  color='red', s=120, alpha=0.8, zorder=5, marker='x', linewidths=2.5,
                  label=f'Fitting failure (T$_{{2,echo}}$ < T$_{{2,FID}}$)')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau_c$ (μs)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Echo Gain (T$_{2,echo}$ / T$_{2,FID}$)', fontsize=13, fontweight='bold')
    ax.set_title('Hahn Echo Gain vs Correlation Time', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # IMPROVEMENT: Add text annotation explaining red points - positioned better
    if len(red_points) > 0:
        ax.text(0.98, 0.05, 
               f'Note: {len(red_points)} points show fitting failure\n(T$_{{2,echo}}$ < T$_{{2,FID}}$), set to gain = 1',
               transform=ax.transAxes, fontsize=8,
               horizontalalignment='right', verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=1.5))
    
    # IMPROVEMENT: Set y-axis limits for better presentation
    # Cap at 6.0 for clarity (most gains are in 1-6 range)
    y_min = 0.8
    y_max = 6.5  # Slightly above max cap (6.0) for visibility
    ax.set_ylim([y_min, y_max])
    
    # Add horizontal line at gain = 6.0 to show cap
    ax.axhline(6.0, color='gray', linestyle=':', linewidth=1.0,
              alpha=0.4, zorder=1, label='Maximum cap (gain = 6)')
    
    plt.tight_layout()
    output_path = output_dir / 'fig3_echo_gain.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_noise_trajectories(data, output_dir):
    """Figure: OU Noise Trajectories (fast vs slow comparison)."""
    # Check if noise trajectory files exist
    data_dir = Path('results')
    fast_file = data_dir / 'noise_trajectory_fast.csv'
    slow_file = data_dir / 'noise_trajectory_slow.csv'
    
    if not fast_file.exists() or not slow_file.exists():
        print("⚠️  Noise trajectory files not found, generating them...")
        # Try to generate them
        try:
            import subprocess
            subprocess.run(['python3', 'generate_noise_data.py'], check=True)
        except Exception as e:
            print(f"❌ Could not generate noise trajectories: {e}")
            print("   Please run: python3 generate_noise_data.py")
            return
    
    # Load data
    df_fast = pd.read_csv(fast_file)
    df_slow = pd.read_csv(slow_file)
    
    # Physics parameters for regime calculation
    gamma_e = 1.76e11  # rad/(s·T)
    B_rms = 0.57e-6    # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration
    
    # Calculate xi for each
    tau_c_fast = 1e-8  # s (10 ns)
    tau_c_slow = 1e-4  # s (100 μs)
    xi_fast = gamma_e * B_rms * tau_c_fast
    xi_slow = gamma_e * B_rms * tau_c_slow
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Panel A: Fast noise (MN regime)
    ax1 = axes[0]
    t_fast = df_fast['time (s)'].values
    B_fast = df_fast['B_z (T)'].values
    
    # Plot trajectory - subsample for clarity
    step = max(1, len(t_fast) // 5000)  # Show up to 5000 points
    ax1.plot(t_fast[::step] * 1e9, B_fast[::step] * 1e6, 
            '-', color=COLORS.get('fid', '#2E86AB'), linewidth=1.2, alpha=0.85)
    
    # Add tau_c indicator lines
    tau_c_fast_ns = tau_c_fast * 1e9
    ax1.axvline(tau_c_fast_ns, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\tau_c$')
    ax1.axvline(2*tau_c_fast_ns, color='red', linestyle=':', linewidth=1.0, alpha=0.4)
    ax1.axvline(3*tau_c_fast_ns, color='red', linestyle=':', linewidth=1.0, alpha=0.4)
    
    ax1.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'$\delta B_z$ (μT)', fontsize=13, fontweight='bold')
    ax1.set_title(r'Fast Fluctuation: $\tau_c = 10$ ns, $\xi = {:.3f}$ (Motional Narrowing)'.format(xi_fast), 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, 
            fontsize=14, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add statistics text
    B_rms_emp_fast = np.std(B_fast) * 1e6
    ax1.text(0.98, 0.05, f'RMS: {B_rms_emp_fast:.2f} μT', 
            transform=ax1.transAxes, fontsize=10,
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Panel B: Slow noise (QS regime)
    ax2 = axes[1]
    t_slow = df_slow['time (s)'].values
    B_slow = df_slow['B_z (T)'].values
    
    # Plot trajectory - subsample for clarity
    step = max(1, len(t_slow) // 5000)  # Show up to 5000 points
    ax2.plot(t_slow[::step] * 1e6, B_slow[::step] * 1e6,
            '-', color=COLORS.get('qs', '#A23B72'), linewidth=1.2, alpha=0.85)
    
    # Add tau_c indicator lines
    tau_c_slow_us = tau_c_slow * 1e6
    ax2.axvline(tau_c_slow_us, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\tau_c$')
    ax2.axvline(2*tau_c_slow_us, color='red', linestyle=':', linewidth=1.0, alpha=0.4)
    ax2.axvline(3*tau_c_slow_us, color='red', linestyle=':', linewidth=1.0, alpha=0.4)
    
    ax2.set_xlabel('Time (μs)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'$\delta B_z$ (μT)', fontsize=13, fontweight='bold')
    ax2.set_title(r'Slow Fluctuation: $\tau_c = 100$ μs, $\xi = {:.1f}$ (Quasi-Static)'.format(xi_slow), 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes,
            fontsize=14, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add statistics text
    B_rms_emp_slow = np.std(B_slow) * 1e6
    ax2.text(0.98, 0.05, f'RMS: {B_rms_emp_slow:.2f} μT', 
            transform=ax2.transAxes, fontsize=10,
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / 'fig_noise_trajectories.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_conceptual_diagrams(output_dir):
    """Conceptual diagrams (fast vs slow noise, three regimes) - Supplementary figure."""
    # Create supplementary directory
    supp_dir = output_dir / 'supplementary'
    supp_dir.mkdir(exist_ok=True)
    
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.patches import Rectangle, Circle
    import matplotlib.patches as mpatches
    
    # ===== Figure 1: Fast vs Slow Noise Conceptual =====
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    fig1.patch.set_facecolor('white')
    
    # Left panel: Fast noise concept
    ax1 = axes1[0]
    ax1.set_xlim([0, 10])
    ax1.set_ylim([-1.5, 1.5])
    
    # Draw fast oscillating noise
    t_fast = np.linspace(0, 10, 200)
    noise_fast = 0.8 * np.sin(20 * t_fast) * np.exp(-0.1 * t_fast)
    ax1.plot(t_fast, noise_fast, '-', color=COLORS['fid'], linewidth=2.5, label='Fast fluctuations')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.fill_between(t_fast, noise_fast, 0, alpha=0.2, color=COLORS['fid'])
    
    ax1.set_xlabel('Time (arbitrary units)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'$\delta B_z$', fontsize=12, fontweight='bold')
    ax1.set_title(r'Fast Noise ($\tau_c \ll T_2$)', fontsize=13, fontweight='bold', pad=10)
    ax1.text(0.5, 0.95, 'Motional Narrowing', transform=ax1.transAxes,
            fontsize=11, style='italic', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Right panel: Slow noise concept
    ax2 = axes1[1]
    ax2.set_xlim([0, 10])
    ax2.set_ylim([-1.5, 1.5])
    
    # Draw slow varying noise
    t_slow = np.linspace(0, 10, 200)
    noise_slow = 0.8 * np.sin(0.5 * t_slow) + 0.2 * np.random.randn(200) * 0.1
    ax2.plot(t_slow, noise_slow, '-', color=COLORS['qs'], linewidth=2.5, label='Slow fluctuations')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax2.fill_between(t_slow, noise_slow, 0, alpha=0.2, color=COLORS['qs'])
    
    ax2.set_xlabel('Time (arbitrary units)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'$\delta B_z$', fontsize=12, fontweight='bold')
    ax2.set_title(r'Slow Noise ($\tau_c \gg T_2$)', fontsize=13, fontweight='bold', pad=10)
    ax2.text(0.5, 0.95, 'Quasi-Static', transform=ax2.transAxes,
            fontsize=11, style='italic', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    supp_dir = output_dir / 'supplementary'
    supp_dir.mkdir(exist_ok=True)
    output_path1 = supp_dir / 'conceptual_noise.png'
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path1}")
    plt.close(fig1)
    
    # ===== Figure 2: Three Regime Schematic =====
    fig2, ax = plt.subplots(figsize=(10, 7))
    fig2.patch.set_facecolor('white')
    
    # Create log-log plot
    tau_c = np.logspace(-9, -3, 1000)  # 1 ns to 1 ms
    xi = gamma_e * B_rms * tau_c
    
    # Calculate T2 for each regime (schematic)
    T2_mn = 1.0 / (gamma_e * B_rms)**2 / tau_c  # MN: T2 ∝ tau_c^-1
    T2_qs = 1.0 / (gamma_e * B_rms) * np.ones_like(tau_c)  # QS: T2 ≈ const
    
    # Transition region (smooth interpolation)
    xi_transition = 1.0
    transition_width = 0.5
    weight_mn = 1.0 / (1.0 + np.exp((xi - xi_transition) / transition_width))
    weight_qs = 1.0 - weight_mn
    T2_crossover = weight_mn * T2_mn + weight_qs * T2_qs
    
    # Plot regimes
    mn_mask = xi < 0.2
    crossover_mask = (xi >= 0.2) & (xi < 3)
    qs_mask = xi >= 3
    
    # MN regime
    ax.plot(tau_c[mn_mask] * 1e6, T2_mn[mn_mask] * 1e6, 
           '-', color=COLORS['fid'], linewidth=3, label='Motional Narrowing', zorder=3)
    
    # Crossover
    ax.plot(tau_c[crossover_mask] * 1e6, T2_crossover[crossover_mask] * 1e6,
           '-', color=COLORS['crossover'], linewidth=3, label='Crossover', zorder=3)
    
    # QS regime
    ax.plot(tau_c[qs_mask] * 1e6, T2_qs[qs_mask] * 1e6,
           '-', color=COLORS['qs'], linewidth=3, label='Quasi-Static', zorder=3)
    
    # Add regime boundaries
    tau_c_boundary1 = 0.2 / (gamma_e * B_rms)
    tau_c_boundary2 = 3.0 / (gamma_e * B_rms)
    T2_boundary1 = 1.0 / (gamma_e * B_rms)**2 / tau_c_boundary1
    T2_boundary2 = 1.0 / (gamma_e * B_rms)
    
    ax.axvline(tau_c_boundary1 * 1e6, color='gray', linestyle='--', 
              linewidth=2, alpha=0.5, zorder=2)
    ax.axvline(tau_c_boundary2 * 1e6, color='gray', linestyle='--',
              linewidth=2, alpha=0.5, zorder=2)
    
    # Add text annotations
    ax.text(0.15, 0.85, r'$\xi < 0.2$', transform=ax.transAxes,
           fontsize=12, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(0.5, 0.5, r'$0.2 \leq \xi < 3$', transform=ax.transAxes,
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    ax.text(0.85, 0.15, r'$\xi \geq 3$', transform=ax.transAxes,
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Add slope annotations
    # MN regime slope = -1
    x_slope = tau_c[mn_mask][len(tau_c[mn_mask])//2] * 1e6
    y_slope = T2_mn[mn_mask][len(T2_mn[mn_mask])//2] * 1e6
    ax.annotate(r'$T_2 \propto \tau_c^{-1}$', 
               xy=(x_slope, y_slope), xytext=(x_slope*0.7, y_slope*1.5),
               fontsize=11, style='italic',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # QS regime constant
    x_const = tau_c[qs_mask][len(tau_c[qs_mask])//2] * 1e6
    y_const = T2_qs[qs_mask][len(T2_qs[qs_mask])//2] * 1e6
    ax.annotate(r'$T_2 \approx$ const', 
               xy=(x_const, y_const), xytext=(x_const*1.3, y_const*1.2),
               fontsize=11, style='italic',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau_c$ (μs)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$T_2$ (μs)', fontsize=13, fontweight='bold')
    ax.set_title('Three Regime Schematic', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    supp_dir = output_dir / 'supplementary'
    supp_dir.mkdir(exist_ok=True)
    output_path2 = supp_dir / 'three_regimes.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path2}")
    plt.close(fig2)

def plot_representative_curves(data, output_dir):
    """Figure 4: Representative coherence decay curves - IMPROVED VERSION.
    
    Improvements:
    1. Dynamic x-axis ranges based on data availability
    2. Echo peak detection and annotation
    3. Better visualization for small tau_c differences
    4. Clear indication of data ranges
    5. Enhanced visual quality
    """
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))  # Increased for better visibility
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
            tau_c_str = tau_c_key.replace('tau_c_', '').replace('e-', 'e-').replace('e+', 'e+')
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
        
        # Determine data ranges for dynamic axis limits
        t_fid = None
        E_fid = None
        t_echo = None
        E_echo = None
        echo_peak_idx = None
        echo_peak_time = None
        echo_peak_value = None
        
        # Get FID data
        if 'time (s)' in fid_df.columns and E_col is not None:
            t_fid = fid_df['time (s)'].values
            E_fid = fid_df[E_col].values
        
        # Get Echo data
        # CRITICAL FIX: Echo data contains pulse delay τ, but echo is measured at time 2τ
        # For Hahn echo sequence (π/2 - τ - π - τ), the total time is 2τ
        # To compare with FID on the same time axis, we need to use 2τ as the time coordinate
        if 'echo_curves' in data and tau_c_key in data['echo_curves']:
            echo_df = data['echo_curves'][tau_c_key]
            if 'time (s)' in echo_df.columns and 'P_echo(t)' in echo_df.columns:
                tau_echo = echo_df['time (s)'].values  # This is actually τ (pulse delay)
                t_echo = 2.0 * tau_echo  # Convert to actual time: 2τ (total sequence time)
                E_echo = echo_df['P_echo(t)'].values
                # Find echo peak (maximum value, but skip initial points to find actual peak)
                # Hahn Echo typically increases from t=0, reaches a peak, then decays
                # Skip first 5% of points to avoid using initial value as peak
                if len(E_echo) > 10:
                    skip_points = max(1, int(len(E_echo) * 0.05))
                    # Find peak in the remaining data
                    E_echo_search = E_echo[skip_points:]
                    t_echo_search = t_echo[skip_points:]
                    if len(E_echo_search) > 0:
                        echo_peak_idx_search = np.argmax(E_echo_search)
                        echo_peak_idx = skip_points + echo_peak_idx_search
                        echo_peak_time = t_echo[echo_peak_idx] * 1e6  # Convert to μs (already 2τ)
                        echo_peak_value = E_echo[echo_peak_idx]
                        # Only mark as peak if it's significantly larger than initial value
                        if echo_peak_value > E_echo[0] * 1.01:  # At least 1% larger than initial
                            pass  # Valid peak
                        else:
                            # If peak is not significantly larger, don't mark it
                            echo_peak_time = None
                            echo_peak_value = None
                    else:
                        echo_peak_time = None
                        echo_peak_value = None
                elif len(E_echo) > 0:
                    # For very short arrays, just use maximum
                    echo_peak_idx = np.argmax(E_echo)
                    echo_peak_time = t_echo[echo_peak_idx] * 1e6
                    echo_peak_value = E_echo[echo_peak_idx]
                    # Only mark if not at t=0
                    if echo_peak_idx == 0:
                        echo_peak_time = None
                        echo_peak_value = None
                else:
                    echo_peak_time = None
                    echo_peak_value = None
        
        # Determine optimal x-axis range based on data availability
        # Strategy: Show actual data range, but ensure meaningful decay is visible
        # CRITICAL: Echo cannot start at t=0! Minimum time is 2*τ_min (Hahn echo sequence)
        x_min = 0.0
        echo_min_time = None
        if t_echo is not None and len(t_echo) > 0:
            # Echo starts at minimum 2τ, not at t=0
            echo_min_time = t_echo.min() * 1e6  # Convert to μs
            # For comparison with FID, we can still start x_min at 0, but Echo won't have data there
            # This is physically correct: Echo requires τ > 0, so 2τ > 0
        
        # Get actual data ranges first
        t_fid_max = 0.0
        t_echo_max = 0.0
        if t_fid is not None and len(t_fid) > 0:
            t_fid_max = t_fid.max() * 1e6  # Convert to μs
        if t_echo is not None and len(t_echo) > 0:
            t_echo_max = t_echo.max() * 1e6  # Convert to μs
        
        # Find where FID decays to significant level (based on regime)
        # Use data-driven approach: show meaningful decay for each regime
        x_max_fid = t_fid_max
        if t_fid is not None and E_fid is not None and len(t_fid) > 0:
            t_fid_us = t_fid * 1e6
            if tau_c_val <= 1e-7:  # MN regime - need much longer range to see decay
                # For very small tau_c (1e-8), show where it decays to 0.99 (about 95 μs)
                # For 1e-7, show where it decays to 0.90 (about 103 μs)
                if tau_c_val <= 1e-8:
                    # Show decay to 0.99 (visible decay in MN regime)
                    decay_idx = np.where(E_fid <= 0.99)[0]
                    if len(decay_idx) > 0:
                        x_max_fid = min(t_fid_us[decay_idx[0]] * 1.3, t_fid_max, 150.0)
                    else:
                        x_max_fid = min(t_fid_max, 100.0)
                else:  # 1e-7
                    # Show decay to 0.90 (more visible decay)
                    decay_idx = np.where(E_fid <= 0.90)[0]
                    if len(decay_idx) > 0:
                        x_max_fid = min(t_fid_us[decay_idx[0]] * 1.2, t_fid_max, 150.0)
                    else:
                        x_max_fid = min(t_fid_max, 100.0)
            elif tau_c_val >= 1e-6:  # QS regime
                # Show up to where it decays to 0.5 (significant decay)
                # For 1e-6, decay to 0.5 is at ~69 μs
                # For 1e-5, decay to 0.5 is at ~13 μs
                decay_idx = np.where(E_fid <= 0.5)[0]
                if len(decay_idx) > 0:
                    x_max_fid = min(t_fid_us[decay_idx[0]] * 1.3, t_fid_max, 100.0)
                else:
                    x_max_fid = min(t_fid_max, 50.0)
            else:  # Crossover
                # Show up to where it decays to 0.5
                decay_idx = np.where(E_fid <= 0.5)[0]
                if len(decay_idx) > 0:
                    x_max_fid = min(t_fid_us[decay_idx[0]] * 1.2, t_fid_max, 80.0)
                else:
                    x_max_fid = min(t_fid_max, 40.0)
        
        # Similar logic for Echo
        x_max_echo = t_echo_max
        if t_echo is not None and E_echo is not None and len(t_echo) > 0:
            t_echo_us = t_echo * 1e6
            if tau_c_val <= 1e-7:  # MN regime - need longer range
                # Similar logic to FID
                if tau_c_val <= 1e-8:
                    decay_idx = np.where(E_echo <= 0.99)[0]
                    if len(decay_idx) > 0:
                        x_max_echo = min(t_echo_us[decay_idx[0]] * 1.3, t_echo_max, 150.0)
                    else:
                        x_max_echo = min(t_echo_max, 100.0)
                else:  # 1e-7
                    decay_idx = np.where(E_echo <= 0.90)[0]
                    if len(decay_idx) > 0:
                        x_max_echo = min(t_echo_us[decay_idx[0]] * 1.2, t_echo_max, 150.0)
                    else:
                        x_max_echo = min(t_echo_max, 100.0)
            elif tau_c_val >= 1e-6:  # QS regime
                # Show decay to 0.5 (significant decay)
                decay_idx = np.where(E_echo <= 0.5)[0]
                if len(decay_idx) > 0:
                    x_max_echo = min(t_echo_us[decay_idx[0]] * 1.3, t_echo_max, 200.0)
                else:
                    x_max_echo = min(t_echo_max, 100.0)
            else:  # Crossover
                decay_idx = np.where(E_echo <= 0.5)[0]
                if len(decay_idx) > 0:
                    x_max_echo = min(t_echo_us[decay_idx[0]] * 1.2, t_echo_max, 80.0)
                else:
                    x_max_echo = min(t_echo_max, 40.0)
        
        # Set x-axis range: FIXED LOGIC - Prioritize showing FID decay
        # CRITICAL: Even if Echo is short, show FID decay properly
        if tau_c_val <= 1e-7:  # MN regime
            # For MN regime, FID decay is the main feature
            # Always show FID decay even if Echo is short
            if t_fid_max > t_echo_max * 5:  # FID is much longer than Echo
                # Use FID calculated range to show meaningful decay
                x_max = min(x_max_fid, t_fid_max * 1.05)
                # Ensure we show at least up to decay point
                if x_max < 50.0 and x_max_fid > 50.0:
                    x_max = min(x_max_fid, 150.0)  # Show at least up to decay point
            elif t_echo_max < t_fid_max * 0.1:  # Echo is extremely short (< 10% of FID)
                # Echo is too short - show FID decay instead
                # Use FID range to show meaningful decay
                x_max = min(x_max_fid, t_fid_max * 1.05)
                if x_max < 50.0 and x_max_fid > 50.0:
                    x_max = min(x_max_fid, 150.0)
            else:
                # Use the longer of the two
                x_max = min(max(x_max_fid, x_max_echo), max(t_fid_max, t_echo_max) * 1.05)
            
            # Removed: Warning annotation - not appropriate for publication
        elif tau_c_val >= 1e-6:  # QS regime
            # For QS, Echo might be longer - use the maximum that shows decay
            x_max = min(max(x_max_fid, x_max_echo), max(t_fid_max, t_echo_max) * 1.05)
        else:  # Crossover
            x_max = min(max(x_max_fid, x_max_echo), max(t_fid_max, t_echo_max) * 1.05)
        
        # Final safety check: ensure we have a reasonable range
        if x_max < 1.0:
            x_max = max(x_max, min(t_fid_max, t_echo_max) if (t_fid_max > 0 or t_echo_max > 0) else 1.0)
        
        # Plot FID with intelligent sampling for large datasets
        if t_fid is not None and E_fid is not None:
            mask_fid = (t_fid * 1e6 >= x_min) & (t_fid * 1e6 <= x_max)
            if mask_fid.sum() > 0:
                t_fid_plot = t_fid[mask_fid] * 1e6
                E_fid_plot = E_fid[mask_fid]
                # If too many points, sample intelligently (keep more points where decay is fast)
                if len(t_fid_plot) > 5000:
                    # Sample more densely at the beginning and end, less in the middle
                    n_samples = 5000
                    # Get indices for sampling
                    indices = np.linspace(0, len(t_fid_plot) - 1, n_samples, dtype=int)
                    t_fid_plot = t_fid_plot[indices]
                    E_fid_plot = E_fid_plot[indices]
                # DEBUG: Ensure we have valid data
                if len(t_fid_plot) > 0 and len(E_fid_plot) > 0:
                    ax.plot(t_fid_plot, E_fid_plot, 
                           '-', color=COLORS['fid'], linewidth=2.5,
                           label='FID', alpha=0.9, zorder=2)
                else:
                    print(f"⚠️  WARNING: No FID data to plot for {tau_c_key}")
        
        # Plot Echo curve if available with intelligent sampling
        if t_echo is not None and E_echo is not None:
            mask_echo = (t_echo * 1e6 >= x_min) & (t_echo * 1e6 <= x_max)
            if mask_echo.sum() > 0:
                t_echo_plot = t_echo[mask_echo] * 1e6
                E_echo_plot = E_echo[mask_echo]
                # If too many points, sample intelligently
                if len(t_echo_plot) > 5000:
                    n_samples = 5000
                    indices = np.linspace(0, len(t_echo_plot) - 1, n_samples, dtype=int)
                    t_echo_plot = t_echo_plot[indices]
                    E_echo_plot = E_echo_plot[indices]
                ax.plot(t_echo_plot, E_echo_plot, 
                       '--', color=COLORS['echo'], linewidth=2.5,
                       label='Hahn Echo', alpha=0.9, zorder=3)
                
                # CRITICAL: Mark the actual starting point of Echo (2τ_min)
                # This makes it visually clear that Echo doesn't start at t=0
                if echo_min_time is not None and echo_min_time >= x_min and echo_min_time <= x_max:
                    # Find the index of the first point
                    first_idx = np.argmin(np.abs(t_echo_plot - echo_min_time))
                    if first_idx < len(E_echo_plot):
                        ax.plot(echo_min_time, E_echo_plot[first_idx], 'o',
                               color=COLORS['echo'], markersize=6,
                               markeredgecolor='white', markeredgewidth=1.0,
                               zorder=4, label='Echo start (2τ_min)')
                
                # Mark echo peak if it's within the visible range
                if echo_peak_time is not None and x_min <= echo_peak_time <= x_max:
                    ax.plot(echo_peak_time, echo_peak_value, 'o', 
                           color=COLORS['echo'], markersize=8, 
                           markeredgecolor='white', markeredgewidth=1.5,
                           zorder=4, label='Echo Peak')
                    # Add annotation for echo peak
                    ax.annotate(f'Peak\n{echo_peak_time:.2f} μs', 
                              xy=(echo_peak_time, echo_peak_value),
                              xytext=(echo_peak_time + 0.15*x_max, echo_peak_value + 0.1),
                              fontsize=9, color=COLORS['echo'],
                              arrowprops=dict(arrowstyle='->', color=COLORS['echo'], 
                                           lw=1.5, alpha=0.7),
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor=COLORS['echo'], alpha=0.8),
                              zorder=5)
        
        # Set axis limits
        ax.set_xlim([x_min, x_max])
        
        # Set y-axis range: Always use 0 to 1 for consistent visualization (복사본 style)
        # This matches the reference graph where y-axis is fixed from 0 to 1
        y_min = 0.0
        y_max = 1.0
        ax.set_ylim([y_min, y_max])
        
        # Labels and title - Increased font sizes for publication
        ax.set_xlabel('Time (μs)', fontsize=16, fontweight='bold')
        ax.set_ylabel(r'$|E(t)|$', fontsize=16, fontweight='bold')
        ax.set_title(f'τc = {tau_c_val*1e6:.2f} μs', fontsize=18, fontweight='bold', pad=10)
        
        # Legend - only show if we have data
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            # Remove duplicate labels (in case echo peak is marked)
            seen = set()
            unique_handles = []
            unique_labels = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    unique_handles.append(h)
                    unique_labels.append(l)
            ax.legend(unique_handles, unique_labels, loc='upper right', 
                     fontsize=14, frameon=True, fancybox=True, shadow=True,
                     framealpha=0.95)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        
        # Add text annotation if echo data is truncated
        # Removed: "Echo data extends to ..." - technical info, not needed for publication
        
        # CRITICAL: Add note that Echo starts at 2τ_min, not t=0
        # This is physically important: Hahn echo requires τ > 0, so 2τ > 0
        if echo_min_time is not None and echo_min_time > 0.001:  # Only show if > 0.001 μs
            ax.text(0.02, 0.25, f'Note: Echo starts at t = {echo_min_time:.3f} μs\n(2τ_min, not t=0)', 
                   transform=ax.transAxes, fontsize=12,  # Increased for publication
                   ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                            alpha=0.7, edgecolor='orange'),
                   zorder=1)
        
        # Removed: "Echo decay is very small in MN regime" - can be explained in text
    
    # IMPROVEMENT: Adjusted spacing for better space efficiency
    plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=2.5)
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
    
    output_dir = Path('results/figures')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    data = load_data()
    
    # Generate figures
    print("\n📈 Generating figures...")
    
    # Figure 1: T2 vs tau_c (Main result)
    print("\n[1/5] Figure 1: T2 vs tau_c (FID) - Main result")
    plot_T2_vs_tau_c(data, output_dir)
    
    # Figure 2: Motional narrowing validation
    print("\n[2/5] Figure 2: Motional Narrowing regime validation")
    plot_MN_regime_slope(data, output_dir)
    
    # Figure 3: Echo gain
    print("\n[3/5] Figure 3: Echo gain")
    plot_echo_gain(data, output_dir)
    
    # Figure 4: Representative curves
    print("\n[4/6] Figure 4: Representative coherence curves")
    plot_representative_curves(data, output_dir)
    
    # Figure 5: Convergence test
    print("\n[5/6] Figure 5: Convergence test")
    plot_convergence_test(data, output_dir)
    
    # Figure 6: OU Noise Trajectories (fast vs slow)
    print("\n[6/6] Figure 6: OU Noise Trajectories (fast vs slow)")
    plot_noise_trajectories(data, output_dir)
    
    print("\n" + "="*80)
    print("✅ All figures generated!")
    print(f"📁 Output directory: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()

