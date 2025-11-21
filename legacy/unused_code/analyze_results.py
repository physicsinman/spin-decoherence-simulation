"""
Analysis and visualization for material comparison study.

This module creates publication-quality figures for comparing spin decoherence
across different materials (Si:P, GaAs) and noise models (OU, Double-OU).

Author: Material Comparison Project
Date: 2025-01-10
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from pathlib import Path
import pandas as pd


# ============================================================================
# PUBLICATION-QUALITY STYLE CONFIGURATION
# ============================================================================

# Apply PRB/APS style
rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 1.6,
    'lines.markersize': 6,
    'axes.linewidth': 1.2,
    'figure.figsize': (8, 6),
    'text.usetex': False,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    'axes.unicode_minus': False,
})

# Material Comparison Color Scheme
MATERIAL_COLORS = {
    'Si_P': '#1E40AF',   # Navy blue
    'GaAs': '#DC2626',   # Red
}

MATERIAL_MARKERS = {
    'Si_P': 'o',  # Circle
    'GaAs': 's',  # Square
}

NOISE_LINESTYLES = {
    'OU': '-',         # Solid
    'Double_OU': '--',  # Dashed
}

SEQUENCE_STYLES = {
    'FID': {'fillstyle': 'full', 'markeredgewidth': 1.0, 'zorder': 2, 'markersize': 7, 'alpha': 0.9},
    'Hahn': {'fillstyle': 'none', 'markeredgewidth': 1.5, 'zorder': 3, 'markersize': 7, 'alpha': 1.0},
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_results(result_file):
    """
    Load simulation results from JSON file.
    
    Parameters
    ----------
    result_file : str
        Path to JSON results file
        
    Returns
    -------
    results : list of dict
        List of result dictionaries
    """
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Handle both single result and list of results
    if isinstance(results, dict):
        results = [results]
    
    return results


def extract_data(all_results, material, noise_model, sequence):
    """
    Extract data for specific combination.
    
    Parameters
    ----------
    all_results : list
        List of all result dictionaries
    material : str
        'Si_P' or 'GaAs'
    noise_model : str
        'OU' or 'Double_OU'
    sequence : str
        'FID' or 'Hahn'
        
    Returns
    -------
    data : dict or None
        Extracted data arrays, or None if not found
    """
    # Find matching results
    matches = [r for r in all_results 
               if r.get('material') == material 
               and r.get('noise_model') == noise_model 
               and r.get('sequence') == sequence]
    
    if not matches:
        return None
    
    result = matches[0]
    data_list = result.get('data', [])
    
    if not data_list:
        return None
    
    # Extract arrays
    if noise_model == 'OU':
        tau_c = np.array([d['tau_c'] for d in data_list if d.get('T2') is not None])
        xi = np.array([d.get('xi', np.nan) for d in data_list if d.get('T2') is not None])
        tau_c1 = None
    else:  # Double_OU
        tau_c = np.array([d.get('tau_c2', d.get('tau_c', np.nan)) for d in data_list if d.get('T2') is not None])
        tau_c1 = data_list[0].get('tau_c1', None) if data_list else None
        if tau_c1 is None and 'parameters' in result:
            tau_c1 = result['parameters'].get('tau_c1')
        xi = np.array([d.get('xi2', d.get('xi', np.nan)) for d in data_list if d.get('T2') is not None])
    
    # Extract T2 values (only valid ones)
    valid_indices = [i for i, d in enumerate(data_list) if d.get('T2') is not None]
    
    if not valid_indices:
        return None
    
    T2 = np.array([data_list[i]['T2'] for i in valid_indices])
    T2_lower = np.array([data_list[i].get('T2_lower', np.nan) for i in valid_indices])
    T2_upper = np.array([data_list[i].get('T2_upper', np.nan) for i in valid_indices])
    beta = np.array([data_list[i].get('beta', np.nan) for i in valid_indices])
    
    return {
        'tau_c': tau_c,
        'xi': xi,
        'T2': T2,
        'T2_lower': T2_lower,
        'T2_upper': T2_upper,
        'beta': beta,
        'tau_c1': tau_c1,
    }


def apply_grid(ax, which='both'):
    """Apply standardized grid to axes."""
    ax.grid(True, which=which, alpha=0.25, linewidth=0.8, linestyle='--')
    ax.set_axisbelow(True)


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_T2_comparison(all_results, save_path=None, show_ci=True):
    """
    Create publication-quality T₂ vs τ_c comparison.
    
    Main result figure showing all material/noise/sequence combinations.
    
    Figure structure:
    - 2×2 subplots (rows: materials, cols: noise models)
    - Each subplot: FID (filled) + Hahn Echo (open)
    - Log-log scale
    - Error bars for FID (optional)
    
    Parameters
    ----------
    all_results : list
        List of all result dictionaries
    save_path : str, optional
        Base path for saving (without extension)
    show_ci : bool, default=True
        Whether to show error bars (confidence intervals)
        Set to False to remove error bars due to regime-dependent bootstrap degeneracy
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    materials = ['Si_P', 'GaAs']
    noise_models = ['OU', 'Double_OU']
    
    # Panel labels
    panel_labels = {
        (0, 0): '(a)', (0, 1): '(b)',
        (1, 0): '(c)', (1, 1): '(d)',
    }
    
    for i, mat in enumerate(materials):
        for j, noise in enumerate(noise_models):
            ax = axes[i, j]
            
            # Extract FID data
            fid_data = extract_data(all_results, mat, noise, 'FID')
            
            # Extract Hahn data
            hahn_data = extract_data(all_results, mat, noise, 'Hahn')
            
            # Get parameters for theory
            matches = [r for r in all_results 
                      if r.get('material') == mat and r.get('noise_model') == noise]
            if matches:
                params = matches[0].get('parameters', {})
                gamma_e = params.get('gamma_e', 1.76e11)
                if noise == 'OU':
                    B_rms = params.get('B_rms', None)
                else:  # Double_OU
                    B_rms1 = params.get('B_rms1', 0)
                    B_rms2 = params.get('B_rms2', 0)
                    B_rms = np.sqrt(B_rms1**2 + B_rms2**2)  # Effective RMS
            else:
                gamma_e = 1.76e11
                B_rms = None
            
            # Determine units based on material
            if mat == 'Si_P':
                tau_c_scale = 1e6  # to μs
                T2_scale = 1e3     # to ms
                T2_unit = 'ms'
            else:  # GaAs
                tau_c_scale = 1e6  # to μs
                T2_scale = 1e6     # to μs
                T2_unit = r'$\mu$s'
            
            # Plot FID
            if fid_data is not None:
                tau_c_plot = fid_data['tau_c'] * tau_c_scale
                T2_plot = fid_data['T2'] * T2_scale
                
                # Calculate error bars (handle None values)
                T2_lower = fid_data.get('T2_lower')
                T2_upper = fid_data.get('T2_upper')
                
                # Handle None or list with None values
                if T2_lower is not None:
                    T2_lower_arr = np.array([x if x is not None else np.nan for x in T2_lower])
                    T2_lower_plot = T2_lower_arr * T2_scale
                else:
                    T2_lower_plot = np.full_like(T2_plot, np.nan)
                
                if T2_upper is not None:
                    T2_upper_arr = np.array([x if x is not None else np.nan for x in T2_upper])
                    T2_upper_plot = T2_upper_arr * T2_scale
                else:
                    T2_upper_plot = np.full_like(T2_plot, np.nan)
                
                T2_err_low = T2_plot - T2_lower_plot
                T2_err_high = T2_upper_plot - T2_plot
                
                # Ensure error bars are non-negative and finite
                T2_err_low = np.maximum(0, np.nan_to_num(T2_err_low, nan=0, posinf=0, neginf=0))
                T2_err_high = np.maximum(0, np.nan_to_num(T2_err_high, nan=0, posinf=0, neginf=0))
                
                # Remove NaN/Inf
                valid = np.isfinite(T2_plot) & np.isfinite(tau_c_plot) & (T2_plot > 0)
                # Also ensure error bars are valid and non-negative
                valid = valid & np.isfinite(T2_err_low) & np.isfinite(T2_err_high)
                valid = valid & (T2_err_low >= 0) & (T2_err_high >= 0)
                
                if np.any(valid):
                    if show_ci:
                        # Final safety check: ensure no negative values in filtered arrays
                        err_low_final = T2_err_low[valid]
                        err_high_final = T2_err_high[valid]
                        err_low_final = np.maximum(0, err_low_final)
                        err_high_final = np.maximum(0, err_high_final)
                        
                        ax.errorbar(
                            tau_c_plot[valid], T2_plot[valid],
                            yerr=[err_low_final, err_high_final],
                            fmt=MATERIAL_MARKERS[mat],
                            color=MATERIAL_COLORS[mat],
                            linestyle=NOISE_LINESTYLES[noise],
                            fillstyle=SEQUENCE_STYLES['FID']['fillstyle'],
                            markersize=SEQUENCE_STYLES['FID']['markersize'],
                            linewidth=1.6,
                            markeredgewidth=SEQUENCE_STYLES['FID']['markeredgewidth'],
                            alpha=SEQUENCE_STYLES['FID']['alpha'],
                            capsize=3.5,
                            capthick=1.3,
                            elinewidth=1.2,
                            label='FID',
                            zorder=SEQUENCE_STYLES['FID']['zorder'],
                        )
                    else:
                        # Plot without error bars
                        ax.plot(
                            tau_c_plot[valid], T2_plot[valid],
                            marker=MATERIAL_MARKERS[mat],
                            color=MATERIAL_COLORS[mat],
                            linestyle=NOISE_LINESTYLES[noise],
                            fillstyle=SEQUENCE_STYLES['FID']['fillstyle'],
                            markersize=SEQUENCE_STYLES['FID']['markersize'],
                            linewidth=1.6,
                            markeredgewidth=SEQUENCE_STYLES['FID']['markeredgewidth'],
                            alpha=SEQUENCE_STYLES['FID']['alpha'],
                            label='FID',
                            zorder=SEQUENCE_STYLES['FID']['zorder'],
                        )
            
            # Plot Hahn Echo
            if hahn_data is not None:
                tau_c_plot = hahn_data['tau_c'] * tau_c_scale
                T2_plot = hahn_data['T2'] * T2_scale
                
                valid = np.isfinite(T2_plot) & np.isfinite(tau_c_plot) & (T2_plot > 0)
                
                if np.any(valid):
                    ax.plot(
                        tau_c_plot[valid], T2_plot[valid],
                        marker=MATERIAL_MARKERS[mat],
                        color=MATERIAL_COLORS[mat],
                        linestyle=NOISE_LINESTYLES[noise],
                        fillstyle=SEQUENCE_STYLES['Hahn']['fillstyle'],
                        markersize=SEQUENCE_STYLES['Hahn']['markersize'],
                        linewidth=1.8,
                        markeredgewidth=SEQUENCE_STYLES['Hahn']['markeredgewidth'],
                        alpha=SEQUENCE_STYLES['Hahn']['alpha'],
                        label='Echo',
                        zorder=SEQUENCE_STYLES['Hahn']['zorder'],
                    )
            
            # Calculate theoretical predictions (for OU only, Double-OU is more complex)
            # Store for both plotting and Y-axis range calculation
            tau_c_theory = None
            T2_theory = None
            Delta_omega = None
            xi_theory = None
            tau_c_mn_boundary = None
            tau_c_qs_boundary = None
            T2_static = None
            
            if noise == 'OU' and B_rms is not None and fid_data is not None:
                from simulate import estimate_characteristic_T2
                tau_c_data = fid_data['tau_c']
                if len(tau_c_data) > 0:
                    # Generate smooth theory curve
                    tau_c_theory = np.logspace(
                        np.log10(tau_c_data.min()),
                        np.log10(tau_c_data.max()),
                        200
                    )
                    T2_theory = np.array([estimate_characteristic_T2(tc, gamma_e, B_rms) 
                                         for tc in tau_c_theory])
                    Delta_omega = gamma_e * B_rms
                    xi_theory = Delta_omega * tau_c_theory
                    
                    # Plot theory line
                    ax.loglog(tau_c_theory * tau_c_scale, T2_theory * T2_scale,
                             '--', color='#991B1B', linewidth=2, alpha=0.7,
                             label='Theory (OU)', zorder=1)
                    
                    # Store regime boundary info for later shading (after axis limits are set)
                    tau_c_mn_boundary = 0.2 / Delta_omega  # ξ = 0.2
                    tau_c_qs_boundary = 2.0 / Delta_omega   # ξ = 2.0
                    T2_static = np.sqrt(2.0) / Delta_omega
            
            # Formatting
            ax.set_xscale('log')
            ax.set_yscale('log')
            apply_grid(ax, which='both')
            
            # Auto-adjust Y-axis range for better visibility
            # Collect all valid T2 values (including theory if available)
            all_T2_values = []
            
            if fid_data is not None:
                fid_valid = np.isfinite(fid_data['T2']) & (fid_data['T2'] > 0)
                if np.any(fid_valid):
                    all_T2_values.extend((fid_data['T2'][fid_valid] * T2_scale).tolist())
                    # Include error bars if available
                    if show_ci and T2_upper is not None:
                        T2_upper_valid = np.array([x if x is not None and np.isfinite(x) else np.nan 
                                                   for x in T2_upper])
                        valid_upper = fid_valid & np.isfinite(T2_upper_valid) & (T2_upper_valid > 0)
                        if np.any(valid_upper):
                            all_T2_values.extend((T2_upper_valid[valid_upper] * T2_scale).tolist())
            
            if hahn_data is not None:
                hahn_valid = np.isfinite(hahn_data['T2']) & (hahn_data['T2'] > 0)
                if np.any(hahn_valid):
                    all_T2_values.extend((hahn_data['T2'][hahn_valid] * T2_scale).tolist())
            
            # Include theory curve if available
            if T2_theory is not None:
                all_T2_values.extend((T2_theory * T2_scale).tolist())
            
            # Set Y-axis limits with padding
            if all_T2_values:
                all_T2_array = np.array(all_T2_values)
                all_T2_array = all_T2_array[all_T2_array > 0]  # Only positive values
                if len(all_T2_array) > 0:
                    y_min = np.min(all_T2_array)
                    y_max = np.max(all_T2_array)
                    # Add 20% padding on both sides (log scale)
                    y_range = y_max / y_min
                    y_min_adj = y_min / (y_range ** 0.1)
                    y_max_adj = y_max * (y_range ** 0.1)
                    ax.set_ylim(y_min_adj, y_max_adj)
            
            # Add regime shading and boundary lines (after axis limits are set)
            if noise == 'OU' and B_rms is not None and tau_c_mn_boundary is not None:
                # Get axis limits after they're set
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # Shade regime regions (behind data, zorder=0)
                # Motional-Narrowing regime (ξ < 0.2)
                mn_boundary_plot = tau_c_mn_boundary * tau_c_scale
                if mn_boundary_plot > xlim[0]:
                    ax.axvspan(xlim[0], min(mn_boundary_plot, xlim[1]),
                              color='#DBEAFE', alpha=0.25, zorder=0, label='_MN_region')
                
                # Transition regime (0.2 < ξ < 2.0)
                qs_boundary_plot = tau_c_qs_boundary * tau_c_scale
                if qs_boundary_plot > mn_boundary_plot:
                    ax.axvspan(mn_boundary_plot, min(qs_boundary_plot, xlim[1]),
                              color='#FEF3C7', alpha=0.15, zorder=0, label='_Transition_region')
                
                # Quasi-static regime (ξ > 2.0)
                if qs_boundary_plot < xlim[1]:
                    ax.axvspan(max(qs_boundary_plot, xlim[0]), xlim[1],
                              color='#FEE2E2', alpha=0.25, zorder=0, label='_QS_region')
                
                # Add regime boundary lines
                # MN/Transition boundary
                if mn_boundary_plot > xlim[0] and mn_boundary_plot < xlim[1]:
                    ax.axvline(mn_boundary_plot, 
                              color='#059669', linestyle='--', linewidth=1.3, 
                              alpha=0.65, zorder=2, dashes=(5, 3))
                
                # Transition/QS boundary
                if qs_boundary_plot > xlim[0] and qs_boundary_plot < xlim[1]:
                    ax.axvline(qs_boundary_plot, 
                              color='#DC2626', linestyle='--', linewidth=1.3, 
                              alpha=0.65, zorder=2, dashes=(5, 3))
                
                # Quasi-static limit line
                T2_static_plot = T2_static * T2_scale
                if T2_static_plot > ylim[0] and T2_static_plot < ylim[1]:
                    ax.axhline(T2_static_plot, color='#DC2626', 
                              linestyle=':', linewidth=1.8, alpha=0.7, zorder=2)
            
            # Labels
            if noise == 'OU':
                ax.set_xlabel(r'$\tau_c$ ($\mu$s)', fontsize=13)
            else:
                ax.set_xlabel(r'$\tau_{c2}$ ($\mu$s)', fontsize=13)
            ax.set_ylabel(f'$T_2$ ({T2_unit})', fontsize=13)
            
            # Title
            mat_label = mat.replace('_', ':')
            noise_label = 'Single OU' if noise == 'OU' else 'Double-OU'
            if noise == 'Double_OU' and fid_data is not None and fid_data.get('tau_c1') is not None:
                tau_c1_us = fid_data['tau_c1'] * 1e6
                title = f"{panel_labels[(i, j)]} {mat_label} | {noise_label} ($\\tau_{{c1}} = {tau_c1_us:.2f}$ $\\mu$s)"
            else:
                title = f"{panel_labels[(i, j)]} {mat_label} | {noise_label}"
            ax.set_title(title, fontsize=14, fontweight='bold', loc='left')
            
            # Legend
            if (fid_data is not None and np.any(np.isfinite(fid_data['T2']))) or \
               (hahn_data is not None and np.any(np.isfinite(hahn_data['T2']))):
                ax.legend(
                    loc='best',
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='black',
                    fancybox=True,
                    fontsize=11,
                )
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.95, 
                       wspace=0.25, hspace=0.25)
    
    # Save
    if save_path:
        for fmt in ['png', 'pdf']:
            filename = f"{save_path}.{fmt}"
            dpi = 300 if fmt == 'png' else None
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                       pad_inches=0.05, facecolor='white', format=fmt)
            print(f"✓ Saved: {filename}")
    
    return fig


def plot_echo_enhancement(all_results, save_path=None, show_ci=False):
    """
    Plot echo enhancement factor η = T₂,echo / T₂,FID.
    
    Shows how much Hahn echo improves T₂ compared to FID.
    Physical constraint: η ≥ 1 always.
    
    Parameters
    ----------
    all_results : list
        List of all result dictionaries
    save_path : str, optional
        Base path for saving (without extension)
    show_ci : bool, default=False
        Whether to show error bars (currently not implemented for echo enhancement)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    materials = ['Si_P', 'GaAs']
    noise_models = ['OU', 'Double_OU']
    
    colors = {
        'OU': '#1E40AF',         # Navy blue
        'Double_OU': '#DC2626',  # Red
    }
    
    for i, mat in enumerate(materials):
        ax = axes[i]
        
        for noise in noise_models:
            # Extract FID and Hahn data
            fid_data = extract_data(all_results, mat, noise, 'FID')
            hahn_data = extract_data(all_results, mat, noise, 'Hahn')
            
            if fid_data is not None and hahn_data is not None:
                # Match tau_c values (they should be the same)
                # For OU: use tau_c, for Double_OU: use tau_c2
                if noise == 'OU':
                    tau_c_fid = fid_data['tau_c']
                    tau_c_hahn = hahn_data['tau_c']
                else:
                    tau_c_fid = fid_data['tau_c']
                    tau_c_hahn = hahn_data['tau_c']
                
                # Find common tau_c values
                common_tau_c = np.intersect1d(tau_c_fid, tau_c_hahn)
                
                if len(common_tau_c) > 0:
                    # Get indices for common values
                    fid_indices = [np.where(tau_c_fid == tc)[0][0] for tc in common_tau_c]
                    hahn_indices = [np.where(tau_c_hahn == tc)[0][0] for tc in common_tau_c]
                    
                    T2_fid_common = fid_data['T2'][fid_indices]
                    T2_hahn_common = hahn_data['T2'][hahn_indices]
                    
                    # Compute enhancement
                    eta = T2_hahn_common / T2_fid_common
                    
                    # Filter valid values
                    valid = np.isfinite(eta) & np.isfinite(common_tau_c) & (eta > 0) & (T2_fid_common > 0)
                    
                    if np.any(valid):
                        tau_c_plot = common_tau_c[valid] * 1e6  # to μs
                        eta_plot = eta[valid]
                        
                        noise_label = 'Single OU' if noise == 'OU' else 'Double-OU'
                        
                        ax.plot(
                            tau_c_plot, eta_plot,
                            marker='o',
                            color=colors[noise],
                            linestyle=NOISE_LINESTYLES[noise],
                            linewidth=1.6,
                            markersize=6,
                            label=noise_label,
                        )
                        
                        # Physical check: η should be ≥ 1
                        if np.any(eta_plot < 1.0):
                            print(f"⚠️  Warning: {mat} {noise} has η < 1 (unphysical)")
        
        # Reference line η = 1
        ax.axhline(1, color='black', linestyle=':', linewidth=1.5, 
                   alpha=0.5, zorder=1, label=r'$\eta = 1$')
        
        # Formatting
        ax.set_xscale('log')
        apply_grid(ax, which='both')
        ax.set_xlabel(r'$\tau_c$ ($\mu$s)', fontsize=13)
        ax.set_ylabel(r'$\eta = T_{2,\mathrm{echo}} / T_{2,\mathrm{FID}}$', fontsize=13)
        
        # Title
        mat_label = mat.replace('_', ':')
        panel_label = '(a)' if i == 0 else '(b)'
        title = f"{panel_label} {mat_label}"
        ax.set_title(title, fontsize=14, fontweight='bold', loc='left')
        
        # Legend
        ax.legend(loc='best', frameon=True, framealpha=0.9, 
                 edgecolor='black', fancybox=True, fontsize=11)
        
        # Set y-axis to start at 0.8 or so
        ax.set_ylim(bottom=0.8)
    
    # Adjust layout
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.92, wspace=0.30)
    
    # Save
    if save_path:
        for fmt in ['png', 'pdf']:
            filename = f"{save_path}.{fmt}"
            dpi = 300 if fmt == 'png' else None
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                       pad_inches=0.05, facecolor='white', format=fmt)
            print(f"✓ Saved: {filename}")
    
    return fig


def create_summary_table(all_results, save_path=None):
    """
    Create summary table with key statistics.
    
    Table columns:
    - Material
    - Noise Model
    - T₂,FID (mean ± std)
    - T₂,Hahn (mean ± std)
    - η (mean)
    
    Parameters
    ----------
    all_results : list
        List of all result dictionaries
    save_path : str, optional
        Path for saving CSV file
        
    Returns
    -------
    df : pandas.DataFrame
        Summary table
    """
    rows = []
    
    materials = ['Si_P', 'GaAs']
    noise_models = ['OU', 'Double_OU']
    
    for mat in materials:
        for noise in noise_models:
            # Extract FID data
            fid_data = extract_data(all_results, mat, noise, 'FID')
            
            # Extract Hahn data
            hahn_data = extract_data(all_results, mat, noise, 'Hahn')
            
            # Compute statistics
            if fid_data is not None:
                T2_fid = fid_data['T2']
                valid_fid = np.isfinite(T2_fid) & (T2_fid > 0)
                T2_fid_mean = np.mean(T2_fid[valid_fid]) if np.any(valid_fid) else np.nan
                T2_fid_std = np.std(T2_fid[valid_fid]) if np.any(valid_fid) else np.nan
            else:
                T2_fid_mean = np.nan
                T2_fid_std = np.nan
            
            if hahn_data is not None:
                T2_hahn = hahn_data['T2']
                valid_hahn = np.isfinite(T2_hahn) & (T2_hahn > 0)
                T2_hahn_mean = np.mean(T2_hahn[valid_hahn]) if np.any(valid_hahn) else np.nan
                T2_hahn_std = np.std(T2_hahn[valid_hahn]) if np.any(valid_hahn) else np.nan
            else:
                T2_hahn_mean = np.nan
                T2_hahn_std = np.nan
            
            # Enhancement
            if not np.isnan(T2_fid_mean) and not np.isnan(T2_hahn_mean) and T2_fid_mean > 0:
                eta_mean = T2_hahn_mean / T2_fid_mean
            else:
                eta_mean = np.nan
            
            # Units
            if mat == 'Si_P':
                T2_fid_mean *= 1e3  # to ms
                T2_fid_std *= 1e3
                T2_hahn_mean *= 1e3
                T2_hahn_std *= 1e3
                unit = 'ms'
            else:
                T2_fid_mean *= 1e6  # to μs
                T2_fid_std *= 1e6
                T2_hahn_mean *= 1e6
                T2_hahn_std *= 1e6
                unit = 'μs'
            
            # Format strings
            if not np.isnan(T2_fid_mean):
                T2_fid_str = f"{T2_fid_mean:.2f} ± {T2_fid_std:.2f} {unit}"
            else:
                T2_fid_str = "N/A"
            
            if not np.isnan(T2_hahn_mean):
                T2_hahn_str = f"{T2_hahn_mean:.2f} ± {T2_hahn_std:.2f} {unit}"
            else:
                T2_hahn_str = "N/A"
            
            if not np.isnan(eta_mean):
                eta_str = f"{eta_mean:.2f}"
            else:
                eta_str = "N/A"
            
            rows.append({
                'Material': mat.replace('_', ':'),
                'Noise Model': 'Single OU' if noise == 'OU' else 'Double-OU',
                'T2_FID': T2_fid_str,
                'T2_Hahn': T2_hahn_str,
                'Enhancement η': eta_str,
            })
    
    df = pd.DataFrame(rows)
    
    # Save
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✓ Saved: {save_path}")
    
    return df


def create_clean_summary_table(all_results, save_path=None):
    """
    Create clean summary table for publication with essential columns only.
    
    Table columns:
    - Material
    - Noise_Model
    - Sequence
    - tau_c_mean (μs)
    - T2_mean (ms for Si:P, μs for GaAs)
    - Regime (QS/MN/Crossover)
    - eta (for Echo sequences only)
    
    Parameters
    ----------
    all_results : list
        List of all result dictionaries
    save_path : str, optional
        Path for saving CSV file
        
    Returns
    -------
    df : pandas.DataFrame
        Clean summary table
    """
    rows = []
    
    materials = ['Si_P', 'GaAs']
    noise_models = ['OU', 'Double_OU']
    sequences = ['FID', 'Hahn']
    
    for mat in materials:
        for noise in noise_models:
            for seq in sequences:
                data = extract_data(all_results, mat, noise, seq)
                
                if data is not None:
                    tau_c = data['tau_c']
                    T2 = data['T2']
                    
                    # Compute mean values
                    valid = np.isfinite(tau_c) & np.isfinite(T2) & (T2 > 0) & (tau_c > 0)
                    
                    if np.any(valid):
                        tau_c_mean = np.mean(tau_c[valid])
                        T2_mean = np.mean(T2[valid])
                        
                        # Determine regime (approximate based on xi)
                        matches = [r for r in all_results 
                                  if r.get('material') == mat and r.get('noise_model') == noise]
                        if matches:
                            params = matches[0].get('parameters', {})
                            gamma_e = params.get('gamma_e', 1.76e11)
                            if noise == 'OU':
                                B_rms = params.get('B_rms', None)
                            else:
                                B_rms1 = params.get('B_rms1', 0)
                                B_rms2 = params.get('B_rms2', 0)
                                B_rms = np.sqrt(B_rms1**2 + B_rms2**2)
                            
                            if B_rms is not None:
                                Delta_omega = gamma_e * B_rms
                                xi_mean = Delta_omega * tau_c_mean
                                
                                if xi_mean < 0.2:
                                    regime = 'MN'  # Motional narrowing
                                elif xi_mean > 2.0:
                                    regime = 'QS'  # Quasi-static
                                else:
                                    regime = 'Crossover'
                            else:
                                regime = 'Unknown'
                        else:
                            regime = 'Unknown'
                        
                        # Units conversion
                        if mat == 'Si_P':
                            tau_c_mean_us = tau_c_mean * 1e6
                            T2_mean_ms = T2_mean * 1e3
                            T2_unit = 'ms'
                            T2_value = T2_mean_ms
                        else:  # GaAs
                            tau_c_mean_us = tau_c_mean * 1e6
                            T2_mean_us = T2_mean * 1e6
                            T2_unit = 'μs'
                            T2_value = T2_mean_us
                        
                        # Compute eta for Echo sequences
                        if seq == 'Hahn':
                            fid_data = extract_data(all_results, mat, noise, 'FID')
                            if fid_data is not None:
                                T2_fid = fid_data['T2']
                                valid_fid = np.isfinite(T2_fid) & (T2_fid > 0)
                                if np.any(valid_fid):
                                    # Match tau_c values for eta calculation
                                    common_tau_c = np.intersect1d(tau_c, fid_data['tau_c'])
                                    if len(common_tau_c) > 0:
                                        fid_indices = [np.where(fid_data['tau_c'] == tc)[0][0] 
                                                      for tc in common_tau_c]
                                        hahn_indices = [np.where(tau_c == tc)[0][0] 
                                                       for tc in common_tau_c]
                                        T2_fid_common = fid_data['T2'][fid_indices]
                                        T2_hahn_common = T2[hahn_indices]
                                        eta = np.mean(T2_hahn_common / T2_fid_common)
                                    else:
                                        eta = np.nan
                                else:
                                    eta = np.nan
                            else:
                                eta = np.nan
                        else:
                            eta = np.nan
                        
                        rows.append({
                            'Material': mat.replace('_', ':'),
                            'Noise_Model': 'Single OU' if noise == 'OU' else 'Double-OU',
                            'Sequence': seq,
                            'tau_c_mean': f"{tau_c_mean_us:.2f}",
                            'T2_mean': f"{T2_value:.2f}",
                            'T2_unit': T2_unit,
                            'Regime': regime,
                            'eta': f"{eta:.2f}" if not np.isnan(eta) else '-'
                        })
    
    df = pd.DataFrame(rows)
    
    # Save
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✓ Saved: {save_path}")
    
    return df


def plot_dimensionless_collapse(all_results, save_path=None):
    """
    Create dimensionless collapse plot: Y = T₂·(γ_e·B_rms)²·τ_c vs ξ = γ_e·B_rms·τ_c.
    
    This allows fair comparison between materials with different B_rms and tau_c ranges.
    Theory: Y → 1 in MN regime (ξ << 1), Y ∝ ξ in quasi-static regime (ξ >> 1).
    
    Parameters
    ----------
    all_results : list
        List of all result dictionaries
    save_path : str, optional
        Base path for saving (without extension)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    materials = ['Si_P', 'GaAs']
    noise_models = ['OU', 'Double_OU']
    
    for i, mat in enumerate(materials):
        ax = axes[i]
        
        # Get parameters
        matches = [r for r in all_results if r.get('material') == mat]
        if matches:
            params = matches[0].get('parameters', {})
            gamma_e = params.get('gamma_e', 1.76e11)
        else:
            gamma_e = 1.76e11
        
        for noise in noise_models:
            fid_data = extract_data(all_results, mat, noise, 'FID')
            
            if fid_data is not None:
                # Get B_rms
                matches = [r for r in all_results 
                          if r.get('material') == mat and r.get('noise_model') == noise]
                if matches:
                    params = matches[0].get('parameters', {})
                    if noise == 'OU':
                        B_rms = params.get('B_rms', None)
                    else:
                        B_rms1 = params.get('B_rms1', 0)
                        B_rms2 = params.get('B_rms2', 0)
                        B_rms = np.sqrt(B_rms1**2 + B_rms2**2)
                else:
                    B_rms = None
                
                if B_rms is not None:
                    tau_c = fid_data['tau_c']
                    T2 = fid_data['T2']
                    
                    # Compute dimensionless variables
                    Delta_omega = gamma_e * B_rms
                    xi = Delta_omega * tau_c
                    Y = T2 * Delta_omega**2 * tau_c
                    
                    # Filter valid values
                    valid = np.isfinite(xi) & np.isfinite(Y) & (xi > 0) & (Y > 0)
                    
                    if np.any(valid):
                        noise_label = 'Single OU' if noise == 'OU' else 'Double-OU'
                        
                        ax.loglog(xi[valid], Y[valid],
                                marker=MATERIAL_MARKERS[mat],
                                color=MATERIAL_COLORS[mat],
                                linestyle=NOISE_LINESTYLES[noise],
                                linewidth=1.6,
                                markersize=6,
                                label=noise_label,
                                zorder=2)
        
        # Theoretical lines
        # MN limit: Y = 1 (horizontal line)
        ax.axhline(1, color='#059669', linestyle='--', linewidth=2,
                  alpha=0.7, zorder=1, label=r'Theory (MN: $Y = 1$)')
        
        # Quasi-static limit: Y = sqrt(2) * xi (slope = 1)
        xi_qs = np.logspace(-2, 2, 100)
        Y_qs = np.sqrt(2.0) * xi_qs
        ax.loglog(xi_qs, Y_qs, '--', color='#DC2626', linewidth=2,
                 alpha=0.7, zorder=1, label=r'Theory (QS: $Y = \sqrt{2}\xi$)')
        
        # Regime markers
        ax.axvline(0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                  zorder=0, label=r'$\xi = 0.2$ (MN boundary)')
        ax.axvline(2.0, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                  zorder=0, label=r'$\xi = 2$ (QS boundary)')
        
        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        apply_grid(ax, which='both')
        ax.set_xlabel(r'$\xi = \gamma_e B_{\mathrm{rms}} \tau_c$', fontsize=13)
        ax.set_ylabel(r'$Y = T_2 \cdot (\gamma_e B_{\mathrm{rms}})^2 \tau_c$', fontsize=13)
        
        # Title
        mat_label = mat.replace('_', ':')
        panel_label = '(a)' if i == 0 else '(b)'
        title = f"{panel_label} {mat_label}"
        ax.set_title(title, fontsize=14, fontweight='bold', loc='left')
        
        # Legend
        ax.legend(loc='best', frameon=True, framealpha=0.9,
                 edgecolor='black', fancybox=True, fontsize=10, ncol=1)
    
    # Adjust layout
    plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12, top=0.92, wspace=0.25)
    
    # Save
    if save_path:
        for fmt in ['png', 'pdf']:
            filename = f"{save_path}.{fmt}"
            dpi = 300 if fmt == 'png' else None
            fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                      pad_inches=0.05, facecolor='white', format=fmt)
            print(f"✓ Saved: {filename}")
    
    return fig


def plot_eta_dimensionless_collapse(all_results, save_path=None):
    """
    Create dimensionless collapse plot: η = T_{2,echo}/T_{2,FID} vs τ_c/T_{2,echo}.
    
    This plot shows the echo enhancement factor as a function of normalized correlation time.
    Key physics:
    - Static regime (τ_c >> T_{2,echo}): η → large enhancement
    - Motional narrowing (τ_c << T_{2,echo}): η → 1 (no enhancement)
    - Regime boundary markers at ξ = 0.2 (MN) and ξ = 2.0 (QS)
    
    Parameters
    ----------
    all_results : list
        List of all result dictionaries
    save_path : str, optional
        Base path for saving (without extension)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    materials = ['Si_P', 'GaAs']
    noise_models = ['OU', 'Double_OU']
    
    colors = {
        'OU': '#1E40AF',         # Navy blue
        'Double_OU': '#DC2626',  # Red
    }
    
    for i, mat in enumerate(materials):
        ax = axes[i]
        
        for noise in noise_models:
            # Extract FID and Hahn data
            fid_data = extract_data(all_results, mat, noise, 'FID')
            hahn_data = extract_data(all_results, mat, noise, 'Hahn')
            
            if fid_data is not None and hahn_data is not None:
                # Match tau_c values
                if noise == 'OU':
                    tau_c_fid = fid_data['tau_c']
                    tau_c_hahn = hahn_data['tau_c']
                else:
                    tau_c_fid = fid_data['tau_c']
                    tau_c_hahn = hahn_data['tau_c']
                
                # Find common tau_c values
                common_tau_c = np.intersect1d(tau_c_fid, tau_c_hahn)
                
                if len(common_tau_c) > 0:
                    # Get indices for common values
                    fid_indices = [np.where(tau_c_fid == tc)[0][0] for tc in common_tau_c]
                    hahn_indices = [np.where(tau_c_hahn == tc)[0][0] for tc in common_tau_c]
                    
                    T2_fid_common = fid_data['T2'][fid_indices]
                    T2_hahn_common = hahn_data['T2'][hahn_indices]
                    
                    # Compute η and normalized τ_c
                    eta = T2_hahn_common / T2_fid_common
                    tau_c_norm = common_tau_c / T2_hahn_common
                    
                    # Filter valid values
                    valid = (np.isfinite(eta) & np.isfinite(tau_c_norm) & 
                            (eta > 0) & (tau_c_norm > 0) & (T2_fid_common > 0) & 
                            (T2_hahn_common > 0))
                    
                    if np.any(valid):
                        tau_c_norm_plot = tau_c_norm[valid]
                        eta_plot = eta[valid]
                        
                        noise_label = 'Single OU' if noise == 'OU' else 'Double-OU'
                        
                        ax.loglog(
                            tau_c_norm_plot, eta_plot,
                            marker='o',
                            color=colors[noise],
                            linestyle=NOISE_LINESTYLES[noise],
                            linewidth=1.6,
                            markersize=6,
                            label=noise_label,
                            zorder=2
                        )
        
        # Reference line η = 1 (no enhancement)
        ax.axhline(1, color='black', linestyle=':', linewidth=1.5, 
                  alpha=0.5, zorder=1, label=r'$\eta = 1$ (no enhancement)')
        
        # Get parameters for regime boundaries
        matches = [r for r in all_results if r.get('material') == mat]
        if matches:
            params = matches[0].get('parameters', {})
            gamma_e = params.get('gamma_e', 1.76e11)
            
            # For OU, get B_rms to compute regime boundaries
            ou_match = [r for r in all_results 
                       if r.get('material') == mat and r.get('noise_model') == 'OU']
            if ou_match:
                ou_params = ou_match[0].get('parameters', {})
                B_rms = ou_params.get('B_rms', None)
                
                if B_rms is not None:
                    Delta_omega = gamma_e * B_rms
                    
                    # Regime boundaries in terms of τ_c/T_{2,echo}
                    # For static limit: T_{2,echo} ≈ sqrt(2) / Delta_omega
                    T2_echo_static = np.sqrt(2.0) / Delta_omega
                    
                    # For MN boundary: ξ = 0.2 → τ_c = 0.2 / Delta_omega
                    # For QS boundary: ξ = 2.0 → τ_c = 2.0 / Delta_omega
                    # Normalized: (τ_c / T_{2,echo}) = (ξ / Delta_omega) / T_{2,echo}
                    
                    # Approximate boundaries (will vary with regime, but give rough guide)
                    # In static regime, T2_echo ≈ constant, so boundaries are:
                    tau_c_norm_mn = 0.2 / (Delta_omega * T2_echo_static)
                    tau_c_norm_qs = 2.0 / (Delta_omega * T2_echo_static)
                    
                    # Add vertical lines for regime boundaries (approximate)
                    if tau_c_norm_mn > 0:
                        ax.axvline(tau_c_norm_mn, color='gray', linestyle=':', 
                                  linewidth=1, alpha=0.5, zorder=0)
                    if tau_c_norm_qs > 0:
                        ax.axvline(tau_c_norm_qs, color='gray', linestyle=':', 
                                  linewidth=1, alpha=0.5, zorder=0)
        
        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        apply_grid(ax, which='both')
        ax.set_xlabel(r'$\tau_c / T_{2,\mathrm{echo}}$', fontsize=13)
        ax.set_ylabel(r'$\eta = T_{2,\mathrm{echo}} / T_{2,\mathrm{FID}}$', fontsize=13)
        
        # Title
        mat_label = mat.replace('_', ':')
        panel_label = '(a)' if i == 0 else '(b)'
        title = f"{panel_label} {mat_label}"
        ax.set_title(title, fontsize=14, fontweight='bold', loc='left')
        
        # Legend
        ax.legend(loc='best', frameon=True, framealpha=0.9,
                 edgecolor='black', fancybox=True, fontsize=11)
        
        # Set reasonable y-axis limits
        ax.set_ylim(bottom=0.8, top=None)
    
    # Adjust layout
    plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12, top=0.92, wspace=0.25)
    
    # Save
    if save_path:
        for fmt in ['png', 'pdf']:
            filename = f"{save_path}.{fmt}"
            dpi = 300 if fmt == 'png' else None
            fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                      pad_inches=0.05, facecolor='white', format=fmt)
            print(f"✓ Saved: {filename}")
    
    return fig


def plot_noise_PSD_comparison(profiles, save_path=None):
    """
    Compare power spectral density of OU vs Double-OU noise.
    
    Shows how two Lorentzians add up in Double-OU model.
    
    Parameters
    ----------
    profiles : dict
        Material profiles from YAML (from load_profiles())
    save_path : str, optional
        Base path for saving (without extension)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    from noise_models import compute_double_OU_PSD_theory
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    materials = ['Si_P', 'GaAs']
    
    for i, mat in enumerate(materials):
        if mat not in profiles:
            continue
        
        ax = axes[i]
        profile = profiles[mat]
        
        # OU parameters
        B_rms_ou = profile['OU']['B_rms']
        tau_c_ou = np.sqrt(profile['OU']['tau_c_min'] * profile['OU']['tau_c_max'])  # Geometric mean
        
        # Double-OU parameters
        B_rms1 = profile['Double_OU']['B_rms1']
        tau_c1 = profile['Double_OU']['tau_c1']
        B_rms2 = profile['Double_OU']['B_rms2']
        tau_c2 = np.sqrt(profile['Double_OU']['tau_c2_min'] * profile['Double_OU']['tau_c2_max'])
        
        # Frequency range
        f_min = 1e3  # 1 kHz
        f_max = 1e9  # 1 GHz
        f = np.logspace(np.log10(f_min), np.log10(f_max), 1000)
        
        # OU PSD (single Lorentzian)
        S_ou = (2 * B_rms_ou**2 * tau_c_ou) / (1 + (2*np.pi*f*tau_c_ou)**2)
        
        # Double-OU PSD (sum of two Lorentzians)
        S_total, S1, S2 = compute_double_OU_PSD_theory(f, tau_c1, tau_c2, B_rms1, B_rms2)
        
        # Plot
        ax.loglog(f, S_ou, '-', color='#1E40AF', linewidth=2, label='Single OU', zorder=3)
        ax.loglog(f, S_total, '--', color='#DC2626', linewidth=2, label='Double-OU (total)', zorder=3)
        ax.loglog(f, S1, ':', color='#059669', linewidth=1.5, alpha=0.7, label='Fast component', zorder=2)
        ax.loglog(f, S2, ':', color='#F59E0B', linewidth=1.5, alpha=0.7, label='Slow component', zorder=2)
        
        # Formatting
        apply_grid(ax, which='both')
        ax.set_xlabel(r'Frequency $f$ (Hz)', fontsize=13)
        ax.set_ylabel(r'PSD $S(f)$ (T$^2$/Hz)', fontsize=13)
        
        mat_label = mat.replace('_', ':')
        panel_label = '(a)' if i == 0 else '(b)'
        title = f"{panel_label} {mat_label}"
        ax.set_title(title, fontsize=14, fontweight='bold', loc='left')
        
        ax.legend(loc='best', frameon=True, framealpha=0.9, 
                 edgecolor='black', fancybox=True, fontsize=10)
    
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.92, wspace=0.30)
    
    # Save
    if save_path:
        for fmt in ['png', 'pdf']:
            filename = f"{save_path}.{fmt}"
            dpi = 300 if fmt == 'png' else None
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                       pad_inches=0.05, facecolor='white', format=fmt)
            print(f"✓ Saved: {filename}")
    
    return fig


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_all(result_file, output_dir='results_comparison', 
                profiles_file='profiles.yaml'):
    """
    Run complete analysis and generate all publication-quality figures.
    
    Parameters
    ----------
    result_file : str
        Path to JSON results file (all_results_*.json)
    output_dir : str
        Directory to save output figures
    profiles_file : str
        Path to profiles.yaml (for PSD plot)
        
    Returns
    -------
    None
    """
    print("=" * 70)
    print("Material Comparison Analysis")
    print("=" * 70)
    
    # Load results
    print(f"\nLoading results from: {result_file}")
    all_results = load_results(result_file)
    print(f"✓ Loaded {len(all_results)} result(s)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir}/")
    
    print("\nGenerating figures...")
    
    # Figure 1: T₂ vs τ_c comparison
    print("\n[1/4] T₂ vs τ_c comparison...")
    fig1 = plot_T2_comparison(all_results, 
                              save_path=str(output_path / 'T2_comparison'))
    plt.close(fig1)
    
    # Figure 2: Echo enhancement
    print("[2/4] Echo enhancement...")
    fig2 = plot_echo_enhancement(all_results,
                                 save_path=str(output_path / 'echo_enhancement'))
    plt.close(fig2)
    
    # Table: Summary statistics
    print("[3/4] Summary table...")
    df = create_summary_table(all_results,
                              save_path=str(output_path / 'summary.csv'))
    print("\nSummary Table:")
    print(df.to_string(index=False))
    
    # Figure 3: Dimensionless collapse
    print("[3/5] Dimensionless collapse...")
    fig3 = plot_dimensionless_collapse(all_results,
                                      save_path=str(output_path / 'dimensionless_collapse'))
    plt.close(fig3)
    
    # Figure 4: PSD comparison (if profiles available)
    print("[4/5] PSD comparison...")
    try:
        import yaml
        with open(profiles_file, 'r') as f:
            profiles = yaml.safe_load(f)['materials']
        
        fig4 = plot_noise_PSD_comparison(profiles,
                                        save_path=str(output_path / 'psd_comparison'))
        plt.close(fig4)
    except Exception as e:
        print(f"  Warning: Could not generate PSD plot: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print(f"✓ All figures saved to: {output_dir}/")
    print("=" * 70)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze material comparison results and generate publication-quality figures'
    )
    
    parser.add_argument('result_file', type=str,
                       help='Path to JSON results file')
    parser.add_argument('--output-dir', type=str, default='results_comparison',
                       help='Output directory for figures')
    parser.add_argument('--profiles', type=str, default='profiles.yaml',
                       help='Path to profiles.yaml')
    
    args = parser.parse_args()
    
    analyze_all(args.result_file, args.output_dir, args.profiles)
