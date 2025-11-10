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
    'FID': {'fillstyle': 'full', 'markeredgewidth': 0.8, 'zorder': 2},
    'Hahn': {'fillstyle': 'none', 'markeredgewidth': 1.2, 'zorder': 3},
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

def plot_T2_comparison(all_results, save_path=None):
    """
    Create publication-quality T₂ vs τ_c comparison.
    
    Main result figure showing all material/noise/sequence combinations.
    
    Figure structure:
    - 2×2 subplots (rows: materials, cols: noise models)
    - Each subplot: FID (filled) + Hahn Echo (open)
    - Log-log scale
    - Error bars for FID
    
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
                
                # Calculate error bars
                T2_lower_plot = fid_data['T2_lower'] * T2_scale
                T2_upper_plot = fid_data['T2_upper'] * T2_scale
                T2_err_low = T2_plot - T2_lower_plot
                T2_err_high = T2_upper_plot - T2_plot
                
                # Remove NaN/Inf
                valid = np.isfinite(T2_plot) & np.isfinite(tau_c_plot) & (T2_plot > 0)
                
                if np.any(valid):
                    ax.errorbar(
                        tau_c_plot[valid], T2_plot[valid],
                        yerr=[T2_err_low[valid], T2_err_high[valid]],
                        fmt=MATERIAL_MARKERS[mat],
                        color=MATERIAL_COLORS[mat],
                        linestyle=NOISE_LINESTYLES[noise],
                        fillstyle=SEQUENCE_STYLES['FID']['fillstyle'],
                        markersize=6,
                        linewidth=1.6,
                        markeredgewidth=SEQUENCE_STYLES['FID']['markeredgewidth'],
                        capsize=3,
                        capthick=1.2,
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
                        markersize=6,
                        linewidth=1.6,
                        markeredgewidth=SEQUENCE_STYLES['Hahn']['markeredgewidth'],
                        label='Echo',
                        zorder=SEQUENCE_STYLES['Hahn']['zorder'],
                    )
            
            # Formatting
            ax.set_xscale('log')
            ax.set_yscale('log')
            apply_grid(ax, which='both')
            
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


def plot_echo_enhancement(all_results, save_path=None):
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
    
    # Figure 3: PSD comparison (if profiles available)
    print("[4/4] PSD comparison...")
    try:
        import yaml
        with open(profiles_file, 'r') as f:
            profiles = yaml.safe_load(f)['materials']
        
        fig3 = plot_noise_PSD_comparison(profiles,
                                        save_path=str(output_path / 'psd_comparison'))
        plt.close(fig3)
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
