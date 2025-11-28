"""
Visualization tools for simulation results.

This module provides functions to plot coherence curves, T_2 vs tau_c,
and compare with theoretical predictions.
"""

"""
Basic plotting functions for spin decoherence simulation results.

This module provides functions for plotting coherence curves, T_2 vs tau_c,
and other basic visualization tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import os

from spin_decoherence.visualization.styles import (
    COLOR_SCHEME,
    ANNOTATION_STYLE,
    get_tau_c_color,
    setup_publication_style,
)

# Ensure publication style is set
setup_publication_style()


def get_tau_c_color(tau_c, tau_c_min, tau_c_max):
    """
    Get color for tau_c value based on position in range.
    Returns color from purple (fast) to yellow (slow).
    """
    if tau_c_max == tau_c_min:
        return COLOR_SCHEME['tau_c_mid_low']
    
    # Normalize tau_c to [0, 1] in log space
    log_tau_c = np.log10(tau_c)
    log_min = np.log10(tau_c_min)
    log_max = np.log10(tau_c_max)
    normalized = (log_tau_c - log_min) / (log_max - log_min)
    
    # Color interpolation: purple → blue → green → yellow
    if normalized < 0.33:
        # Purple to blue
        t_local = normalized / 0.33
        r = int(139 + (59 - 139) * t_local)
        g = int(92 + (130 - 92) * t_local)
        b = int(246 + (246 - 246) * t_local)
    elif normalized < 0.67:
        # Blue to green
        t_local = (normalized - 0.33) / 0.34
        r = int(59 + (16 - 59) * t_local)
        g = int(130 + (185 - 130) * t_local)
        b = int(246 + (129 - 246) * t_local)
    else:
        # Green to yellow
        t_local = (normalized - 0.67) / 0.33
        r = int(16 + (245 - 16) * t_local)
        g = int(185 + (158 - 185) * t_local)
        b = int(129 + (11 - 129) * t_local)
    
    return f'#{r:02X}{g:02X}{b:02X}'


def plot_coherence_curve(result, ax=None, show_fit=True, show_error=True, 
                        show_analytical=False, show_residuals=False,
                        inset_position='upper right'):
    """
    Plot coherence function E(t) vs time.
    
    Parameters
    ----------
    result : dict
        Single simulation result dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_fit : bool
        Whether to overlay fitted curve
    show_error : bool
        Whether to show error bands
    show_analytical : bool
        Whether to show analytical solution
    show_residuals : bool
        Whether to show residual inset
    inset_position : str or list
        Position of residual inset: 'upper right', 'upper left', 'lower right', 'lower left'
        or [x, y, width, height] in axes coordinates
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    t = np.array(result['t'])
    E_magnitude = np.array(result['E_magnitude'])
    
    tau_c_us = result['tau_c'] * 1e6
    
    # Get tau_c range for color scheme (if available from results context)
    # For now use default color, but can be enhanced with tau_c range
    line_color = COLOR_SCHEME['simulation']
    
    # Plot coherence (navy blue solid, linewidth 1.3)
    ax.plot(t * 1e6, E_magnitude, '-', color=line_color,
            label=r'$\tau_c = {:.2f}$ $\mu$s'.format(tau_c_us), 
            linewidth=1.3, zorder=1)
    
    # Error bands (standard error of |E|) - reduced opacity for better contrast
    if show_error and 'E_se' in result:
        E_se = np.array(result['E_se'])
        ax.fill_between(t * 1e6, 
                        E_magnitude - E_se,
                        E_magnitude + E_se,
                        alpha=0.15, color=line_color, label=r'$\pm$ Standard error', zorder=0)
    
    # Analytical solution (green dotted, linewidth 1.0)
    if show_analytical and 'params' in result:
        from fitting import analytical_ou_coherence
        params = result['params']
        E_analytical = analytical_ou_coherence(
            t, params['gamma_e'], params['B_rms'], result['tau_c']
        )
        ax.plot(t * 1e6, E_analytical, ':', 
                color=COLOR_SCHEME['analytical'],
                label='Analytical solution', linewidth=1.0, alpha=0.8, zorder=2)
    
    # Fitted curve (red dashed, linewidth 1.5)
    if show_fit and result.get('fit_result') is not None:
        fit_result = result['fit_result']
        fit_curve = fit_result['fit_curve']
        T2_us = fit_result['T2'] * 1e6
        
        # Standardized label format (remove model name for consistency)
        label_text = r'Fit: $T_2 = {:.2f}$ $\mu$s'.format(T2_us)
        ax.plot(t * 1e6, fit_curve, '--', 
                color=COLOR_SCHEME['theory'],
                label=label_text, linewidth=1.5, zorder=3)
    
    # Standardized axis labels and title (LaTeX notation)
    ax.set_xlabel(r'$t$ ($\mu$s)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$|E(t)|$', fontsize=12, fontweight='bold')
    ax.set_title(r'$\tau_c = {:.2f}$ $\mu$s'.format(tau_c_us), fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Simplified legend
    # Remove "Analytical" from legend if present, keep only essential items
    handles, labels = ax.get_legend_handles_labels()
    # Filter out 'Analytical' label if it exists (will be mentioned in caption)
    filtered_handles = []
    filtered_labels = []
    for h, l in zip(handles, labels):
        if 'Analytical' not in l:
            filtered_handles.append(h)
            filtered_labels.append(l)
    if filtered_handles:
        ax.legend(filtered_handles, filtered_labels, loc='lower left', fontsize=9, framealpha=0.9)
    else:
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, None])
    
    # Residuals inset with standardized position and size
    if show_residuals and result.get('fit_result') is not None:
        fit_result = result['fit_result']
        fit_curve = fit_result['fit_curve']
        residuals = E_magnitude - fit_curve
        
        # Use inset_position parameter
        inset_pos = inset_position
        
        # Standardized inset position based on input
        # Positions adjusted to upper corners (y > 0.6) to avoid overlap with decay curves
        # Most decay curves pass through lower/mid regions, so upper region is safest
        if isinstance(inset_pos, str):
            position_map = {
                'upper right': [0.55, 0.65, 0.38, 0.30],  # Upper right, higher y, larger size
                'upper left': [0.02, 0.65, 0.38, 0.30],   # Upper left, higher y, larger size
                'lower right': [0.55, 0.02, 0.38, 0.30],  # Lower right (for slow decay cases)
                'lower left': [0.02, 0.02, 0.38, 0.30]    # Lower left (for slow decay cases)
            }
            inset_coords = position_map.get(inset_pos, [0.55, 0.65, 0.38, 0.30])
        else:
            inset_coords = inset_pos
        
        axins = ax.inset_axes(inset_coords)
        # Add semi-transparent white background with border for better visibility
        axins.patch.set_facecolor('white')
        axins.patch.set_alpha(0.90)  # More opaque to clearly separate from main plot
        axins.patch.set_edgecolor('gray')
        axins.patch.set_linewidth(1.5)
        axins.set_zorder(10)  # Ensure inset is on top
        
        axins.plot(t * 1e6, residuals, '-', color=COLOR_SCHEME['residual'], 
                  linewidth=0.8, alpha=0.7)
        axins.axhline(0, color=COLOR_SCHEME['theory'], linestyle='--', linewidth=0.8)
        axins.set_ylim(-0.02, 0.02)  # Reduced range (±0.02) for better visibility
        axins.set_xlabel(r'$t$ ($\mu$s)', fontsize=7)
        axins.set_ylabel('Residual', fontsize=7)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=6)
        # Remove ticks for cleaner look (optional)
        # axins.set_xticks([])
        # axins.set_yticks([])
    
    return ax


def plot_T2_vs_tauc(results, ax=None, show_theory=True, show_ci=True, 
                   gamma_e=None, B_rms=None, show_crossover=True):
    """
    Plot T_2 vs tau_c on log-log scale with unified styling.
    
    Parameters
    ----------
    results : list
        List of simulation result dictionaries
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_theory : bool
        Whether to show theoretical prediction
    show_ci : bool
        Whether to show 95% CI error bars
    gamma_e : float, optional
        Electron gyromagnetic ratio (for crossover line)
    B_rms : float, optional
        RMS noise amplitude (for crossover line)
    show_crossover : bool
        Whether to show crossover line at τ_c = 1/Δω
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data
    tau_c_values = []
    T2_fitted = []
    T2_theory = []
    T2_ci_lower = []
    T2_ci_upper = []
    
    for r in results:
        if r.get('fit_result') is not None:
            tau_c_values.append(r['tau_c'])
            T2_fitted.append(r['fit_result']['T2'])
            if 'T2_theory' in r:
                T2_theory.append(r['T2_theory'])
            # Extract CI if available (T2_ci is stored in fit_result)
            if show_ci and 'T2_ci' in r.get('fit_result', {}) and r['fit_result']['T2_ci'] is not None:
                T2_ci = r['fit_result']['T2_ci']
                T2_ci_lower.append(T2_ci[0])
                T2_ci_upper.append(T2_ci[1])
            else:
                T2_ci_lower.append(None)
                T2_ci_upper.append(None)
    
    tau_c_values = np.array(tau_c_values)
    T2_fitted = np.array(T2_fitted)
    
    # Plot fitted T_2 with error bars (navy/royal blue for better contrast)
    if show_ci and any(ci is not None for ci in T2_ci_lower):
        T2_ci_lower_arr = np.array([ci if ci is not None else T2_fitted[i] 
                                   for i, ci in enumerate(T2_ci_lower)])
        T2_ci_upper_arr = np.array([ci if ci is not None else T2_fitted[i] 
                                   for i, ci in enumerate(T2_ci_upper)])
        # Use navy/royal blue for better contrast
        ax.loglog(tau_c_values * 1e6, T2_fitted * 1e6, 'o-',
                  color='#1E40AF', markersize=9, 
                  label=r'Simulation', linewidth=2.5, zorder=5,
                  markeredgecolor='#1E3A8A', markeredgewidth=1.5)
        # Add error bars manually for log scale
        for i in range(len(tau_c_values)):
            if T2_ci_lower_arr[i] is not None:
                ax.plot([tau_c_values[i] * 1e6, tau_c_values[i] * 1e6],
                       [T2_ci_lower_arr[i] * 1e6, T2_ci_upper_arr[i] * 1e6],
                       color='#1E40AF', linewidth=2, alpha=0.6)
                # Caps
                ax.plot(tau_c_values[i] * 1e6, T2_ci_lower_arr[i] * 1e6, '_',
                       color='#1E40AF', markersize=10, markeredgewidth=2)
                ax.plot(tau_c_values[i] * 1e6, T2_ci_upper_arr[i] * 1e6, '_',
                       color='#1E40AF', markersize=10, markeredgewidth=2)
    else:
        ax.loglog(tau_c_values * 1e6, T2_fitted * 1e6, 'o-',
                  color='#1E40AF', markersize=9, 
                  label=r'Simulation', linewidth=2.5, zorder=5,
                  markeredgecolor='#1E3A8A', markeredgewidth=1.5)
    
    # Plot theoretical predictions
    if show_theory and gamma_e is not None and B_rms is not None:
        from fitting import theoretical_T2_motional_narrowing, analytical_ou_coherence
        Delta_omega = gamma_e * B_rms
        
        # Compute theoretical T2 for all tau_c values
        tau_c_theory = np.logspace(np.log10(tau_c_values.min()), 
                                   np.log10(tau_c_values.max()), 200)
        
        # MN limit: T2 = 1/(Delta_omega^2 * tau_c)
        T2_mn_theory = theoretical_T2_motional_narrowing(gamma_e, B_rms, tau_c_theory)
        ax.loglog(tau_c_theory * 1e6, T2_mn_theory * 1e6, '--',
                  color='#991B1B', linewidth=2.5, alpha=0.9,
                  label=r'Theory (MN limit: $T_2 = 1/(\Delta\omega^2 \tau_c)$)', zorder=3)
        
        # Static limit: T2 ≈ sqrt(2)/Delta_omega (for large tau_c)
        T2_static = np.sqrt(2.0) / Delta_omega
        if tau_c_values.max() * 1e6 > 1.0:  # Only show if we have slow noise data
            ax.axhline(T2_static * 1e6, color='#DC2626', linestyle=':', 
                      linewidth=2, alpha=0.8, zorder=2,
                      label=rf'Static limit: $T_2 = \sqrt{{2}}/\Delta\omega = {T2_static*1e6:.2f}$ μs')
        
        # Also plot from stored T2_theory if available (for comparison)
        if len(T2_theory) > 0:
            T2_theory = np.array(T2_theory)
            ax.loglog(tau_c_values * 1e6, T2_theory * 1e6, 'o',
                     color='#991B1B', markersize=4, alpha=0.6, zorder=4,
                     label='Theory (stored)', markerfacecolor='none', markeredgewidth=1)
    
    # Add crossover line and region highlighting (quantitative marker)
    if show_crossover and gamma_e is not None and B_rms is not None:
        Delta_omega = gamma_e * B_rms
        tau_c_crossover = 1.0 / Delta_omega  # ξ = 1 boundary
        y_min = T2_fitted.min() * 1e6
        y_max = T2_fitted.max() * 1e6
        
        # Highlight Motional Narrowing region (ξ < 1, τ_c < 1/Δω)
        tau_c_mn_max = tau_c_crossover * 1e6
        tau_c_mn_min = tau_c_values.min() * 1e6
        ax.axvspan(tau_c_mn_min, tau_c_mn_max, 
                  color=COLOR_SCHEME['mn_region'], alpha=0.2, zorder=0,
                  label='Motional narrowing regime')
        
        # Highlight quasi-static region (ξ > 1, τ_c > 1/Δω)
        tau_c_static_min = tau_c_crossover * 1e6
        tau_c_static_max = tau_c_values.max() * 1e6
        if tau_c_static_max > tau_c_static_min:
            ax.axvspan(tau_c_static_min, tau_c_static_max, 
                      color=COLOR_SCHEME['static_region'], alpha=0.15, zorder=0,
                      label='Quasi-static regime')
        
        # Add boundary line
        ax.axvline(tau_c_crossover * 1e6, color='gray', linestyle=':', 
                  linewidth=2, alpha=0.8, zorder=2,
                  label=r'$\tau_c = 1/\Delta\omega$ (ξ = 1)')
        # Add annotation
        ax.text(tau_c_crossover * 1e6, y_max * 0.7, 
               r'$\xi = 1$', rotation=90, fontsize=10, 
               verticalalignment='bottom', color='gray', fontweight='bold')
    
    # Reference line: T_2 ∝ 1/τ_c (full range) - removed for clarity
    
    # LaTeX labels with scientific storytelling
    ax.set_xlabel(r'$\tau_c$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$T_2$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_title(r'$T_2$ vs $\tau_c$', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, which='both', linestyle='--', linewidth=0.8)  # Reduced grid density
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Removed excessive annotation boxes and inset graph that obscured the main plot
    
    return ax


def plot_mn_slope_fit(results, gamma_e, B_rms, ax, xi_threshold=0.2):
    """
    Add MN regime fit overlay to existing T_2 vs tau_c plot.
    
    Parameters
    ----------
    results : list
        List of simulation result dictionaries
    gamma_e : float
        Electron gyromagnetic ratio
    B_rms : float
        RMS noise amplitude
    ax : matplotlib.axes.Axes
        Existing axes to add overlay to
    xi_threshold : float
        Maximum ξ for MN regime
    """
    from fitting import fit_mn_slope
    
    # Fit MN regime
    mn_fit = fit_mn_slope(results, gamma_e, B_rms, xi_threshold=xi_threshold)
    
    if mn_fit is not None:
        # Extract MN data points
        Delta_omega = gamma_e * B_rms
        tau_c_mn = []
        T2_mn = []
        
        for r in results:
            if r.get('fit_result') is not None:
                tau_c = r['tau_c']
                T2 = r['fit_result']['T2']
                xi = Delta_omega * tau_c
                
                if xi < xi_threshold and T2 > 0:
                    tau_c_mn.append(tau_c)
                    T2_mn.append(T2)
        
        tau_c_mn = np.array(tau_c_mn)
        T2_mn = np.array(T2_mn)
        
        # Plot MN fit line
        tau_mn_extrap = np.logspace(np.log10(tau_c_mn.min() * 1e6),
                                    np.log10(tau_c_mn.max() * 1e6),
                                    100)
        T2_mn_extrap = 10**(mn_fit['intercept']) * (tau_mn_extrap / 1e6)**mn_fit['slope'] * 1e6
        
        ax.loglog(tau_mn_extrap, T2_mn_extrap, 'm-', alpha=0.7, linewidth=2,
                 label=f"MN fit (ξ<{xi_threshold}, slope={mn_fit['slope']:.2f}±{mn_fit['slope_std']:.2f})")
        
        # Highlight MN points
        ax.scatter(tau_c_mn * 1e6, T2_mn * 1e6, c='orange', s=100, 
                  marker='o', edgecolors='darkorange', linewidths=2,
                  label=f'MN regime (ξ<{xi_threshold})', zorder=5)
        
        # Removed text annotation box that obscured the graph
    
    return ax


def plot_multiple_coherence_curves(results, tau_c_indices=None, n_curves=5):
    """
    Plot multiple coherence curves for different tau_c values.
    
    Parameters
    ----------
    results : list
        List of simulation results
    tau_c_indices : list, optional
        Indices of results to plot (if None, selects evenly spaced)
    n_curves : int
        Number of curves to plot (if tau_c_indices not specified)
    """
    if tau_c_indices is None:
        n_total = len(results)
        tau_c_indices = np.linspace(0, n_total - 1, n_curves, dtype=int)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use viridis colormap (perceptually uniform)
    import matplotlib.cm as cm
    try:
        colormap = cm.get_cmap('viridis')
    except AttributeError:
        colormap = cm.viridis
    
    # Sort tau_c values for proper color mapping
    tau_c_list = []
    for idx in tau_c_indices:
        if idx < len(results):
            tau_c_list.append((idx, results[idx]['tau_c']))
    
    # Sort by tau_c value (ascending)
    tau_c_list.sort(key=lambda x: x[1])
    tau_c_min = tau_c_list[0][1]
    tau_c_max = tau_c_list[-1][1]
    norm = plt.Normalize(tau_c_min, tau_c_max)
    
    # Plot with sorted order
    for idx, tau_c in tau_c_list:
        result = results[idx]
        t = np.array(result['t'])
        E_magnitude = np.array(result['E_magnitude'])
        tau_c_us = tau_c * 1e6
        color = colormap(norm(tau_c))
        
        # Plot main curve
        ax.plot(t * 1e6, E_magnitude, '-', color=color,
               label=f'$\\tau_c = {tau_c_us:.2f}$ μs', linewidth=2.5, alpha=0.9)
    
    # Add gray shading for noise floor region (below 0.05)
    ax.axhspan(0, 0.05, alpha=0.1, color='gray', zorder=0, 
              label='Noise floor region')
    
    ax.set_xlabel(r'Time $t$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$|E(t)|$', fontsize=13, fontweight='bold')
    ax.set_title(r'Coherence Decay for Different Correlation Times\n'
                r'Fast noise (small $\tau_c$): slow decay; Slow noise: fast decay', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    
    # Sort legend by tau_c value (ascending order)
    handles, labels = ax.get_legend_handles_labels()
    # Extract tau_c values from labels and sort
    label_pairs = [(h, l) for h, l in zip(handles, labels)]
    # Sort by tau_c value (extracted from label)
    def extract_tau_c(label):
        try:
            return float(label.split('=')[1].split('μ')[0].strip())
        except:
            return 0
    label_pairs.sort(key=lambda x: extract_tau_c(x[1]))
    
    # Recreate legend with sorted order
    ax.legend([h for h, l in label_pairs], [l for h, l in label_pairs], 
              loc='best', ncol=2, fontsize=9, framealpha=0.9)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return fig, ax


def create_summary_plots(results, output_dir='results', save=True, 
                        compute_bootstrap=True, gamma_e=None, B_rms=None):
    """
    Create and save summary plots.
    
    Parameters
    ----------
    results : list
        List of simulation results
    output_dir : str
        Directory to save plots
    save : bool
        Whether to save plots to files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: T_2 vs tau_c (with CI and theory if available)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_T2_vs_tauc(results, ax=ax1, show_ci=compute_bootstrap,
                   gamma_e=gamma_e, B_rms=B_rms, show_crossover=True,
                   show_theory=(gamma_e is not None and B_rms is not None))
    
    # Add MN regime fit if gamma_e and B_rms provided
    if gamma_e is not None and B_rms is not None:
        plot_mn_slope_fit(results, gamma_e, B_rms, ax=ax1, xi_threshold=0.2)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, 'T2_vs_tauc.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'T2_vs_tauc.png')}")
    
    # Plot 1b: Dimensionless collapse (if gamma_e and B_rms provided)
    if gamma_e is not None and B_rms is not None:
        fig_collapse, ax_collapse = plt.subplots(figsize=(8, 6))
        plot_dimensionless_collapse(results, gamma_e, B_rms, ax=ax_collapse, 
                                   show_ci=compute_bootstrap)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(output_dir, 'dimensionless_collapse.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(output_dir, 'dimensionless_collapse.png')}")
    
    # Plot 1c: β(τ_c) plot
    fig_beta, ax_beta = plt.subplots(figsize=(8, 6))
    plot_beta_vs_tauc(results, ax=ax_beta)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, 'beta_vs_tauc.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'beta_vs_tauc.png')}")
    
    # Plot 1d: OU PSD verification (multi-panel for multiple tau_c values)
    if len(results) > 0:
        # Select representative tau_c values (logarithmically spaced)
        n_psd_examples = min(6, len(results))
        psd_indices = np.linspace(0, len(results) - 1, n_psd_examples, dtype=int)
        
        # Create multi-panel figure (2 rows x 3 columns)
        nrows = 2
        ncols = 3
        fig_psd, axes_psd = plt.subplots(nrows, ncols, figsize=(15, 10))
        axes_psd = axes_psd.flatten()
        
        from ornstein_uhlenbeck import generate_ou_noise
        
        for i, idx in enumerate(psd_indices):
            if idx >= len(results):
                axes_psd[i].axis('off')
                continue
                
            r = results[idx]
            tau_c = r['tau_c']
            B_rms = r['params']['B_rms']
            dt = r['params']['dt']
            
            # Generate or use existing delta_B sample
            if 'delta_B_sample' in r:
                delta_B = np.array(r['delta_B_sample'])
            else:
                N_steps = len(r['t'])
                delta_B = generate_ou_noise(
                    tau_c, B_rms, dt, N_steps, 
                    seed=r['params']['seed'] + idx
                )
            
            # Plot PSD verification
            plot_ou_psd_verification(delta_B, tau_c, B_rms, dt, ax=axes_psd[i])
            
            # Update title to be more compact (remove redundant title from plot_ou_psd_verification)
            axes_psd[i].set_title(f'$\\tau_c = {tau_c*1e6:.2f}$ μs', 
                                 fontsize=11, fontweight='bold', pad=8)
            
            # Simplify legend for multi-panel (only if legend exists)
            legend = axes_psd[i].get_legend()
            if legend is not None and len(legend.get_texts()) > 0:
                axes_psd[i].legend(loc='upper right', fontsize=7, framealpha=0.8)
        
        # Hide unused subplots
        for i in range(len(psd_indices), len(axes_psd)):
            axes_psd[i].axis('off')
        
        # Add overall figure title
        fig_psd.suptitle('OU Noise PSD Verification (Multiple $\\tau_c$ Values)', 
                        fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if save:
            plt.savefig(os.path.join(output_dir, 'ou_psd_verification.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(output_dir, 'ou_psd_verification.png')}")
    
    # Plot 2: Multiple coherence curves
    fig2, ax2 = plot_multiple_coherence_curves(results)
    if save:
        plt.savefig(os.path.join(output_dir, 'coherence_curves.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'coherence_curves.png')}")
    
    # Plot 3: Individual coherence curves (selected examples)
    # Layout: 2 rows x 4 columns (top: main plots, bottom: residual plots)
    n_examples = min(4, len(results))
    example_indices = np.linspace(0, len(results) - 1, n_examples, dtype=int)
    
    # Create 2x4 grid: top row for main plots, bottom row for residuals
    fig3, axes = plt.subplots(2, 4, figsize=(16, 9), 
                              sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.3, hspace=0.35,
                        left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    main_axes = axes[0, :]  # Top row: main plots
    residual_axes = axes[1, :]  # Bottom row: residual plots
    
    # Plot main coherence curves
    for i, idx in enumerate(example_indices):
        if idx < len(results):
            result = results[idx]
            plot_coherence_curve(result, ax=main_axes[i], 
                               show_fit=True,
                               show_analytical=True, 
                               show_residuals=False)
            
            # Remove individual subplot title and labels
            main_axes[i].set_title('')
            if i > 0:  # Remove Y-axis label for non-left subplots
                main_axes[i].set_ylabel('')
            # X-axis labels removed for top row (will be on bottom row)
            main_axes[i].set_xlabel('')
            
            # Add tau_c label to each subplot
            tau_c_us = result['tau_c'] * 1e6
            main_axes[i].text(0.05, 0.95, f'$\\tau_c = {tau_c_us:.2f}$ μs', 
                             transform=main_axes[i].transAxes,
                             fontsize=11, fontweight='bold',
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot residuals: all as separate subplots in bottom row
    for i, idx in enumerate(example_indices):
        if idx < len(results):
            result = results[idx]
            if result.get('fit_result') is not None:
                fit_result = result['fit_result']
                fit_curve = fit_result['fit_curve']
                t = np.array(result['t'])
                E_magnitude = np.array(result['E_magnitude'])
                residuals = E_magnitude - fit_curve
                
                # Calculate residual statistics
                residual_std = np.std(residuals)
                residual_mean = np.mean(residuals)
                residual_max = max(abs(residuals.max()), abs(residuals.min()))
                ylim_max = max(0.02, residual_max * 1.2)
                
                # Plot residual in separate subplot
                residual_axes[i].plot(t * 1e6, residuals, '-', 
                          color=COLOR_SCHEME['residual'], 
                          linewidth=1.5, alpha=0.9)
                residual_axes[i].axhline(0, color=COLOR_SCHEME['theory'], 
                             linestyle='--', linewidth=1.5, alpha=0.8)
                residual_axes[i].set_ylim(-ylim_max, ylim_max)
                
                # Add ±2σ bands
                residual_axes[i].axhline(2 * residual_std, color='gray', 
                             linestyle=':', linewidth=1.0, alpha=0.5)
                residual_axes[i].axhline(-2 * residual_std, color='gray', 
                             linestyle=':', linewidth=1.0, alpha=0.5)
                
                residual_axes[i].grid(True, alpha=0.3)
                residual_axes[i].tick_params(labelsize=10)
                residual_axes[i].set_xlabel(r'$t$ ($\mu$s)', fontsize=11, fontweight='bold')
                residual_axes[i].set_ylabel('Residual', fontsize=11, fontweight='bold')
                
                # Add statistics
                if residual_std > 0:
                    residual_axes[i].text(0.98, 0.95, 
                              f'Mean: {residual_mean:.4f}\nStd: {residual_std:.4f}',
                              transform=residual_axes[i].transAxes,
                              fontsize=10, verticalalignment='top',
                              horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='white', 
                                      alpha=0.9, edgecolor='gray', linewidth=1.0))
            else:
                # No fit result, hide the subplot
                residual_axes[i].axis('off')
    
    # Add common axis labels for main plots
    fig3.text(0.02, 0.75, r'$|E(t)|$', va='center', rotation='vertical', fontsize=13, fontweight='bold')
    
    # Add figure title
    fig3.suptitle(r'Detailed Coherence Examples with Residuals\n'
                 r'Analytical vs fitted decay curves for selected $\tau_c$ values',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save:
        plt.savefig(os.path.join(output_dir, 'coherence_examples.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'coherence_examples.png')}")
    
    if not save:
        plt.show()
    else:
        plt.close('all')


def _compute_psd_hash(delta_B, dt):
    """
    Compute a hash for delta_B array and dt for caching purposes.
    
    Parameters
    ----------
    delta_B : ndarray
        Array to hash
    dt : float
        Time step
        
    Returns
    -------
    hash_str : str
        Hash string for caching
    """
    # Create a hash from array shape, first/last few values, and dt
    # This is faster than hashing the entire array
    arr_hash = hashlib.md5()
    arr_hash.update(delta_B.shape[0].to_bytes(8, 'little'))
    arr_hash.update(delta_B[:10].tobytes())  # First 10 values
    arr_hash.update(delta_B[-10:].tobytes())  # Last 10 values
    arr_hash.update(delta_B[::max(1, len(delta_B)//100)].tobytes())  # Sample
    arr_hash.update(np.float64(dt).tobytes())
    return arr_hash.hexdigest()


def plot_ou_psd_verification(delta_B, tau_c, B_rms, dt, ax=None):
    """
    Plot OU noise power spectral density (PSD) and compare with theory.
    
    This function creates Figure 4: OU Noise PSD Verification, showing:
    - Theoretical PSD (red dotted line): Lorentzian shape
    - Simulated PSD (blue line): FFT-based from time series
    - Corner frequency annotation
    - Behavior annotations for low/high frequency regions
    - Verification result summary
    
    Parameters
    ----------
    delta_B : ndarray
        OU noise time series (Tesla)
    tau_c : float
        Correlation time (seconds)
    B_rms : float
        RMS noise amplitude (Tesla)
    dt : float
        Time step (seconds)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    N = len(delta_B)
    # Frequency array
    freq = np.fft.rfftfreq(N, d=dt)
    
    # Compute PSD with caching (using hash of delta_B and dt as key)
    # Since delta_B is not hashable, we use a hash-based cache key
    cache_key = (_compute_psd_hash(delta_B, dt), N, dt)
    
    # Check if we have a cached result
    if not hasattr(plot_ou_psd_verification, '_psd_cache'):
        plot_ou_psd_verification._psd_cache = {}
    
    if cache_key in plot_ou_psd_verification._psd_cache:
        # Use cached PSD
        PSD = plot_ou_psd_verification._psd_cache[cache_key]
    else:
        # Compute PSD (one-sided)
        fft_dB = np.fft.rfft(delta_B)
        PSD = (np.abs(fft_dB)**2) * dt / N
        # Cache the result (limit cache size to 128 entries)
        if len(plot_ou_psd_verification._psd_cache) >= 128:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(plot_ou_psd_verification._psd_cache))
            del plot_ou_psd_verification._psd_cache[oldest_key]
        plot_ou_psd_verification._psd_cache[cache_key] = PSD
    
    # Theoretical PSD: S(ω) = 2 B_rms^2 τ_c / (1 + ω^2 τ_c^2)
    omega = 2 * np.pi * freq
    omega_c = 1.0 / tau_c  # Corner frequency (rad/s)
    f_c = omega_c / (2 * np.pi)  # Corner frequency (Hz)
    S_th = 2 * (B_rms**2) * tau_c / (1 + (omega * tau_c)**2)
    
    # Low-frequency plateau value
    S_0 = 2 * (B_rms**2) * tau_c
    
    # Plot simulated and theoretical PSD
    ax.loglog(freq, PSD, 'b-', label='Simulated PSD', linewidth=2, alpha=0.8)
    ax.loglog(freq, S_th, 'r--', label='Theoretical PSD', linewidth=2.5, 
             dashes=(3, 2), alpha=0.9)
    
    # Mark corner frequency (simple vertical line)
    ax.axvline(f_c, color='green', linestyle=':', linewidth=2, alpha=0.6, zorder=3,
               label=f'Corner frequency: $f_c = {f_c:.2e}$ Hz')
    
    ax.set_xlabel('Frequency $f$ (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power spectral density $S(f)$ (T²/Hz)', fontsize=12, fontweight='bold')
    # Title will be set by caller for multi-panel plots
    # Check if this is a standalone plot (single axes in figure)
    try:
        fig = ax.get_figure()
        is_standalone = len(fig.get_axes()) == 1
    except:
        is_standalone = True
    
    if is_standalone:
        ax.set_title(f'OU Noise PSD Verification\n'
                    f'($\\tau_c = {tau_c*1e6:.2f}$ μs, $f_c = {f_c:.2e}$ Hz)', 
                     fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    # Only show legend if this is a standalone plot (not part of multi-panel)
    if ax is None or len(ax.get_figure().get_axes()) == 1:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    else:
        # For multi-panel, use simpler legend
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8, ncol=1)
    
    # Set reasonable limits
    ax.set_xlim(left=freq[freq > 0][0], right=freq[-1])
    
    return ax


def plot_dimensionless_collapse(results, gamma_e, B_rms, ax=None, show_ci=True):
    """
    Plot dimensionless collapse: Y = T₂(Δω)²τ_c vs ξ = Δω·τ_c.
    
    In motional-narrowing limit (ξ << 1), Y → 1.
    
    Parameters
    ----------
    results : list
        List of simulation result dictionaries
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    B_rms : float
        RMS noise amplitude (Tesla)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_ci : bool
        Whether to show 95% CI error bars
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute dimensionless parameters
    # CRITICAL: Ensure Delta_omega is in rad/s (gamma_e is rad·s⁻¹·T⁻¹, B_rms is Tesla)
    Delta_omega = gamma_e * B_rms  # rad/s
    Delta_omega_sq = Delta_omega**2  # (rad/s)²
    
    # Verify units are consistent
    if Delta_omega < 1e3 or Delta_omega > 1e9:
        print(f"WARNING: Delta_omega = {Delta_omega:.3e} rad/s seems unusual.")
        print(f"  Check: gamma_e = {gamma_e:.3e} rad·s⁻¹·T⁻¹, B_rms = {B_rms:.3e} T")
    
    xi_values = []
    Y_values = []
    Y_ci_lower = []
    Y_ci_upper = []
    
    for r in results:
        if r.get('fit_result') is not None:
            tau_c = r['tau_c']  # seconds
            T2 = r['fit_result']['T2']  # seconds
            
            # ξ = Δω · τ_c (dimensionless)
            xi = Delta_omega * tau_c
            
            # Y = T₂ · (Δω)² · τ_c (dimensionless)
            # Units check: [s] · [(rad/s)²] · [s] = rad² (dimensionless, ✓)
            Y = T2 * Delta_omega_sq * tau_c
            
            # Sanity check: Y/xi = T2 * Delta_omega (should be ~constant for large tau_c)
            # This is a useful diagnostic: if Y/xi grows unbounded, there's a unit mismatch
            if xi > 1.0:  # Static regime
                ratio = Y / xi  # Should equal T2 * Delta_omega
                expected = T2 * Delta_omega
                if abs(ratio - expected) > 1e-6:
                    print(f"WARNING: Y/xi mismatch at tau_c={tau_c*1e6:.2f}μs:")
                    print(f"  Y/xi = {ratio:.6e}, expected T2*Delta_omega = {expected:.6e}")
            
            xi_values.append(xi)
            Y_values.append(Y)
            
            # Extract CI if available
            if show_ci and 'T2_ci' in r.get('fit_result', {}) and r['fit_result']['T2_ci'] is not None:
                T2_ci = r['fit_result']['T2_ci']
                Y_ci_lower.append(T2_ci[0] * Delta_omega_sq * tau_c)
                Y_ci_upper.append(T2_ci[1] * Delta_omega_sq * tau_c)
            else:
                Y_ci_lower.append(None)
                Y_ci_upper.append(None)
    
    xi_values = np.array(xi_values)
    Y_values = np.array(Y_values)
    
    # Plot with error bars if CI available - Add gray shading for CI
    if show_ci and any(ci is not None for ci in Y_ci_lower):
        Y_ci_lower_arr = np.array([ci if ci is not None else Y_values[i] 
                                  for i, ci in enumerate(Y_ci_lower)])
        Y_ci_upper_arr = np.array([ci if ci is not None else Y_values[i] 
                                  for i, ci in enumerate(Y_ci_upper)])
        
        # Plot main line
        ax.loglog(xi_values, Y_values, 'o-', color='#1E40AF', markersize=9, 
                 label='Simulation', linewidth=2.5, zorder=5,
                 markeredgecolor='#1E3A8A', markeredgewidth=1.5)
        
        # Add gray shading for CI (more visible than error bars)
        for i in range(len(xi_values)):
            if Y_ci_lower_arr[i] is not None and Y_ci_upper_arr[i] is not None:
                # Plot error bar with gray shading
                ax.plot([xi_values[i], xi_values[i]],
                       [Y_ci_lower_arr[i], Y_ci_upper_arr[i]],
                       'k-', linewidth=2, alpha=0.3, color='gray', zorder=1)
                # Add caps
                ax.plot(xi_values[i], Y_ci_lower_arr[i], '_',
                       color='gray', markersize=8, markeredgewidth=2, alpha=0.5)
                ax.plot(xi_values[i], Y_ci_upper_arr[i], '_',
                       color='gray', markersize=8, markeredgewidth=2, alpha=0.5)
    else:
        ax.loglog(xi_values, Y_values, 'o-', color='#1E40AF', markersize=9, 
                 label='Simulation', linewidth=2.5, zorder=5,
                 markeredgecolor='#1E3A8A', markeredgewidth=1.5)
    
    # Highlight regions: MN (ξ < 1) and quasi-static (ξ > 1)
    if len(xi_values) > 0:
        xi_min = xi_values.min()
        xi_max = xi_values.max()
        y_min = Y_values.min()
        y_max = Y_values.max()
        
        # MN region (ξ < 1): Y ≈ 1
        if xi_min < 1.0:
            ax.axvspan(xi_min, min(1.0, xi_max), 
                      color=COLOR_SCHEME['mn_region'], alpha=0.2, zorder=0)
        
        # Quasi-static region (ξ > 1)
        if xi_max > 1.0:
            ax.axvspan(max(1.0, xi_min), xi_max, 
                      color=COLOR_SCHEME['static_region'], alpha=0.15, zorder=0)
    
    # Theoretical limit: Y = 1 for ξ << 1
    ax.axhline(1.0, color='#991B1B', linestyle='--', linewidth=2.5, alpha=0.9,
              label='Theory (Y = 1)', zorder=3)
    
    # Mark motional-narrowing regime boundary
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=2, alpha=0.8, zorder=2,
              label=r'$\xi = 1$ (MN boundary)')
    
    # LaTeX labels with unified notation and scientific storytelling
    ax.set_xlabel(r'$\xi = \Delta\omega \cdot \tau_c$', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$Y = T_2 \cdot (\Delta\omega)^2 \cdot \tau_c$', fontsize=13, fontweight='bold')
    ax.set_title(r'Dimensionless Scaling and Motional Narrowing Collapse', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, which='both', linestyle='--', linewidth=0.8)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Fix font issue: Use LogFormatterSciNotation with ASCII minus
    # This ensures tick labels use ASCII minus instead of Unicode minus (U+2212)
    from matplotlib.ticker import LogFormatterSciNotation
    # Note: useMathText parameter removed in matplotlib 3.9+
    try:
        formatter = LogFormatterSciNotation(base=10, useMathText=False)
    except TypeError:
        # Fallback for matplotlib 3.9+ where useMathText was removed
        formatter = LogFormatterSciNotation(base=10)
    formatter.set_scientific(False)  # Use regular notation instead of scientific
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    # Alternative: Force ASCII minus in tick labels by replacing Unicode minus
    # This is a more robust solution
    for axis in [ax.xaxis, ax.yaxis]:
        for tick in axis.get_major_ticks():
            label = tick.label1.get_text()
            if label:
                # Replace Unicode minus (U+2212) with ASCII minus
                label = label.replace('\u2212', '-')
                tick.label1.set_text(label)
    
    return ax


def plot_beta_vs_tauc(results, ax=None):
    """
    Plot stretched exponent β vs correlation time τ_c.
    
    Shows how the decay shape changes from Gaussian (β=2) to exponential (β=1)
    as a function of correlation time.
    
    Parameters
    ----------
    results : list
        List of simulation result dictionaries
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    tau_c_values = []
    beta_values = []
    beta_errors = []
    model_names = []
    
    for r in results:
        if r.get('fit_result') is not None:
            fit_result = r['fit_result']
            model = fit_result.get('model', '')
            
            tau_c_values.append(r['tau_c'])
            
            if model == 'stretched' and 'beta' in fit_result:
                beta_values.append(fit_result['beta'])
                # Get beta uncertainty (95% CI = 1.96 * std)
                beta_std = fit_result.get('beta_std')
                beta_err = beta_std * 1.96 if beta_std is not None else None
                beta_errors.append(beta_err)
                model_names.append('stretched')
            elif model == 'gaussian':
                beta_values.append(2.0)  # Gaussian decay
                beta_errors.append(None)  # No uncertainty for fixed value
                model_names.append('gaussian')
            elif model == 'exponential':
                beta_values.append(1.0)  # Exponential decay
                beta_errors.append(None)  # No uncertainty for fixed value
                model_names.append('exponential')
            else:
                # If model doesn't have beta, skip or infer
                continue
    
    if len(tau_c_values) == 0:
        return ax
    
    tau_c_values = np.array(tau_c_values)
    beta_values = np.array(beta_values)
    beta_errors = np.array(beta_errors)
    
    # Separate data by model for different colors
    tau_c_exp = []
    beta_exp = []
    beta_err_exp = []
    tau_c_stretch = []
    beta_stretch = []
    beta_err_stretch = []
    tau_c_gauss = []
    beta_gauss = []
    
    for i, model in enumerate(model_names):
        if model == 'exponential':
            tau_c_exp.append(tau_c_values[i])
            beta_exp.append(beta_values[i])
            if beta_errors[i] is not None:
                beta_err_exp.append(beta_errors[i])
            else:
                beta_err_exp.append(None)
        elif model == 'stretched':
            tau_c_stretch.append(tau_c_values[i])
            beta_stretch.append(beta_values[i])
            if beta_errors[i] is not None:
                beta_err_stretch.append(beta_errors[i])
            else:
                beta_err_stretch.append(None)
        elif model == 'gaussian':
            tau_c_gauss.append(tau_c_values[i])
            beta_gauss.append(beta_values[i])
    
    # Plot exponential points (green)
    if len(tau_c_exp) > 0:
        tau_c_exp = np.array(tau_c_exp)
        beta_exp = np.array(beta_exp)
        beta_err_exp = np.array(beta_err_exp)
        has_errors = np.array([e is not None for e in beta_err_exp])
        
        if np.any(has_errors):
            # Points with errors
            ax.errorbar(tau_c_exp[has_errors] * 1e6, beta_exp[has_errors],
                       yerr=beta_err_exp[has_errors], fmt='o', markersize=9,
                       color='#059669', label='Exponential ($\\beta=1$)', 
                       markeredgecolor='#047857', markeredgewidth=1.5,
                       capsize=4, capthick=1.5, alpha=0.8, zorder=5)
        if np.any(~has_errors):
            # Points without errors
            ax.plot(tau_c_exp[~has_errors] * 1e6, beta_exp[~has_errors], 'o',
                   markersize=9, color='#059669', markeredgecolor='#047857',
                   markeredgewidth=1.5, zorder=5)
    
    # Plot stretched points (blue) with error bars
    if len(tau_c_stretch) > 0:
        tau_c_stretch = np.array(tau_c_stretch)
        beta_stretch = np.array(beta_stretch)
        beta_err_stretch = np.array(beta_err_stretch)
        has_errors = np.array([e is not None for e in beta_err_stretch])
        
        if np.any(has_errors):
            # Points with errors
            ax.errorbar(tau_c_stretch[has_errors] * 1e6, beta_stretch[has_errors],
                       yerr=beta_err_stretch[has_errors], fmt='o', markersize=9,
                       color='#1E40AF', label='Stretched ($1 < \\beta < 2$)', 
                       markeredgecolor='#1E3A8A', markeredgewidth=1.5,
                       capsize=4, capthick=1.5, alpha=0.8, zorder=5)
        if np.any(~has_errors):
            # Points without errors
            ax.plot(tau_c_stretch[~has_errors] * 1e6, beta_stretch[~has_errors], 'o',
                   markersize=9, color='#1E40AF', markeredgecolor='#1E3A8A',
                   markeredgewidth=1.5, zorder=5)
    
    # Plot gaussian points (red)
    if len(tau_c_gauss) > 0:
        tau_c_gauss = np.array(tau_c_gauss)
        beta_gauss = np.array(beta_gauss)
        ax.plot(tau_c_gauss * 1e6, beta_gauss, 'o', markersize=9,
               color='#DC2626', label='Gaussian ($\\beta=2$)', 
               markeredgecolor='#991B1B', markeredgewidth=1.5, zorder=5)
    
    # Connect ALL points with lines in tau_c order (regardless of model) for smooth transition
    # This shows the continuous evolution of β from 1 to 2
    if len(tau_c_values) > 1:
        # Sort all data by tau_c
        sort_idx = np.argsort(tau_c_values)
        tau_c_sorted = tau_c_values[sort_idx] * 1e6
        beta_sorted = beta_values[sort_idx]
        
        # Plot connecting line through all points
        ax.plot(tau_c_sorted, beta_sorted,
               '-', color='#1E40AF', linewidth=2.5, alpha=0.5, zorder=3,
               label='_nolegend_')  # Don't add to legend (points already labeled)
    
    # Highlight transition region (β = 1 to 2) with gradient shading
    tau_c_us = tau_c_values * 1e6
    if len(tau_c_us) > 1:
        # Find transition region (where β changes from 1 to 2)
        beta_min = beta_values.min()
        beta_max = beta_values.max()
        
        # Identify transition region (where β is between 1.2 and 1.8)
        transition_mask = (beta_values >= 1.2) & (beta_values <= 1.8)
        if np.sum(transition_mask) > 0:
            tau_c_trans_min = tau_c_us[transition_mask].min()
            tau_c_trans_max = tau_c_us[transition_mask].max()
            
            # Add gradient-like shading for transition region
            ax.axvspan(tau_c_trans_min, tau_c_trans_max, 
                      color=COLOR_SCHEME['transition'], alpha=0.25, zorder=0,
                      label='Transition region')
    
    # Reference lines
    ax.axhline(2.0, color='#DC2626', linestyle='--', linewidth=1.8, alpha=0.8,
              label='Gaussian limit ($\\beta=2$)', zorder=1)
    ax.axhline(1.0, color='#059669', linestyle='--', linewidth=1.8, alpha=0.8,
              label='Exponential ($\\beta=1$)', zorder=1)
    
    ax.set_xlabel(r'$\tau_c$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Stretched exponent $\beta$', fontsize=13, fontweight='bold')
    ax.set_title(r'$\beta$ vs $\tau_c$', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_ylim([0.5, 2.5])
    
    return ax


def plot_hahn_echo_vs_fid(tau_echo, E_echo, E_echo_se, E_echo_theory,
                          t_fid, E_fid, E_fid_theory,
                          tau_c, ax=None):
    """
    Plot Hahn echo vs FID coherence comparison.
    
    Parameters
    ----------
    tau_echo : ndarray
        Echo times 2τ (seconds)
    E_echo : ndarray
        Simulated echo coherence |E_echo(2τ)|
    E_echo_se : ndarray
        Standard error of echo coherence
    E_echo_theory : ndarray
        Theoretical echo coherence
    t_fid : ndarray
        FID time points (seconds)
    E_fid : ndarray
        FID coherence |E_FID(t)|
    E_fid_theory : ndarray
        Theoretical FID coherence
    tau_c : float
        Correlation time (seconds)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot FID
    ax.plot(t_fid * 1e6, E_fid, '-', color='#1E40AF',  # Navy blue
           linewidth=2.5, label=r'FID: $|E_{\rm FID}(t)|$', alpha=0.9, zorder=3)
    ax.plot(t_fid * 1e6, E_fid_theory, '--', color='#DC2626',  # Red
           linewidth=2, label=r'FID Theory', alpha=0.8, dashes=(5, 2), zorder=2)
    
    # Plot Hahn echo
    ax.plot(tau_echo * 1e6, E_echo, 'o-', color='#F97316',  # Orange
           markersize=8, linewidth=2.5, label=r'Hahn Echo: $|E_{\rm echo}(2\tau)|$', 
           markeredgecolor='#C2410C', markeredgewidth=1.5, zorder=5, alpha=0.9)
    
    # Error bars for echo
    if E_echo_se is not None:
        ax.errorbar(tau_echo * 1e6, E_echo, yerr=E_echo_se,
                   fmt='none', color='#F97316', alpha=0.4, capsize=3, capthick=1.5)
        ax.fill_between(tau_echo * 1e6, 
                       E_echo - E_echo_se,
                       E_echo + E_echo_se,
                       alpha=0.15, color='#F97316')
    
    # Theoretical echo (filter-function based)
    if E_echo_theory is not None:
        ax.plot(tau_echo * 1e6, E_echo_theory, '-.', 
               color='#7C3AED',  # Purple/violet for better contrast
               linewidth=2.5, label=r'Echo Theory (filter-function)', alpha=0.9, zorder=1)
    
    # LaTeX labels
    ax.set_xlabel(r'Time $t$ or $2\tau$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$|E(t)|$', fontsize=13, fontweight='bold')
    
    tau_c_us = tau_c * 1e6
    
    ax.set_title(f'Hahn Echo vs FID\n($\\tau_c = {tau_c_us:.2f}$ μs)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, 1.05])
    
    return ax


def plot_multiple_hahn_echo_comparisons(echo_results, n_examples=3, figsize=None):
    """
    Plot multiple Hahn echo vs FID comparisons in a single figure with subplots.
    
    Parameters
    ----------
    echo_results : list
        List of result dictionaries from run_hahn_echo_sweep
    n_examples : int
        Number of tau_c values to show (default: 3)
    figsize : tuple, optional
        Figure size (default: auto-calculated based on n_examples)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if len(echo_results) == 0:
        return None
    
    # Select example indices
    n_examples = min(n_examples, len(echo_results))
    example_indices = np.linspace(0, len(echo_results) - 1, n_examples, dtype=int)
    
    # Determine layout (prefer horizontal layout for 3 or fewer)
    if n_examples <= 3:
        nrows = 1
        ncols = n_examples
    else:
        nrows = 2
        ncols = (n_examples + 1) // 2
    
    # Auto-calculate figure size
    if figsize is None:
        width = 6 * ncols
        height = 5 * nrows
        figsize = (width, height)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, idx in enumerate(example_indices):
        if idx >= len(echo_results):
            continue
        
        result = echo_results[idx]
        ax = axes[i]
        
        # Extract data
        tau_echo = np.array(result['tau_echo'])
        E_echo = np.array(result['E_echo_abs'])
        E_echo_se = result.get('E_echo_se')
        if E_echo_se is not None:
            E_echo_se = np.array(E_echo_se)
        E_echo_theory = result.get('E_echo_theory')
        if E_echo_theory is not None:
            E_echo_theory = np.array(E_echo_theory)
        t_fid = np.array(result['t_fid'])
        E_fid = np.array(result['E_fid_abs'])
        E_fid_theory = result.get('E_fid_theory')
        if E_fid_theory is not None:
            E_fid_theory = np.array(E_fid_theory)
        tau_c = result['tau_c']
        
        # Plot FID
        ax.plot(t_fid * 1e6, E_fid, '-', color='#1E40AF',
               linewidth=2.5, label=r'FID: $|E_{\rm FID}(t)|$', alpha=0.9, zorder=3)
        if E_fid_theory is not None and not np.all(np.isnan(E_fid_theory)):
            ax.plot(t_fid * 1e6, E_fid_theory, '--', color='#DC2626',
                   linewidth=2, label=r'FID Theory', alpha=0.8, dashes=(5, 2), zorder=2)
        
        # Plot Hahn echo
        ax.plot(tau_echo * 1e6, E_echo, 'o-', color='#F97316',
               markersize=6, linewidth=2, label=r'Hahn Echo: $|E_{\rm echo}(2\tau)|$',
               markeredgecolor='#C2410C', markeredgewidth=1.2, zorder=5, alpha=0.9)
        
        # Error bars for echo
        if E_echo_se is not None and not np.all(np.isnan(E_echo_se)):
            ax.errorbar(tau_echo * 1e6, E_echo, yerr=E_echo_se,
                       fmt='none', color='#F97316', alpha=0.4, capsize=2, capthick=1)
            ax.fill_between(tau_echo * 1e6,
                           E_echo - E_echo_se,
                           E_echo + E_echo_se,
                           alpha=0.15, color='#F97316')
        
        # Theoretical echo
        if E_echo_theory is not None and not np.all(np.isnan(E_echo_theory)):
            ax.plot(tau_echo * 1e6, E_echo_theory, '-.',
                   color='#7C3AED',
                   linewidth=2, label=r'Echo Theory', alpha=0.9, zorder=1)
        
        # Labels
        ax.set_xlabel(r'Time $t$ or $2\tau$ ($\mu$s)', fontsize=11, fontweight='bold')
        if i % ncols == 0:  # Leftmost column
            ax.set_ylabel(r'$|E(t)|$', fontsize=11, fontweight='bold')
        
        tau_c_us = tau_c * 1e6
        ax.set_title(f'$\\tau_c = {tau_c_us:.2f}$ μs',
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Hide unused subplots
    for i in range(len(example_indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_echo_envelope(echo_results, ax=None, show_fid_comparison=True):
    """
    Plot Hahn echo envelope (coherence curves) for multiple tau_c values.
    
    Shows |E_echo(2τ)| vs 2τ for different correlation times, with optional FID comparison.
    
    Parameters
    ----------
    echo_results : list
        List of result dictionaries from run_hahn_echo_sweep
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_fid_comparison : bool
        Whether to overlay FID curves for comparison
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select a few tau_c values for clarity (e.g., 4-5 examples)
    n_examples = min(5, len(echo_results))
    example_indices = np.linspace(0, len(echo_results) - 1, n_examples, dtype=int)
    
    for i, idx in enumerate(example_indices):
        if idx >= len(echo_results):
            continue
        
        result = echo_results[idx]
        tau_c = result['tau_c']
        tau_echo = np.array(result['tau_echo'])
        E_echo_abs = np.array(result['E_echo_abs'])
        E_echo_se = np.array(result['E_echo_se'])
        
        # Use sequential colormap (viridis) for better distinction
        import matplotlib.cm as cm
        tau_c_min = min([r['tau_c'] for r in echo_results])
        tau_c_max = max([r['tau_c'] for r in echo_results])
        norm = plt.Normalize(tau_c_min, tau_c_max)
        try:
            colormap = cm.get_cmap('viridis')
        except AttributeError:
            colormap = cm.viridis
        color = colormap(norm(tau_c))
        
        # Plot echo with tau_c label in legend only (removed overlapping labels on curves)
        tau_c_us = tau_c * 1e6
        label = rf'$\tau_c = {tau_c_us:.2f}$ μs'
        ax.plot(tau_echo * 1e6, E_echo_abs, 'o-', color=color,
               markersize=6, linewidth=2.5, label=label, alpha=0.9, zorder=5-i)
        
        # Error bands
        if E_echo_se is not None:
            ax.fill_between(tau_echo * 1e6,
                           E_echo_abs - E_echo_se,
                           E_echo_abs + E_echo_se,
                           alpha=0.15, color=color)
        
        # Optionally plot FID for comparison - dashed gray for clarity
        if show_fid_comparison and i == 0:  # Only show FID for first example
            t_fid = np.array(result['t_fid'])
            E_fid_abs = np.array(result['E_fid_abs'])
            # Only plot FID up to max echo time for clarity
            fid_mask = t_fid <= tau_echo.max()
            ax.plot(t_fid[fid_mask] * 1e6, E_fid_abs[fid_mask], '--',
                   color='gray', linewidth=2, alpha=0.6,
                   label='FID (reference)', zorder=1)
    
    ax.set_xlabel(r'Time $t$ or $2\tau$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$|E(t)|$', fontsize=13, fontweight='bold')
    ax.set_title('Hahn Echo Envelope', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, ncol=1, framealpha=0.9)
    ax.set_ylim([0, 1.05])
    
    # Removed excessive "Patterns" annotation box that obscured the graph
    
    return ax


def plot_T2_echo_vs_tauc(echo_results, ax=None, show_fid_comparison=True, show_theory=True):
    """
    Plot T_2,echo vs tau_c on log-log scale, with optional FID comparison.
    
    Includes bootstrap 95% CI error bars and verifies T₂,echo ≥ T₂,FID.
    
    Parameters
    ----------
    echo_results : list
        List of result dictionaries from run_hahn_echo_sweep
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_fid_comparison : bool
        Whether to overlay FID T_2 for comparison
    show_theory : bool
        Whether to show theoretical T_2 (motional-narrowing limit)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract echo T2 data
    tau_c_values = []
    T2_echo = []
    T2_echo_ci_lower = []
    T2_echo_ci_upper = []
    
    for r in echo_results:
        if r.get('fit_result_echo') is not None:
            tau_c_values.append(r['tau_c'])
            T2_echo.append(r['fit_result_echo']['T2'])
            
            # CI if available (bootstrap 95% CI)
            if r.get('T2_echo_ci') is not None:
                T2_echo_ci_lower.append(r['T2_echo_ci'][0])
                T2_echo_ci_upper.append(r['T2_echo_ci'][1])
            elif r.get('fit_result_echo', {}).get('T2_ci') is not None:
                T2_echo_ci = r['fit_result_echo']['T2_ci']
                T2_echo_ci_lower.append(T2_echo_ci[0])
                T2_echo_ci_upper.append(T2_echo_ci[1])
            else:
                T2_echo_ci_lower.append(None)
                T2_echo_ci_upper.append(None)
    
    if len(tau_c_values) == 0:
        return ax
    
    tau_c_values = np.array(tau_c_values)
    T2_echo = np.array(T2_echo)
    
    # Plot echo T2 with consistent color
    ax.loglog(tau_c_values * 1e6, T2_echo * 1e6, 'o-', 
             color='#F97316', markersize=10, 
             linewidth=2.5, label=r'Hahn Echo: $T_{2,\rm echo}$', zorder=5,
             markeredgecolor='#C2410C', markeredgewidth=1.5)
    
    # Error bars (bootstrap 95% CI)
    has_ci = any(ci is not None for ci in T2_echo_ci_lower)
    if has_ci:
        # Calculate symmetric error bars for log scale
        for i in range(len(tau_c_values)):
            if T2_echo_ci_lower[i] is not None and T2_echo_ci_upper[i] is not None:
                err_low = T2_echo[i] - T2_echo_ci_lower[i]
                err_up = T2_echo_ci_upper[i] - T2_echo[i]
                ax.errorbar(tau_c_values[i] * 1e6, T2_echo[i] * 1e6,
                           yerr=[[err_low * 1e6], [err_up * 1e6]],
                           fmt='none', color='#F97316', alpha=0.6, 
                           capsize=4, capthick=2, linewidth=2, zorder=4)
    
    # FID comparison with bootstrap CI
    if show_fid_comparison:
        tau_c_fid = []
        T2_fid = []
        T2_fid_ci_lower = []
        T2_fid_ci_upper = []
        
        for r in echo_results:
            if r.get('fit_result_fid') is not None:
                tau_c_fid.append(r['tau_c'])
                T2_fid.append(r['fit_result_fid']['T2'])
                
                # CI if available
                if r.get('fit_result_fid', {}).get('T2_ci') is not None:
                    T2_fid_ci = r['fit_result_fid']['T2_ci']
                    T2_fid_ci_lower.append(T2_fid_ci[0])
                    T2_fid_ci_upper.append(T2_fid_ci[1])
                else:
                    T2_fid_ci_lower.append(None)
                    T2_fid_ci_upper.append(None)
        
        if len(tau_c_fid) > 0:
            tau_c_fid = np.array(tau_c_fid)
            T2_fid = np.array(T2_fid)
            ax.loglog(tau_c_fid * 1e6, T2_fid * 1e6, 's--',
                     color='#1E40AF', markersize=8,
                     linewidth=2.5, alpha=0.9, label=r'FID: $T_{2,\rm FID}$',
                     markeredgecolor='#1E3A8A', markeredgewidth=1.5, zorder=4)
            
            # Error bars for FID
            for i in range(len(tau_c_fid)):
                if T2_fid_ci_lower[i] is not None and T2_fid_ci_upper[i] is not None:
                    err_low = T2_fid[i] - T2_fid_ci_lower[i]
                    err_up = T2_fid_ci_upper[i] - T2_fid[i]
                    ax.errorbar(tau_c_fid[i] * 1e6, T2_fid[i] * 1e6,
                               yerr=[[err_low * 1e6], [err_up * 1e6]],
                               fmt='none', color='#1E40AF', alpha=0.4, 
                               capsize=3, capthick=1.5, linewidth=1.5, zorder=3)
    
    # Verify T₂,echo ≥ T₂,FID (physical requirement)
    violations = []
    if show_fid_comparison and len(tau_c_fid) > 0:
        # Match tau_c values
        for i, tau_c_echo in enumerate(tau_c_values):
            # Find matching FID point
            idx_fid = np.argmin(np.abs(tau_c_fid - tau_c_echo))
            if abs(tau_c_fid[idx_fid] - tau_c_echo) < 1e-10:  # Same tau_c
                if T2_echo[i] < T2_fid[idx_fid]:
                    violations.append((tau_c_echo, T2_echo[i], T2_fid[idx_fid]))
    
    # Theoretical line (motional-narrowing: T2 ∝ 1/τ_c)
    if show_theory and len(echo_results) > 0:
        from fitting import theoretical_T2_motional_narrowing
        from config import CONSTANTS
        
        params = echo_results[0]['params']
        gamma_e = params['gamma_e']
        B_rms = params['B_rms']
        
        tau_c_theory = np.logspace(np.log10(tau_c_values.min()),
                                  np.log10(tau_c_values.max()), 100)
        T2_theory = [theoretical_T2_motional_narrowing(gamma_e, B_rms, tc) 
                     for tc in tau_c_theory]
        
        ax.loglog(tau_c_theory * 1e6, np.array(T2_theory) * 1e6, 
                 'k:', linewidth=1.5, alpha=0.5, zorder=1,
                 label=r'Theoretical (MN limit)')
    
    # Crossover line
    if len(echo_results) > 0:
        params = echo_results[0]['params']
        from config import CONSTANTS
        Delta_omega = CONSTANTS.GAMMA_E * params['B_rms']
        tau_c_crossover = 1.0 / Delta_omega
        ax.axvline(tau_c_crossover * 1e6, color='gray', linestyle=':', 
                  linewidth=1.5, alpha=0.5, label=r'$\tau_c = 1/\Delta\omega$')
    
    ax.set_xlabel(r'$\tau_c$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$T_2$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_title(r'$T_{2,\rm echo}$ vs $\tau_c$', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9)
    
    return ax


def plot_beta_echo_vs_tauc(echo_results, ax=None, show_fid_comparison=True):
    """
    Plot stretched exponent β_echo vs tau_c, with optional FID comparison.
    
    Parameters
    ----------
    echo_results : list
        List of result dictionaries from run_hahn_echo_sweep
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_fid_comparison : bool
        Whether to overlay FID β for comparison
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract echo beta data
    tau_c_echo = []
    beta_echo = []
    
    for r in echo_results:
        if r.get('fit_result_echo') is not None:
            fit_result = r['fit_result_echo']
            model = fit_result.get('model', '')
            
            if model == 'stretched' and 'beta' in fit_result:
                tau_c_echo.append(r['tau_c'])
                beta_echo.append(fit_result['beta'])
            elif model == 'gaussian':
                tau_c_echo.append(r['tau_c'])
                beta_echo.append(2.0)
            elif model == 'exponential':
                tau_c_echo.append(r['tau_c'])
                beta_echo.append(1.0)
    
    if len(tau_c_echo) == 0:
        return ax
    
    tau_c_echo = np.array(tau_c_echo)
    beta_echo = np.array(beta_echo)
    
    # Plot echo beta
    ax.semilogx(tau_c_echo * 1e6, beta_echo, 'o-',
               color='#F97316', markersize=10,
               linewidth=2.5, label=r'Hahn Echo: $\beta_{\rm echo}$',
               markeredgecolor='#C2410C', markeredgewidth=1.5, zorder=5)
    
    # FID comparison
    if show_fid_comparison:
        tau_c_fid = []
        beta_fid = []
        beta_fid_err = []  # For error bars if available
        
        for r in echo_results:
            if r.get('fit_result_fid') is not None:
                fit_result = r['fit_result_fid']
                model = fit_result.get('model', '')
                
                if model == 'stretched' and 'beta' in fit_result:
                    tau_c_fid.append(r['tau_c'])
                    beta_fid.append(fit_result['beta'])
                    # Try to get beta error if available
                    beta_err = fit_result.get('beta_err', 0.1)  # Default 10% error
                    beta_fid_err.append(beta_err)
                elif model == 'gaussian':
                    tau_c_fid.append(r['tau_c'])
                    beta_fid.append(2.0)
                    beta_fid_err.append(0.05)  # Small error for Gaussian
                elif model == 'exponential':
                    tau_c_fid.append(r['tau_c'])
                    beta_fid.append(1.0)
                    beta_fid_err.append(0.05)  # Small error for exponential
        
        if len(tau_c_fid) > 0:
            tau_c_fid = np.array(tau_c_fid)
            beta_fid = np.array(beta_fid)
            beta_fid_err = np.array(beta_fid_err)
            ax.semilogx(tau_c_fid * 1e6, beta_fid, 's--',
                       color='#1E40AF', markersize=8,
                       linewidth=2.5, alpha=0.9, label=r'FID: $\beta_{\rm FID}$',
                       markeredgecolor='#1E3A8A', markeredgewidth=1.5, zorder=4)
            # Add error bars for FID
            ax.errorbar(tau_c_fid * 1e6, beta_fid, yerr=beta_fid_err,
                       fmt='none', color='#1E40AF', alpha=0.4, capsize=3, capthick=1.5)
    
    # Reference lines
    ax.axhline(2.0, color='#DC2626', linestyle='--', linewidth=2, alpha=0.8,
              label='Gaussian limit ($\\beta=2$)', zorder=1)
    ax.axhline(1.0, color='#059669', linestyle='--', linewidth=2, alpha=0.8,
              label='Exponential ($\\beta=1$)', zorder=1)
    
    ax.set_xlabel(r'$\tau_c$ ($\mu$s)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Stretched exponent $\beta$', fontsize=13, fontweight='bold')
    ax.set_title(r'$\beta$ vs $\tau_c$', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0.3, 2.5])
    
    # Minimal annotations - removed excessive text boxes that obscure the graph
    
    return ax


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded.")
    print("Use create_summary_plots(results) to generate all plots.")

