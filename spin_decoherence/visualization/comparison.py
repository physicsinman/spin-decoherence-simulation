"""
Comparison plots for FID vs Hahn echo sequences.

This module provides functions for comparing FID and Hahn echo results.
"""

import numpy as np
import matplotlib.pyplot as plt
from spin_decoherence.visualization.styles import (
    COLOR_SCHEME,
    ANNOTATION_STYLE,
    setup_publication_style,
)

# Ensure publication style is set
setup_publication_style()


def plot_hahn_echo_vs_fid(tau_echo, E_echo, E_echo_se, E_echo_theory,
                          t_fid, E_fid, E_fid_se, E_fid_theory,
                          ax=None, show_error=True):
    """
    Plot Hahn echo vs FID coherence comparison.
    
    This function creates a comparison plot showing both FID and Hahn echo
    coherence curves on the same axes.
    
    Parameters
    ----------
    tau_echo : ndarray
        Echo times 2τ (seconds)
    E_echo : ndarray
        Echo coherence |E_echo(2τ)|
    E_echo_se : ndarray
        Standard error of echo coherence
    E_echo_theory : ndarray
        Theoretical echo coherence
    t_fid : ndarray
        FID time array (seconds)
    E_fid : ndarray
        FID coherence |E(t)|
    E_fid_se : ndarray
        Standard error of FID coherence
    E_fid_theory : ndarray
        Theoretical FID coherence
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_error : bool
        Whether to show error bands
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot FID
    ax.semilogy(t_fid * 1e6, E_fid, 'o-', color=COLOR_SCHEME['simulation'],
                label='FID (simulation)', markersize=4, linewidth=1.5)
    if show_error and E_fid_se is not None:
        ax.fill_between(t_fid * 1e6, E_fid - E_fid_se, E_fid + E_fid_se,
                        color=COLOR_SCHEME['simulation'], alpha=0.2)
    if E_fid_theory is not None:
        ax.semilogy(t_fid * 1e6, E_fid_theory, '--', color=COLOR_SCHEME['theory'],
                   label='FID (theory)', linewidth=2)
    
    # Plot Echo
    ax.semilogy(tau_echo * 1e6, E_echo, 's-', color=COLOR_SCHEME['echo'],
                label='Hahn Echo (simulation)', markersize=4, linewidth=1.5)
    if show_error and E_echo_se is not None:
        ax.fill_between(tau_echo * 1e6, E_echo - E_echo_se, E_echo + E_echo_se,
                        color=COLOR_SCHEME['echo'], alpha=0.2)
    if E_echo_theory is not None:
        ax.semilogy(tau_echo * 1e6, E_echo_theory, '--', color=COLOR_SCHEME['analytical'],
                   label='Hahn Echo (theory)', linewidth=2)
    
    ax.set_xlabel('Time (μs)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coherence |E|', fontsize=13, fontweight='bold')
    ax.set_title('FID vs Hahn Echo Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    
    return ax


def plot_multiple_hahn_echo_comparisons(echo_results, n_examples=3, figsize=None):
    """
    Plot multiple Hahn echo vs FID comparisons for different tau_c values.
    
    Parameters
    ----------
    echo_results : list
        List of result dictionaries containing both FID and echo data
    n_examples : int
        Number of examples to plot
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if figsize is None:
        figsize = (15, 5)
    
    n_examples = min(n_examples, len(echo_results))
    fig, axes = plt.subplots(1, n_examples, figsize=figsize)
    
    if n_examples == 1:
        axes = [axes]
    
    indices = np.linspace(0, len(echo_results) - 1, n_examples, dtype=int)
    
    for i, idx in enumerate(indices):
        result = echo_results[idx]
        tau_c = result['tau_c']
        
        # Extract data
        tau_echo = np.array(result.get('tau_echo', []))
        E_echo = np.array(result.get('E_echo_abs', []))
        E_echo_se = np.array(result.get('E_echo_se', []))
        t_fid = np.array(result.get('t_fid', result.get('t', [])))
        E_fid = np.array(result.get('E_fid_abs', result.get('E_abs', [])))
        E_fid_se = np.array(result.get('E_fid_se', result.get('E_se', [])))
        
        # Plot comparison
        plot_hahn_echo_vs_fid(
            tau_echo, E_echo, E_echo_se, None,
            t_fid, E_fid, E_fid_se, None,
            ax=axes[i], show_error=True
        )
        axes[i].set_title(f'$\\tau_c = {tau_c*1e6:.2f}$ μs', fontsize=12, fontweight='bold')
    
    fig.suptitle('FID vs Hahn Echo Comparison (Multiple $\\tau_c$ Values)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_T2_echo_vs_tauc(echo_results, ax=None, show_fid_comparison=True, show_theory=True):
    """
    Plot T_2 (echo) vs tau_c with FID comparison.
    
    Parameters
    ----------
    echo_results : list
        List of result dictionaries containing echo T_2 values
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_fid_comparison : bool
        Whether to show FID T_2 for comparison
    show_theory : bool
        Whether to show theoretical predictions
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    tau_c_list = [r['tau_c'] for r in echo_results]
    T2_echo_list = []
    T2_fid_list = []
    T2_echo_ci_list = []
    T2_fid_ci_list = []
    
    for r in echo_results:
        if r.get('fit_result_echo') is not None:
            T2_echo_list.append(r['fit_result_echo'].get('T2'))
            T2_echo_ci = r.get('T2_echo_ci')
            T2_echo_ci_list.append(T2_echo_ci if T2_echo_ci is not None else (None, None))
        else:
            T2_echo_list.append(None)
            T2_echo_ci_list.append((None, None))
        
        if show_fid_comparison and r.get('fit_result_fid') is not None:
            T2_fid_list.append(r['fit_result_fid'].get('T2'))
            T2_fid_ci = r['fit_result_fid'].get('T2_ci')
            T2_fid_ci_list.append(T2_fid_ci if T2_fid_ci is not None else (None, None))
        else:
            T2_fid_list.append(None)
            T2_fid_ci_list.append((None, None))
    
    tau_c_array = np.array(tau_c_list) * 1e6  # Convert to μs
    
    # Plot Echo T_2
    valid_echo = [i for i, t2 in enumerate(T2_echo_list) if t2 is not None]
    if valid_echo:
        T2_echo_valid = np.array([T2_echo_list[i] for i in valid_echo]) * 1e6
        tau_c_echo_valid = tau_c_array[valid_echo]
        ax.loglog(tau_c_echo_valid, T2_echo_valid, 's-', color=COLOR_SCHEME['echo'],
                 label='T₂ (Hahn Echo)', markersize=6, linewidth=2)
        
        # Plot CI if available
        for i, idx in enumerate(valid_echo):
            ci = T2_echo_ci_list[idx]
            if ci[0] is not None and ci[1] is not None:
                ax.plot([tau_c_echo_valid[i], tau_c_echo_valid[i]],
                       [ci[0]*1e6, ci[1]*1e6], '|-', color=COLOR_SCHEME['echo'],
                       alpha=0.5, linewidth=1)
    
    # Plot FID T_2
    if show_fid_comparison:
        valid_fid = [i for i, t2 in enumerate(T2_fid_list) if t2 is not None]
        if valid_fid:
            T2_fid_valid = np.array([T2_fid_list[i] for i in valid_fid]) * 1e6
            tau_c_fid_valid = tau_c_array[valid_fid]
            ax.loglog(tau_c_fid_valid, T2_fid_valid, 'o-', color=COLOR_SCHEME['simulation'],
                     label='T₂ (FID)', markersize=6, linewidth=2)
            
            # Plot CI if available
            for i, idx in enumerate(valid_fid):
                ci = T2_fid_ci_list[idx]
                if ci[0] is not None and ci[1] is not None:
                    ax.plot([tau_c_fid_valid[i], tau_c_fid_valid[i]],
                           [ci[0]*1e6, ci[1]*1e6], '|-', color=COLOR_SCHEME['simulation'],
                           alpha=0.5, linewidth=1)
    
    # Plot theory if available
    if show_theory and len(echo_results) > 0:
        from spin_decoherence.simulation.engine import estimate_characteristic_T2
        params = echo_results[0].get('params', {})
        gamma_e = params.get('gamma_e', 1.76e11)
        B_rms = params.get('B_rms', 0.57e-6)  # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration
        
        tau_c_theory = np.logspace(np.log10(tau_c_array.min()), np.log10(tau_c_array.max()), 100) * 1e-6
        T2_theory = [estimate_characteristic_T2(tc, gamma_e, B_rms) for tc in tau_c_theory]
        T2_theory = np.array(T2_theory) * 1e6
        
        ax.loglog(tau_c_theory * 1e6, T2_theory, '--', color=COLOR_SCHEME['theory'],
                 label='T₂ (theory)', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('$\\tau_c$ (μs)', fontsize=13, fontweight='bold')
    ax.set_ylabel('$T_2$ (μs)', fontsize=13, fontweight='bold')
    ax.set_title('$T_2$ vs $\\tau_c$: FID vs Hahn Echo', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    
    return ax

