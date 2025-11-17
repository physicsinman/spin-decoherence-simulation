"""
Analytical comparison plot for spin decoherence simulation.

Compares simulated coherence curves with analytical predictions
for all tau_c values and generates residual plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from glob import glob
from spin_decoherence.physics import compute_ensemble_coherence, analytical_ou_coherence
from spin_decoherence.config import CONSTANTS


def load_results_from_json(json_path):
    """Load simulation results from JSON file."""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def compare_simulation_analytical(result, ax=None, show_residuals=False):
    """
    Compare simulation with analytical prediction for a single result.
    
    Parameters
    ----------
    result : dict
        Single simulation result dictionary
    ax : matplotlib.axes, optional
        Axes to plot on
    show_residuals : bool
        Whether to show residual subplot
        
    Returns
    -------
    ax : matplotlib.axes
    """
    if ax is None:
        if show_residuals:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                          height_ratios=[2, 1])
            ax = ax1
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data
    t = np.array(result['t'])
    E_abs = np.array(result['E_magnitude'])
    E_se = np.array(result.get('E_se', []))
    tau_c = result['tau_c']
    B_rms = result['params']['B_rms']
    
    # Analytical prediction
    E_analytical = analytical_ou_coherence(t, CONSTANTS.GAMMA_E, B_rms, tau_c)
    
    # Plot simulation
    if len(E_se) > 0 and np.any(E_se > 0):
        ax.errorbar(t * 1e6, E_abs, yerr=E_se, fmt='o', markersize=4,
                   alpha=0.6, label='Simulation', capsize=2, capthick=1)
    else:
        ax.plot(t * 1e6, E_abs, 'o', markersize=4, alpha=0.6, label='Simulation')
    
    # Plot analytical
    ax.plot(t * 1e6, E_analytical, '-', linewidth=2, label='Analytical (OU)')
    
    # Plot fitted curve if available
    if result.get('fit_result') is not None:
        fit_curve = result['fit_result'].get('fit_curve', [])
        if len(fit_curve) > 0:
            ax.plot(t * 1e6, fit_curve, '--', linewidth=2, alpha=0.7,
                   label='Fitted')
    
    ax.set_xlabel('Time $t$ (μs)', fontsize=12)
    ax.set_ylabel('$|E(t)|$', fontsize=12)
    ax.set_title(f'Coherence: Simulation vs Analytical (τ$_c$ = {tau_c*1e6:.3f} μs)',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Residual plot
    if show_residuals:
        residual = E_abs - E_analytical
        relative_residual = residual / (E_analytical + 1e-10)  # Avoid division by zero
        
        ax2.plot(t * 1e6, relative_residual, 'o-', markersize=3, linewidth=1)
        ax2.axhline(0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time $t$ (μs)', fontsize=12)
        ax2.set_ylabel('Relative Residual', fontsize=12)
        ax2.set_title('Residual: $(E_{\\text{sim}} - E_{\\text{analytical}}) / E_{\\text{analytical}}$',
                     fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return ax


def plot_multiple_comparisons(results, n_examples=4, output_dir='results'):
    """
    Plot analytical comparison for multiple tau_c values.
    
    Parameters
    ----------
    results : list
        List of simulation result dictionaries
    n_examples : int
        Number of examples to plot
    output_dir : str
        Output directory
    """
    # Select representative examples
    valid_results = [r for r in results if r.get('fit_result') is not None]
    if len(valid_results) == 0:
        print("No valid results found!")
        return
    
    n_examples = min(n_examples, len(valid_results))
    indices = np.linspace(0, len(valid_results) - 1, n_examples, dtype=int)
    
    # Create subplot grid
    n_cols = 2
    n_rows = (n_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_examples > 1 else [axes]
    
    for i, idx in enumerate(indices):
        result = valid_results[idx]
        compare_simulation_analytical(result, ax=axes[i], show_residuals=False)
    
    # Hide extra subplots
    for i in range(n_examples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'analytical_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_residual_summary(results, output_dir='results'):
    """
    Plot summary of residuals for all tau_c values.
    
    Parameters
    ----------
    results : list
        List of simulation result dictionaries
    output_dir : str
        Output directory
    """
    tau_c_list = []
    max_residual_list = []
    mean_residual_list = []
    rms_residual_list = []
    
    for result in results:
        if result.get('fit_result') is None:
            continue
        
        t = np.array(result['t'])
        E_abs = np.array(result['E_magnitude'])
        tau_c = result['tau_c']
        B_rms = result['params']['B_rms']
        
        # Analytical prediction
        E_analytical = analytical_ou_coherence(t, CONSTANTS.GAMMA_E, B_rms, tau_c)
        
        # Residual
        residual = E_abs - E_analytical
        relative_residual = residual / (E_analytical + 1e-10)
        
        # Statistics
        tau_c_list.append(tau_c * 1e6)
        max_residual_list.append(np.abs(relative_residual).max())
        mean_residual_list.append(np.abs(relative_residual).mean())
        rms_residual_list.append(np.sqrt(np.mean(relative_residual**2)))
    
    if len(tau_c_list) == 0:
        print("No valid results for residual summary!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tau_c_arr = np.array(tau_c_list)
    max_residual_arr = np.array(max_residual_list)
    mean_residual_arr = np.array(mean_residual_list)
    rms_residual_arr = np.array(rms_residual_list)
    
    ax.semilogx(tau_c_arr, max_residual_arr * 100, 'o-', label='Max |Relative Residual| (%)',
               linewidth=2, markersize=8)
    ax.semilogx(tau_c_arr, mean_residual_arr * 100, 's-', label='Mean |Relative Residual| (%)',
               linewidth=2, markersize=8)
    ax.semilogx(tau_c_arr, rms_residual_arr * 100, '^-', label='RMS Relative Residual (%)',
               linewidth=2, markersize=8)
    
    ax.set_xlabel('Correlation Time τ$_c$ (μs)', fontsize=12)
    ax.set_ylabel('Relative Residual (%)', fontsize=12)
    ax.set_title('Analytical Comparison: Residual Statistics', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'analytical_residual_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analytical Comparison Plot'
    )
    parser.add_argument('--json-file', type=str, default=None,
                       help='JSON results file (default: latest)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--n-examples', type=int, default=4,
                       help='Number of example plots (default: 4)')
    
    args = parser.parse_args()
    
    # Find JSON file
    if args.json_file is None:
        # Find latest simulation results
        json_files = glob(os.path.join(args.output_dir, 'simulation_results_*.json'))
        if len(json_files) == 0:
            print(f"Error: No simulation results found in {args.output_dir}")
            return
        json_file = max(json_files, key=os.path.getctime)
        print(f"Using latest results: {json_file}")
    else:
        json_file = args.json_file
    
    # Load results
    results = load_results_from_json(json_file)
    
    print(f"Loaded {len(results)} results")
    
    # Plot multiple comparisons
    print("\nGenerating analytical comparison plots...")
    plot_multiple_comparisons(results, n_examples=args.n_examples, 
                             output_dir=args.output_dir)
    
    # Plot residual summary
    print("Generating residual summary...")
    plot_residual_summary(results, output_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print("Analytical comparison completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

