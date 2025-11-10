"""
Main entry point for running the full simulation sweep.

This script runs the complete parameter sweep and generates all plots.
"""

import numpy as np
import argparse
import os
import json
from datetime import datetime
from simulate import run_simulation_sweep, save_results, get_default_config, config_to_dict
from visualize import create_summary_plots
from config import CONSTANTS
from units import Units


def main():
    parser = argparse.ArgumentParser(
        description='Spin Decoherence Simulation - Parameter Sweep'
    )
    parser.add_argument('--tau-c-min', type=float, default=0.01,
                       help='Minimum tau_c in μs (default: 0.01)')
    parser.add_argument('--tau-c-max', type=float, default=10.0,
                       help='Maximum tau_c in μs (default: 10.0)')
    parser.add_argument('--tau-c-num', type=int, default=20,
                       help='Number of tau_c values (default: 20)')
    parser.add_argument('--T-max', type=float, default=30.0,
                       help='Maximum simulation time in μs (default: 30.0)')
    parser.add_argument('--dt', type=float, default=0.2,
                       help='Time step in ns (default: 0.2)')
    parser.add_argument('--M', type=int, default=1000,
                       help='Number of realizations (default: 1000)')
    parser.add_argument('--B-rms', type=float, default=5.0,
                       help='RMS noise amplitude in μT (default: 5.0)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no-bootstrap', action='store_true',
                       help='Skip bootstrap CI computation (faster)')
    parser.add_argument('--save-psd-sample', action='store_true',
                       help='Save OU noise sample for PSD verification')
    parser.add_argument('--hahn-echo', action='store_true',
                       help='Include Hahn echo simulation')
    parser.add_argument('--tau-echo-min', type=float, default=50.0,
                       help='Minimum echo delay τ in μs (default: 50.0)')
    parser.add_argument('--tau-echo-max', type=float, default=2000.0,
                       help='Maximum echo delay τ in μs (default: 2000.0)')
    parser.add_argument('--tau-echo-num', type=int, default=30,
                       help='Number of τ values (default: 30)')
    
    args = parser.parse_args()
    
    # Set up parameters with explicit unit conversion and validation
    default_config = get_default_config()
    params = config_to_dict(default_config)
    
    # Unit conversion: CLI inputs (μs, ns, μT) → internal SI units (s, T)
    tau_c_min_si = Units.us_to_s(args.tau_c_min)
    tau_c_max_si = Units.us_to_s(args.tau_c_max)
    T_max_si = Units.us_to_s(args.T_max)
    dt_si = Units.ns_to_s(args.dt)
    B_rms_si = Units.uT_to_T(args.B_rms)
    
    # Validation: critical parameter checks
    assert tau_c_min_si > 0, f"tau_c_min must be positive, got {args.tau_c_min} μs"
    assert tau_c_max_si > tau_c_min_si, f"tau_c_max ({args.tau_c_max} μs) must be > tau_c_min ({args.tau_c_min} μs)"
    assert dt_si > 0, f"dt must be positive, got {args.dt} ns"
    assert T_max_si > 0, f"T_max must be positive, got {args.T_max} μs"
    assert B_rms_si > 0, f"B_rms must be positive, got {args.B_rms} μT"
    
    # Stability check: dt << tau_c (should be at least 50x smaller for accuracy)
    # README recommends: dt ≤ min(τc/50, 0.2ns) for numerical stability
    min_tau_c = min(tau_c_min_si, tau_c_max_si)
    max_tau_c = max(tau_c_min_si, tau_c_max_si)
    
    # Recommended dt: min(tau_c/50, 0.2ns) for all tau_c values
    dt_recommended = min(min_tau_c / 50.0, 0.2e-9)
    dt_minimum = min_tau_c / 10.0  # Absolute minimum: 10 samples per tau_c
    
    if dt_si > dt_recommended:
        print(f"⚠️  WARNING: dt ({args.dt:.2f} ns) may be too large relative to tau_c.")
        print(f"  Recommended: dt ≤ min(τc_min/50, 0.2ns) = {dt_recommended*1e9:.2f} ns")
        print(f"  Minimum: dt ≤ τc_min/10 = {dt_minimum*1e9:.2f} ns")
        print(f"  Current dt = {dt_si*1e9:.2f} ns")
        
        # Check severity
        if dt_si >= min_tau_c / 5:  # Very large dt (critical)
            dt_suggested = dt_recommended
            print(f"  🔴 CRITICAL: dt is very large, may cause numerical instability in OU process")
            print(f"  🔧 SUGGESTION: Use dt = {dt_suggested*1e9:.2f} ns for better accuracy")
        elif dt_si >= min_tau_c / 10:  # Large dt (warning)
            print(f"  ⚠️  WARNING: dt is large, may cause minor numerical errors")
            print(f"  🔧 SUGGESTION: Consider dt ≤ {dt_recommended*1e9:.2f} ns")
    
    # Set parameters
    params['tau_c_range'] = (tau_c_min_si, tau_c_max_si)
    params['tau_c_num'] = args.tau_c_num
    params['T_max'] = T_max_si
    params['dt'] = dt_si
    params['M'] = args.M
    params['B_rms'] = B_rms_si
    params['seed'] = args.seed
    params['output_dir'] = args.output_dir
    params['compute_bootstrap'] = not args.no_bootstrap
    params['save_delta_B_sample'] = args.save_psd_sample
    
    # Log unit conversions for verification
    if not args.no_plots:
        print(f"\nUnit conversions (CLI → SI):")
        print(f"  τ_c: {args.tau_c_min:.2f}-{args.tau_c_max:.2f} μs → {tau_c_min_si:.2e}-{tau_c_max_si:.2e} s")
        print(f"  T_max: {args.T_max:.2f} μs → {T_max_si:.2e} s")
        print(f"  dt: {args.dt:.2f} ns → {dt_si:.2e} s")
        print(f"  B_rms: {args.B_rms:.2f} μT → {B_rms_si:.2e} T")
    
    print("=" * 70)
    print("Spin Decoherence Simulation - Full Parameter Sweep")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  τ_c range: {args.tau_c_min:.2f} - {args.tau_c_max:.2f} μs ({args.tau_c_num} values)")
    print(f"  T_max: {args.T_max:.2f} μs")
    print(f"  dt: {args.dt:.2f} ns")
    print(f"  M (realizations): {args.M}")
    print(f"  B_rms: {args.B_rms:.2f} μT")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Run simulation with Hahn echo if requested
    if args.hahn_echo:
        from simulate import run_hahn_echo_sweep
        from visualize import (plot_hahn_echo_vs_fid, plot_echo_envelope,
                               plot_multiple_hahn_echo_comparisons, 
                              plot_T2_echo_vs_tauc, plot_beta_echo_vs_tauc)
        import matplotlib.pyplot as plt
        
        print("\n" + "="*70)
        print("Running Hahn Echo Sweep Simulation")
        print("="*70)
        print("Using dimensionless scan: υ = τ/τc ∈ [0.05, 0.8]")
        print("(tau_list will be auto-generated for each tau_c)")
        
        # Use None to trigger dimensionless scan in run_hahn_echo_sweep
        # Each tau_c will use get_dimensionless_tau_range automatically
        tau_list = None
        
        # Run full sweep for all tau_c values
        echo_results = run_hahn_echo_sweep(params, tau_list=tau_list, verbose=True)
        
        # Save echo results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        echo_json_path = os.path.join(args.output_dir, f"hahn_echo_results_{timestamp}.json")
        save_results(echo_results, output_dir=args.output_dir,
                    filename=f"hahn_echo_results_{timestamp}.json")
        print(f"\nHahn echo results saved to: {echo_json_path}")
        
        # Generate Hahn echo plots
        if not args.no_plots:
            print("\n" + "="*70)
            print("Generating Hahn Echo Plots")
            print("="*70)
            
            # Plot 1: Echo envelope (multiple tau_c)
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            plot_echo_envelope(echo_results, ax=ax1, show_fid_comparison=True)
            plt.tight_layout()
            filename1 = os.path.join(args.output_dir, "hahn_echo_envelope.png")
            plt.savefig(filename1, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename1}")
            plt.close()
            
            # Plot 2: T_2,echo vs tau_c (with FID comparison)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            plot_T2_echo_vs_tauc(echo_results, ax=ax2, show_fid_comparison=True, show_theory=True)
            plt.tight_layout()
            filename2 = os.path.join(args.output_dir, "hahn_echo_T2_vs_tauc.png")
            plt.savefig(filename2, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename2}")
            plt.close()
            
            # Plot 3: β_echo vs tau_c (with FID comparison)
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            plot_beta_echo_vs_tauc(echo_results, ax=ax3, show_fid_comparison=True)
            plt.tight_layout()
            filename3 = os.path.join(args.output_dir, "hahn_echo_beta_vs_tauc.png")
            plt.savefig(filename3, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename3}")
            plt.close()
            
            # Plot 4: Multiple comparisons in a single figure
            n_examples = min(3, len(echo_results))
            fig = plot_multiple_hahn_echo_comparisons(echo_results, n_examples=n_examples)
            if fig is not None:
                filename = os.path.join(args.output_dir, "hahn_echo_comparison.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")
                plt.close(fig)
            
            print("\n" + "="*70)
            print("All Hahn echo plots generated successfully!")
            print("="*70)
    
    # Run simulation
    results = run_simulation_sweep(params, verbose=True)
    
    # Save results
    json_path = save_results(results, output_dir=args.output_dir)
    
    # Generate plots
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("Generating plots...")
        print("=" * 70)
        # CONSTANTS already imported at top
        create_summary_plots(results, output_dir=args.output_dir, save=True,
                           compute_bootstrap=True,
                           gamma_e=CONSTANTS.GAMMA_E, B_rms=params['B_rms'])
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Simulation Summary")
    print("=" * 70)
    
    successful_fits = sum(1 for r in results if r.get('fit_result') is not None)
    print(f"Total simulations: {len(results)}")
    print(f"Successful fits: {successful_fits}")
    
    if successful_fits > 0:
        T2_values = [r['fit_result']['T2'] for r in results if r.get('fit_result')]
        tau_c_values = [r['tau_c'] for r in results if r.get('fit_result')]
        
        print(f"\nT_2 range: {min(T2_values)*1e6:.2f} - {max(T2_values)*1e6:.2f} μs")
        print(f"τ_c range: {min(tau_c_values)*1e6:.2f} - {max(tau_c_values)*1e6:.2f} μs")
        
        # MN regime fit
        from fitting import fit_mn_slope
        # CONSTANTS already imported at top
        mn_fit = fit_mn_slope(results, CONSTANTS.GAMMA_E, params['B_rms'], xi_threshold=0.2)
        if mn_fit is not None:
            print(f"\n{'='*70}")
            print("Motional-Narrowing Regime Analysis (ξ < 0.2)")
            print(f"{'='*70}")
            print(f"Slope: {mn_fit['slope']:.3f} ± {mn_fit['slope_std']:.3f}")
            print(f"  (Expected: -1.000 in MN limit)")
            print(f"R²: {mn_fit['R2']:.4f}")
            print(f"Number of points: {mn_fit['n_points']}")
            print(f"τ_c range: {mn_fit['tau_c_range'][0]*1e6:.3f} - {mn_fit['tau_c_range'][1]*1e6:.3f} μs")
            print(f"ξ range: {mn_fit['xi_range'][0]:.4f} - {mn_fit['xi_range'][1]:.4f}")
        
        # Full range scaling
        if len(T2_values) > 1:
            log_tau = np.log10(tau_c_values)
            log_T2 = np.log10(T2_values)
            slope = np.polyfit(log_tau, log_T2, 1)[0]
            print(f"\nFull range scaling: T_2 ∝ τ_c^{slope:.2f}")
            print(f"  (Note: includes crossover regime)")
    
    print("\n" + "=" * 70)
    print("Simulation completed successfully!")
    print(f"Results saved to: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

