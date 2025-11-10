"""
High-resolution scan for Motional-Narrowing (MN) regime.

This script performs a focused parameter sweep in the MN regime (ξ < 0.2)
to improve the resolution of the T_2 vs τ_c scaling analysis.

For MN regime: ξ = γ_e * B_rms * τ_c < 0.2
With B_rms = 5 μT, γ_e = 1.76e11 rad·s⁻¹·T⁻¹:
    τ_c < 0.2 / (1.76e11 * 5e-6) ≈ 0.227 μs

We scan τ_c = 0.005 - 0.2 μs with 15-20 points for better resolution.
"""

import numpy as np
import argparse
import os
from datetime import datetime
from simulate import run_simulation_sweep, save_results, get_default_config, config_to_dict
from visualize import create_summary_plots
from config import CONSTANTS
from units import Units


def main():
    parser = argparse.ArgumentParser(
        description='High-Resolution MN Regime Scan'
    )
    parser.add_argument('--tau-c-min', type=float, default=0.005,
                       help='Minimum tau_c in μs (default: 0.005)')
    parser.add_argument('--tau-c-max', type=float, default=0.2,
                       help='Maximum tau_c in μs (default: 0.2)')
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
    
    args = parser.parse_args()
    
    # Unit conversion: CLI inputs (μs, ns, μT) → internal SI units (s, T)
    tau_c_min_si = Units.us_to_s(args.tau_c_min)
    tau_c_max_si = Units.us_to_s(args.tau_c_max)
    T_max_si = Units.us_to_s(args.T_max)
    dt_si = Units.ns_to_s(args.dt)
    B_rms_si = Units.uT_to_T(args.B_rms)
    
    # Validation
    assert tau_c_min_si > 0, f"tau_c_min must be positive, got {args.tau_c_min} μs"
    assert tau_c_max_si > tau_c_min_si, f"tau_c_max ({args.tau_c_max} μs) must be > tau_c_min ({args.tau_c_min} μs)"
    
    # Check MN regime condition
    Delta_omega = CONSTANTS.GAMMA_E * B_rms_si
    xi_max = Delta_omega * tau_c_max_si
    xi_min = Delta_omega * tau_c_min_si
    
    print("=" * 70)
    print("High-Resolution Motional-Narrowing Regime Scan")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  τ_c range: {args.tau_c_min:.3f} - {args.tau_c_max:.3f} μs ({args.tau_c_num} values)")
    print(f"  T_max: {args.T_max:.2f} μs")
    print(f"  dt: {args.dt:.2f} ns")
    print(f"  M (realizations): {args.M}")
    print(f"  B_rms: {args.B_rms:.2f} μT")
    print(f"\nMN Regime Check:")
    print(f"  ξ = γ_e * B_rms * τ_c")
    print(f"  ξ_min = {xi_min:.4f} (at τ_c = {args.tau_c_min:.3f} μs)")
    print(f"  ξ_max = {xi_max:.4f} (at τ_c = {args.tau_c_max:.3f} μs)")
    
    if xi_max > 0.2:
        print(f"\n  ⚠️  WARNING: ξ_max = {xi_max:.4f} > 0.2")
        print(f"  Some points may be outside MN regime!")
    else:
        print(f"\n  ✅ All points in MN regime (ξ < 0.2)")
    
    print("=" * 70)
    
    # Set parameters
    default_config = get_default_config()
    params = config_to_dict(default_config)
    params['tau_c_range'] = (tau_c_min_si, tau_c_max_si)
    params['tau_c_num'] = args.tau_c_num
    params['T_max'] = T_max_si
    params['dt'] = dt_si
    params['M'] = args.M
    params['B_rms'] = B_rms_si
    params['seed'] = args.seed
    params['output_dir'] = args.output_dir
    params['compute_bootstrap'] = True
    
    # Run simulation
    results = run_simulation_sweep(params, verbose=True)
    
    # Save results with special naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = save_results(
        results, 
        output_dir=args.output_dir,
        filename=f"mn_regime_scan_{timestamp}.json"
    )
    
    # Generate plots
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("Generating plots...")
        print("=" * 70)
        create_summary_plots(
            results, 
            output_dir=args.output_dir, 
            save=True,
            compute_bootstrap=True,
            gamma_e=CONSTANTS.GAMMA_E, 
            B_rms=params['B_rms']
        )
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("MN Regime Scan Summary")
    print("=" * 70)
    
    successful_fits = sum(1 for r in results if r.get('fit_result') is not None)
    print(f"Total simulations: {len(results)}")
    print(f"Successful fits: {successful_fits}")
    
    if successful_fits > 0:
        T2_values = [r['fit_result']['T2'] for r in results if r.get('fit_result')]
        tau_c_values = [r['tau_c'] for r in results if r.get('fit_result')]
        
        print(f"\nT_2 range: {min(T2_values)*1e6:.2f} - {max(T2_values)*1e6:.2f} μs")
        print(f"τ_c range: {min(tau_c_values)*1e6:.3f} - {max(tau_c_values)*1e6:.3f} μs")
        
        # MN regime fit
        from fitting import fit_mn_slope
        mn_fit = fit_mn_slope(results, CONSTANTS.GAMMA_E, params['B_rms'], xi_threshold=0.2)
        if mn_fit is not None:
            print(f"\n{'='*70}")
            print("Motional-Narrowing Regime Analysis (ξ < 0.2)")
            print(f"{'='*70}")
            print(f"Slope: {mn_fit['slope']:.4f} ± {mn_fit['slope_std']:.4f}")
            print(f"  (Expected: -1.000 in MN limit)")
            print(f"R²: {mn_fit['R2']:.6f}")
            print(f"Number of points: {mn_fit['n_points']}")
            print(f"τ_c range: {mn_fit['tau_c_range'][0]*1e6:.4f} - {mn_fit['tau_c_range'][1]*1e6:.4f} μs")
            print(f"ξ range: {mn_fit['xi_range'][0]:.6f} - {mn_fit['xi_range'][1]:.6f}")
            
            # Deviation from theory
            slope_error = abs(mn_fit['slope'] - (-1.0))
            print(f"\nDeviation from theory: {slope_error:.6f}")
            if slope_error < 0.01:
                print("  ✅ Excellent agreement with theory!")
            elif slope_error < 0.05:
                print("  ✅ Good agreement with theory")
            else:
                print("  ⚠️  Significant deviation - may need more points or longer T_max")
    
    print("\n" + "=" * 70)
    print("MN regime scan completed successfully!")
    print(f"Results saved to: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

