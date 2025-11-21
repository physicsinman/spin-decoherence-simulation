#!/usr/bin/env python3
"""
Regenerate all plots from existing simulation results.

This script loads the latest simulation results and regenerates all plots
with the improved visualization code.
"""
import json
import os
import glob
from visualize import create_summary_plots
from spin_decoherence.config import CONSTANTS

def find_latest_results(results_dir='results'):
    """Find the latest simulation results file."""
    pattern = os.path.join(results_dir, 'simulation_results_*.json')
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, get latest
    latest = max(files, key=os.path.getmtime)
    return latest

def extract_parameters(results):
    """Extract gamma_e and B_rms from results."""
    # Try to get from first result's params
    if len(results) > 0:
        first_result = results[0]
        if 'params' in first_result:
            params = first_result['params']
            gamma_e = params.get('gamma_e', CONSTANTS.GAMMA_E)
            B_rms = params.get('B_rms', 5e-6)  # Default: 5 μT
            return gamma_e, B_rms
    
    # Fallback to constants
    return CONSTANTS.GAMMA_E, 5e-6

def main():
    results_dir = 'results'
    latest_file = find_latest_results(results_dir)
    
    if latest_file is None:
        print("No simulation results found. Please run the simulation first.")
        return
    
    print("=" * 70)
    print("Regenerating All Plots from Existing Results")
    print("=" * 70)
    print(f"\nLoading results from: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Handle different result file structures
    if isinstance(data, dict):
        if 'results' in data:
            results = data['results']
            print(f"Found {len(results)} results in structured format")
        else:
            # Single result or old format
            results = [data]
            print(f"Found 1 result (single result format)")
    else:
        results = data
        print(f"Found {len(results)} results")
    
    # Extract parameters for theory plots
    gamma_e, B_rms = extract_parameters(results)
    print(f"\nUsing parameters:")
    print(f"  gamma_e = {gamma_e:.3e} rad·s⁻¹·T⁻¹")
    print(f"  B_rms = {B_rms*1e6:.2f} μT")
    
    print("\n" + "=" * 70)
    print("Regenerating all summary plots...")
    print("=" * 70)
    print("\nThis will generate:")
    print("  - T2_vs_tauc.png")
    print("  - coherence_curves.png")
    print("  - coherence_examples.png")
    print("  - dimensionless_collapse.png")
    print("  - beta_vs_tauc.png")
    print("  - ou_psd_verification.png")
    
    # Regenerate all plots with improved code
    create_summary_plots(
        results, 
        output_dir=results_dir, 
        save=True,
        compute_bootstrap=False,  # Skip bootstrap to save time (CI already in results)
        gamma_e=gamma_e,
        B_rms=B_rms
    )
    
    print("\n" + "=" * 70)
    print("✅ All plots regenerated successfully!")
    print("=" * 70)
    print(f"\nCheck the '{results_dir}' directory for updated plots.")

if __name__ == '__main__':
    main()

