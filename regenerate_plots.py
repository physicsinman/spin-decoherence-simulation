#!/usr/bin/env python3
"""
Quick script to regenerate coherence_examples.png from existing results.
"""
import json
import os
import glob
from visualize import create_summary_plots

def find_latest_results(results_dir='results'):
    """Find the latest simulation results file."""
    pattern = os.path.join(results_dir, 'simulation_results_*.json')
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, get latest
    latest = max(files, key=os.path.getmtime)
    return latest

def main():
    results_dir = 'results'
    latest_file = find_latest_results(results_dir)
    
    if latest_file is None:
        print("No simulation results found. Please run the simulation first.")
        return
    
    print(f"Loading results from: {latest_file}")
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"Found {len(results)} results")
    print("Regenerating coherence_examples.png...")
    
    # Regenerate only the coherence_examples plot
    create_summary_plots(results, output_dir=results_dir, save=True, 
                        compute_bootstrap=False)
    
    print("Done! Check results/coherence_examples.png")

if __name__ == '__main__':
    main()

