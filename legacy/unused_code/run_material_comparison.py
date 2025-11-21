"""
Main execution script.

Usage:
------
# Quick test (1 combination):
python main_comparison.py --test

# Full run (8 combinations):
python main_comparison.py --full

# Custom run:
python main_comparison.py --materials Si_P GaAs --noise OU --sequences FID Hahn
"""

import argparse
from simulate_materials import run_full_comparison, test_single_run
from analyze_results import (plot_T2_comparison, 
                             plot_echo_enhancement,
                             create_summary_table,
                             plot_noise_PSD_comparison,
                             load_results)
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(
        description='Material Comparison Study for Spin Decoherence'
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Run quick test (single combination)')
    parser.add_argument('--full', action='store_true',
                       help='Run full comparison (all combinations)')
    parser.add_argument('--materials', nargs='+', 
                       default=['Si_P', 'GaAs'],
                       help='Materials to simulate')
    parser.add_argument('--noise', nargs='+',
                       default=['OU', 'Double_OU'],
                       help='Noise models to use')
    parser.add_argument('--sequences', nargs='+',
                       default=['FID', 'Hahn'],
                       help='Pulse sequences')
    parser.add_argument('--output-dir', type=str,
                       default='results_comparison',
                       help='Output directory')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing results and create plots')
    parser.add_argument('--result-file', type=str,
                       help='Result file to analyze')
    parser.add_argument('--profiles', type=str,
                       default='profiles.yaml',
                       help='Path to profiles.yaml file')
    
    args = parser.parse_args()
    
    # ===== Test mode =====
    if args.test:
        print("="*70)
        print("QUICK TEST MODE")
        print("="*70)
        result = test_single_run()
        return
    
    # ===== Analysis mode =====
    if args.analyze:
        if not args.result_file:
            print("Error: --result-file required for analysis mode")
            return
        
        print("="*70)
        print("ANALYSIS MODE")
        print("="*70)
        
        # Load results
        if Path(args.result_file).is_file():
            # Single file (load_results already returns a list)
            all_results = load_results(args.result_file)
        else:
            # Directory - load all JSON files
            result_dir = Path(args.result_file)
            all_results = []
            for json_file in result_dir.glob('*.json'):
                if 'all_results' not in json_file.name:
                    # load_results returns a list, so extend
                    all_results.extend(load_results(json_file))
        
        # Create output directory for plots
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create plots (save_path without extension - will generate .png and .pdf)
        plot_T2_comparison(all_results, 
                          save_path=str(output_path / "T2_comparison"))
        plot_echo_enhancement(all_results,
                             save_path=str(output_path / "echo_enhancement"))
        create_summary_table(all_results,
                           save_path=str(output_path / "summary.csv"))
        
        # Load profiles for PSD comparison
        try:
            with open(args.profiles, 'r') as f:
                profiles_data = yaml.safe_load(f)
            plot_noise_PSD_comparison(profiles_data['materials'],
                                     save_path=str(output_path / "psd_comparison"))
        except Exception as e:
            print(f"⚠️  Could not create PSD comparison plot: {e}")
        
        print("✓ Analysis complete!")
        return
    
    # ===== Full simulation mode =====
    print("="*70)
    print("FULL SIMULATION MODE")
    print("="*70)
    
    all_results = run_full_comparison(
        materials=args.materials,
        noise_models=args.noise,
        sequences=args.sequences,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*70)
    print("CREATING ANALYSIS PLOTS...")
    print("="*70)
    
    # Auto-analyze
    output_path = Path(args.output_dir)
    plot_T2_comparison(all_results,
                      save_path=str(output_path / "T2_comparison"))
    plot_echo_enhancement(all_results,
                         save_path=str(output_path / "echo_enhancement"))
    create_summary_table(all_results,
                       save_path=str(output_path / "summary.csv"))
    
    # Load profiles for PSD comparison
    try:
        with open(args.profiles, 'r') as f:
            profiles_data = yaml.safe_load(f)
        plot_noise_PSD_comparison(profiles_data['materials'],
                                 save_path=str(output_path / "psd_comparison"))
    except Exception as e:
        print(f"⚠️  Could not create PSD comparison plot: {e}")
    
    print("\n" + "="*70)
    print("✓ ALL DONE!")
    print("="*70)


if __name__ == '__main__':
    main()

