#!/usr/bin/env python3
"""
Master script to run all simulations in the correct order
Executes all simulation scripts according to the checklist
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}\n")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Error: {script_path} not found!")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        return False

def main():
    print("="*80)
    print("Master Simulation Runner")
    print("="*80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will run all simulations in the correct order.")
    print("Estimated total time: ~3-4 hours")
    print("\nPress Ctrl+C to stop at any time.\n")
    
    # Define execution order
    scripts = [
        # Step 1: FID Full Sweep
        ("sim_fid_sweep.py", "FID Full Sweep (20 points)"),
        
        # Step 2: FID Representative Curves
        ("sim_fid_curves.py", "FID Representative Curves (4 files)"),
        
        # Step 3: Motional Narrowing Fit
        ("analyze_mn.py", "Motional Narrowing Fit Analysis"),
        
        # Step 4: Hahn Echo Full Sweep
        ("sim_echo_sweep.py", "Hahn Echo Full Sweep (20 points)"),
        
        # Step 5: Hahn Echo Representative Curves
        ("sim_echo_curves.py", "Hahn Echo Representative Curves (4 files)"),
        
        # Step 6: Echo Gain Analysis
        ("analyze_echo_gain.py", "Echo Gain Analysis"),
        
        # Step 7: Noise Trajectories
        ("generate_noise_data.py", "Noise Trajectory Examples"),
        
        # Step 8: Optional - Bootstrap
        # ("run_bootstrap.py", "Bootstrap Distribution (Optional)"),
        
        # Step 9: Optional - Convergence Test
        # ("run_convergence_test.py", "Convergence Test (Optional)"),
    ]
    
    results = []
    
    for i, (script, description) in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] {description}")
        success = run_script(script, description)
        results.append((script, description, success))
        
        if not success:
            print(f"\n⚠️  Stopping due to error in {script}")
            print("You can continue manually by running the remaining scripts.")
            break
    
    # Summary
    print(f"\n{'='*80}")
    print("Execution Summary")
    print(f"{'='*80}\n")
    
    for script, description, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {description}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*80}")
    
    # Check output files
    output_dir = Path("results")
    if output_dir.exists():
        print("\nGenerated files:")
        csv_files = sorted(output_dir.glob("*.csv"))
        txt_files = sorted(output_dir.glob("*.txt"))
        
        for f in csv_files + txt_files:
            size_kb = f.stat().st_size / 1024
            print(f"  ✅ {f.name} ({size_kb:.1f} KB)")

if __name__ == '__main__':
    main()

