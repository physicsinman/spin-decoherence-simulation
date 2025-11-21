#!/usr/bin/env python3
"""
Post-Meeting Improvements - All-in-One Python Script
모든 개선 작업을 순차적으로 실행
"""

import subprocess
import sys
import time
from datetime import datetime, timedelta

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
        print(f"\n✅ Completed in {elapsed_str}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False

def main():
    print("="*80)
    print("Post-Meeting Improvements - All-in-One Execution")
    print("="*80)
    print("\nThis will run all improvement scripts sequentially:")
    print("  1. FID re-simulation (R² < 0.8 points)")
    print("  2. Echo re-simulation (problematic points)")
    print("  3. Echo gain re-analysis")
    print("  4. Convergence test improvement")
    print("  5. Final validation")
    print("  6. Graph regeneration")
    print("\nEstimated total time: ~5-6 hours")
    print()
    
    response = input("Press Enter to start, or 'q' to quit: ")
    if response.lower() == 'q':
        print("Cancelled.")
        return
    
    start_time = time.time()
    steps = [
        ("python3 rerun_poor_fid_points.py", "Step 1/6: Re-running FID simulations"),
        ("python3 rerun_echo_problem_points.py", "Step 2/6: Re-running Echo simulations"),
        ("python3 analyze_echo_gain.py", "Step 3/6: Re-analyzing Echo gain"),
        ("python3 improve_convergence_test.py", "Step 4/6: Improving convergence test"),
        ("python3 final_validation.py", "Step 5/6: Final validation"),
        ("python3 generate_dissertation_plots.py", "Step 6/6: Regenerating all figures"),
    ]
    
    for i, (cmd, desc) in enumerate(steps, 1):
        success = run_command(cmd, desc)
        if not success:
            print(f"\n❌ Step {i} failed. Stopping.")
            sys.exit(1)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "="*80)
    print("✅ All improvements completed successfully!")
    print("="*80)
    print(f"Total elapsed time: {hours}h {minutes}m {seconds}s")
    print("\nNext steps:")
    print("  1. Check results_comparison/figures/ for updated graphs")
    print("  2. Review final_validation.py output for quality metrics")
    print("  3. Verify all improvements meet expectations")
    print("="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Partial results may be available.")
        sys.exit(1)

