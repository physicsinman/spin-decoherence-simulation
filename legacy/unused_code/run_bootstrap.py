#!/usr/bin/env python3
"""
Bootstrap Distribution Analysis (Optional)
Generates bootstrap_distribution.csv for one representative tau_c in MN regime
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single
from spin_decoherence.analysis.bootstrap import bootstrap_T2

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (0.05 mT)
tau_c_representative = 1e-7  # s (100 ns, MN regime)
N_traj = 1000              # trajectories
N_bootstrap = 1000         # bootstrap iterations

# Adaptive parameters
def get_dt(tau_c):
    return tau_c / 100

def get_tmax(tau_c, B_rms, gamma_e):
    xi = gamma_e * B_rms * tau_c
    if xi < 0.3:
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
    elif xi > 3:
        T2_est = 1.0 / (gamma_e * B_rms)
    else:
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
    return 10 * T2_est

def main():
    print("="*80)
    print("Bootstrap Distribution Analysis")
    print("="*80)
    print(f"\nParameters:")
    print(f"  tau_c = {tau_c_representative*1e6:.3f} μs (MN regime)")
    print(f"  N_traj = {N_traj}")
    print(f"  N_bootstrap = {N_bootstrap}")
    print(f"\nExpected time: ~30 minutes")
    print("\nStarting simulation...\n")
    
    # Create output directory
    output_dir = Path("results_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Calculate adaptive parameters
    dt = get_dt(tau_c_representative)
    T_max = get_tmax(tau_c_representative, B_rms, gamma_e)
    xi = gamma_e * B_rms * tau_c_representative
    
    print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.3e}")
    
    # Setup parameters
    params = {
        'B_rms': B_rms,
        'tau_c_range': (tau_c_representative, tau_c_representative),
        'tau_c_num': 1,
        'gamma_e': gamma_e,
        'dt': dt,
        'T_max': T_max,
        'M': N_traj,
        'seed': 42,
        'output_dir': str(output_dir),
        'compute_bootstrap': False,  # We'll do it manually
        'save_delta_B_sample': False,
    }
    
    # Run simulation
    result = run_simulation_single(tau_c_representative, params=params, verbose=True)
    
    # Extract data
    t = np.array(result['t'])
    E_abs_all = np.array(result['E_abs_all'])
    E_se = np.array(result['E_se'])
    
    if E_abs_all.size == 0:
        print("\n❌ Error: E_abs_all is empty! Cannot run bootstrap.")
        print("   Please ensure use_online=False in compute_ensemble_coherence")
        return
    
    print(f"\nRunning bootstrap with {N_bootstrap} iterations...")
    
    # Run bootstrap
    T2_mean, T2_ci, T2_bootstrap = bootstrap_T2(
        t, E_abs_all, E_se=E_se, B=N_bootstrap, verbose=True,
        tau_c=tau_c_representative, gamma_e=gamma_e, B_rms=B_rms
    )
    
    print(f"\nBootstrap Results:")
    print(f"  T2_mean = {T2_mean*1e6:.3f} μs")
    print(f"  95% CI: [{T2_ci[0]*1e6:.3f}, {T2_ci[1]*1e6:.3f}] μs")
    
    # Save bootstrap distribution
    df = pd.DataFrame({
        'iteration': np.arange(len(T2_bootstrap)),
        'T2_bootstrap (s)': T2_bootstrap
    })
    
    output_file = output_dir / "bootstrap_distribution.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Bootstrap distribution saved to: {output_file}")
    print(f"{'='*80}")
    
    return df

if __name__ == '__main__':
    df = main()

