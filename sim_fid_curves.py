#!/usr/bin/env python3
"""
FID Representative Curves
Generates fid_tau_c_*.csv files for representative tau_c values
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single
from spin_decoherence.physics.coherence import compute_ensemble_coherence

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (0.05 mT)

# Representative tau_c values
# Option 1: Minimum (4 points) - 체크리스트 최소 요구사항
# tau_c_representative = np.array([1e-8, 1e-7, 1e-6, 1e-5])  # s

# Option 2: Recommended (7 points) - 더 나은 분석을 위해
tau_c_representative = np.array([1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5])  # s

N_traj = 2000              # trajectories per point (increased for better statistics)

# Adaptive parameters
def get_dt(tau_c):
    """Adaptive timestep selection"""
    return tau_c / 100

def get_tmax(tau_c, B_rms, gamma_e):
    """Calculate appropriate simulation duration"""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est
    elif xi > 3:  # QS regime
        T2_est = 1.0 / (gamma_e * B_rms)
        # IMPROVEMENT 2: QS regime에서 더 긴 T_max로 T2 saturation 방지
        return 30 * T2_est  # 증가: 10 → 30
    else:  # Crossover
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est

def main():
    print("="*80)
    print("FID Representative Curves")
    print("="*80)
    print(f"\nParameters:")
    print(f"  gamma_e = {gamma_e:.3e} rad/(s·T)")
    print(f"  B_rms = {B_rms*1e3:.3f} mT")
    print(f"  tau_c values: {tau_c_representative}")
    print(f"  N_traj = {N_traj}")
    print(f"\nExpected time: ~10 minutes")
    print("\nStarting simulation...\n")
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    for i, tau_c in enumerate(tau_c_representative):
        print(f"\n[{i+1}/{len(tau_c_representative)}] Processing τ_c = {tau_c*1e6:.3f} μs")
        
        # Calculate adaptive parameters
        dt = get_dt(tau_c)
        T_max = get_tmax(tau_c, B_rms, gamma_e)
        xi = gamma_e * B_rms * tau_c
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.3e}")
        
        # Run simulation
        E, E_abs, E_se, t, E_abs_all = compute_ensemble_coherence(
            tau_c=tau_c,
            B_rms=B_rms,
            gamma_e=gamma_e,
            dt=dt,
            T_max=T_max,
            M=N_traj,
            seed=42 + i,
            progress=True,
            use_online=False  # Save full trajectories for std calculation
        )
        
        # Calculate P(t) = |E(t)| and standard deviation
        P_t = E_abs
        P_std = np.std(E_abs_all, axis=0) if E_abs_all.size > 0 else E_se
        
        # Create DataFrame
        df = pd.DataFrame({
            'time (s)': t,
            'P(t)': P_t,
            'P_std': P_std
        })
        
        # Save to CSV (format: fid_tau_c_1e-8.csv)
        tau_c_str = f"{tau_c:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        output_file = output_dir / f"fid_tau_c_{tau_c_str}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"  ✅ Saved to: {output_file}")
        print(f"  Points: {len(t)}, Time range: {t[0]*1e6:.2f} to {t[-1]*1e6:.2f} μs")
    
    print(f"\n{'='*80}")
    print(f"✅ All curves saved!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

