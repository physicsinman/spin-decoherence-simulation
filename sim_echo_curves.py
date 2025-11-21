#!/usr/bin/env python3
"""
Hahn Echo Representative Curves
Generates echo_tau_c_*.csv files for representative tau_c values
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.echo import run_simulation_with_hahn_echo
from spin_decoherence.simulation.engine import get_dimensionless_tau_range

# ============================================
# SI:P SIMULATION PARAMETERS (FINAL: Si-28:P)
# ============================================
import yaml
gamma_e = 1.76e11          # rad/(s·T)
# Load from profiles.yaml
try:
    with open('profiles.yaml', 'r') as f:
        profiles = yaml.safe_load(f)
    if 'materials' in profiles and 'Si_P' in profiles['materials']:
        B_rms = profiles['materials']['Si_P']['OU']['B_rms']
        print(f"✅ Loaded B_rms from profiles.yaml: {B_rms*1e9:.1f} nT")
    else:
        B_rms = 4.0e-9  # Default: 4.0 nT (Si-28:P)
        print(f"⚠️  Using default B_rms: {B_rms*1e9:.1f} nT")
except Exception as e:
    B_rms = 4.0e-9  # Default: 4.0 nT (Si-28:P)
    print(f"⚠️  Error loading profiles.yaml: {e}, using default B_rms: {B_rms*1e9:.1f} nT")

# Representative tau_c values
# Use 4 representative points for Figure 4 (one from each regime)
tau_c_representative = np.array([1e-8, 1e-7, 1e-6, 1e-5])  # s

N_traj = 2000              # trajectories per point (increased for better statistics)

# Adaptive parameters
def get_dt(tau_c, max_memory_gb=4.0):
    """Adaptive timestep selection with memory constraints"""
    dt_base = tau_c / 100
    
    # For very small tau_c, dt might be too small, leading to huge N_steps
    # Cap dt at a reasonable minimum (e.g., 1 ps) to avoid memory issues
    dt_min = 1.0e-12  # 1 ps minimum
    
    # For very large T_max, we need to increase dt to fit memory
    # Estimate N_steps for base dt
    T_max_est = 10.0e-3  # 10 ms
    N_steps_est = int(T_max_est / dt_base)
    
    # Memory estimate: ~8 bytes per step per trajectory
    memory_gb = (N_steps_est * 8 * 2000) / (1024**3)
    
    if memory_gb > max_memory_gb:
        # Increase dt to reduce N_steps
        dt = T_max_est / (max_memory_gb * (1024**3) / (8 * 2000))
        dt = max(dt, dt_min)
        print(f"  ⚠️  Memory constraint: dt adjusted from {dt_base*1e12:.2f} ps to {dt*1e12:.2f} ps")
    else:
        dt = max(dt_base, dt_min)
    
    return dt

def get_tmax(tau_c, B_rms, gamma_e, max_T_max_sip=10.0e-3):
    """Calculate appropriate simulation duration"""
    xi = gamma_e * B_rms * tau_c
    
    # Use T_max from profiles.yaml as base, but adapt for different regimes
    # For Si:P, T_max is typically 10 ms
    base_T_max = max_T_max_sip
    
    if xi < 0.3:  # MN regime
        # For MN, use base T_max (10 ms) but ensure it's at least 10*tau_c
        return max(base_T_max, 10.0 * tau_c)
    elif xi > 3:  # QS regime
        # For QS, use longer T_max to capture decay
        T2_est = np.sqrt(2) / (gamma_e * B_rms)  # QS T2
        return min(30 * T2_est, base_T_max * 2.0)  # Cap at 2x base
    else:  # Crossover
        return base_T_max

def main():
    print("="*80)
    print("Hahn Echo Representative Curves")
    print("="*80)
    print(f"\nParameters:")
    print(f"  gamma_e = {gamma_e:.3e} rad/(s·T)")
    print(f"  B_rms = {B_rms*1e9:.1f} nT ({B_rms*1e3:.6f} mT)")
    print(f"  tau_c values: {tau_c_representative}")
    print(f"  N_traj = {N_traj}")
    print(f"\nExpected time: ~10 minutes")
    print("\nStarting simulation...\n")
    
    # Get max_T_max from profiles.yaml
    try:
        with open('profiles.yaml', 'r') as f:
            profiles = yaml.safe_load(f)
        if 'materials' in profiles and 'Si_P' in profiles['materials']:
            max_T_max_sip = profiles['materials']['Si_P'].get('T_max', 10.0e-3)
        else:
            max_T_max_sip = 10.0e-3
    except:
        max_T_max_sip = 10.0e-3
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    for i, tau_c in enumerate(tau_c_representative):
        print(f"\n[{i+1}/{len(tau_c_representative)}] Processing τ_c = {tau_c*1e6:.3f} μs")
        
        # Calculate adaptive parameters
        dt = get_dt(tau_c, max_memory_gb=4.0)
        T_max = get_tmax(tau_c, B_rms, gamma_e, max_T_max_sip)
        xi = gamma_e * B_rms * tau_c
        
        # Ensure dt < tau_c/5 for numerical stability
        dt_max_stable = tau_c / 5.0
        if dt > dt_max_stable:
            print(f"  ⚠️  dt ({dt*1e12:.2f} ps) > tau_c/5 ({dt_max_stable*1e12:.2f} ps), reducing T_max instead")
            # Reduce T_max to fit memory while maintaining dt stability
            memory_gb = 4.0
            N_steps_max = int(memory_gb * (1024**3) / (8 * 2000))
            T_max = min(T_max, N_steps_max * dt_max_stable)
            dt = dt_max_stable
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.3e}")
        
        # Setup parameters
        # IMPROVEMENT: T_max_echo를 T_max보다 더 크게 증가하여 Echo decay 충분히 관측
        # For longer echo curves, use longer T_max_echo
        if xi > 3:  # QS regime
            T_max_echo = T_max * 5.0  # Increased: 2.5 → 5.0 for longer range
        elif xi < 0.3:  # MN regime
            T_max_echo = T_max * 4.0  # Increased: 2.0 → 4.0
        else:  # Crossover
            T_max_echo = T_max * 4.0  # Increased: 2.0 → 4.0
        
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'T_max_echo': T_max_echo,  # IMPROVEMENT: 1.5배 증가
            'M': N_traj,
            'seed': 42 + i,
            'output_dir': str(output_dir),
            'compute_bootstrap': False,  # Not needed for curves
            'save_delta_B_sample': False,
        }
        
        # IMPROVEMENT: Calculate optimal upsilon_max based on T_max_echo
        # For longer echo curves, use larger upsilon_max
        # Target: echo should cover at least 2-3x the FID time range
        # For Si:P with small B_rms, we need very long echo ranges
        if xi > 3:  # QS regime
            # For QS, echo decay is slow, need very long range
            upsilon_max = min(2.0 * T_max_echo / tau_c, 20.0)  # Increased: 1.0 → 2.0, cap: 10.0 → 20.0
        elif xi < 0.3:  # MN regime
            # For MN, echo decay is fast, but still need longer range than before
            upsilon_max = min(1.5 * T_max_echo / tau_c, 15.0)  # Increased
        else:  # Crossover
            upsilon_max = min(1.5 * T_max_echo / tau_c, 15.0)  # Increased
        
        # Generate tau_list for echo with adaptive upsilon_max - MORE POINTS for smoother curves
        tau_list = get_dimensionless_tau_range(
            tau_c, n_points=100, upsilon_min=0.05, upsilon_max=upsilon_max,  # Increased: 50 → 100 points
            dt=dt, T_max=T_max_echo
        )
        
        # Run simulation
        result = run_simulation_with_hahn_echo(
            tau_c, params=params, tau_list=tau_list, verbose=True
        )
        
        # Extract echo data
        tau_echo = np.array(result['tau_echo'])
        E_echo_abs = np.array(result['E_echo_abs'])
        E_echo_se = np.array(result['E_echo_se'])
        E_echo_abs_all = np.array(result['E_echo_abs_all']) if result.get('E_echo_abs_all') else None
        
        # Calculate P_echo(t) = |E_echo(t)| and standard deviation
        P_echo_t = E_echo_abs
        if E_echo_abs_all is not None and E_echo_abs_all.size > 0:
            P_echo_std = np.std(E_echo_abs_all, axis=0)
        else:
            P_echo_std = E_echo_se
        
        # Create DataFrame
        df = pd.DataFrame({
            'time (s)': tau_echo,
            'P_echo(t)': P_echo_t,
            'P_echo_std': P_echo_std
        })
        
        # Save to CSV (format: echo_tau_c_1e-8.csv)
        tau_c_str = f"{tau_c:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        output_file = output_dir / f"echo_tau_c_{tau_c_str}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"  ✅ Saved to: {output_file}")
        print(f"  Points: {len(tau_echo)}, Time range: {tau_echo[0]*1e6:.2f} to {tau_echo[-1]*1e6:.2f} μs")
    
    print(f"\n{'='*80}")
    print(f"✅ All curves saved!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

