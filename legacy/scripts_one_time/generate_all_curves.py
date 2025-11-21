#!/usr/bin/env python3
"""
Generate FID and Echo curves for all tau_c values in echo_gain.csv
This enables direct measurement method for echo gain calculation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.physics.coherence import compute_ensemble_coherence
from spin_decoherence.simulation.echo import run_simulation_with_hahn_echo
from spin_decoherence.simulation.engine import get_dimensionless_tau_range
import yaml

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (0.05 mT)
N_traj = 1000              # Reduced for faster generation (can increase later)

# Load B_rms from profiles.yaml if available
try:
    with open('profiles.yaml', 'r') as f:
        profiles = yaml.safe_load(f)
    if 'materials' in profiles and 'Si_P' in profiles['materials']:
        B_rms = profiles['materials']['Si_P']['OU']['B_rms']
        print(f"✅ Loaded B_rms from profiles.yaml: {B_rms*1e9:.1f} nT")
except Exception as e:
    print(f"⚠️  Using default B_rms: {B_rms*1e3:.3f} mT")

# Adaptive parameters
def get_dt(tau_c, T_max=None):
    """Adaptive timestep selection with memory constraints"""
    # Numerical stability: dt < tau_c / 5
    dt_max_stable = tau_c / 5.0
    
    # Base dt: tau_c / 100 (smaller for better accuracy)
    dt_base = tau_c / 100.0
    
    # For very small tau_c, ensure dt is not too small
    dt_min = 1.0e-12  # 1 ps minimum
    
    # Start with base dt
    dt = max(dt_base, dt_min)
    
    # Ensure numerical stability
    dt = min(dt, dt_max_stable)
    
    # If T_max is provided, check memory constraints
    if T_max is not None:
        N_steps_est = int(T_max / dt)
        # Memory estimate: online algorithm uses much less memory
        # But still cap dt to avoid too many steps (10M steps max)
        if N_steps_est > 1e7:  # More than 10M steps
            # Increase dt to reduce steps, but respect stability constraint
            dt_memory = T_max / 1e7
            dt = min(dt_memory, dt_max_stable)
            dt = max(dt, dt_min)
    
    return dt

def get_tmax_fid(tau_c, B_rms, gamma_e):
    """Calculate appropriate FID simulation duration"""
    xi = gamma_e * B_rms * tau_c
    
    # Cap T_max at reasonable value to avoid memory issues
    T_max_max = 100.0e-3  # 100 ms maximum
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        T_max = 10 * T2_est
    elif xi > 3:  # QS regime
        T2_est = 1.0 / (gamma_e * B_rms)
        T_max = 30 * T2_est
    else:  # Crossover
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        T_max = 10 * T2_est
    
    return min(T_max, T_max_max)

def get_tmax_echo(tau_c, B_rms, gamma_e):
    """Calculate appropriate Echo simulation duration"""
    xi = gamma_e * B_rms * tau_c
    
    # Cap T_max at reasonable value to avoid memory issues
    T_max_max = 100.0e-3  # 100 ms maximum
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        T_max = 20 * T2_est
    elif xi > 3:  # QS regime
        T2_est = 1.0 / (gamma_e * B_rms)
        T_max = 50 * T2_est
    else:  # Crossover
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        T_max = 20 * T2_est
    
    return min(T_max, T_max_max)

def main(priority_only=False, max_points=None):
    """
    Parameters
    ----------
    priority_only : bool
        If True, only generate curves for points where direct measurement failed
    max_points : int, optional
        Maximum number of points to process (for testing)
    """
    print("="*80)
    print("Generate All Curves for Echo Gain Analysis")
    print("="*80)
    
    # Load tau_c values from echo_gain.csv
    gain_file = Path("results_comparison/echo_gain.csv")
    if not gain_file.exists():
        print(f"\n❌ Error: {gain_file} not found!")
        print("   Please run analyze_echo_gain.py first.")
        return
    
    df_gain = pd.read_csv(gain_file)
    
    if priority_only:
        # Only generate for points where direct measurement failed
        failed_mask = df_gain['method_used'] != 'direct_measurement'
        tau_c_list = df_gain[failed_mask]['tau_c'].unique()
        print(f"\nPriority mode: Generating curves for {len(tau_c_list)} points with failed direct measurement")
    else:
        tau_c_list = df_gain['tau_c'].unique()
        print(f"\nFound {len(tau_c_list)} unique tau_c values")
    
    tau_c_list = np.sort(tau_c_list)
    
    if max_points is not None:
        tau_c_list = tau_c_list[:max_points]
        print(f"Limited to first {max_points} points for testing")
    
    # Check which curves already exist
    output_dir = Path("results_comparison")
    output_dir.mkdir(exist_ok=True)
    
    missing_fid = []
    missing_echo = []
    
    for tau_c in tau_c_list:
        tau_c_str = f"{tau_c:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        fid_file = output_dir / f"fid_tau_c_{tau_c_str}.csv"
        echo_file = output_dir / f"echo_tau_c_{tau_c_str}.csv"
        
        if not fid_file.exists():
            missing_fid.append(tau_c)
        if not echo_file.exists():
            missing_echo.append(tau_c)
    
    print(f"\nMissing FID curves: {len(missing_fid)}")
    print(f"Missing Echo curves: {len(missing_echo)}")
    
    if len(missing_fid) == 0 and len(missing_echo) == 0:
        print("\n✅ All curves already exist!")
        return
    
    # Generate FID curves
    if len(missing_fid) > 0:
        print(f"\n{'='*80}")
        print(f"Generating {len(missing_fid)} FID curves...")
        print(f"{'='*80}")
        
        for i, tau_c in enumerate(missing_fid):
            print(f"\n[{i+1}/{len(missing_fid)}] FID: τ_c = {tau_c*1e6:.3f} μs")
            
            try:
                T_max = get_tmax_fid(tau_c, B_rms, gamma_e)
                dt = get_dt(tau_c, T_max)  # Pass T_max for memory check
                xi = gamma_e * B_rms * tau_c
                
                print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.3e}")
                
                # Run simulation with online algorithm to save memory
                E, E_abs, E_se, t = compute_ensemble_coherence(
                    tau_c=tau_c,
                    B_rms=B_rms,
                    gamma_e=gamma_e,
                    dt=dt,
                    T_max=T_max,
                    M=N_traj,
                    seed=42 + i,
                    progress=True,
                    use_online=True  # Use online algorithm to save memory
                )
                
                # Calculate P(t) = |E(t)| and standard deviation
                P_t = E_abs
                P_std = E_se  # Standard error from online algorithm
                
                # Create DataFrame
                df = pd.DataFrame({
                    'time (s)': t,
                    'P(t)': P_t,
                    'P_std': P_std
                })
                
                # Save to CSV
                tau_c_str = f"{tau_c:.0e}".replace("e-0", "e-").replace("e+0", "e+")
                output_file = output_dir / f"fid_tau_c_{tau_c_str}.csv"
                df.to_csv(output_file, index=False)
                
                print(f"  ✅ Saved: {output_file}")
                print(f"  Points: {len(t)}, Time range: {t[0]*1e6:.2f} to {t[-1]*1e6:.2f} μs")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
    
    # Generate Echo curves
    if len(missing_echo) > 0:
        print(f"\n{'='*80}")
        print(f"Generating {len(missing_echo)} Echo curves...")
        print(f"{'='*80}")
        
        # Get max_T_max from profiles.yaml if available
        max_T_max_sip = 10.0e-3
        try:
            with open('profiles.yaml', 'r') as f:
                profiles = yaml.safe_load(f)
            if 'materials' in profiles and 'Si_P' in profiles['materials']:
                max_T_max_sip = profiles['materials']['Si_P'].get('T_max', 10.0e-3)
        except:
            pass
        
        for i, tau_c in enumerate(missing_echo):
            print(f"\n[{i+1}/{len(missing_echo)}] Echo: τ_c = {tau_c*1e6:.3f} μs")
            
            try:
                T_max_echo = get_tmax_echo(tau_c, B_rms, gamma_e)
                T_max_echo = min(T_max_echo, max_T_max_sip * 2.0)  # Cap at 2x max_T_max
                dt = get_dt(tau_c, T_max_echo)  # Pass T_max for memory check
                xi = gamma_e * B_rms * tau_c
                
                print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max_echo*1e6:.2f} μs, ξ = {xi:.3e}")
                
                # Parameters
                params = {
                    'gamma_e': gamma_e,
                    'B_rms': B_rms,
                    'dt': dt,
                    'T_max': T_max_echo,
                    'M': N_traj,
                    'seed': 42 + i,
                }
                
                # Calculate optimal upsilon_max
                if xi > 3:  # QS regime
                    upsilon_max = min(2.0 * T_max_echo / tau_c, 20.0)
                elif xi < 0.3:  # MN regime
                    upsilon_max = min(1.5 * T_max_echo / tau_c, 15.0)
                else:  # Crossover
                    upsilon_max = min(1.5 * T_max_echo / tau_c, 15.0)
                
                # Generate tau_list for echo
                tau_list = get_dimensionless_tau_range(
                    tau_c, n_points=100, upsilon_min=0.05, upsilon_max=upsilon_max,
                    dt=dt, T_max=T_max_echo
                )
                
                # Run simulation
                result = run_simulation_with_hahn_echo(
                    tau_c, params=params, tau_list=tau_list, verbose=False
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
                
                # Save to CSV
                tau_c_str = f"{tau_c:.0e}".replace("e-0", "e-").replace("e+0", "e+")
                output_file = output_dir / f"echo_tau_c_{tau_c_str}.csv"
                df.to_csv(output_file, index=False)
                
                print(f"  ✅ Saved: {output_file}")
                print(f"  Points: {len(tau_echo)}, Time range: {tau_echo[0]*1e6:.2f} to {tau_echo[-1]*1e6:.2f} μs")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print(f"✅ Curve generation complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

