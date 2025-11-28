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
# SI:P SIMULATION PARAMETERS (FINAL)
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.57e-6            # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration

# Representative tau_c values
# Use 4 representative points for Figure 4 (one from each regime)
tau_c_representative = np.array([1e-8, 1e-7, 1e-6, 1e-5])  # s

N_traj = 3000              # FID trajectories (increased for more detailed data)
# CRITICAL: Reduce M_echo for MN regime to avoid memory issues
# For MN regime, echo ≈ FID so fewer trajectories are acceptable
# CRITICAL: For MN regime, reduce M_echo further to avoid memory issues
# MN regime: echo ≈ FID, so fewer trajectories are acceptable
M_echo = 1000              # Echo trajectories (reduced for memory efficiency)

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
    
    if xi < 0.1:  # MN regime (논문 기준: ξ < 0.1)
        # For MN, use base T_max (10 ms) but ensure it's at least 10*tau_c
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        T_max_from_T2 = 10 * T2_est
        # CRITICAL: Cap T_max to prevent memory issues when B_rms is very small
        return min(max(base_T_max, 10.0 * tau_c, T_max_from_T2), 10e-3)  # Cap at 10 ms for speed
    elif xi > 10:  # QS regime (논문 기준: ξ > 10)
        # For QS, use longer T_max to capture decay
        T2_est = np.sqrt(2) / (gamma_e * B_rms)  # QS T2
        T_max_from_T2 = 30 * T2_est
        return min(T_max_from_T2, base_T_max * 2.0, 10e-3)  # Cap at 10 ms for speed
    else:  # Crossover
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        T_max_from_T2 = 20 * T2_est
        return min(max(base_T_max, T_max_from_T2), 10e-3)  # Cap at 10 ms for speed

def main():
    print("="*80)
    print("Hahn Echo Representative Curves")
    print("="*80)
    print(f"\nParameters:")
    print(f"  gamma_e = {gamma_e:.3e} rad/(s·T)")
    print(f"  B_rms = {B_rms*1e3:.3f} mT")
    print(f"  tau_c values: {tau_c_representative}")
    print(f"  FID trajectories: {N_traj}")
    print(f"  Echo trajectories: {M_echo} (10x FID for maximum accuracy)")
    print(f"\n개선 사항:")
    print(f"  - Echo trajectories: {M_echo} (기존: {N_traj})")
    print(f"  - dt_echo: dt/2 (더 정밀한 phase alignment)")
    print(f"  - T_max_echo: Regime별 적응형 (최대 1000 ms)")
    print(f"  - n_points_echo: 150-200 (기존: 100)")
    print(f"  - upsilon_max: Regime별 적응형 (최대 50.0)")
    print(f"\nExpected time: ~60-90 minutes (높은 정확도, 더 상세한 데이터)")
    print("\nStarting simulation...\n")
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    for i, tau_c in enumerate(tau_c_representative):
        print(f"\n[{i+1}/{len(tau_c_representative)}] Processing τ_c = {tau_c*1e6:.3f} μs")
        
        # Calculate adaptive parameters (matching sim_echo_sweep.py)
        def get_tmax_improved(tau_c, B_rms, gamma_e):
            """Calculate appropriate simulation duration - matching sim_echo_sweep.py"""
            xi = gamma_e * B_rms * tau_c
            if xi < 0.1:  # MN regime (논문 기준: ξ < 0.1)
                T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
                return 10 * T2_est
            elif xi > 10:  # QS regime (논문 기준: ξ > 10)
                # CRITICAL FIX: QS regime T2 = sqrt(2) / (gamma_e * B_rms)
                # This comes from Gaussian decay: E(t) = exp(-(Δω·t)²/2)
                # At t = T2, E(T2) = 1/e, so (Δω·T2)²/2 = 1, giving T2 = sqrt(2)/Δω
                T2_est = np.sqrt(2.0) / (gamma_e * B_rms)
                if xi < 10:
                    multiplier = 100
                elif xi < 50:
                    multiplier = 150
                else:
                    multiplier = 200
                T_max_from_T2 = multiplier * T2_est
                burnin_time = 5.0 * tau_c
                T_max_final = max(T_max_from_T2, burnin_time)
                return min(T_max_final, 10e-3)  # Cap at 10 ms for speed
            else:  # Crossover
                T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
                return 20 * T2_est
        
        def get_dt_improved(tau_c, T_max=None, max_memory_gb=8.0):
            """Adaptive timestep selection with memory limit"""
            dt_target = tau_c / 150  # Finer timestep for more detailed data
            if T_max is not None:
                N_traj = 2000
                N_steps = int(T_max / dt_target) + 1
                memory_gb = (N_steps * N_traj * 8) / (1024**3)
                if memory_gb > max_memory_gb:
                    max_N_steps = int((max_memory_gb * 1024**3) / (N_traj * 8))
                    dt_required = T_max / max_N_steps if max_N_steps > 0 else dt_target
                    dt_min = tau_c / 50
                    dt = max(dt_required, dt_min)
                    return dt
            return dt_target
        
        T_max = get_tmax_improved(tau_c, B_rms, gamma_e)
        dt = get_dt_improved(tau_c, T_max=T_max, max_memory_gb=8.0)
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
        
        # CRITICAL IMPROVEMENT: T_max_echo를 더 길게 설정 (sim_echo_sweep.py와 동일한 로직)
        # Echo decay를 충분히 관찰하려면 FID보다 훨씬 긴 시간 필요
        if xi > 3:  # QS regime
            if xi > 50:  # Deep QS regime
                T_max_echo = T_max * 10.0
                T_max_echo = min(T_max_echo, 1000e-3)  # Cap at 1000 ms
            elif xi > 20:  # Intermediate QS regime
                T_max_echo = T_max * 8.0
                T_max_echo = min(T_max_echo, 800e-3)  # Cap at 800 ms
            else:  # Shallow QS regime
                T_max_echo = T_max * 5.0
                T_max_echo = min(T_max_echo, 500e-3)  # Cap at 500 ms
        elif xi < 0.1:  # MN regime (논문 기준: ξ < 0.1) - CRITICAL IMPROVEMENT
            # MN regime: Echo는 거의 decay하지 않으므로 매우 긴 시간 필요
            # But we need to balance with memory constraints
            # For representative curves, match FID time range (1000 μs)
            # But cap based on memory: N_steps = T_max_echo / dt_echo
            # With dt_echo = dt/2 and dt = tau_c/150, we need reasonable T_max_echo
            T2_fid_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)  # MN regime FID T2
            T_FID_est = T2_fid_est
            
            # Memory constraint: N_steps = T_max_echo / dt_echo
            # For 30000 trajectories, we need to limit memory usage
            # Target: ~8GB memory = 1B floats = 125M steps for 30000 trajectories
            # But we want T_max_echo = 1000 μs to match FID
            # So we need to adjust dt_echo or reduce trajectories
            
            # Calculate dt_echo - use larger dt for MN regime to save memory
            dt_base = tau_c / 150
            # For MN regime with long T_max, use larger dt_echo to save memory
            # Still maintain dt_echo < tau_c/5 for stability
            dt_echo_max = tau_c / 5.0
            dt_echo_target = dt_base / 2
            
            # Memory limit: ~8GB for M_echo trajectories
            # With M_echo = 1000, max steps = 8GB / (1000 * 8 bytes) = 1B steps
            max_steps = 1_000_000_000  # More steps since M_echo is smaller
            # CRITICAL FIX: For MN regime, we need to match FID time range to show Echo ≈ FID
            # FID goes up to ~1000 μs, so we need Echo to go at least to 150 μs for comparison
            # This ensures we can show Echo ≈ FID behavior in MN regime
            T_max_target = 150e-6  # 150 μs - match FID range for proper comparison
            
            # Calculate required dt_echo to fit T_max_target in memory
            dt_echo_required = T_max_target / max_steps
            # Use larger dt_echo to fit in memory - for MN regime, use 2*dt_base to save memory
            # This is still much smaller than tau_c/5 for stability
            dt_echo = max(dt_echo_required, dt_base * 2)  # Use 2*dt_base for memory efficiency
            dt_echo = min(dt_echo, dt_echo_max)  # Don't exceed stability limit
            
            # Now calculate T_max_echo based on memory
            T_max_from_memory = max_steps * dt_echo
            T_max_echo = min(T_max_target, T_max_from_memory, 2000e-3)
            
            print(f'  Memory optimization: dt_echo={dt_echo*1e12:.2f} ps, T_max_echo={T_max_echo*1e6:.2f} μs')
            print(f'  N_steps: {int(T_max_echo / dt_echo):,}, Memory: {int(T_max_echo / dt_echo) * M_echo * 8 / (1024**3):.1f} GB')
        else:  # Crossover
            T_max_echo = T_max * 8.0
            T_max_echo = min(T_max_echo, 300e-3)  # Cap at 300 ms
        
        # Echo-specific parameters for maximum accuracy
        # CRITICAL: For MN regime, dt_echo was already calculated above with memory constraints
        # For other regimes, use standard dt_echo
        if not (tau_c <= 1e-7 and xi < 0.1):
            # For non-MN regimes, use standard dt_echo
            dt_echo = dt / 2  # Finer time step for accurate π pulse phase alignment
        # For MN regime, dt_echo was already set in T_max_echo calculation above
        
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'T_max_echo': T_max_echo,
            'M': N_traj,  # FID uses N_traj
            'M_echo': M_echo,  # Echo uses more trajectories for accuracy
            'dt_echo': dt_echo,  # Echo uses finer time step
            'seed': 42 + i,
            'output_dir': str(output_dir),
            'compute_bootstrap': False,  # Not needed for curves
            'save_delta_B_sample': False,
        }
        
        # CRITICAL IMPROVEMENT: Calculate optimal upsilon_max based on T_max_echo
        # For QS regime, we need MUCH larger upsilon_max to capture echo decay
        if xi > 3:  # QS regime
            if xi > 50:  # Deep QS regime
                upsilon_max = min(0.4 * T_max_echo / tau_c, 50.0)
            elif xi > 20:  # Intermediate QS regime
                upsilon_max = min(0.4 * T_max_echo / tau_c, 30.0)
            else:  # Shallow QS regime
                upsilon_max = min(0.4 * T_max_echo / tau_c, 20.0)
        elif xi < 0.1:  # MN regime (논문 기준) - CRITICAL FIX: Increase upsilon_max to see echo decay
            # MN regime에서 Echo decay를 관찰하려면 훨씬 더 긴 시간 범위 필요
            # upsilon_max를 크게 증가시켜 Echo decay를 충분히 관찰
            # For representative curves, use T_max_echo-based calculation
            # Hahn echo is measured at 2*tau, so tau_max <= T_max_echo/2
            # Therefore: upsilon_max = tau_max/tau_c <= (T_max_echo/2)/tau_c
            upsilon_max_from_Tmax = (T_max_echo / 2.0) / tau_c if tau_c > 0 else 200.0
            # For MN regime, use T_max-based calculation without cap to match FID range
            # This ensures echo data goes as long as FID data
            upsilon_max = upsilon_max_from_Tmax  # No cap for MN regime - use full T_max_echo range
        else:  # Crossover
            upsilon_max = min(0.4 * T_max_echo / tau_c, 20.0)  # Increased from 10.0 to 20.0
        
        # CRITICAL: Increase n_points for smoother curves and better decay observation
        if xi > 50:  # Deep QS regime
            n_points_echo = 300  # More points for deep QS (increased for detail)
        elif xi > 20:  # Intermediate QS regime
            n_points_echo = 300  # Increased for better accuracy
        elif xi < 0.1:  # MN regime (논문 기준) - need more points to see small decay
            n_points_echo = 400  # Even more points to capture small echo decay in MN regime
        else:
            n_points_echo = 300  # Increased for better accuracy
        
        # Generate tau_list with adaptive upsilon_max
        tau_list = get_dimensionless_tau_range(
            tau_c, n_points=n_points_echo, upsilon_min=0.05, upsilon_max=upsilon_max,
            dt=dt_echo, T_max=T_max_echo  # Use dt_echo for tau_list generation
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

