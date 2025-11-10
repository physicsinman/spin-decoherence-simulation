"""
Main simulation script for spin decoherence under stochastic magnetic fields.

This script performs parameter sweeps over correlation time tau_c and
generates coherence curves and T_2 relaxation times.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from coherence import compute_ensemble_coherence
from fitting import fit_coherence_decay, theoretical_T2_motional_narrowing, bootstrap_T2
from config import CONSTANTS, SimulationConfig
from units import Units


# Default simulation configuration (from Chapter 4)
# For motional-narrowing regime: ξ = γ_e * B_rms * τ_c < 1
# Set A: B_rms = 5 μT, τ_c = 0.01-10 μs → ξ < 8.8 for smallest τ_c
def get_default_config():
    """Get default simulation configuration."""
    return SimulationConfig(
        B_rms=Units.uT_to_T(5.0),  # 5 μT in Tesla (reduced for motional-narrowing)
        tau_c_range=(Units.us_to_s(0.01), Units.us_to_s(10.0)),  # 0.01 to 10 μs in seconds
        tau_c_num=20,  # Number of tau_c values to sweep
        dt=Units.ns_to_s(0.2),  # 0.2 ns (finer time step)
        T_max=Units.us_to_s(30.0),  # 30 μs (longer to capture full decay)
        M=1000,  # Number of realizations
        seed=42,
        output_dir='results',
        compute_bootstrap=True,  # Compute bootstrap CI for T_2
        save_delta_B_sample=False,  # Save first trajectory's delta_B for PSD verification
        T_max_echo=Units.us_to_s(20.0),  # Cap echo delays to keep simulations tractable
    )


# Backward compatibility: create dict-like interface for existing code
# This allows gradual migration
def config_to_dict(config: SimulationConfig) -> dict:
    """Convert SimulationConfig to dict for backward compatibility."""
    return {
        'B_rms': config.B_rms,
        'tau_c_range': config.tau_c_range,
        'tau_c_num': config.tau_c_num,
        'gamma_e': CONSTANTS.GAMMA_E,  # Use global constant
        'dt': config.dt,
        'T_max': config.T_max,
        'M': config.M,
        'seed': config.seed,
        'output_dir': config.output_dir,
        'compute_bootstrap': config.compute_bootstrap,
        'save_delta_B_sample': config.save_delta_B_sample,
        'T_max_echo': config.T_max_echo,
    }


def _solve_T2_exact(tau_c, delta_omega):
    """Solve for T2 from the analytical OU coherence by bisection."""
    if delta_omega <= 0:
        return np.inf

    def coherence_argument(t):
        return delta_omega**2 * tau_c**2 * (
            np.exp(-t / tau_c) + t / tau_c - 1.0
        ) - 1.0

    t_low = 0.0
    t_high = max(10.0 * tau_c, 10.0 / delta_omega)

    # Increase upper bound until the coherence argument becomes positive
    while coherence_argument(t_high) < 0:
        t_high *= 2.0
        if t_high > 1e3 / delta_omega:
            break

    # Bisection search for root
    for _ in range(80):
        t_mid = 0.5 * (t_low + t_high)
        value = coherence_argument(t_mid)
        if value > 0:
            t_high = t_mid
        else:
            t_low = t_mid

    return t_high


def estimate_characteristic_T2(tau_c, gamma_e, B_rms):
    """Estimate characteristic T2 for OU noise across regimes."""
    delta_omega = abs(gamma_e * B_rms)
    if delta_omega == 0:
        return np.inf

    xi = delta_omega * tau_c
    mn_T2 = 1.0 / (delta_omega**2 * tau_c)
    static_T2 = np.sqrt(2.0) / delta_omega

    if xi < 0.05:
        return mn_T2
    if xi > 20.0:
        return static_T2

    return _solve_T2_exact(tau_c, delta_omega)


def get_dimensionless_tau_range(tau_c, n_points=28, upsilon_min=0.05, upsilon_max=0.8, 
                                 gamma_e=None, B_rms=None, T_max=None):
    """
    Get dimensionless tau range for Hahn echo: υ = τ/τc ∈ [upsilon_min, upsilon_max].
    
    This replaces the absolute τ sweep with a normalized range that ensures
    consistent coverage across different correlation times.
    
    For fast noise (motional-narrowing regime), uses extended range to ensure
    sufficient time coverage for proper fitting. The range is automatically
    adjusted based on estimated T2_echo to ensure complete decay is captured.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    n_points : int
        Number of tau points (25-30, default: 28)
    upsilon_min : float
        Minimum normalized delay υ_min = τ_min/τc (default: 0.05)
    upsilon_max : float
        Maximum normalized delay υ_max = τ_max/τc (default: 0.8)
    gamma_e : float, optional
        Electron gyromagnetic ratio (for regime detection)
    B_rms : float, optional
        RMS noise amplitude (for regime detection)
    T_max : float, optional
        Maximum simulation time (for absolute bounds)
        
    Returns
    -------
    tau_list : ndarray
        Tau range in seconds: τ = υ * τc
    """
    # Adjust range for fast noise (motional-narrowing regime)
    # Fast noise: ξ = γ_e * B_rms * τ_c << 1
    # In fast noise, echo ≈ FID, but we need larger time range for fitting
    if gamma_e is not None and B_rms is not None:
        Delta_omega = gamma_e * B_rms
        xi = Delta_omega * tau_c
        
        # Estimate T2_echo for range determination
        # In fast noise (ξ << 1), echo ≈ FID, so T2_echo ≈ T2_FID ≈ 1/(Δω²τc)
        # In slow noise (ξ >> 1), echo > FID, but we still need sufficient range
        T2_est = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
        
        # For fast noise (ξ < 0.5), extend range significantly
        # Need to capture at least 3-5 T2_echo for proper fitting
        if xi < 0.5:
            # Calculate required upsilon_max based on T2_echo
            # Echo time is 2τ, so we need 2τ_max ≈ 3-5 * T2_echo
            # Therefore: τ_max ≈ 1.5-2.5 * T2_echo
            # In dimensionless units: υ_max = τ_max/τc ≈ (1.5-2.5)*T2_echo/τc
            if T2_est > 0 and tau_c > 0:
                # For fast noise, we need to capture significant portion of echo decay
                # Target: echo_max_time = 2*τ_max >= 5 * T2_echo for proper fitting
                # Therefore: τ_max >= 2.5 * T2_echo
                # In dimensionless: υ_max >= 2.5 * T2_echo / τc
                upsilon_max_from_T2 = 2.5 * T2_est / tau_c  # 2τ_max = 5 * T2_echo
                
                # For fast noise, allow larger upsilon_max if T_max permits
                # Calculate maximum allowed upsilon based on T_max
                if T_max is not None:
                    upsilon_max_from_Tmax = (T_max / 2.0) / tau_c if tau_c > 0 else 50.0
                    # Use the minimum of T2-based and T_max-based limits
                    # But ensure we get at least 3*T2_echo coverage
                    min_upsilon_for_coverage = 1.5 * T2_est / tau_c  # Minimum for 3*T2_echo
                    upsilon_max_cap = min(upsilon_max_from_T2, upsilon_max_from_Tmax)
                    upsilon_max_cap = max(upsilon_max_cap, min_upsilon_for_coverage)
                else:
                    # No T_max limit, use reasonable cap
                    upsilon_max_cap = min(upsilon_max_from_T2, 100.0)  # Cap at υ=100 for very fast noise
                
                # Use the larger of: extended default (2.0) or T2-based estimate
                upsilon_max_adjusted = max(upsilon_max * 2.0, upsilon_max_from_T2, 5.0)
                upsilon_max_adjusted = min(upsilon_max_adjusted, upsilon_max_cap)
            else:
                # Fallback to extended default
                upsilon_max_adjusted = min(upsilon_max * 3.0, 10.0)  # Extend up to υ=10.0
            
            # Use smaller upsilon_min to start earlier
            upsilon_min_adjusted = max(upsilon_min * 0.5, 0.01)  # Start from υ=0.01
        else:
            # For slow noise, use default range but ensure it's sufficient
            # In slow noise, echo can be longer than FID, so we may need extended range too
            if T2_est > 0 and tau_c > 0:
                # Estimate echo T2 (typically 1.5-3x FID T2 in slow noise)
                T2_echo_est = T2_est * 2.0  # Conservative estimate
                upsilon_max_from_T2 = min(3.0 * T2_echo_est / tau_c, 8.0)
                upsilon_max_adjusted = max(upsilon_max, upsilon_max_from_T2, 2.0)
                upsilon_max_adjusted = min(upsilon_max_adjusted, 8.0)  # Cap at υ=8.0
            else:
                upsilon_max_adjusted = max(upsilon_max, 2.0)  # At least υ=2.0
            
            upsilon_min_adjusted = upsilon_min
    else:
        upsilon_max_adjusted = upsilon_max
        upsilon_min_adjusted = upsilon_min
    
    # Apply absolute bounds if T_max is provided
    # For fast noise, T_max constraint is already considered in upsilon_max_cap calculation above
    # So we only apply it if it's more restrictive than what we already calculated
    if T_max is not None:
        # Echo time is 2τ, so we need 2τ_max <= T_max
        # Therefore: τ_max <= T_max / 2
        # In dimensionless: υ_max <= (T_max / 2) / τc
        upsilon_max_absolute = (T_max / 2.0) / tau_c if tau_c > 0 else upsilon_max_adjusted
        
        # Only apply T_max constraint if it's more restrictive than current estimate
        # (For fast noise, we already considered T_max in the cap calculation)
        upsilon_max_adjusted = min(upsilon_max_adjusted, upsilon_max_absolute)
    
    # Generate evenly spaced points in υ space
    upsilon_list = np.linspace(upsilon_min_adjusted, upsilon_max_adjusted, n_points)
    
    # Convert to absolute tau values
    tau_list = upsilon_list * tau_c
    
    return tau_list


def get_optimal_tau_range(
    tau_c,
    n_points=30,
    factor_min=0.1,
    factor_max=10,
    dt=None,
    T_max=None,
    gamma_e=None,
    B_rms=None,
):
    """
    Get optimal tau range for Hahn echo based on correlation time.
    
    DEPRECATED: Use get_dimensionless_tau_range instead for consistent
    dimensionless scanning.
    
    For effective refocusing, tau should be on the order of tau_c.
    Too small: refocusing effect negligible
    Too large: echo already decayed below noise floor
    
    For fast noise (tau_c < 0.1 μs), use narrower range to capture
    the regime where echo ≈ FID.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    n_points : int
        Number of tau points
    factor_min : float
        tau_min = factor_min * tau_c (adjusted for fast noise)
    factor_max : float
        tau_max = factor_max * tau_c (adjusted for fast noise)
        
    Returns
    -------
    tau_list : ndarray
        Optimal tau range (seconds)
    """
    if gamma_e is None:
        gamma_e = CONSTANTS.GAMMA_E
    if B_rms is None:
        default_config = get_default_config()
        B_rms = default_config.B_rms
    if dt is None:
        default_config = get_default_config()
        dt = default_config.dt
    if T_max is None:
        default_config = get_default_config()
        T_max = default_config.T_max_echo

    tau_min = factor_min * tau_c
    tau_max = factor_max * tau_c

    T2_est = estimate_characteristic_T2(tau_c, gamma_e, B_rms)

    # Determine practical time window for echo measurement (2τ)
    max_time = min(6.0 * T2_est, T_max)
    min_time = max(0.02 * T2_est, 20.0 * dt)

    tau_min = max(tau_min, 0.5 * min_time)
    tau_max = max(tau_max, 0.5 * max_time)

    # Enforce absolute bounds to avoid extremely small/large delays
    tau_min = max(tau_min, 0.01e-6)
    tau_max = min(tau_max, 0.5 * T_max)

    if tau_max <= tau_min:
        tau_max = tau_min * 1.5

    tau_list = np.logspace(np.log10(tau_min), np.log10(tau_max), n_points)

    return tau_list


def run_simulation_with_hahn_echo(tau_c, params=None, tau_list=None, verbose=True):
    """
    Run simulation with both FID and Hahn echo sequences.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    params : dict, optional
        Simulation parameters
    tau_list : array-like, optional
        List of echo delays τ (seconds). If None, auto-generate.
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    result : dict
        Dictionary containing FID and echo results
    """
    from coherence import compute_ensemble_coherence, compute_hahn_echo_coherence
    from fitting import fit_coherence_decay, analytical_ou_coherence
    
    if params is None:
        default_config = get_default_config()
        params = config_to_dict(default_config)
    
    T_max_echo = params.get('T_max_echo', params['T_max'])
    
    # For fast noise, adaptively increase T_max_echo to capture full echo decay
    Delta_omega = params['gamma_e'] * params['B_rms']
    xi = Delta_omega * tau_c
    if xi < 0.5:  # Fast noise regime
        T2_est = estimate_characteristic_T2(tau_c, params['gamma_e'], params['B_rms'])
        # Need at least 5 * T2_echo for proper fitting
        # Echo time = 2τ, so we need 2τ_max >= 5 * T2_est
        # Therefore: τ_max >= 2.5 * T2_est
        # Echo max time = 2 * τ_max >= 5 * T2_est
        required_echo_time = 5.0 * T2_est
        if required_echo_time > T_max_echo:
            # Extend T_max_echo for fast noise
            # For very fast noise (ξ < 0.1), use larger cap (5x default) to capture full decay
            # For moderate fast noise (0.1 ≤ ξ < 0.5), use standard cap (3x default)
            cap_multiplier = 5.0 if xi < 0.1 else 3.0
            T_max_echo_extended = min(required_echo_time * 1.2, params['T_max'] * cap_multiplier)
            T_max_echo = T_max_echo_extended
            
            # Check if coverage is sufficient after extension
            coverage_ratio = (T_max_echo / 2.0) / T2_est if T2_est > 0 else 0.0
            if verbose:
                print(f"  [INFO] Fast noise detected (ξ={xi:.3f}): Extended T_max_echo to {T_max_echo*1e6:.2f} μs "
                      f"(required: {required_echo_time*1e6:.2f} μs for T2_est={T2_est*1e6:.2f} μs)")
                if coverage_ratio < 3.0:
                    print(f"  ⚠️  WARNING: Echo coverage may be insufficient (coverage = {coverage_ratio:.2f}x T2_est, "
                          f"recommended ≥ 3.0x). Consider increasing T_max for better accuracy.")
                elif coverage_ratio < 5.0:
                    print(f"  ℹ️  Echo coverage is adequate (coverage = {coverage_ratio:.2f}x T2_est)")
                else:
                    print(f"  ✅ Echo coverage is excellent (coverage = {coverage_ratio:.2f}x T2_est)")

    # Auto-generate tau_list if not provided
    if tau_list is None:
        # Use dimensionless scan: υ = τ/τc ∈ [0.05, 0.8] with 25-30 evenly spaced points
        # Range is automatically adjusted for fast noise (extended range)
        # T_max_echo is used to set absolute bounds (echo time = 2τ <= T_max_echo)
        tau_list = get_dimensionless_tau_range(
            tau_c,
            n_points=28,  # 25-30 points, using 28 as default
            upsilon_min=0.05,
            upsilon_max=0.8,
            gamma_e=params['gamma_e'],
            B_rms=params['B_rms'],
            T_max=T_max_echo
        )
        if verbose:
            # Check if range was adjusted for fast noise
            Delta_omega = params['gamma_e'] * params['B_rms']
            xi = Delta_omega * tau_c
            upsilon_min_actual = tau_list.min() / tau_c
            upsilon_max_actual = tau_list.max() / tau_c
            
            if xi < 0.5:
                print(f"Dimensionless scan: υ = τ/τc ∈ [{upsilon_min_actual:.2f}, {upsilon_max_actual:.2f}] (extended for fast noise)")
            else:
                print(f"Dimensionless scan: υ = τ/τc ∈ [{upsilon_min_actual:.2f}, {upsilon_max_actual:.2f}]")
            print(f"Auto-generated tau range: {tau_list.min()*1e6:.2f} - {tau_list.max()*1e6:.2f} μs")
            print(f"  (tau_c = {tau_c*1e6:.2f} μs, ξ = {xi:.3f})")
    else:
        tau_list = np.array(tau_list)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running simulation with Hahn echo (τ_c = {tau_c*1e6:.2f} μs)")
        print(f"{'='*70}")
    
    # Compute FID coherence
    if verbose:
        print("Computing FID coherence...")
    
    E_fid, E_fid_abs, E_fid_se, t_fid, E_fid_abs_all = compute_ensemble_coherence(
        tau_c=tau_c, B_rms=params['B_rms'],
        gamma_e=params['gamma_e'], dt=params['dt'], T_max=params['T_max'],
        M=params['M'], seed=params['seed'], progress=verbose
    )
    
    # Compute Hahn echo coherence
    if verbose:
        print("Computing Hahn echo coherence...")
    
    # Use larger ensemble for echo if specified (echo has higher variance)
    M_echo = params.get('M_echo', params['M'])
    if M_echo > params['M'] and verbose:
        print(f"  Using larger ensemble for echo: M = {M_echo} (FID: M = {params['M']})")
    
    tau_echo, E_echo, E_echo_abs, E_echo_se, E_echo_abs_all = compute_hahn_echo_coherence(
        tau_c=tau_c, B_rms=params['B_rms'],
        gamma_e=params['gamma_e'], dt=params['dt'], tau_list=tau_list,
        M=M_echo, seed=params['seed'], progress=verbose
    )
    
    # Theoretical predictions
    E_fid_theory = analytical_ou_coherence(t_fid, params['gamma_e'], params['B_rms'], tau_c)
    
    # Use filter-function integral for Hahn echo (exact OU noise integral)
    # E_echo(2τ) = exp[-1/π ∫₀^∞ S_ω(ω)/ω² |F_echo(ω,τ)|² dω]
    from fitting import analytical_hahn_echo_filter_function, analytical_hahn_echo_coherence
    tau_list_for_theory = np.array(tau_list)
    if verbose:
        print("  Computing filter-function integral for echo theory...")
    E_echo_theory = analytical_hahn_echo_filter_function(
        tau_list_for_theory, params['gamma_e'], params['B_rms'], tau_c, 
        omega_max=None, n_points=5000
    )
    
    # CROSS-VALIDATION: Compare closed-form (analytical) vs numerical integration
    # This ensures numerical implementation matches theory
    E_echo_closed = analytical_hahn_echo_coherence(
        tau_list_for_theory, params['gamma_e'], params['B_rms'], tau_c
    )
    
    # Compute relative error between closed-form and numerical integration
    if verbose and len(E_echo_theory) > 0 and len(E_echo_closed) > 0:
        # Avoid division by zero
        mask = E_echo_closed > 1e-10
        if np.any(mask):
            rel_err = np.abs(E_echo_theory[mask] - E_echo_closed[mask]) / E_echo_closed[mask]
            max_rel_err = np.max(rel_err)
            if max_rel_err > 5e-3:  # Threshold: 0.5%
                print(f"  ⚠️  WARNING: Closed-form vs numerical integration mismatch")
                print(f"     Max relative error: {max_rel_err:.2e} (threshold: 5e-3)")
            else:
                print(f"  ✅ Cross-validation: Closed-form vs numerical integration")
                print(f"     Max relative error: {max_rel_err:.2e} (within tolerance)")
    
    # Fit FID decay with scale and offset
    # T₂ is automatically extracted as time where χ(t) = 1 (E = 1/e) in fit_coherence_decay_with_offset
    from fitting import fit_coherence_decay_with_offset
    fit_result_fid = fit_coherence_decay_with_offset(
        t_fid, E_fid_abs, E_se=E_fid_se, model='auto',
        tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms'],
        M=params['M']  # For weighted fitting
    )
    
    # Fit echo decay with scale and offset (use echo-optimized window selection)
    # T₂ is automatically extracted as time where χ(t) = 1 (E = 1/e) in fit_coherence_decay_with_offset
    M_echo = params.get('M_echo', params['M'])
    fit_result_echo = fit_coherence_decay_with_offset(
        tau_echo, E_echo_abs, E_se=E_echo_se, model='auto', is_echo=True,
        tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms'],
        M=M_echo  # For weighted fitting
    )
    
    tau_info = {
        'tau_min': float(tau_list.min()),
        'tau_max': float(tau_list.max()),
        'n_points': int(len(tau_list)),
    }

    result = {
        'tau_c': tau_c,
        't_fid': t_fid.tolist(),
        'E_fid': E_fid.tolist(),
        'E_fid_abs': E_fid_abs.tolist(),
        'E_fid_se': E_fid_se.tolist(),
        'E_fid_abs_all': E_fid_abs_all.tolist(),  # For bootstrap analysis
        'E_fid_theory': E_fid_theory.tolist(),
        'tau_echo': tau_echo.tolist(),
        'tau_list': tau_list.tolist(),
        'E_echo': E_echo.tolist(),
        'E_echo_abs': E_echo_abs.tolist(),
        'E_echo_se': E_echo_se.tolist(),
        'E_echo_abs_all': E_echo_abs_all.tolist(),  # For bootstrap analysis
        'E_echo_theory': E_echo_theory.tolist() if E_echo_theory is not None else None,
        'fit_result_fid': fit_result_fid,
        'fit_result_echo': fit_result_echo,
        'params': params,
        'tau_info': tau_info,
    }
    
    if verbose:
        if fit_result_fid:
            print(f"FID T_2: {fit_result_fid['T2']*1e6:.2f} μs ({fit_result_fid['model']})")
        else:
            print("FID fit failed")
        
        if fit_result_echo:
            print(f"Echo T_2: {fit_result_echo['T2']*1e6:.2f} μs ({fit_result_echo['model']})")
            if fit_result_fid:
                ratio = fit_result_echo['T2'] / fit_result_fid['T2']
                print(f"Ratio (Echo/FID): {ratio:.3f}")
                
                # Check regime: ξ = γ_e * B_rms * τ_c
                Delta_omega = params['gamma_e'] * params['B_rms']
                xi = Delta_omega * tau_c
                
                if xi > 0.5:  # Slow noise (quasi-static regime)
                    if ratio > 1.0:
                        print("  ✅ Echo > FID (expected for slow noise)")
                    else:
                        print("  ⚠️  Echo < FID (unphysical for slow noise)")
                else:  # Fast noise (motional-narrowing regime)
                    # In fast noise, echo ≈ FID, so ratio should be close to 1
                    if ratio < 0.5:
                        print("  ⚠️  Echo << FID (may indicate fitting issue in fast noise)")
                    elif ratio > 2.0:
                        print("  ⚠️  Echo >> FID (unexpected in fast noise)")
                    else:
                        print("  ℹ️  Echo ≈ FID (expected in fast noise)")
        else:
            print("Echo fit failed")
        print(f"Echo max time: {tau_echo.max()*1e6:.2f} μs")
    
    return result


def run_hahn_echo_sweep(params=None, tau_list=None, verbose=True):
    """
    Run Hahn echo simulation for a range of tau_c values.
    
    Parameters
    ----------
    params : dict, optional
        Simulation parameters
    tau_list : array-like, optional
        List of echo delays τ (seconds). If None, auto-generate.
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : list
        List of result dictionaries for each tau_c, containing both FID and echo data
    """
    if params is None:
        default_config = get_default_config()
        params = config_to_dict(default_config)
    
    # Auto-generate tau_list if not provided
    # Note: For sweep, we use a representative tau_c to set range
    # In practice, each tau_c will use get_optimal_tau_range
    if tau_list is None:
        tau_c_ref = np.sqrt(params['tau_c_range'][0] * params['tau_c_range'][1])
        tau_list = get_optimal_tau_range(
            tau_c_ref,
            n_points=30,
            dt=params['dt'],
            T_max=params.get('T_max_echo', params['T_max']),
            gamma_e=params['gamma_e'],
            B_rms=params['B_rms'],
        )
    else:
        tau_list = np.array(tau_list)
    
    # Generate tau_c values (logarithmic spacing)
    tau_c_min, tau_c_max = params['tau_c_range']
    tau_c_values = np.logspace(
        np.log10(tau_c_min),
        np.log10(tau_c_max),
        params['tau_c_num']
    )
    
    if verbose:
        print(f"Running Hahn echo sweep: {params['tau_c_num']} tau_c values")
        print(f"Range: {tau_c_min*1e6:.2f} to {tau_c_max*1e6:.2f} μs")
        print(
            "Echo delays: "
            f"{len(tau_list)} points from {tau_list.min()*1e6:.1f} to "
            f"{tau_list.max()*1e6:.1f} μs"
        )
    
    results = []
    for i, tau_c in enumerate(tau_c_values):
        if verbose:
            print(f"\n[{i+1}/{params['tau_c_num']}] tau_c = {tau_c*1e6:.2f} μs")
        
        # Use dimensionless scan: υ = τ/τc ∈ [0.05, 0.8]
        # Range is automatically adjusted for fast noise (extended range)
        T_max_echo = params.get('T_max_echo', params['T_max'])
        
        # For fast noise, adaptively increase T_max_echo to capture full echo decay
        Delta_omega = params['gamma_e'] * params['B_rms']
        xi = Delta_omega * tau_c
        if xi < 0.5:  # Fast noise regime
            T2_est = estimate_characteristic_T2(tau_c, params['gamma_e'], params['B_rms'])
            # Need at least 5 * T2_echo for proper fitting
            required_echo_time = 5.0 * T2_est
            if required_echo_time > T_max_echo:
                # Extend T_max_echo for fast noise
                # For very fast noise (ξ < 0.1), use larger cap (5x default) to capture full decay
                # For moderate fast noise (0.1 ≤ ξ < 0.5), use standard cap (3x default)
                cap_multiplier = 5.0 if xi < 0.1 else 3.0
                T_max_echo_extended = min(required_echo_time * 1.2, params['T_max'] * cap_multiplier)
                T_max_echo = T_max_echo_extended
                
                # Check if coverage is sufficient after extension
                coverage_ratio = (T_max_echo / 2.0) / T2_est if T2_est > 0 else 0.0
                if verbose:
                    print(f"  [INFO] Fast noise (ξ={xi:.3f}): Extended T_max_echo to {T_max_echo*1e6:.2f} μs "
                          f"(required: {required_echo_time*1e6:.2f} μs)")
                    if coverage_ratio < 3.0:
                        print(f"  ⚠️  WARNING: Echo coverage may be insufficient (coverage = {coverage_ratio:.2f}x T2_est, "
                              f"recommended ≥ 3.0x). Consider increasing T_max for better accuracy.")
                    elif coverage_ratio < 5.0:
                        print(f"  ℹ️  Echo coverage is adequate (coverage = {coverage_ratio:.2f}x T2_est)")
                    else:
                        print(f"  ✅ Echo coverage is excellent (coverage = {coverage_ratio:.2f}x T2_est)")
        
        tau_list_optimal = get_dimensionless_tau_range(
            tau_c,
            n_points=28,
            upsilon_min=0.05,
            upsilon_max=0.8,
            gamma_e=params['gamma_e'],
            B_rms=params['B_rms'],
            T_max=T_max_echo
        )
        if verbose:
            upsilon_min_actual = tau_list_optimal.min() / tau_c
            upsilon_max_actual = tau_list_optimal.max() / tau_c
            if xi < 0.5:
                print(f"  Dimensionless scan: υ = τ/τc ∈ [{upsilon_min_actual:.2f}, {upsilon_max_actual:.2f}] (extended for fast noise)")
            else:
                print(f"  Dimensionless scan: υ = τ/τc ∈ [{upsilon_min_actual:.2f}, {upsilon_max_actual:.2f}]")
            print(f"  Tau range: {tau_list_optimal.min()*1e6:.2f} - {tau_list_optimal.max()*1e6:.2f} μs")
        
        result = run_simulation_with_hahn_echo(tau_c, params, tau_list=tau_list_optimal, verbose=verbose)
        
        # Fit echo decay with scale and offset (already done in run_simulation_with_hahn_echo)
        # Bootstrap CI for echo T2 (only if compute_bootstrap is True)
        if params.get('compute_bootstrap', True):
            from fitting import bootstrap_T2
            tau_echo = np.array(result['tau_echo'])
            E_echo_abs = np.array(result['E_echo_abs'])
            E_echo_se = np.array(result['E_echo_se'])
            
            # Bootstrap CI for echo T2 (if E_echo_abs_all available)
            T2_echo_ci = None
            if 'E_echo_abs_all' in result:
                E_echo_abs_all = np.array(result['E_echo_abs_all'])  # Convert to numpy array
                T2_mean, T2_echo_ci, _ = bootstrap_T2(
                    tau_echo, E_echo_abs_all, E_se=E_echo_se, B=500, verbose=False,
                    tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms']
                )
            
            result['T2_echo_ci'] = T2_echo_ci
            
            # Also compute bootstrap CI for FID if available
            if 'E_fid_abs_all' in result:
                t_fid = np.array(result['t_fid'])
                E_fid_abs_all = np.array(result['E_fid_abs_all'])  # Convert to numpy array
                E_fid_se = np.array(result['E_fid_se'])
                T2_mean, T2_fid_ci, _ = bootstrap_T2(
                    t_fid, E_fid_abs_all, E_se=E_fid_se, B=500, verbose=False,
                    tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms']
                )
                if result.get('fit_result_fid') is not None:
                    result['fit_result_fid']['T2_ci'] = T2_fid_ci
        else:
            result['T2_echo_ci'] = None
        
        results.append(result)
    
    return results


def run_simulation_single(tau_c, params=None, verbose=True):
    """
    Run simulation for a single tau_c value.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    params : dict, optional
        Simulation parameters (uses defaults if None)
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    result : dict
        Dictionary containing E(t), fitting results, etc.
    """
    if params is None:
        default_config = get_default_config()
        params = config_to_dict(default_config)
    
    if verbose:
        print(f"\nRunning simulation for τ_c = {tau_c*1e6:.2f} μs")
    
    # DEBUG: Print key parameters for diagnostics
    Delta_omega = params['gamma_e'] * params['B_rms']
    xi = Delta_omega * tau_c
    T2_th = estimate_characteristic_T2(tau_c, params['gamma_e'], params['B_rms'])
    
    if verbose:
        print(f"[DEBUG] INPUT (user): Brms_input = {params['B_rms']*1e6:.2f} μT")
        print(f"[DEBUG] INTERNAL (SI): Brms_SI = {params['B_rms']:.3e} T, "
              f"tau_c_SI = {tau_c:.3e} s, dt_SI = {params['dt']:.3e} s")
        print(f"[DEBUG] Delta_omega = gamma_e * B_rms = {Delta_omega:.3e} rad/s")
        print(f"[DEBUG] xi = Delta_omega * tau_c = {xi:.3e} "
              f"(<<1 for MN, >>1 for static)")
        print(f"[DEBUG] T2_th (estimated) ~ {T2_th*1e6:.2f} microseconds")
    
    # Adaptive T_max: For static regime (ξ >> 1), use T_max based on T2_th
    # For MN regime, use default T_max (already long enough)
    T_max_adaptive = params['T_max']
    if xi > 2.0:  # Static/quasi-static regime
        # In static regime, coherence decays very quickly (Gaussian decay)
        # Use T_max = max(5*T2_th, 1.0 μs) to capture full decay including noise floor
        # Static regime: T2 ≈ sqrt(2)/Delta_omega (independent of tau_c)
        # Increased from 3*T2_th to 5*T2_th to show complete decay curve in plots
        T_max_from_T2 = max(5.0 * T2_th, 1.0e-6)  # At least 1.0 μs (increased from 0.5 μs)
        
        # CRITICAL FIX: Ensure T_max is at least 5*tau_c to guarantee OU noise burn-in
        # OU noise needs burn-in time of ~5*tau_c to reach stationary distribution
        # This ensures empirical std matches theoretical B_rms
        burnin_time = 5.0 * tau_c
        T_max_from_burnin = max(T_max_from_T2, burnin_time)
        
        # Allow up to default T_max to ensure sufficient coverage
        T_max_adaptive = min(T_max_from_burnin, params['T_max'])
        if verbose:
            print(f"[DEBUG] Static regime detected (ξ = {xi:.3f} >> 1)")
            if T_max_from_burnin > T_max_from_T2:
                print(f"[DEBUG] T_max extended for OU burn-in: {burnin_time*1e6:.2f} μs "
                      f"(5×τ_c = {tau_c*1e6:.2f} μs)")
            print(f"[DEBUG] Adaptive T_max: {T_max_adaptive*1e6:.2f} μs "
                  f"(based on T2_th = {T2_th*1e6:.2f} μs, default = {params['T_max']*1e6:.2f} μs)")
    elif xi < 0.1:  # Motional narrowing regime
        # In MN regime, T2 is long, so default T_max should be fine
        # But ensure it's at least 5*T2_th for proper fitting and visualization
        # Increased from 3*T2_th to 5*T2_th for better coverage
        T_max_from_T2 = max(5.0 * T2_th, params['T_max'])
        # Increased cap from 2x to 3x default to allow longer simulations for slow decay
        T_max_adaptive = min(T_max_from_T2, params['T_max'] * 3.0)  # Cap at 3x default (increased from 2x)
        if verbose and T_max_adaptive != params['T_max']:
            print(f"[DEBUG] MN regime: Extended T_max to {T_max_adaptive*1e6:.2f} μs "
                  f"(for T2_th = {T2_th*1e6:.2f} μs)")
    
    # Compute ensemble coherence
    E, E_abs, E_se, t, E_abs_all = compute_ensemble_coherence(
        tau_c=tau_c,
        B_rms=params['B_rms'],
        gamma_e=params['gamma_e'],
        dt=params['dt'],
        T_max=T_max_adaptive,
        M=params['M'],
        seed=params['seed'],
        progress=verbose
    )
    
    # Use E_abs (mean |E|) for fitting
    E_magnitude = E_abs
    
    # Fit coherence decay with scale and offset
    # T₂ is automatically extracted as time where χ(t) = 1 (E = 1/e) in fit_coherence_decay_with_offset
    from fitting import fit_coherence_decay_with_offset
    fit_result = fit_coherence_decay_with_offset(
        t, E_magnitude, E_se=E_se, model='auto',
        tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms'],
        M=params['M']  # For weighted fitting
    )
    
    # Bootstrap CI for T_2 (if fit successful)
    T2_ci = None
    if fit_result is not None and params.get('compute_bootstrap', True):
        if verbose:
            print("  Computing bootstrap CI...")
        T2_mean, T2_ci, _ = bootstrap_T2(
            t, E_abs_all, E_se=E_se, B=500, verbose=verbose,
            tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms']
        )
        if T2_ci is not None:
            fit_result['T2_ci'] = T2_ci
            if verbose:
                print(f"  T_2 95% CI: [{T2_ci[0]*1e6:.2f}, {T2_ci[1]*1e6:.2f}] μs")
    
    # Get first trajectory's delta_B for PSD verification (optional)
    delta_B_sample = None
    if params.get('save_delta_B_sample', False):
        from ornstein_uhlenbeck import generate_ou_noise
        N_steps = len(t)
        delta_B_sample = generate_ou_noise(
            tau_c, params['B_rms'], params['dt'], N_steps, seed=params['seed']
        )
    
    # Theoretical prediction: Use regime-aware estimate (not just MN limit)
    # This gives correct T2 for all regimes (MN, crossover, static)
    T2_theory = estimate_characteristic_T2(
        tau_c, params['gamma_e'], params['B_rms']
    )
    
    result = {
        'tau_c': tau_c,
        't': t.tolist(),
        'E': E.tolist(),
        'E_magnitude': E_magnitude.tolist(),
        'E_se': E_se.tolist(),  # Standard error of |E|
        'fit_result': fit_result,
        'T2_theory': T2_theory,
        'params': params
    }
    
    # Add delta_B_sample if available
    if delta_B_sample is not None:
        result['delta_B_sample'] = delta_B_sample.tolist()
    
    if fit_result:
        result['T2_fitted'] = fit_result['T2']
        result['fit_model'] = fit_result['model']
        if verbose:
            print(f"  Fitted T_2 = {fit_result['T2']*1e6:.2f} μs ({fit_result['model']} model)")
            print(f"  Theoretical T_2 = {T2_theory*1e6:.2f} μs")
    
    return result


def run_simulation_sweep(params=None, verbose=True):
    """
    Run simulation for a range of tau_c values.
    
    Parameters
    ----------
    params : dict, optional
        Simulation parameters
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : list
        List of result dictionaries for each tau_c
    """
    if params is None:
        default_config = get_default_config()
        params = config_to_dict(default_config)
    
    # Generate tau_c values (logarithmic spacing)
    tau_c_min, tau_c_max = params['tau_c_range']
    tau_c_values = np.logspace(
        np.log10(tau_c_min),
        np.log10(tau_c_max),
        params['tau_c_num']
    )
    
    if verbose:
        print(f"Running parameter sweep: {params['tau_c_num']} tau_c values")
        print(f"Range: {tau_c_min*1e6:.2f} to {tau_c_max*1e6:.2f} μs")
    
    results = []
    for i, tau_c in enumerate(tau_c_values):
        if verbose:
            print(f"\n[{i+1}/{params['tau_c_num']}]")
        result = run_simulation_single(tau_c, params, verbose=verbose)
        results.append(result)
    
    return results


def _make_json_serializable(obj):
    """
    Recursively convert numpy arrays and complex numbers to JSON-serializable format.
    """
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                'real': np.real(obj).tolist(),
                'imag': np.imag(obj).tolist()
            }
        else:
            return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, complex):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def save_results(results, output_dir=None, filename=None):
    """
    Save simulation results to JSON file.
    
    Parameters
    ----------
    results : list or dict
        Simulation results
    output_dir : str, optional
        Output directory
    filename : str, optional
        Output filename
    """
    if output_dir is None:
        default_config = get_default_config()
        output_dir = default_config.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert to JSON-serializable format
    json_results = _make_json_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Example: run a quick test with a single tau_c
    print("=" * 60)
    print("Spin Decoherence Simulation")
    print("=" * 60)
    
    # Test with single tau_c
    test_result = run_simulation_single(1e-6, verbose=True)
    
    # Save test result
    save_results(test_result, filename="test_single.json")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

