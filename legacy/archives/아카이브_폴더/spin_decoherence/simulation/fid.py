"""
FID (Free Induction Decay) simulation routines.

This module provides functions for running FID simulations and parameter sweeps.
"""

import numpy as np
from typing import Dict, List, Optional
from spin_decoherence.physics.coherence import compute_ensemble_coherence
from spin_decoherence.simulation.engine import estimate_characteristic_T2
from spin_decoherence.config.constants import CONSTANTS
from spin_decoherence.config.simulation import SimulationConfig
from spin_decoherence.config.units import Units


def get_default_config():
    """Get default simulation configuration."""
    return SimulationConfig(
        B_rms=Units.uT_to_T(5.0),  # 5 μT in Tesla
        tau_c_range=(Units.us_to_s(0.01), Units.us_to_s(10.0)),  # 0.01 to 10 μs
        tau_c_num=20,
        dt=Units.ns_to_s(0.2),  # 0.2 ns
        T_max=Units.us_to_s(30.0),  # 30 μs
        M=1000,
        seed=42,
        output_dir='results',
        compute_bootstrap=True,
        save_delta_B_sample=False,
        T_max_echo=Units.us_to_s(20.0),
    )


def config_to_dict(config: SimulationConfig) -> dict:
    """Convert SimulationConfig to dict for backward compatibility."""
    return {
        'B_rms': config.B_rms,
        'tau_c_range': config.tau_c_range,
        'tau_c_num': config.tau_c_num,
        'gamma_e': CONSTANTS.GAMMA_E,
        'dt': config.dt,
        'T_max': config.T_max,
        'M': config.M,
        'seed': config.seed,
        'output_dir': config.output_dir,
        'compute_bootstrap': config.compute_bootstrap,
        'save_delta_B_sample': config.save_delta_B_sample,
        'T_max_echo': config.T_max_echo,
    }


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
    
    # Compute key parameters
    Delta_omega = params['gamma_e'] * params['B_rms']
    xi = Delta_omega * tau_c
    T2_th = estimate_characteristic_T2(tau_c, params['gamma_e'], params['B_rms'])
    
    if verbose:
        print(f"[DEBUG] Delta_omega = {Delta_omega:.3e} rad/s")
        print(f"[DEBUG] xi = {xi:.3e}")
        print(f"[DEBUG] T2_th (estimated) ~ {T2_th*1e6:.2f} μs")
    
    # Adaptive T_max based on regime
    T_max_adaptive = params['T_max']
    if xi > 2.0:  # Static regime
        T_max_from_T2 = max(5.0 * T2_th, 1.0e-6)
        burnin_time = 5.0 * tau_c
        T_max_from_burnin = max(T_max_from_T2, burnin_time)
        T_max_adaptive = min(T_max_from_burnin, params['T_max'])
    elif xi < 0.1:  # Motional narrowing regime
        T_max_from_T2 = max(5.0 * T2_th, params['T_max'])
        T_max_adaptive = min(T_max_from_T2, params['T_max'] * 3.0)
    
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
    
    # Fit coherence decay
    from spin_decoherence.analysis.fitting import fit_coherence_decay_with_offset
    fit_result = fit_coherence_decay_with_offset(
        t, E_abs, E_se=E_se, model='auto',
        tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms'],
        M=params['M']
    )
    
    # Bootstrap CI for T_2 (if requested)
    T2_ci = None
    T2_samples = None  # Initialize T2_samples
    if params.get('compute_bootstrap', True) and fit_result is not None:
        from spin_decoherence.analysis.bootstrap import bootstrap_T2
        T2_mean, T2_ci, T2_samples = bootstrap_T2(
            t, E_abs_all, E_se=E_se, B=500, verbose=verbose,
            tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms']
        )
        
        # Fallback to analytical error if bootstrap CI is degenerate
        if T2_ci is None and fit_result is not None:
            if verbose:
                print(f"  Bootstrap CI is degenerate, using analytical error estimate")
            # Use fit error if available
            T2 = fit_result.get('T2', np.nan)
            T2_error = fit_result.get('T2_error', np.nan)
            if not np.isnan(T2) and not np.isnan(T2_error):
                # Use 1.96 * SE for 95% CI (assuming normal distribution)
                T2_ci = (T2 - 1.96 * T2_error, T2 + 1.96 * T2_error)
                if verbose:
                    print(f"  Analytical CI: [{T2_ci[0]*1e6:.3f}, {T2_ci[1]*1e6:.3f}] μs")
            elif T2_samples is not None and len(T2_samples) > 0:
                # Use std of bootstrap samples even if CI is degenerate
                T2_std = np.std(T2_samples, ddof=1)
                T2 = fit_result.get('T2', T2_mean if T2_mean is not None else np.nan)
                if not np.isnan(T2) and T2_std > 0:
                    T2_ci = (T2 - 1.96 * T2_std, T2 + 1.96 * T2_std)
                    if verbose:
                        print(f"  Bootstrap std-based CI: [{T2_ci[0]*1e6:.3f}, {T2_ci[1]*1e6:.3f}] μs")
        
        if fit_result is not None:
            fit_result['T2_ci'] = T2_ci
    
    # Prepare result dictionary
    result = {
        'tau_c': tau_c,
        'params': params,
        't': t,
        'E': E,
        'E_abs': E_abs,
        'E_se': E_se,
        'E_abs_all': E_abs_all,
        'fit_result': fit_result,
        'T2_ci': T2_ci,
        'T2_samples': T2_samples,  # Include bootstrap samples
    }
    
    # Save delta_B sample if requested
    if params.get('save_delta_B_sample', False):
        from spin_decoherence.noise.ou import generate_ou_noise
        delta_B_sample = generate_ou_noise(
            tau_c, params['B_rms'], params['dt'], len(t),
            seed=params['seed']
        )
        result['delta_B_sample'] = delta_B_sample.tolist()
    
    return result


def run_simulation_sweep(params=None, verbose=True):
    """
    Run parameter sweep over tau_c values.
    
    Parameters
    ----------
    params : dict, optional
        Simulation parameters (uses defaults if None)
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : list
        List of result dictionaries, one per tau_c value
    """
    if params is None:
        default_config = get_default_config()
        params = config_to_dict(default_config)
    
    # Generate tau_c values
    tau_c_min, tau_c_max = params['tau_c_range']
    tau_c_num = params.get('tau_c_num', 20)
    tau_c_list = np.logspace(np.log10(tau_c_min), np.log10(tau_c_max), tau_c_num)
    
    results = []
    for i, tau_c in enumerate(tau_c_list):
        if verbose:
            print(f"\n[{i+1}/{len(tau_c_list)}] Processing τ_c = {tau_c*1e6:.2f} μs")
        
        result = run_simulation_single(tau_c, params=params, verbose=verbose)
        results.append(result)
    
    return results

