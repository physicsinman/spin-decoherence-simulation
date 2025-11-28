"""
Hahn echo simulation routines.

This module provides functions for running Hahn echo simulations and parameter sweeps.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from spin_decoherence.physics.coherence import (
    compute_ensemble_coherence,
    compute_hahn_echo_coherence,
)
from spin_decoherence.simulation.engine import (
    estimate_characteristic_T2,
    get_dimensionless_tau_range,
)
from spin_decoherence.simulation.fid import get_default_config, config_to_dict


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
    from spin_decoherence.analysis.fitting import fit_coherence_decay_with_offset
    from spin_decoherence.simulation.fid import run_simulation_single
    
    if params is None:
        default_config = get_default_config()
        params = config_to_dict(default_config)
    
    # Run FID simulation
    fid_result = run_simulation_single(tau_c, params=params, verbose=verbose)
    
    # Generate tau_list if not provided
    if tau_list is None:
        T_max_echo = params.get('T_max_echo', params['T_max'])
        tau_list = get_dimensionless_tau_range(
            tau_c, n_points=28, upsilon_min=0.05, upsilon_max=0.8,
            dt=params['dt'], T_max=T_max_echo
        )
    
    # Run echo simulation
    if verbose:
        print(f"  Computing Hahn echo: {len(tau_list)} tau values")
    
    # IMPROVEMENT 1: Echo requires more trajectories for stability
    # Echo is more sensitive than FID, so use 8-10x more trajectories
    M_echo = params.get('M_echo', params.get('M', 1000) * 8)  # Default: 8x FID M
    
    # IMPROVEMENT 2: Echo requires finer time step for π pulse phase alignment
    # Smaller dt improves phase evaluation accuracy at τ and 2τ points
    dt_echo = params.get('dt_echo', params['dt'] / 2)  # Default: half of FID dt
    
    if verbose:
        print(f"  Echo parameters: M={M_echo}, dt={dt_echo*1e9:.2f} ns (FID: M={params['M']}, dt={params['dt']*1e9:.2f} ns)")
    
    tau_echo, E_echo, E_echo_abs, E_echo_se, E_echo_abs_all = compute_hahn_echo_coherence(
        tau_c=tau_c,
        B_rms=params['B_rms'],
        gamma_e=params['gamma_e'],
        dt=dt_echo,  # Use echo-specific dt
        tau_list=tau_list,
        M=M_echo,  # Use echo-specific M
        seed=params['seed'],
        progress=verbose
    )
    
    # IMPROVEMENT: Enhanced echo decay fitting
    # Use 'auto' model selection which automatically chooses best model based on data
    # For echo, we want to ensure proper decay observation window
    fit_result_echo = fit_coherence_decay_with_offset(
        tau_echo, E_echo_abs, E_se=E_echo_se, model='auto',
        is_echo=True,  # CRITICAL: Mark as echo for proper window selection
        tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms'],
        M=M_echo  # Use echo-specific M for weighted fitting
    )
    
    # Bootstrap CI for echo T2 - IMPROVED
    T2_echo_ci = None
    if params.get('compute_bootstrap', True) and E_echo_abs_all.size > 0:
        try:
            from spin_decoherence.analysis.bootstrap import bootstrap_T2
            T2_mean, T2_echo_ci, _ = bootstrap_T2(
                tau_echo, E_echo_abs_all, E_se=E_echo_se, B=500, verbose=False,
                tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms']
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Echo bootstrap CI failed: {e}")
    
    # Combine results
    result = fid_result.copy()
    result.update({
        'tau_list': tau_list.tolist() if isinstance(tau_list, np.ndarray) else tau_list,
        'tau_echo': tau_echo.tolist(),
        'E_echo': E_echo.tolist(),
        'E_echo_abs': E_echo_abs.tolist(),
        'E_echo_se': E_echo_se.tolist(),
        'E_echo_abs_all': E_echo_abs_all.tolist() if E_echo_abs_all.size > 0 else [],
        'fit_result_echo': fit_result_echo,
        'T2_echo_ci': T2_echo_ci,  # CRITICAL: Add CI to result
        't_fid': result['t'].tolist(),
        'E_fid_abs': result['E_abs'].tolist(),
        'E_fid_se': result['E_se'].tolist(),
        'E_fid_abs_all': result['E_abs_all'].tolist() if result['E_abs_all'].size > 0 else [],
        'fit_result_fid': result['fit_result'],
    })
    
    return result


def run_hahn_echo_sweep(params=None, tau_list=None, verbose=True):
    """
    Run parameter sweep with Hahn echo for multiple tau_c values.
    
    Parameters
    ----------
    params : dict, optional
        Simulation parameters
    tau_list : array-like, optional
        List of echo delays τ (seconds). If None, auto-generate per tau_c.
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
        
        # Generate optimal tau_list for this tau_c if not provided
        tau_list_optimal = tau_list
        if tau_list_optimal is None:
            T_max_echo = params.get('T_max_echo', params['T_max'])
            tau_list_optimal = get_dimensionless_tau_range(
                tau_c, n_points=28, upsilon_min=0.05, upsilon_max=0.8,
                dt=params['dt'], T_max=T_max_echo
            )
        
        result = run_simulation_with_hahn_echo(
            tau_c, params=params, tau_list=tau_list_optimal, verbose=verbose
        )
        
        # Bootstrap CI for echo T2
        if params.get('compute_bootstrap', True):
            from spin_decoherence.analysis.bootstrap import bootstrap_T2
            tau_echo = np.array(result['tau_echo'])
            E_echo_abs_all = np.array(result['E_echo_abs_all'])
            E_echo_se = np.array(result['E_echo_se'])
            
            if E_echo_abs_all.size > 0:
                T2_mean, T2_echo_ci, _ = bootstrap_T2(
                    tau_echo, E_echo_abs_all, E_se=E_echo_se, B=500, verbose=False,
                    tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms']
                )
                result['T2_echo_ci'] = T2_echo_ci
            
            # Bootstrap CI for FID
            if result.get('E_fid_abs_all') and len(result['E_fid_abs_all']) > 0:
                t_fid = np.array(result['t_fid'])
                E_fid_abs_all = np.array(result['E_fid_abs_all'])
                E_fid_se = np.array(result['E_fid_se'])
                T2_mean, T2_fid_ci, _ = bootstrap_T2(
                    t_fid, E_fid_abs_all, E_se=E_fid_se, B=500, verbose=False,
                    tau_c=tau_c, gamma_e=params['gamma_e'], B_rms=params['B_rms']
                )
                if result.get('fit_result_fid') is not None:
                    result['fit_result_fid']['T2_ci'] = T2_fid_ci
        
        results.append(result)
    
    return results

