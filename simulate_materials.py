"""
Main simulation script for material comparison.

Workflow:
1. Load material profiles from YAML
2. Loop over materials, noise models, sequences
3. Run simulations and extract T₂
4. Save results to JSON
"""

import numpy as np
import yaml
import json
from datetime import datetime
from pathlib import Path

# Import existing modules
from ornstein_uhlenbeck import generate_ou_noise
from noise_models import generate_double_OU_noise
from coherence import (compute_ensemble_coherence, 
                       compute_hahn_echo_coherence,
                       compute_ensemble_coherence_double_OU,
                       compute_hahn_echo_coherence_double_OU,
                       compute_phase_accumulation)
from fitting import fit_coherence_decay_with_offset, bootstrap_T2
from simulate import get_dimensionless_tau_range
from config import CONSTANTS


def load_profiles(yaml_file='profiles.yaml'):
    """
    Load material profiles from YAML file.
    
    Returns
    -------
    materials : dict
        Dictionary with material parameters
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data['materials']


# Note: Double-OU functions are now in coherence.py
# They are imported at the top of this file

def run_single_case(material_name, profile, noise_model, 
                    sequence_type, verbose=True, save_curves=False):
    """
    Run simulation for a single combination.
    
    Parameters
    ----------
    material_name : str
        'Si_P' or 'GaAs'
    profile : dict
        Material parameters from YAML
    noise_model : str
        'OU' or 'Double_OU'
    sequence_type : str
        'FID' or 'Hahn'
    verbose : bool
        Print progress
    
    Returns
    -------
    results : dict
        Simulation results with T₂ values, curves, etc.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {material_name} | {noise_model} | {sequence_type}")
        print(f"{'='*60}")
    
    # Extract common parameters
    gamma_e = float(profile['gamma_e'])
    T_max = float(profile['T_max'])
    dt = float(profile['dt'])
    M = int(profile['M'])
    seed = int(profile['seed'])
    B_0 = 0.0  # Pure dephasing (no static field)
    
    # Initialize results storage
    results = {
        'material': material_name,
        'noise_model': noise_model,
        'sequence': sequence_type,
        'parameters': {
            'gamma_e': gamma_e,
            'T_max': T_max,
            'dt': dt,
            'M': M,
        },
        'data': []
    }
    
    # ===== Case 1: Single OU =====
    if noise_model == 'OU':
        B_rms = profile['OU']['B_rms']
        tau_c_values = np.logspace(
            np.log10(profile['OU']['tau_c_min']),
            np.log10(profile['OU']['tau_c_max']),
            profile['OU']['tau_c_num']
        )
        
        results['parameters']['B_rms'] = B_rms
        
        for i, tau_c in enumerate(tau_c_values):
            if verbose:
                print(f"  [{i+1}/{len(tau_c_values)}] tau_c = {tau_c*1e6:.3f} μs")
            
            # Compute coherence
            if sequence_type == 'FID':
                # Use use_online=False to get E_abs_all for bootstrap CI
                E, E_abs, E_se, t_out, E_abs_all = compute_ensemble_coherence(
                    tau_c, B_rms, gamma_e, dt, T_max, M, 
                    seed=seed+i, progress=verbose and i == 0,
                    use_online=False  # Need full trajectories for bootstrap
                )
                
                # Fit and extract T₂
                fit_result = fit_coherence_decay_with_offset(
                    t_out, E_abs, E_se=E_se, model='auto',
                    tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms, M=M
                )
                
                # Compute bootstrap CI if we have trajectories
                # Use regime-aware bootstrap (Phase 2 improvement)
                T2_lower = None
                T2_upper = None
                if fit_result and len(E_abs_all) > 0 and E_abs_all.shape[0] > 0:
                    try:
                        from regime_aware_bootstrap import regime_aware_bootstrap_T2
                        T2_mean, T2_ci, _, method = regime_aware_bootstrap_T2(
                            t_out, E_abs_all, E_se=E_se, B=500, verbose=False,
                            tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms,
                            use_analytical_ci=True
                        )
                        if T2_ci is not None:
                            T2_lower, T2_upper = T2_ci
                            fit_result['T2_ci'] = T2_ci
                            fit_result['ci_method'] = method  # Store method used
                    except ImportError:
                        # Fallback to standard bootstrap if regime-aware not available
                        from fitting import bootstrap_T2
                        xi = gamma_e * B_rms * tau_c
                        n_bootstrap = 150 if xi < 0.3 else 200
                        T2_mean, T2_ci, _ = bootstrap_T2(
                            t_out, E_abs_all, E_se=E_se, B=n_bootstrap, verbose=False,
                            tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
                        )
                        if T2_ci is not None:
                            T2_lower, T2_upper = T2_ci
                            fit_result['T2_ci'] = T2_ci
                
                if fit_result:
                    T2 = fit_result['T2']
                    # Use bootstrap CI if available, otherwise try to get from fit_result
                    if T2_lower is None:
                        T2_lower = fit_result.get('T2_ci', [None, None])[0]
                    if T2_upper is None:
                        T2_upper = fit_result.get('T2_ci', [None, None])[1]
                    beta = fit_result.get('beta', None)
                    model = fit_result.get('model', None)
                else:
                    T2 = None
                    T2_lower = None
                    T2_upper = None
                    beta = None
                    model = None
                
                # Compute xi parameter
                xi = gamma_e * B_rms * tau_c
                
                # Store results
                data_entry = {
                    'tau_c': float(tau_c),
                    'xi': float(xi),
                    'T2': float(T2) if T2 is not None else None,
                    'T2_lower': float(T2_lower) if T2_lower is not None else None,
                    'T2_upper': float(T2_upper) if T2_upper is not None else None,
                    'beta': float(beta) if beta is not None else None,
                    'model': model,
                }
                # Save full curves if requested (can be large files)
                if save_curves:
                    data_entry['t'] = t_out.tolist()
                    data_entry['E_abs'] = E_abs.tolist()
                results['data'].append(data_entry)
                
            elif sequence_type == 'Hahn':
                # Use dimensionless scan for Hahn echo
                tau_list = get_dimensionless_tau_range(
                    tau_c, n_points=20, gamma_e=gamma_e, 
                    B_rms=B_rms, T_max=T_max
                )
                
                # Compute echo coherence
                tau_echo, E_echo, E_echo_abs, E_echo_se, E_echo_abs_all = compute_hahn_echo_coherence(
                    tau_c, B_rms, gamma_e, dt, tau_list,
                    M, seed=seed+i, progress=verbose and i == 0
                )
                
                # Fit and extract T₂
                fit_result = fit_coherence_decay_with_offset(
                    tau_echo, E_echo_abs, E_se=E_echo_se, model='auto', is_echo=True,
                    tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms, M=M
                )
                
                # Compute bootstrap CI if we have trajectories
                # Use regime-aware bootstrap (Phase 2 improvement)
                T2_lower = None
                T2_upper = None
                if fit_result and len(E_echo_abs_all) > 0 and E_echo_abs_all.shape[0] > 0:
                    try:
                        from regime_aware_bootstrap import regime_aware_bootstrap_T2
                        T2_mean, T2_ci, _, method = regime_aware_bootstrap_T2(
                            tau_echo, E_echo_abs_all, E_se=E_echo_se, B=500, verbose=False,
                            tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms,
                            use_analytical_ci=True
                        )
                        if T2_ci is not None:
                            T2_lower, T2_upper = T2_ci
                            fit_result['T2_ci'] = T2_ci
                            fit_result['ci_method'] = method
                    except ImportError:
                        from fitting import bootstrap_T2
                        xi = gamma_e * B_rms * tau_c
                        n_bootstrap = 150 if xi < 0.3 else 200
                        T2_mean, T2_ci, _ = bootstrap_T2(
                            tau_echo, E_echo_abs_all, E_se=E_echo_se, B=n_bootstrap, verbose=False,
                            tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
                        )
                        if T2_ci is not None:
                            T2_lower, T2_upper = T2_ci
                            fit_result['T2_ci'] = T2_ci
                
                if fit_result:
                    T2 = fit_result['T2']
                    # Use bootstrap CI if available, otherwise try to get from fit_result
                    if T2_lower is None:
                        T2_lower = fit_result.get('T2_ci', [None, None])[0]
                    if T2_upper is None:
                        T2_upper = fit_result.get('T2_ci', [None, None])[1]
                    beta = fit_result.get('beta', None)
                    model = fit_result.get('model', None)
                else:
                    T2 = None
                    T2_lower = None
                    T2_upper = None
                    beta = None
                    model = None
                
                # Compute xi parameter
                xi = gamma_e * B_rms * tau_c
                
                # Store results
                data_entry = {
                    'tau_c': float(tau_c),
                    'xi': float(xi),
                    'T2': float(T2) if T2 is not None else None,
                    'T2_lower': float(T2_lower) if T2_lower is not None else None,
                    'T2_upper': float(T2_upper) if T2_upper is not None else None,
                    'beta': float(beta) if beta is not None else None,
                    'model': model,
                }
                # Save full echo curves if requested (can be large files)
                if save_curves:
                    data_entry['tau_echo'] = tau_echo.tolist()
                    data_entry['E_echo_abs'] = E_echo_abs.tolist()
                    data_entry['tau_list'] = tau_list.tolist()
                results['data'].append(data_entry)
    
    # ===== Case 2: Double OU =====
    elif noise_model == 'Double_OU':
        # Fast component (fixed)
        B_rms1 = profile['Double_OU']['B_rms1']
        tau_c1 = profile['Double_OU']['tau_c1']
        
        # Slow component (sweep)
        B_rms2 = profile['Double_OU']['B_rms2']
        tau_c2_values = np.logspace(
            np.log10(profile['Double_OU']['tau_c2_min']),
            np.log10(profile['Double_OU']['tau_c2_max']),
            profile['Double_OU']['tau_c2_num']
        )
        
        results['parameters'].update({
            'B_rms1': float(B_rms1),
            'tau_c1': float(tau_c1),
            'B_rms2': float(B_rms2),
        })
        
        for i, tau_c2 in enumerate(tau_c2_values):
            if verbose:
                print(f"  [{i+1}/{len(tau_c2_values)}] tau_c2 = {tau_c2*1e6:.3f} μs")
            
            # Compute coherence
            if sequence_type == 'FID':
                # Use use_online=False to get E_abs_all for bootstrap CI
                E, E_abs, E_se, t_out, E_abs_all = compute_ensemble_coherence_double_OU(
                    tau_c1, tau_c2, B_rms1, B_rms2, gamma_e, dt, T_max, M,
                    seed=seed+i, progress=verbose and i == 0,
                    use_online=False  # Need full trajectories for bootstrap
                )
                
                # Fit and extract T₂
                # For double-OU, use effective tau_c for fitting guidance
                tau_c_eff = max(tau_c1, tau_c2)  # Use slower component as reference
                B_rms_eff = np.sqrt(B_rms1**2 + B_rms2**2)  # Effective RMS
                
                fit_result = fit_coherence_decay_with_offset(
                    t_out, E_abs, E_se=E_se, model='auto',
                    tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff, M=M
                )
                
                # Compute bootstrap CI if we have trajectories
                # Use regime-aware bootstrap (Phase 2 improvement)
                T2_lower = None
                T2_upper = None
                if fit_result and len(E_abs_all) > 0 and E_abs_all.shape[0] > 0:
                    try:
                        from regime_aware_bootstrap import regime_aware_bootstrap_T2
                        T2_mean, T2_ci, _, method = regime_aware_bootstrap_T2(
                            t_out, E_abs_all, E_se=E_se, B=500, verbose=False,
                            tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff,
                            use_analytical_ci=True
                        )
                        if T2_ci is not None:
                            T2_lower, T2_upper = T2_ci
                            fit_result['T2_ci'] = T2_ci
                            fit_result['ci_method'] = method
                    except ImportError:
                        from fitting import bootstrap_T2
                        xi_eff = gamma_e * B_rms_eff * tau_c_eff
                        n_bootstrap = 150 if xi_eff < 0.3 else 200
                        T2_mean, T2_ci, _ = bootstrap_T2(
                            t_out, E_abs_all, E_se=E_se, B=n_bootstrap, verbose=False,
                            tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff
                        )
                        if T2_ci is not None:
                            T2_lower, T2_upper = T2_ci
                            fit_result['T2_ci'] = T2_ci
                
                if fit_result:
                    T2 = fit_result['T2']
                    # Use bootstrap CI if available, otherwise try to get from fit_result
                    if T2_lower is None:
                        T2_lower = fit_result.get('T2_ci', [None, None])[0]
                    if T2_upper is None:
                        T2_upper = fit_result.get('T2_ci', [None, None])[1]
                    beta = fit_result.get('beta', None)
                    model = fit_result.get('model', None)
                else:
                    T2 = None
                    T2_lower = None
                    T2_upper = None
                    beta = None
                    model = None
                
                # Compute xi parameters
                xi1 = gamma_e * B_rms1 * tau_c1
                xi2 = gamma_e * B_rms2 * tau_c2
                
                # Store results
                data_entry = {
                    'tau_c1': float(tau_c1),
                    'tau_c2': float(tau_c2),
                    'xi1': float(xi1),
                    'xi2': float(xi2),
                    'T2': float(T2) if T2 is not None else None,
                    'T2_lower': float(T2_lower) if T2_lower is not None else None,
                    'T2_upper': float(T2_upper) if T2_upper is not None else None,
                    'beta': float(beta) if beta is not None else None,
                    'model': model,
                }
                # Save full curves if requested (can be large files)
                if save_curves:
                    data_entry['t'] = t_out.tolist()
                    data_entry['E_abs'] = E_abs.tolist()
                results['data'].append(data_entry)
                
            elif sequence_type == 'Hahn':
                # Use dimensionless scan based on slower component
                tau_c_ref = tau_c2  # Use slower component as reference
                B_rms_eff = np.sqrt(B_rms1**2 + B_rms2**2)
                
                tau_list = get_dimensionless_tau_range(
                    tau_c_ref, n_points=20, gamma_e=gamma_e,
                    B_rms=B_rms_eff, T_max=T_max
                )
                
                # Compute echo coherence
                tau_echo, E_echo, E_echo_abs, E_echo_se, E_echo_abs_all = compute_hahn_echo_coherence_double_OU(
                    tau_c1, tau_c2, B_rms1, B_rms2, gamma_e, dt, tau_list,
                    M, seed=seed+i, progress=verbose and i == 0
                )
                
                # Fit and extract T₂
                tau_c_eff = max(tau_c1, tau_c2)
                B_rms_eff = np.sqrt(B_rms1**2 + B_rms2**2)
                
                fit_result = fit_coherence_decay_with_offset(
                    tau_echo, E_echo_abs, E_se=E_echo_se, model='auto', is_echo=True,
                    tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff, M=M
                )
                
                # Compute bootstrap CI if we have trajectories
                # Use regime-aware bootstrap (Phase 2 improvement)
                T2_lower = None
                T2_upper = None
                if fit_result and len(E_echo_abs_all) > 0 and E_echo_abs_all.shape[0] > 0:
                    try:
                        from regime_aware_bootstrap import regime_aware_bootstrap_T2
                        T2_mean, T2_ci, _, method = regime_aware_bootstrap_T2(
                            tau_echo, E_echo_abs_all, E_se=E_echo_se, B=500, verbose=False,
                            tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff,
                            use_analytical_ci=True
                        )
                        if T2_ci is not None:
                            T2_lower, T2_upper = T2_ci
                            fit_result['T2_ci'] = T2_ci
                            fit_result['ci_method'] = method
                    except ImportError:
                        from fitting import bootstrap_T2
                        xi_eff = gamma_e * B_rms_eff * tau_c_eff
                        n_bootstrap = 150 if xi_eff < 0.3 else 200
                        T2_mean, T2_ci, _ = bootstrap_T2(
                            tau_echo, E_echo_abs_all, E_se=E_echo_se, B=n_bootstrap, verbose=False,
                            tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff
                        )
                        if T2_ci is not None:
                            T2_lower, T2_upper = T2_ci
                            fit_result['T2_ci'] = T2_ci
                
                if fit_result:
                    T2 = fit_result['T2']
                    # Use bootstrap CI if available, otherwise try to get from fit_result
                    if T2_lower is None:
                        T2_lower = fit_result.get('T2_ci', [None, None])[0]
                    if T2_upper is None:
                        T2_upper = fit_result.get('T2_ci', [None, None])[1]
                    beta = fit_result.get('beta', None)
                    model = fit_result.get('model', None)
                else:
                    T2 = None
                    T2_lower = None
                    T2_upper = None
                    beta = None
                    model = None
                
                # Compute xi parameters
                xi1 = gamma_e * B_rms1 * tau_c1
                xi2 = gamma_e * B_rms2 * tau_c2
                
                # Store results
                data_entry = {
                    'tau_c1': float(tau_c1),
                    'tau_c2': float(tau_c2),
                    'xi1': float(xi1),
                    'xi2': float(xi2),
                    'T2': float(T2) if T2 is not None else None,
                    'T2_lower': float(T2_lower) if T2_lower is not None else None,
                    'T2_upper': float(T2_upper) if T2_upper is not None else None,
                    'beta': float(beta) if beta is not None else None,
                    'model': model,
                }
                # Save full echo curves if requested (can be large files)
                if save_curves:
                    data_entry['tau_echo'] = tau_echo.tolist()
                    data_entry['E_echo_abs'] = E_echo_abs.tolist()
                    data_entry['tau_list'] = tau_list.tolist()
                results['data'].append(data_entry)
    
    return results


def run_full_comparison(materials=['Si_P', 'GaAs'],
                        noise_models=['OU', 'Double_OU'],
                        sequences=['FID', 'Hahn'],
                        output_dir='results_comparison',
                        save_curves=True):
    """
    Run complete material comparison study.
    
    This is the main entry point that orchestrates all simulations.
    
    Parameters
    ----------
    materials : list
        Material names to simulate
    noise_models : list
        Noise models to use
    sequences : list
        Pulse sequences to apply
    output_dir : str
        Directory to save results
    save_curves : bool
        Save full coherence curves (large files)
    
    Returns
    -------
    all_results : list
        List of result dictionaries
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load profiles
    profiles = load_profiles()
    
    # Storage
    all_results = []
    
    # Total number of combinations
    n_total = len(materials) * len(noise_models) * len(sequences)
    counter = 0
    
    print(f"\n{'='*70}")
    print(f"Material Comparison Study")
    print(f"{'='*70}")
    print(f"Materials: {materials}")
    print(f"Noise models: {noise_models}")
    print(f"Sequences: {sequences}")
    print(f"Total combinations: {n_total}")
    print(f"{'='*70}\n")
    
    # Main loop
    for mat_name in materials:
        if mat_name not in profiles:
            print(f"⚠️  Warning: Material '{mat_name}' not found in profiles. Skipping.")
            continue
        
        profile = profiles[mat_name]
        
        for noise_model in noise_models:
            if noise_model not in profile:
                print(f"⚠️  Warning: Noise model '{noise_model}' not found for {mat_name}. Skipping.")
                continue
            
            for sequence in sequences:
                counter += 1
                print(f"\n[{counter}/{n_total}] Starting simulation...")
                
                # Run simulation
                result = run_single_case(
                    mat_name, profile, noise_model, 
                    sequence, verbose=True, save_curves=save_curves
                )
                
                # Save individual result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{mat_name}_{noise_model}_{sequence}_{timestamp}.json"
                filepath = output_path / filename
                
                with open(filepath, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                print(f"  ✓ Saved: {filename}")
                
                all_results.append(result)
    
    # Save combined results
    combined_file = output_path / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"✓ All simulations complete!")
    print(f"✓ Results saved to: {output_dir}/")
    print(f"{'='*70}\n")
    
    return all_results


# ===== Quick test function =====
def test_single_run():
    """
    Quick test: run single simulation to verify everything works.
    
    Use this before running the full comparison.
    """
    print("Running quick test...")
    
    profiles = load_profiles()
    
    # Test: Si_P, OU, FID
    result = run_single_case(
        material_name='Si_P',
        profile=profiles['Si_P'],
        noise_model='OU',
        sequence_type='FID',
        verbose=True
    )
    
    print("\n✓ Test successful!")
    print(f"✓ Computed {len(result['data'])} points")
    
    return result

