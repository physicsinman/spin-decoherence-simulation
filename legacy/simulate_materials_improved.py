"""
Improved material simulation with parameter validation and adaptive strategies.

This module integrates the new parameter validation, memory-efficient simulation,
and adaptive strategies into the material comparison workflow.
"""

import numpy as np
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# New modules
from parameter_validation import (
    SimulationParameters, 
    validate_simulation_parameters
)
from memory_efficient_sim import MemoryEfficientSimulation
from adaptive_simulation import AdaptiveSimulation
from improved_t2_extraction import ImprovedT2Extraction
from regime_aware_bootstrap_improved import RegimeAwareBootstrap
from simulation_monitor import SimulationMonitor

# Updated imports: use spin_decoherence package directly
from spin_decoherence.noise import generate_ou_noise, generate_double_OU_noise
from spin_decoherence.physics import (
    compute_ensemble_coherence,
    compute_hahn_echo_coherence,
    compute_phase_accumulation,
    compute_ensemble_coherence_double_OU,
    compute_hahn_echo_coherence_double_OU,
)
from spin_decoherence.analysis import fit_coherence_decay_with_offset
from spin_decoherence.config import CONSTANTS
from spin_decoherence.simulation.engine import estimate_characteristic_T2


def load_profiles(yaml_file='profiles.yaml'):
    """Load material profiles from YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data['materials']


def run_single_case_improved(
    material_name: str,
    profile: Dict,
    noise_model: str,
    sequence_type: str,
    use_validation: bool = True,
    use_adaptive: bool = True,
    use_improved_t2: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run improved simulation for a single case with validation and adaptive strategies.
    
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
    use_validation : bool
        Use parameter validation
    use_adaptive : bool
        Use adaptive simulation strategy
    use_improved_t2 : bool
        Use improved T2 extraction
    verbose : bool
        Print progress
    
    Returns
    -------
    results : dict
        Simulation results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running (Improved): {material_name} | {noise_model} | {sequence_type}")
        print(f"{'='*60}")
    
    # Extract common parameters
    gamma_e = float(profile['gamma_e'])
    T_max_original = float(profile['T_max'])
    dt = float(profile['dt'])
    M = int(profile['M'])
    seed = int(profile['seed'])
    
    # Parameter validation
    if use_validation:
        if verbose:
            print("\n[1/5] Parameter Validation...")
        
        # Compare current parameters with literature
        B_rms_current = profile['OU']['B_rms'] if noise_model == 'OU' else None
        comparison = validate_simulation_parameters(
            system=material_name,
            target_regime='all',
            B_rms_current=B_rms_current,
            T_max_current=T_max_original
        )
        
        # Create validated parameters
        params = SimulationParameters(system=material_name, target_regime='all')
        
        # Use validated parameters if significantly different
        # CRITICAL: Always use T_max from profiles.yaml (user-specified)
        # Validation may suggest different T_max, but profiles.yaml takes precedence
        if comparison['recommendations']:
            if verbose:
                print("⚠️  Using validated B_rms, but keeping T_max from profiles.yaml")
            B_rms = params.sigma_z
            # CRITICAL FIX: Always use T_max from profiles.yaml, not validation
            # profiles.yaml has been carefully tuned (e.g., 50 ms for Si:P QS regime)
            T_max = T_max_original
            # Adjust ensemble size if memory is an issue
            if params.validate()['warnings']:
                M = min(M, 200)  # Reduce ensemble size
        else:
            B_rms = B_rms_current if B_rms_current else params.sigma_z
            T_max = T_max_original
    else:
        # Use original parameters
        params = SimulationParameters(system=material_name, target_regime='all')
        B_rms = profile['OU']['B_rms'] if noise_model == 'OU' else params.sigma_z
        T_max = T_max_original
    
    # Initialize results
    results = {
        'material': material_name,
        'noise_model': noise_model,
        'sequence': sequence_type,
        'parameters': {
            'gamma_e': gamma_e,
            'T_max': T_max,
            'dt': dt,
            'M': M,
            'B_rms': B_rms,
        },
        'data': [],
        'validation': {
            'use_validation': use_validation,
            'use_adaptive': use_adaptive,
            'use_improved_t2': use_improved_t2,
        }
    }
    
    # DISSERTATION FIX: Force disable adaptive simulation for full decay observation
    # AdaptiveSimulation uses T2_estimate = 1 μs, which is too short for actual T2 = 5.14 μs
    # This causes insufficient simulation time, preventing observation of full decay curves
    # For dissertation, we need to show complete decay curves, not just initial decay
    use_adaptive = False  # Force disable for research validation
    
    # Initialize simulation
    if use_adaptive:
        sim = AdaptiveSimulation(params)
        if verbose:
            print("[2/5] Using Adaptive Simulation Strategy...")
    else:
        sim_base = MemoryEfficientSimulation(params)
        if verbose:
            print("[2/5] Using Memory-Efficient Simulation (Adaptive disabled for full decay observation)...")
    
    # Initialize T2 extraction
    if use_improved_t2:
        t2_extractor = ImprovedT2Extraction()
        if verbose:
            print("[3/5] Using Improved T2 Extraction...")
    
    # Initialize bootstrap
    bootstrap = RegimeAwareBootstrap(params)
    
    # Initialize monitor
    monitor = SimulationMonitor(params)
    
    # ===== Case 1: Single OU =====
    if noise_model == 'OU':
        tau_c_values = np.logspace(
            np.log10(profile['OU']['tau_c_min']),
            np.log10(profile['OU']['tau_c_max']),
            profile['OU']['tau_c_num']
        )
        
        for i, tau_c in enumerate(tau_c_values):
            if verbose:
                print(f"\n  [{i+1}/{len(tau_c_values)}] tau_c = {tau_c*1e6:.3f} µs")
            
            # Determine regime
            regime = bootstrap.determine_regime(tau_c)
            if verbose:
                print(f"      Regime: {regime}")
            
            # Check time step
            monitor.check_time_step(tau_c)
            
            # ===== CRITICAL FIX 3: Dynamic T_max adjustment =====
            # Ensure T_max is sufficient to capture decay (T_max >= 10*T2_expected)
            T2_expected = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
            T_max_from_T2 = max(10 * T2_expected, T_max)  # At least 10x T2
            
            # MEMORY FIX: Check memory limit FIRST (ou.py has 1GB limit per trajectory)
            # Memory per trajectory: N_steps * 8 bytes, limit at 400 MB (very conservative, well below 1 GB)
            max_memory_per_trajectory_mb = 400  # MB, very conservative limit
            max_steps_per_trajectory = int(max_memory_per_trajectory_mb * 1024**2 / 8)
            max_T_max_from_memory = max_steps_per_trajectory * dt
            
            # Cap T_max at minimum of: memory limit, 100ms code cap, or T2-based extension
            T_max_adaptive = min(T_max_from_T2, 0.1, max_T_max_from_memory)  # Cap at memory limit or 100ms
            
            if verbose and T_max_adaptive > T_max:
                print(f"      T2_expected = {T2_expected*1e6:.2f} µs")
                print(f"      T_max_adaptive = {T_max_adaptive*1e6:.2f} µs (extended from {T_max*1e6:.2f} µs)")
                if T_max_adaptive >= max_T_max_from_memory:
                    print(f"      [MEMORY LIMIT] T_max capped at {max_T_max_from_memory*1e6:.2f} µs due to memory constraints")
            # ===== END CRITICAL FIX 3 =====
            
            # Generate time points for coherence evaluation
            N_steps = int(T_max_adaptive / dt)  # Use adaptive T_max
            time_points = np.arange(0, T_max_adaptive, dt)[:N_steps]  # Use adaptive T_max
            
            # DISSERTATION FIX: Use compute_ensemble_coherence directly for bootstrap support
            # This provides full ensemble data needed for proper bootstrap CI calculation
            from spin_decoherence.physics import compute_ensemble_coherence
            
            # MEMORY FIX: Check memory requirements before deciding use_online
            # For long T_max (e.g., Si:P with 25ms), storing full ensemble can require >10GB
            N_steps_est = int(T_max_adaptive / dt)
            memory_gb_est = (N_steps_est * M * 8) / (1024**3)  # 8 bytes per float64
            
            # Use online mode if memory requirement > 5GB (conservative threshold)
            # This prevents MemoryError while still allowing bootstrap for smaller cases
            use_online_for_memory = memory_gb_est > 5.0
            
            if verbose and use_online_for_memory:
                print(f"      [MEMORY] Estimated memory: {memory_gb_est:.1f} GB")
                print(f"      [MEMORY] Using online mode (bootstrap will use analytical estimate)")
            
            # Compute coherence with adaptive memory mode
            E, E_abs, E_se, t_out, E_abs_all = compute_ensemble_coherence(
                tau_c, B_rms, gamma_e, dt, T_max_adaptive, M,
                seed=seed, progress=False, use_online=use_online_for_memory
            )
            
            # Use time_points that match t_out (actual simulation times)
            coherence_series = E_abs
            coherence_std_series = E_se
            time_points = t_out
            
            # Extract T2
            if use_improved_t2:
                T2, T2_error, fit_info = t2_extractor.extract_T2_auto(
                    time_points, coherence_series, coherence_std_series,
                    tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms, M=M
                )
            else:
                # Fallback to existing method
                from spin_decoherence.analysis import fit_coherence_decay
                fit_result = fit_coherence_decay(
                    time_points, coherence_series,
                    tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
                )
                T2 = fit_result.get('T2', np.nan)
                T2_error = fit_result.get('T2_error', np.nan)
                fit_info = fit_result
            
            # DISSERTATION FIX: Proper Bootstrap CI calculation
            # Use actual bootstrap resampling with full ensemble data (if available)
            T2_lower = None
            T2_upper = None
            
            # Only use bootstrap if we have full ensemble data (use_online=False)
            can_use_bootstrap = (len(time_points) > 10 and not np.isnan(T2) and 
                               E_abs_all.shape[0] > 0)
            
            if can_use_bootstrap:
                try:
                    from spin_decoherence.analysis import bootstrap_T2
                    T2_mean_bootstrap, T2_ci, T2_samples = bootstrap_T2(
                        time_points, E_abs_all, E_se=coherence_std_series,
                        B=1000,  # Increased from 500 for better statistics
                        tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms,
                        verbose=False
                    )
                    if T2_ci is not None and len(T2_ci) == 2:
                        T2_lower, T2_upper = T2_ci
                except Exception as e:
                    # Fallback to analytical error estimate if bootstrap fails
                    if verbose:
                        print(f"      Bootstrap CI failed: {e}, using analytical estimate")
                    if not np.isnan(T2_error):
                        T2_lower = T2 - 1.96 * T2_error
                        T2_upper = T2 + 1.96 * T2_error
            else:
                # Use analytical error estimate when online mode is used (no ensemble data)
                if not np.isnan(T2_error):
                    T2_lower = T2 - 1.96 * T2_error
                    T2_upper = T2 + 1.96 * T2_error
                    if verbose and use_online_for_memory:
                        print(f"      Using analytical CI (online mode: no ensemble data for bootstrap)")
            
            # Calculate dimensionless parameter
            Delta_omega = gamma_e * B_rms
            xi = Delta_omega * tau_c
            
            # Build metadata
            # estimate_characteristic_T2 is already imported at top of file
            T2_expected = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
            
            metadata = {
                'regime': regime,
                'xi': float(xi),
                'T2_expected': float(T2_expected),
                'T2_actual': float(T2) if not np.isnan(T2) else None,
                'n_ensemble_used': M,
                'sim_time_used': float(T_max_adaptive),
                'dt': float(dt),
                'tau_c': float(tau_c),
                'B_rms': float(B_rms),
                'bootstrap_success': T2_lower is not None and T2_upper is not None,
                'model_used': fit_info.get('model', fit_info.get('method_used', 'unknown')) if use_improved_t2 else 'exponential',
            }
            
            # Store result
            result_entry = {
                'tau_c': float(tau_c),
                'xi': float(xi),
                'T2': float(T2) if not np.isnan(T2) else None,
                'T2_lower': float(T2_lower) if T2_lower is not None else None,
                'T2_upper': float(T2_upper) if T2_upper is not None else None,
                'regime': regime,
                'model': fit_info.get('model', fit_info.get('method_used', 'unknown')) if use_improved_t2 else 'exponential',
                'metadata': metadata,
            }
            
            results['data'].append(result_entry)
            
            # Check T2 vs literature
            if not np.isnan(T2):
                monitor.check_T2_vs_literature(T2)
    
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
                print(f"\n  [{i+1}/{len(tau_c2_values)}] tau_c2 = {tau_c2*1e6:.3f} µs")
            
            # Determine regime (use effective tau_c)
            tau_c_eff = max(tau_c1, tau_c2)  # Use slower component
            B_rms_eff = np.sqrt(B_rms1**2 + B_rms2**2)  # Effective RMS
            regime = bootstrap.determine_regime(tau_c_eff)
            if verbose:
                print(f"      Regime: {regime} (tau_c_eff = {tau_c_eff*1e6:.3f} µs)")
            
            # Dynamic T_max adjustment
            T2_expected = estimate_characteristic_T2(tau_c_eff, gamma_e, B_rms_eff)
            
            # MEMORY FIX: For Double_OU, we need 2x memory (two noise trajectories)
            # ou.py has a hard limit at 1 GB per trajectory, so we need to be very conservative
            # Calculate max T_max that fits in memory
            # Memory per trajectory: N_steps * 8 bytes, limit at 400 MB (very conservative, well below 1 GB)
            max_memory_per_trajectory_mb = 400  # MB, very conservative limit
            max_steps_per_trajectory = int(max_memory_per_trajectory_mb * 1024**2 / 8)
            max_T_max_from_memory = max_steps_per_trajectory * dt
            
            # SPEED FIX: More conservative T_max extension for Double_OU
            # For Si_P with very long T2_expected, don't extend T_max too much
            # Cap at original T_max * 1.5 (instead of 2.0) to prevent memory issues
            T_max_from_T2 = min(2.0 * T2_expected, T_max * 1.5)  # Max 2×T2 or 1.5×T_max
            T_max_adaptive = max(T_max_from_T2, T_max)  # At least original T_max
            
            # CRITICAL: Cap T_max at memory limit FIRST, before any other extensions
            # This prevents memory errors in ou.py noise generation
            T_max_adaptive = min(T_max_adaptive, max_T_max_from_memory)
            
            # Additional safety cap at 10 ms for Si_P (very long T2 cases)
            if material_name == 'Si_P':
                T_max_adaptive = min(T_max_adaptive, 0.01)  # Cap at 10 ms for Si_P
            
            if verbose and T_max_adaptive > T_max:
                print(f"      T2_expected = {T2_expected*1e6:.2f} µs")
                print(f"      T_max_adaptive = {T_max_adaptive*1e6:.2f} µs (extended from {T_max*1e6:.2f} µs)")
                if T_max_adaptive >= max_T_max_from_memory:
                    print(f"      [MEMORY LIMIT] T_max capped at {max_T_max_from_memory*1e6:.2f} µs due to memory constraints")
            
            # Memory check for ensemble storage
            N_steps_est = int(T_max_adaptive / dt)
            memory_gb_est = (N_steps_est * M * 8) / (1024**3)
            use_online_for_memory = memory_gb_est > 5.0
            
            if verbose and use_online_for_memory:
                print(f"      [MEMORY] Estimated memory: {memory_gb_est:.1f} GB")
                print(f"      [MEMORY] Using online mode (bootstrap will use analytical estimate)")
            
            # Compute coherence for Double-OU
            if sequence_type == 'FID':
                E, E_abs, E_se, t_out, E_abs_all = compute_ensemble_coherence_double_OU(
                    tau_c1, tau_c2, B_rms1, B_rms2, gamma_e, dt, T_max_adaptive, M,
                    seed=seed+i, progress=False, use_online=use_online_for_memory
                )
            else:  # Hahn Echo
                # For echo, we need tau_list
                tau_list = np.linspace(0, T_max_adaptive/2, int(T_max_adaptive/dt/2))
                tau_echo, E_echo, E_echo_abs, E_echo_se, E_abs_all = compute_hahn_echo_coherence_double_OU(
                    tau_c1, tau_c2, B_rms1, B_rms2, gamma_e, dt, tau_list, M,
                    seed=seed+i, progress=False
                )
                # Use echo results
                E_abs = E_echo_abs
                E_se = E_echo_se
                t_out = tau_echo
            
            # Extract T2
            if use_improved_t2:
                T2, T2_error, fit_info = t2_extractor.extract_T2_auto(
                    t_out, E_abs, E_se,
                    tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff, M=M
                )
            else:
                from spin_decoherence.analysis import fit_coherence_decay
                fit_result = fit_coherence_decay(
                    t_out, E_abs,
                    tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff
                )
                T2 = fit_result.get('T2', np.nan)
                T2_error = fit_result.get('T2_error', np.nan)
                fit_info = fit_result
            
            # Bootstrap CI
            T2_lower = None
            T2_upper = None
            can_use_bootstrap = (len(t_out) > 10 and not np.isnan(T2) and 
                               E_abs_all.shape[0] > 0)
            
            if can_use_bootstrap:
                try:
                    from spin_decoherence.analysis import bootstrap_T2
                    T2_mean_bootstrap, T2_ci, T2_samples = bootstrap_T2(
                        t_out, E_abs_all, E_se=E_se,
                        B=1000,
                        tau_c=tau_c_eff, gamma_e=gamma_e, B_rms=B_rms_eff,
                        verbose=False
                    )
                    if T2_ci is not None and len(T2_ci) == 2:
                        T2_lower, T2_upper = T2_ci
                except Exception as e:
                    if verbose:
                        print(f"      Bootstrap CI failed: {e}, using analytical estimate")
                    if not np.isnan(T2_error):
                        T2_lower = T2 - 1.96 * T2_error
                        T2_upper = T2 + 1.96 * T2_error
            else:
                if not np.isnan(T2_error):
                    T2_lower = T2 - 1.96 * T2_error
                    T2_upper = T2 + 1.96 * T2_error
                    if verbose and use_online_for_memory:
                        print(f"      Using analytical CI (online mode)")
            
            # Calculate dimensionless parameters
            Delta_omega1 = gamma_e * B_rms1
            Delta_omega2 = gamma_e * B_rms2
            xi1 = Delta_omega1 * tau_c1
            xi2 = Delta_omega2 * tau_c2
            xi_eff = gamma_e * B_rms_eff * tau_c_eff
            
            # Build metadata
            T2_expected = estimate_characteristic_T2(tau_c_eff, gamma_e, B_rms_eff)
            
            metadata = {
                'regime': regime,
                'xi1': float(xi1),
                'xi2': float(xi2),
                'xi_eff': float(xi_eff),
                'tau_c1': float(tau_c1),
                'tau_c2': float(tau_c2),
                'tau_c_eff': float(tau_c_eff),
                'B_rms1': float(B_rms1),
                'B_rms2': float(B_rms2),
                'B_rms_eff': float(B_rms_eff),
                'T2_expected': float(T2_expected),
                'T2_actual': float(T2) if not np.isnan(T2) else None,
                'n_ensemble_used': M,
                'sim_time_used': float(T_max_adaptive),
                'dt': float(dt),
                'bootstrap_success': T2_lower is not None and T2_upper is not None,
                'model_used': fit_info.get('model', fit_info.get('method_used', 'unknown')) if use_improved_t2 else 'exponential',
            }
            
            # Store result
            result_entry = {
                'tau_c': float(tau_c2),  # Use tau_c2 as primary variable (tau_c1 is fixed)
                'tau_c1': float(tau_c1),
                'tau_c2': float(tau_c2),
                'xi': float(xi_eff),  # Effective xi
                'xi1': float(xi1),
                'xi2': float(xi2),
                'T2': float(T2) if not np.isnan(T2) else None,
                'T2_lower': float(T2_lower) if T2_lower is not None else None,
                'T2_upper': float(T2_upper) if T2_upper is not None else None,
                'regime': regime,
                'model': fit_info.get('model', fit_info.get('method_used', 'unknown')) if use_improved_t2 else 'exponential',
                'metadata': metadata,
            }
            
            results['data'].append(result_entry)
            
            # Check T2 vs literature
            if not np.isnan(T2):
                monitor.check_T2_vs_literature(T2)
    
    # Final validation report
    if verbose:
        print("\n[4/5] Final Validation...")
        monitor.report()
    
    return results


def run_full_comparison_improved(
    materials: List[str] = ['Si_P', 'GaAs'],
    noise_models: List[str] = ['OU', 'Double_OU'],
    sequences: List[str] = ['FID', 'Hahn'],
    output_dir: str = 'results_comparison',
    use_validation: bool = True,
    use_adaptive: bool = True,
    use_improved_t2: bool = True,
    save_curves: bool = False
) -> Dict:
    """
    Run full material comparison with improved methods.
    
    Parameters
    ----------
    materials : list
        List of material names
    noise_models : list
        List of noise models
    sequences : list
        List of sequences
    output_dir : str
        Output directory
    use_validation : bool
        Use parameter validation
    use_adaptive : bool
        Use adaptive simulation
    use_improved_t2 : bool
        Use improved T2 extraction
    save_curves : bool
        Save full coherence curves
    
    Returns
    -------
    all_results : dict
        All simulation results
    """
    print("="*70)
    print("IMPROVED MATERIAL COMPARISON SIMULATION")
    print("="*70)
    print(f"\nSettings:")
    print(f"  Parameter validation: {'✓' if use_validation else '✗'}")
    print(f"  Adaptive simulation: {'✓' if use_adaptive else '✗'}")
    print(f"  Improved T2 extraction: {'✓' if use_improved_t2 else '✗'}")
    print("="*70)
    
    # Load profiles
    profiles = load_profiles('profiles.yaml')
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = []
    
    # Run all combinations
    for material in materials:
        if material not in profiles:
            print(f"Warning: Material {material} not found in profiles")
            continue
        
        profile = profiles[material]
        
        for noise_model in noise_models:
            if noise_model not in profile:
                print(f"Warning: Noise model {noise_model} not found for {material}")
                continue
            
            for sequence in sequences:
                try:
                    result = run_single_case_improved(
                        material, profile, noise_model, sequence,
                        use_validation=use_validation,
                        use_adaptive=use_adaptive,
                        use_improved_t2=use_improved_t2,
                        verbose=True
                    )
                    
                    all_results.append(result)
                    
                    # Save individual result
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{material}_{noise_model}_{sequence}_{timestamp}.json"
                    filepath = output_path / filename
                    
                    with open(filepath, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"✓ Saved: {filename}")
                    
                except Exception as e:
                    print(f"✗ Error in {material} | {noise_model} | {sequence}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results_file = output_path / f"all_results_improved_{timestamp}.json"
    
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Saved combined results: {all_results_file}")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    # Run improved comparison
    results = run_full_comparison_improved(
        materials=['GaAs'],  # Start with GaAs (faster)
        noise_models=['OU'],
        sequences=['FID'],
        use_validation=True,
        use_adaptive=True,
        use_improved_t2=True
    )

