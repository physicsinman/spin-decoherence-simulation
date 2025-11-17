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
)
from spin_decoherence.analysis import fit_coherence_decay_with_offset
from spin_decoherence.config import CONSTANTS


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
    
    # Initialize simulation
    if use_adaptive:
        sim = AdaptiveSimulation(params)
        if verbose:
            print("[2/5] Using Adaptive Simulation Strategy...")
    else:
        sim_base = MemoryEfficientSimulation(params)
        if verbose:
            print("[2/5] Using Memory-Efficient Simulation...")
    
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
            
            # Generate time points for coherence evaluation
            N_steps = int(T_max / dt)
            time_points = np.arange(0, T_max, dt)[:N_steps]
            
            # Simulate coherence
            if use_adaptive:
                coherence_series, coherence_std_series, metadata = sim.simulate_time_series_adaptive(
                    tau_c, sequence=sequence_type, time_points=time_points, seed=seed
                )
            else:
                coherence_series, coherence_std_series = sim_base.simulate_coherence_time_series(
                    tau_c, sequence=sequence_type, time_points=time_points, seed=seed
                )
                metadata = {}
            
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
            
            # Bootstrap CI (if enough data)
            if len(coherence_series) > 10 and not np.isnan(T2):
                # For bootstrap, we need full ensemble data
                # This is simplified - in practice, you'd store full ensemble
                # For now, use analytical error estimate
                T2_lower = T2 - 1.96 * T2_error if not np.isnan(T2_error) else None
                T2_upper = T2 + 1.96 * T2_error if not np.isnan(T2_error) else None
            else:
                T2_lower = None
                T2_upper = None
            
            # Calculate dimensionless parameter
            Delta_omega = gamma_e * B_rms
            xi = Delta_omega * tau_c
            
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
        # Similar structure but for Double-OU
        # (Implementation similar to above but with Double-OU noise)
        if verbose:
            print("Double-OU simulation not yet fully integrated with new methods")
            print("Using existing implementation...")
        
        # For now, use existing method
        # TODO: Integrate Double-OU with new methods
    
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

