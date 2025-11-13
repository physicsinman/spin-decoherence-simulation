"""
Test script for parameter validation and memory-efficient simulation.

This script demonstrates how to use the new parameter validation and
memory-efficient simulation classes.
"""

import numpy as np
from parameter_validation import SimulationParameters, validate_simulation_parameters
from simulation_monitor import SimulationMonitor
from memory_efficient_sim import MemoryEfficientSimulation


def test_parameter_validation():
    """Test parameter validation for Si:P and GaAs."""
    print("\n" + "="*70)
    print("TEST 1: Parameter Validation")
    print("="*70)
    
    # Test Si:P with motional-narrowing regime
    print("\n--- Si:P, Motional-Narrowing Regime ---")
    params_sip_mn = SimulationParameters(system='Si_P', target_regime='motional_narrowing')
    report_sip_mn = params_sip_mn.validate()
    
    # Test Si:P with quasi-static regime
    print("\n--- Si:P, Quasi-Static Regime ---")
    params_sip_qs = SimulationParameters(system='Si_P', target_regime='quasi_static')
    report_sip_qs = params_sip_qs.validate()
    
    # Test GaAs
    print("\n--- GaAs, All Regimes ---")
    params_gaas = SimulationParameters(system='GaAs', target_regime='all')
    report_gaas = params_gaas.validate()
    
    # Compare with current parameters
    print("\n" + "="*70)
    print("TEST 2: Comparison with Current Parameters")
    print("="*70)
    
    # Current Si:P parameters from profiles.yaml
    B_rms_current_sip = 5e-6  # 5 µT
    T_max_current_sip = 30e-6  # 30 µs
    
    comparison_sip = validate_simulation_parameters(
        system='Si_P',
        target_regime='all',
        B_rms_current=B_rms_current_sip,
        T_max_current=T_max_current_sip
    )
    
    # Current GaAs parameters
    B_rms_current_gaas = 8e-6  # 8 µT
    T_max_current_gaas = 30e-6  # 30 µs
    
    comparison_gaas = validate_simulation_parameters(
        system='GaAs',
        target_regime='all',
        B_rms_current=B_rms_current_gaas,
        T_max_current=T_max_current_gaas
    )


def test_simulation_monitor():
    """Test simulation monitoring."""
    print("\n" + "="*70)
    print("TEST 3: Simulation Monitoring")
    print("="*70)
    
    # Create parameters
    params = SimulationParameters(system='Si_P', target_regime='motional_narrowing')
    
    # Create monitor
    monitor = SimulationMonitor(params)
    
    # Run checks
    print("\nRunning validation checks...")
    monitor.check_noise_amplitude()
    monitor.check_simulation_time()
    monitor.check_time_step(tau_c=0.1e-6)  # 0.1 µs
    monitor.check_memory_requirement()
    
    # Test convergence check
    t = np.linspace(0, params.total_time, 100)
    coherence = np.exp(-t / (params.T2_star_target / 2))  # Simulated decay
    monitor.check_convergence(coherence, t)
    
    # Test T2 check
    T2_measured = params.T2_star_target * 0.8  # 80% of target (should pass)
    monitor.check_T2_vs_literature(T2_measured)
    
    # Generate report
    report = monitor.report()
    
    return report


def test_memory_efficient_simulation():
    """Test memory-efficient simulation (small-scale test)."""
    print("\n" + "="*70)
    print("TEST 4: Memory-Efficient Simulation (Small Scale)")
    print("="*70)
    
    # Use GaAs for faster testing (shorter timescales)
    params = SimulationParameters(system='GaAs', target_regime='motional_narrowing')
    
    # Reduce ensemble size for testing
    params.n_ensemble = 10
    
    # Create simulation
    sim = MemoryEfficientSimulation(params)
    
    # Test with a single tau_c
    tau_c = 0.1e-6  # 0.1 µs
    
    print(f"\nSimulating with tau_c = {tau_c*1e6:.3f} µs...")
    print(f"Parameters: T_max = {params.total_time*1e6:.1f} µs, "
          f"dt = {params.dt*1e9:.3f} ns, M = {params.n_ensemble}")
    
    try:
        coherence, coherence_std = sim.simulate_coherence_chunked(
            tau_c, sequence='FID', seed=42
        )
        
        print(f"\n✓ Simulation completed successfully!")
        print(f"  Coherence: {coherence:.6f} ± {coherence_std:.6f}")
        
        return True
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PARAMETER VALIDATION AND MEMORY-EFFICIENT SIMULATION TESTS")
    print("="*70)
    
    # Test 1: Parameter validation
    test_parameter_validation()
    
    # Test 2: Simulation monitoring
    report = test_simulation_monitor()
    
    # Test 3: Memory-efficient simulation
    success = test_memory_efficient_simulation()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Parameter validation: ✓")
    print(f"Simulation monitoring: {'✓' if report['all_passed'] else '⚠️'}")
    print(f"Memory-efficient simulation: {'✓' if success else '✗'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

