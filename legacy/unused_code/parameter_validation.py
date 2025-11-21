"""
Parameter validation and reset for spin decoherence simulation.

This module provides classes for validating simulation parameters against
literature values and ensuring physical consistency.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import warnings


@dataclass
class SimulationParameters:
    """
    Physically consistent parameter set for spin decoherence simulation.
    
    This class ensures that simulation parameters are validated against
    literature values and are physically consistent.
    """
    system: str = 'Si_P'
    target_regime: str = 'all'  # 'motional_narrowing', 'quasi_static', 'all'
    
    # Physical constants (will be set based on system)
    gamma_e: float = 1.76e11  # rad/(s·T), electron gyromagnetic ratio
    T2_star_target: Optional[float] = None  # Target T2* from literature (seconds)
    sigma_z: Optional[float] = None  # Back-calculated noise amplitude (Tesla)
    
    # Simulation time settings
    total_time: Optional[float] = None  # T_max (seconds)
    min_tau_c: Optional[float] = None  # Minimum tau_c (seconds)
    max_tau_c: Optional[float] = None  # Maximum tau_c (seconds)
    
    # Time step
    dt: Optional[float] = None  # Time step (seconds)
    
    # Ensemble size
    n_ensemble: int = 100  # Number of realizations
    
    def __post_init__(self):
        """Initialize parameters based on system and target regime."""
        if self.system == 'Si_P':
            self._init_sip()
        elif self.system == 'GaAs':
            self._init_gaas()
        else:
            raise ValueError(f"Unknown system: {self.system}")
        
        # Set time step based on regime
        if self.dt is None:
            if self.min_tau_c is not None and self.T2_star_target is not None:
                self.dt = min(self.min_tau_c, self.T2_star_target) / 100
            else:
                self.dt = 0.2e-9  # Default: 0.2 ns
    
    def _init_sip(self):
        """Initialize parameters for Si:P system."""
        # Physical constants for Si:P
        self.gamma_e = 1.76e11  # rad/(s·T)
        
        # Target T2* from literature (1-10 ms range, use 2.5 ms as typical)
        if self.T2_star_target is None:
            self.T2_star_target = 2.5e-3  # 2.5 ms
        
        # Back-calculate noise amplitude from target T2*
        # T2* = sqrt(2) / (gamma_e * sigma_z) for Gaussian noise in quasi-static limit
        if self.sigma_z is None:
            self.sigma_z = np.sqrt(2.0) / (self.gamma_e * self.T2_star_target)
        
        # Set simulation time and tau_c range based on target regime
        if self.target_regime == 'motional_narrowing':
            self.total_time = 10 * self.T2_star_target  # Sufficient margin
            self.min_tau_c = 0.01e-6  # 0.01 µs
            self.max_tau_c = 1e-6  # 1 µs
            
        elif self.target_regime == 'quasi_static':
            self.total_time = 10 * self.T2_star_target  # 25 ms
            self.min_tau_c = 10e-6  # 10 µs
            self.max_tau_c = 1000e-6  # 1000 µs
            
        else:  # 'all'
            # Full regime coverage requires very long simulation
            self.total_time = 10 * self.T2_star_target  # 25 ms
            self.min_tau_c = 0.1e-6  # 0.1 µs
            self.max_tau_c = 500e-6  # 500 µs
    
    def _init_gaas(self):
        """Initialize parameters for GaAs system."""
        # Physical constants for GaAs
        self.gamma_e = 1.76e11  # rad/(s·T) (same as Si:P for electron)
        
        # Target T2* from literature (~1 µs)
        if self.T2_star_target is None:
            self.T2_star_target = 1e-6  # 1 µs
        
        # Back-calculate noise amplitude
        if self.sigma_z is None:
            self.sigma_z = np.sqrt(2.0) / (self.gamma_e * self.T2_star_target)
        
        # GaAs timescales are shorter, so simulation is easier
        if self.target_regime == 'all':
            self.total_time = 10 * self.T2_star_target  # 10 µs
            self.min_tau_c = 0.01e-6  # 0.01 µs
            self.max_tau_c = 10e-6  # 10 µs
        else:
            # For other regimes, use similar logic as Si:P but scaled
            self.total_time = 10 * self.T2_star_target
            if self.target_regime == 'motional_narrowing':
                self.min_tau_c = 0.01e-6
                self.max_tau_c = 0.1e-6
            else:  # quasi_static
                self.min_tau_c = 1e-6
                self.max_tau_c = 10e-6
    
    def validate(self) -> Dict[str, any]:
        """
        Validate parameter consistency and return validation report.
        
        Returns
        -------
        report : dict
            Validation report with warnings and recommendations
        """
        report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        print(f"\n{'='*60}")
        print(f"=== {self.system} Parameter Validation ===")
        print(f"{'='*60}")
        print(f"Target T2*: {self.T2_star_target * 1e6:.3f} µs = {self.T2_star_target * 1e3:.3f} ms")
        print(f"Noise amplitude σ_z: {self.sigma_z * 1e9:.3f} nT = {self.sigma_z * 1e6:.6f} µT")
        print(f"Total simulation time: {self.total_time * 1e6:.1f} µs = {self.total_time * 1e3:.3f} ms")
        print(f"Time step dt: {self.dt * 1e9:.3f} ns")
        
        # Calculate number of time steps
        n_steps = int(self.total_time / self.dt)
        print(f"Number of time steps: {n_steps:,}")
        print(f"τ_c range: {self.min_tau_c * 1e6:.3f} - {self.max_tau_c * 1e6:.1f} µs")
        
        # Memory estimate
        memory_per_ensemble = n_steps * 8 / 1e9  # GB (assuming float64)
        total_memory = memory_per_ensemble * self.n_ensemble
        print(f"\nEstimated memory:")
        print(f"  Per ensemble: {memory_per_ensemble:.2f} GB")
        print(f"  Total ({self.n_ensemble} ensembles): {total_memory:.2f} GB")
        
        # Validation checks
        if total_memory > 16:
            report['warnings'].append(
                f"Memory requirement ({total_memory:.2f} GB) exceeds typical limits!"
            )
            report['recommendations'].append(
                "Consider: 1) Shorter total_time, 2) Larger dt, 3) Fewer ensembles"
            )
            report['valid'] = False
        
        # Check time step vs tau_c
        if self.dt > self.min_tau_c / 10:
            report['warnings'].append(
                f"Time step dt={self.dt*1e9:.3f} ns may be too large for "
                f"min tau_c={self.min_tau_c*1e6:.3f} µs"
            )
            report['recommendations'].append(
                f"Consider dt <= {self.min_tau_c/50*1e9:.3f} ns for better accuracy"
            )
        
        # Check simulation time vs T2*
        coverage_ratio = self.total_time / self.T2_star_target
        if coverage_ratio < 5:
            report['warnings'].append(
                f"Simulation time ({self.total_time*1e6:.1f} µs) may be insufficient "
                f"for T2* = {self.T2_star_target*1e6:.3f} µs"
            )
            report['recommendations'].append(
                f"Recommend T_max >= {5*self.T2_star_target*1e6:.1f} µs "
                f"(current: {coverage_ratio:.1f}× coverage)"
            )
        
        # Check noise amplitude consistency
        T2_expected = np.sqrt(2.0) / (self.gamma_e * self.sigma_z)
        ratio = T2_expected / self.T2_star_target
        if not (0.5 < ratio < 2.0):
            report['warnings'].append(
                f"Noise amplitude gives T2* = {T2_expected*1e6:.3f} µs, "
                f"target is {self.T2_star_target*1e6:.3f} µs (ratio = {ratio:.2f})"
            )
            report['valid'] = False
        
        # Print warnings and recommendations
        if report['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in report['warnings']:
                print(f"   - {warning}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   - {rec}")
        
        if report['valid']:
            print(f"\n✓ All critical checks passed")
        else:
            print(f"\n✗ Some validation checks failed")
        
        print(f"{'='*60}\n")
        
        return report
    
    def to_dict(self) -> Dict[str, any]:
        """Convert parameters to dictionary for compatibility."""
        return {
            'gamma_e': self.gamma_e,
            'B_rms': self.sigma_z,  # Note: sigma_z is B_rms
            'T_max': self.total_time,
            'dt': self.dt,
            'M': self.n_ensemble,
            'tau_c_range': (self.min_tau_c, self.max_tau_c),
        }


def validate_simulation_parameters(system='Si_P', target_regime='all',
                                   T2_star_lit: Optional[float] = None,
                                   B_rms_current: Optional[float] = None,
                                   T_max_current: Optional[float] = None) -> Dict[str, any]:
    """
    Compare current simulation parameters with literature values.
    
    Parameters
    ----------
    system : str
        'Si_P' or 'GaAs'
    target_regime : str
        'motional_narrowing', 'quasi_static', or 'all'
    T2_star_lit : float, optional
        Literature T2* value (seconds). If None, uses default for system.
    B_rms_current : float, optional
        Current B_rms value used in simulation (Tesla)
    T_max_current : float, optional
        Current T_max value used in simulation (seconds)
    
    Returns
    -------
    comparison : dict
        Comparison report with recommendations
    """
    # Create validated parameter set
    params = SimulationParameters(system=system, target_regime=target_regime)
    if T2_star_lit is not None:
        params.T2_star_target = T2_star_lit
        params.sigma_z = np.sqrt(2.0) / (params.gamma_e * T2_star_lit)
    
    comparison = {
        'system': system,
        'literature': {
            'T2_star': params.T2_star_target,
            'B_rms_required': params.sigma_z,
        },
        'current': {
            'B_rms': B_rms_current,
            'T_max': T_max_current,
        },
        'recommendations': []
    }
    
    print(f"\n{'='*60}")
    print(f"Parameter Comparison for {system}")
    print(f"{'='*60}")
    
    # Compare B_rms
    if B_rms_current is not None:
        ratio = B_rms_current / params.sigma_z
        print(f"\nB_rms Comparison:")
        print(f"  Literature (required): {params.sigma_z * 1e9:.3f} nT")
        print(f"  Current simulation:   {B_rms_current * 1e9:.3f} nT")
        print(f"  Ratio: {ratio:.1f}×")
        
        if ratio > 10 or ratio < 0.1:
            comparison['recommendations'].append(
                f"B_rms is {ratio:.1f}× {'too large' if ratio > 1 else 'too small'}. "
                f"Update to {params.sigma_z * 1e9:.3f} nT"
            )
    
    # Compare T_max
    if T_max_current is not None:
        required_T_max = 5 * params.T2_star_target
        ratio = T_max_current / required_T_max
        print(f"\nT_max Comparison:")
        print(f"  Required (≥5×T2*): {required_T_max * 1e6:.1f} µs")
        print(f"  Current simulation:  {T_max_current * 1e6:.1f} µs")
        print(f"  Ratio: {ratio:.2f}×")
        
        if ratio < 1.0:
            comparison['recommendations'].append(
                f"T_max is {1/ratio:.1f}× too short. "
                f"Update to ≥ {required_T_max * 1e6:.1f} µs"
            )
    
    if comparison['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in comparison['recommendations']:
            print(f"   - {rec}")
    
    print(f"{'='*60}\n")
    
    return comparison

