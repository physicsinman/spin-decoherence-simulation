"""
Simulation configuration for spin decoherence simulation.

This module defines mutable simulation parameters that can be configured
for different simulation scenarios.
"""

from dataclasses import dataclass
from typing import Tuple
import warnings


@dataclass
class SimulationConfig:
    """
    Simulation configuration (Simulation Configuration) - 변경 가능
    
    Parameters for running spin decoherence simulations.
    These can be modified for different simulation scenarios.
    """
    # Noise parameters
    B_rms: float  # T, RMS magnetic field
    tau_c_range: Tuple[float, float]  # s, correlation time range (min, max)
    tau_c_num: int = 20  # Number of tau_c values to sweep
    
    # Time grid
    dt: float = 0.2e-9  # s, time step
    T_max: float = 30e-6  # s, max simulation time
    
    # Ensemble
    M: int = 1000  # Number of realizations
    seed: int = 42  # Random seed
    
    # Output settings
    output_dir: str = 'results'  # Output directory
    compute_bootstrap: bool = True  # Compute bootstrap CI for T_2
    save_delta_B_sample: bool = False  # Save first trajectory's delta_B for PSD verification
    
    # Echo-specific settings
    T_max_echo: float = 20e-6  # s, cap echo delays to keep simulations tractable
    
    def __post_init__(self):
        """Validate simulation parameters."""
        assert self.B_rms > 0, "B_rms must be positive"
        assert len(self.tau_c_range) == 2, "tau_c_range must be a tuple of (min, max)"
        assert self.tau_c_range[0] < self.tau_c_range[1], "Invalid tau_c range: min must be < max"
        assert self.tau_c_range[0] > 0, "tau_c_range min must be positive"
        assert self.dt > 0, "dt must be positive"
        assert self.T_max > 0, "T_max must be positive"
        assert self.M > 0, "M must be positive"
        assert self.tau_c_num > 0, "tau_c_num must be positive"
        
        # Stability check: dt << tau_c
        min_tau_c = self.tau_c_range[0]
        if self.dt > min_tau_c / 10:
            warnings.warn(
                f"dt={self.dt:.2e}s may be too large for tau_c={min_tau_c:.2e}s. "
                f"Recommend dt <= tau_c/50 for stability.",
                UserWarning
            )
        
        # Check that T_max is reasonable
        if self.T_max < 10 * self.dt:
            warnings.warn(
                f"T_max={self.T_max:.2e}s is very small compared to dt={self.dt:.2e}s. "
                f"Recommend T_max >= 10*dt for meaningful results.",
                UserWarning
            )

