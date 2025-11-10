"""
Physical constants for spin decoherence simulation.

This module defines immutable physical constants used throughout the simulation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)  # Immutable for safety
class PhysicalConstants:
    """
    Physical constants (Physical Constants) - 변경 불가
    
    These are fundamental physical constants that should not be modified
    during simulation runs.
    """
    GAMMA_E: float = 1.76e11  # rad/(s·T), electron gyromagnetic ratio
    HBAR: float = 1.054571817e-34  # J·s, reduced Planck constant
    
    def __post_init__(self):
        """Validate physical constants."""
        assert self.GAMMA_E > 0, "GAMMA_E must be positive"
        assert self.HBAR > 0, "HBAR must be positive"


# Global instance of physical constants (immutable)
CONSTANTS = PhysicalConstants()

