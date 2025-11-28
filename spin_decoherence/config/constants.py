"""
Physical constants for spin decoherence simulation.

This module defines immutable physical constants and regime boundaries
used throughout the simulation.
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


@dataclass(frozen=True)  # Immutable for safety
class RegimeBoundaries:
    """
    Standard regime boundaries based on dimensionless parameter ξ = Δω × τc.
    
    These boundaries are used consistently throughout the codebase for:
    - Plotting and visualization
    - Model selection in fitting
    - Adaptive parameter selection
    - Analysis and interpretation
    
    All code should use these constants instead of hardcoded values.
    
    Standard boundaries (from Methods Section 3.1):
    - Motional Narrowing (MN): ξ < 0.5
    - Crossover: 0.5 ≤ ξ < 2.0
    - Quasi-Static (QS): ξ ≥ 2.0
    """
    # Main regime boundaries (standard, used for plotting and general classification)
    XI_MN_MAX: float = 0.5  # Motional Narrowing: ξ < 0.5
    XI_QS_MIN: float = 2.0  # Quasi-Static: ξ ≥ 2.0
    
    # Stricter boundaries for model selection in fitting
    # These ensure we're well within each regime for accurate model fitting
    XI_MN_FITTING_MAX: float = 0.15  # Strict MN for exponential model
    XI_QS_FITTING_MIN: float = 4.0  # Strict QS for Gaussian model
    
    # Stricter boundary for MN slope fitting (power-law validation)
    # Ensures T₂ ∝ τc⁻¹ is strictly valid
    XI_MN_SLOPE_MAX: float = 0.2  # Strict MN for slope = -1 validation
    
    def is_mn(self, xi: float) -> bool:
        """Check if ξ is in Motional Narrowing regime (standard boundary)."""
        return xi < self.XI_MN_MAX
    
    def is_crossover(self, xi: float) -> bool:
        """Check if ξ is in Crossover regime."""
        return self.XI_MN_MAX <= xi < self.XI_QS_MIN
    
    def is_qs(self, xi: float) -> bool:
        """Check if ξ is in Quasi-Static regime (standard boundary)."""
        return xi >= self.XI_QS_MIN
    
    def is_mn_strict(self, xi: float) -> bool:
        """Check if ξ is in strict MN regime (for fitting)."""
        return xi < self.XI_MN_FITTING_MAX
    
    def is_qs_strict(self, xi: float) -> bool:
        """Check if ξ is in strict QS regime (for fitting)."""
        return xi >= self.XI_QS_FITTING_MIN
    
    def is_mn_slope(self, xi: float) -> bool:
        """Check if ξ is in MN regime for slope fitting."""
        return xi < self.XI_MN_SLOPE_MAX
    
    def get_regime(self, xi: float) -> str:
        """Get regime name for given ξ value (standard boundaries)."""
        if self.is_mn(xi):
            return "MN"
        elif self.is_crossover(xi):
            return "Crossover"
        else:
            return "QS"


# Global instances (immutable)
CONSTANTS = PhysicalConstants()
REGIME_BOUNDARIES = RegimeBoundaries()

