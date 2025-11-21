"""
Ornstein-Uhlenbeck noise generation for spin decoherence simulation.

This module is a compatibility wrapper that re-exports functions from
the spin_decoherence package. For new code, prefer importing directly
from spin_decoherence.noise.

DEPRECATED: This file is maintained for backward compatibility.
New code should use: from spin_decoherence.noise import generate_ou_noise
"""

# Re-export from spin_decoherence package
from spin_decoherence.noise import (
    generate_ou_noise,
    generate_ou_noise_vectorized,
    NumericalStabilityError,
    InvalidParameterError,
)

__all__ = [
    'generate_ou_noise',
    'generate_ou_noise_vectorized',
    'NumericalStabilityError',
    'InvalidParameterError',
]
