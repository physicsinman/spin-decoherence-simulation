"""
Extended noise models for material comparison.

This module is a compatibility wrapper that re-exports functions from
the spin_decoherence package. For new code, prefer importing directly
from spin_decoherence.noise.

DEPRECATED: This file is maintained for backward compatibility.
New code should use: from spin_decoherence.noise import generate_double_OU_noise
"""

# Re-export from spin_decoherence package
from spin_decoherence.noise import (
    generate_double_OU_noise,
    compute_double_OU_PSD_theory,
    verify_double_OU_statistics,
    NumericalStabilityError,
    InvalidParameterError,
)

__all__ = [
    'generate_double_OU_noise',
    'compute_double_OU_PSD_theory',
    'verify_double_OU_statistics',
    'NumericalStabilityError',
    'InvalidParameterError',
]
