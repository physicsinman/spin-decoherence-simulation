"""
Noise models for spin decoherence simulation.

This module provides:
- Base classes and exceptions (base.py)
- Ornstein-Uhlenbeck noise (ou.py)
- Double Ornstein-Uhlenbeck noise (double_ou.py)
"""

from spin_decoherence.noise.base import (
    NumericalStabilityError,
    InvalidParameterError
)
from spin_decoherence.noise.ou import generate_ou_noise, generate_ou_noise_vectorized
from spin_decoherence.noise.double_ou import (
    generate_double_OU_noise,
    compute_double_OU_PSD_theory,
    verify_double_OU_statistics
)

__all__ = [
    'NumericalStabilityError',
    'InvalidParameterError',
    'generate_ou_noise',
    'generate_ou_noise_vectorized',
    'generate_double_OU_noise',
    'compute_double_OU_PSD_theory',
    'verify_double_OU_statistics',
]

