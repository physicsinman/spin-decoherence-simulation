"""
Coherence function calculation for spin decoherence.

This module is a compatibility wrapper that re-exports functions from
the spin_decoherence package. For new code, prefer importing directly
from spin_decoherence.physics.

DEPRECATED: This file is maintained for backward compatibility.
New code should use: from spin_decoherence.physics import compute_ensemble_coherence
"""

# Re-export from spin_decoherence package
from spin_decoherence.physics import (
    compute_phase_accumulation,
    compute_trajectory_coherence,
    compute_ensemble_coherence,
    compute_hahn_echo_coherence,
    compute_ensemble_coherence_double_OU,
    compute_hahn_echo_coherence_double_OU,
)

__all__ = [
    'compute_phase_accumulation',
    'compute_trajectory_coherence',
    'compute_ensemble_coherence',
    'compute_hahn_echo_coherence',
    'compute_ensemble_coherence_double_OU',
    'compute_hahn_echo_coherence_double_OU',
]
