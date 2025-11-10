"""
Physics calculations for spin decoherence.

This module provides:
- Phase accumulation (phase.py)
- Coherence calculations (coherence.py)
- Analytical solutions (analytical.py)
"""

from spin_decoherence.physics.phase import compute_phase_accumulation
from spin_decoherence.physics.coherence import (
    compute_trajectory_coherence,
    compute_ensemble_coherence,
    compute_hahn_echo_coherence,
    compute_ensemble_coherence_double_OU,
    compute_hahn_echo_coherence_double_OU,
)
from spin_decoherence.physics.analytical import (
    analytical_ou_coherence,
    analytical_hahn_echo_coherence,
    theoretical_T2_motional_narrowing,
)

__all__ = [
    'compute_phase_accumulation',
    'compute_trajectory_coherence',
    'compute_ensemble_coherence',
    'compute_hahn_echo_coherence',
    'compute_ensemble_coherence_double_OU',
    'compute_hahn_echo_coherence_double_OU',
    'analytical_ou_coherence',
    'analytical_hahn_echo_coherence',
    'theoretical_T2_motional_narrowing',
]

