"""
Analysis tools for spin decoherence simulation results.

This module provides:
- Curve fitting (fitting.py)
- Statistical analysis (statistics.py)
- Bootstrap confidence intervals (bootstrap.py)
"""

from spin_decoherence.analysis.fitting import (
    fit_coherence_decay,
    fit_coherence_decay_with_offset,
    extract_T2_from_chi,
)
from spin_decoherence.analysis.bootstrap import bootstrap_T2
from spin_decoherence.analysis.statistics import (
    compute_statistics,
    compute_confidence_intervals,
)

__all__ = [
    'fit_coherence_decay',
    'fit_coherence_decay_with_offset',
    'extract_T2_from_chi',
    'bootstrap_T2',
    'compute_statistics',
    'compute_confidence_intervals',
]

