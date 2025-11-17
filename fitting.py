"""
Fitting functions for extracting T_2 relaxation times.

This module is a compatibility wrapper that re-exports functions from
the spin_decoherence package. For new code, prefer importing directly
from spin_decoherence.analysis.

DEPRECATED: This file is maintained for backward compatibility.
New code should use: from spin_decoherence.analysis import fit_coherence_decay
"""

# Re-export from spin_decoherence package
from spin_decoherence.analysis.fitting import (
    gaussian_decay,
    exponential_decay,
    stretched_exponential_decay,
    select_echo_fit_window,
    select_fit_window,
    fit_echo_decay,
    fit_coherence_decay,
    theoretical_T2_motional_narrowing,
    analytical_hahn_echo_coherence,
    filter_function_hahn_echo,
    ou_psd_omega,
    analytical_hahn_echo_filter_function,
    analytical_ou_coherence,
    fit_mn_slope,
    weights_from_E,
    decay_with_offset,
    fit_coherence_decay_with_offset,
    extract_T2_from_chi,
)
from spin_decoherence.analysis.bootstrap import bootstrap_T2

__all__ = [
    'gaussian_decay',
    'exponential_decay',
    'stretched_exponential_decay',
    'select_echo_fit_window',
    'select_fit_window',
    'fit_echo_decay',
    'fit_coherence_decay',
    'theoretical_T2_motional_narrowing',
    'analytical_hahn_echo_coherence',
    'filter_function_hahn_echo',
    'ou_psd_omega',
    'analytical_hahn_echo_filter_function',
    'analytical_ou_coherence',
    'fit_mn_slope',
    'weights_from_E',
    'decay_with_offset',
    'fit_coherence_decay_with_offset',
    'extract_T2_from_chi',
    'bootstrap_T2',
]
