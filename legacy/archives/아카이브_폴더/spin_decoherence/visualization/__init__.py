"""
Visualization tools for spin decoherence simulation results.

This module provides:
- Plot styles and configuration (styles.py)
- Basic plotting functions (plots.py)
- Comparison and analysis plots (comparison.py)
"""

from spin_decoherence.visualization.styles import (
    COLOR_SCHEME,
    ANNOTATION_STYLE,
    get_tau_c_color,
    setup_publication_style,
)
from spin_decoherence.visualization.plots import (
    plot_coherence_curve,
    plot_T2_vs_tauc,
    plot_ou_psd_verification,
    create_summary_plots,
)
from spin_decoherence.visualization.comparison import (
    plot_hahn_echo_vs_fid,
    plot_multiple_hahn_echo_comparisons,
    plot_T2_echo_vs_tauc,
)

__all__ = [
    'COLOR_SCHEME',
    'ANNOTATION_STYLE',
    'get_tau_c_color',
    'setup_publication_style',
    'plot_coherence_curve',
    'plot_T2_vs_tauc',
    'plot_ou_psd_verification',
    'create_summary_plots',
    'plot_hahn_echo_vs_fid',
    'plot_multiple_hahn_echo_comparisons',
    'plot_T2_echo_vs_tauc',
]

