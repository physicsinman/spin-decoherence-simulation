"""
Plot styles and configuration for publication-quality figures.

This module provides unified styling for PRB/APS/Nature-quality figures.
"""

import platform
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality rcParams (PRB/APS style)
PUBLICATION_RCPARAMS = {
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 1.6,
    'lines.markersize': 5,
    'axes.linewidth': 1.2,
    'figure.figsize': (8, 6),
    'text.usetex': False,  # Set to True if LaTeX is installed
    'grid.alpha': 0.25,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    'axes.unicode_minus': False,  # Use ASCII minus instead of U+2212
}

# Unified color scheme (Publication-quality)
COLOR_SCHEME = {
    # Primary data colors (consistent across all figures)
    'simulation': '#1E40AF',      # Navy blue - FID/simulation data
    'echo': '#F97316',            # Orange - Hahn Echo data
    'theory': '#991B1B',          # Dark red - Theoretical predictions
    'analytical': '#059669',      # Dark green - Analytical solutions
    
    # Secondary/auxiliary colors
    'tau_c_low': '#8B5CF6',       # Purple (fast noise)
    'tau_c_mid_low': '#3B82F6',   # Blue
    'tau_c_mid_high': '#10B981',  # Green
    'tau_c_high': '#F59E0B',      # Yellow (slow noise)
    'residual': '#000000',        # Black for residuals
    
    # Region highlighting
    'mn_region': '#DBEAFE',       # Light blue - Motional narrowing region
    'static_region': '#FEE2E2',   # Light red - Static/quasi-static region
    'transition': '#FEF3C7',      # Light yellow - Transition region
}

# Annotation box style (unified)
ANNOTATION_STYLE = {
    'boxstyle': 'round',
    'pad': 0.5,
    'alpha': 0.85,
    'edgecolor': 'black',
    'linewidth': 1.5
}


def setup_publication_style():
    """
    Setup publication-quality matplotlib style.
    
    This function configures matplotlib rcParams for PRB/APS/Nature-quality figures.
    """
    rcParams.update(PUBLICATION_RCPARAMS)
    
    # Set font for Korean characters and handle U+2212 (minus sign) support
    if platform.system() == 'Darwin':  # macOS
        rcParams['font.family'] = ['AppleGothic', 'Arial Unicode MS', 'sans-serif']
    elif platform.system() == 'Windows':
        rcParams['font.family'] = ['Malgun Gothic', 'sans-serif']
    else:  # Linux
        rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        # Fallback: try to use Noto Sans CJK if available
        try:
            import matplotlib.font_manager as fm
            fonts = [f.name for f in fm.fontManager.ttflist]
            if 'Noto Sans CJK KR' in fonts or 'Noto Sans' in fonts:
                rcParams['font.family'] = ['Noto Sans CJK KR', 'Noto Sans', 'DejaVu Sans', 'sans-serif']
        except:
            pass


def get_tau_c_color(tau_c, tau_c_min, tau_c_max):
    """
    Get color for tau_c value based on position in range.
    
    Returns color from purple (fast) to yellow (slow).
    
    Parameters
    ----------
    tau_c : float
        Correlation time value
    tau_c_min : float
        Minimum tau_c in range
    tau_c_max : float
        Maximum tau_c in range
        
    Returns
    -------
    color : str
        Hex color code
    """
    if tau_c_max <= tau_c_min:
        return COLOR_SCHEME['tau_c_mid_low']
    
    import numpy as np
    # Normalize to [0, 1]
    frac = (np.log10(tau_c) - np.log10(tau_c_min)) / (np.log10(tau_c_max) - np.log10(tau_c_min))
    frac = np.clip(frac, 0.0, 1.0)
    
    # Interpolate between colors
    if frac < 0.33:
        return COLOR_SCHEME['tau_c_low']
    elif frac < 0.67:
        return COLOR_SCHEME['tau_c_mid_low']
    else:
        return COLOR_SCHEME['tau_c_high']


# Initialize style on import
setup_publication_style()

