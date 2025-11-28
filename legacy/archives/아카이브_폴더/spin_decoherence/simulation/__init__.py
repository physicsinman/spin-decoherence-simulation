"""
Simulation engine for spin decoherence.

This module provides:
- Core simulation engine (engine.py)
- FID simulation routines (fid.py)
- Hahn echo simulation routines (echo.py)
"""

from spin_decoherence.simulation.engine import (
    estimate_characteristic_T2,
    get_optimal_tau_range,
)
from spin_decoherence.simulation.fid import (
    run_simulation_single,
    run_simulation_sweep,
)
from spin_decoherence.simulation.echo import (
    run_simulation_with_hahn_echo,
    run_hahn_echo_sweep,
)

__all__ = [
    'estimate_characteristic_T2',
    'get_optimal_tau_range',
    'run_simulation_single',
    'run_simulation_sweep',
    'run_simulation_with_hahn_echo',
    'run_hahn_echo_sweep',
]

