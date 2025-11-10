"""
Configuration module for spin decoherence simulation.

This module provides:
- Physical constants (constants.py)
- Simulation configuration (simulation.py)
- Unit conversion utilities (units.py)
"""

from spin_decoherence.config.constants import CONSTANTS, PhysicalConstants
from spin_decoherence.config.simulation import SimulationConfig
from spin_decoherence.config.units import Units

__all__ = [
    'CONSTANTS',
    'PhysicalConstants',
    'SimulationConfig',
    'Units',
]

