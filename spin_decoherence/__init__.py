"""
Spin Decoherence Simulation Package

A comprehensive package for simulating spin decoherence in quantum systems
subject to stochastic magnetic field fluctuations.
"""

__version__ = "1.0.0"

# Import main classes and functions for convenience
from spin_decoherence.config.constants import CONSTANTS, PhysicalConstants
from spin_decoherence.config.simulation import SimulationConfig
from spin_decoherence.config.units import Units

__all__ = [
    'CONSTANTS',
    'PhysicalConstants',
    'SimulationConfig',
    'Units',
]

