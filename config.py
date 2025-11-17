"""
Configuration module for spin decoherence simulation.

This module is a compatibility wrapper that re-exports classes from
the spin_decoherence package. For new code, prefer importing directly
from spin_decoherence.config.

DEPRECATED: This file is maintained for backward compatibility.
New code should use: from spin_decoherence.config import CONSTANTS, SimulationConfig
"""

# Re-export from spin_decoherence package
from spin_decoherence.config import (
    CONSTANTS,
    PhysicalConstants,
    SimulationConfig,
)

__all__ = [
    'CONSTANTS',
    'PhysicalConstants',
    'SimulationConfig',
]
