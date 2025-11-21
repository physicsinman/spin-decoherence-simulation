"""
Unit conversion helpers for spin decoherence simulation.

This module is a compatibility wrapper that re-exports the Units class from
the spin_decoherence package. For new code, prefer importing directly
from spin_decoherence.config.

DEPRECATED: This file is maintained for backward compatibility.
New code should use: from spin_decoherence.config import Units
"""

# Re-export from spin_decoherence package
from spin_decoherence.config import Units

__all__ = ['Units']
