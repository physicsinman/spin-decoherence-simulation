"""
Unit conversion helpers for spin decoherence simulation.

This module provides utilities for converting between different units
commonly used in the simulation (μs, ns, μT, etc.) and SI base units (s, T).
"""

from typing import Union


class Units:
    """Unit conversion helpers (Unit Conversion Helpers)"""
    
    @staticmethod
    def us_to_s(time_us: Union[float, int]) -> float:
        """Convert microseconds to seconds."""
        return float(time_us * 1e-6)
    
    @staticmethod
    def ns_to_s(time_ns: Union[float, int]) -> float:
        """Convert nanoseconds to seconds."""
        return float(time_ns * 1e-9)
    
    @staticmethod
    def uT_to_T(field_uT: Union[float, int]) -> float:
        """Convert microtesla to tesla."""
        return float(field_uT * 1e-6)
    
    @staticmethod
    def s_to_us(time_s: Union[float, int]) -> float:
        """Convert seconds to microseconds."""
        return float(time_s * 1e6)
    
    @staticmethod
    def s_to_ns(time_s: Union[float, int]) -> float:
        """Convert seconds to nanoseconds."""
        return float(time_s * 1e9)
    
    @staticmethod
    def T_to_uT(field_T: Union[float, int]) -> float:
        """Convert tesla to microtesla."""
        return float(field_T * 1e6)
    
    @staticmethod
    def format_time(time_s: Union[float, int]) -> str:
        """
        Format time with appropriate unit.
        
        Automatically selects the most appropriate unit (ps, ns, μs, ms, s)
        based on the magnitude of the time value.
        """
        if time_s < 1e-12:
            return f"{time_s*1e15:.2f} fs"
        elif time_s < 1e-9:
            return f"{time_s*1e12:.2f} ps"
        elif time_s < 1e-6:
            return f"{time_s*1e9:.2f} ns"
        elif time_s < 1e-3:
            return f"{time_s*1e6:.2f} μs"
        elif time_s < 1.0:
            return f"{time_s*1e3:.2f} ms"
        else:
            return f"{time_s:.2f} s"
    
    @staticmethod
    def format_field(field_T: Union[float, int]) -> str:
        """
        Format magnetic field with appropriate unit.
        
        Automatically selects the most appropriate unit (nT, μT, mT, T)
        based on the magnitude of the field value.
        """
        if field_T < 1e-9:
            return f"{field_T*1e12:.2f} pT"
        elif field_T < 1e-6:
            return f"{field_T*1e9:.2f} nT"
        elif field_T < 1e-3:
            return f"{field_T*1e6:.2f} μT"
        elif field_T < 1.0:
            return f"{field_T*1e3:.2f} mT"
        else:
            return f"{field_T:.2f} T"

