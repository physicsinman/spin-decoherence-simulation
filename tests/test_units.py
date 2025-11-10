"""
Unit tests for unit conversion helpers.

Tests cover:
- Unit conversions (μs↔s, ns↔s, μT↔T)
- Formatting functions
- Edge cases
"""

import pytest
from spin_decoherence.config import Units


class TestUnitConversions:
    """단위 변환 테스트"""
    
    def test_us_to_s(self):
        """마이크로초 → 초 변환"""
        assert Units.us_to_s(1.0) == 1e-6
        assert abs(Units.us_to_s(10.0) - 10e-6) < 1e-15
        assert Units.us_to_s(0.5) == 0.5e-6
    
    def test_s_to_us(self):
        """초 → 마이크로초 변환"""
        assert Units.s_to_us(1e-6) == 1.0
        assert Units.s_to_us(10e-6) == 10.0
        assert abs(Units.s_to_us(0.5e-6) - 0.5) < 1e-10
    
    def test_ns_to_s(self):
        """나노초 → 초 변환"""
        assert Units.ns_to_s(1.0) == 1e-9
        assert Units.ns_to_s(10.0) == 10e-9
        assert Units.ns_to_s(0.5) == 0.5e-9
    
    def test_s_to_ns(self):
        """초 → 나노초 변환"""
        assert Units.s_to_ns(1e-9) == 1.0
        assert Units.s_to_ns(10e-9) == 10.0
        assert abs(Units.s_to_ns(0.5e-9) - 0.5) < 1e-10
    
    def test_uT_to_T(self):
        """마이크로테슬라 → 테슬라 변환"""
        assert Units.uT_to_T(1.0) == 1e-6
        assert abs(Units.uT_to_T(10.0) - 10e-6) < 1e-15
        assert Units.uT_to_T(0.5) == 0.5e-6
    
    def test_T_to_uT(self):
        """테슬라 → 마이크로테슬라 변환"""
        assert Units.T_to_uT(1e-6) == 1.0
        assert Units.T_to_uT(10e-6) == 10.0
        assert abs(Units.T_to_uT(0.5e-6) - 0.5) < 1e-10
    
    def test_round_trip_conversions(self):
        """왕복 변환 테스트"""
        # Time conversions
        time_us = 5.0
        time_s = Units.us_to_s(time_us)
        assert abs(Units.s_to_us(time_s) - time_us) < 1e-10
        
        time_ns = 0.2
        time_s = Units.ns_to_s(time_ns)
        assert abs(Units.s_to_ns(time_s) - time_ns) < 1e-10
        
        # Field conversions
        field_uT = 5.0
        field_T = Units.uT_to_T(field_uT)
        assert abs(Units.T_to_uT(field_T) - field_uT) < 1e-10
    
    def test_int_input(self):
        """정수 입력 처리"""
        assert Units.us_to_s(1) == 1e-6
        assert Units.ns_to_s(1) == 1e-9
        assert Units.uT_to_T(1) == 1e-6


class TestFormatting:
    """포맷팅 함수 테스트"""
    
    def test_format_time_ps(self):
        """피코초 포맷팅"""
        time_s = 1e-12
        formatted = Units.format_time(time_s)
        assert "ps" in formatted
        assert "1.00" in formatted
    
    def test_format_time_ns(self):
        """나노초 포맷팅"""
        time_s = 1e-9
        formatted = Units.format_time(time_s)
        assert "ns" in formatted
    
    def test_format_time_us(self):
        """마이크로초 포맷팅"""
        time_s = 1e-6
        formatted = Units.format_time(time_s)
        assert "μs" in formatted or "us" in formatted
    
    def test_format_time_ms(self):
        """밀리초 포맷팅"""
        time_s = 1e-3
        formatted = Units.format_time(time_s)
        assert "ms" in formatted
    
    def test_format_time_s(self):
        """초 포맷팅"""
        time_s = 1.0
        formatted = Units.format_time(time_s)
        assert "s" in formatted
    
    def test_format_field_nT(self):
        """나노테슬라 포맷팅"""
        field_T = 1e-9
        formatted = Units.format_field(field_T)
        assert "nT" in formatted
    
    def test_format_field_uT(self):
        """마이크로테슬라 포맷팅"""
        field_T = 1e-6
        formatted = Units.format_field(field_T)
        assert "μT" in formatted or "uT" in formatted
    
    def test_format_field_mT(self):
        """밀리테슬라 포맷팅"""
        field_T = 1e-3
        formatted = Units.format_field(field_T)
        assert "mT" in formatted
    
    def test_format_field_T(self):
        """테슬라 포맷팅"""
        field_T = 1.0
        formatted = Units.format_field(field_T)
        assert "T" in formatted and "m" not in formatted

