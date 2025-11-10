"""
Unit tests for configuration module.

Tests cover:
- PhysicalConstants validation
- SimulationConfig validation
- Parameter constraints
"""

import pytest
from spin_decoherence.config import PhysicalConstants, SimulationConfig, Units


class TestPhysicalConstants:
    """물리 상수 테스트"""
    
    def test_constants_are_positive(self):
        """상수들이 양수인지 검증"""
        constants = PhysicalConstants()
        
        assert constants.GAMMA_E > 0
        assert constants.HBAR > 0
    
    def test_constants_immutability(self):
        """상수 불변성 검증"""
        constants = PhysicalConstants()
        
        # Should not be able to modify (frozen dataclass)
        with pytest.raises(Exception):  # dataclass.FrozenInstanceError
            constants.GAMMA_E = 0
    
    def test_default_values(self):
        """기본값 검증"""
        constants = PhysicalConstants()
        
        assert constants.GAMMA_E == 1.76e11
        assert constants.HBAR == 1.054571817e-34


class TestSimulationConfig:
    """시뮬레이션 설정 테스트"""
    
    def test_valid_config(self):
        """유효한 설정 생성"""
        config = SimulationConfig(
            B_rms=Units.uT_to_T(5.0),
            tau_c_range=(Units.us_to_s(0.01), Units.us_to_s(10.0))
        )
        
        assert config.B_rms > 0
        assert config.tau_c_range[0] < config.tau_c_range[1]
    
    def test_invalid_B_rms(self):
        """유효하지 않은 B_rms"""
        with pytest.raises(AssertionError):
            SimulationConfig(
                B_rms=-5e-6,
                tau_c_range=(1e-8, 1e-5)
            )
    
    def test_invalid_tau_c_range(self):
        """유효하지 않은 tau_c_range"""
        # min >= max
        with pytest.raises(AssertionError):
            SimulationConfig(
                B_rms=5e-6,
                tau_c_range=(1e-5, 1e-8)  # min > max
            )
        
        # min <= 0
        with pytest.raises(AssertionError):
            SimulationConfig(
                B_rms=5e-6,
                tau_c_range=(0, 1e-5)
            )
    
    def test_invalid_dt(self):
        """유효하지 않은 dt"""
        with pytest.raises(AssertionError):
            SimulationConfig(
                B_rms=5e-6,
                tau_c_range=(1e-8, 1e-5),
                dt=-1e-9
            )
    
    def test_stability_warning(self):
        """안정성 경고 테스트"""
        # dt too large relative to tau_c
        with pytest.warns(UserWarning):
            SimulationConfig(
                B_rms=5e-6,
                tau_c_range=(1e-8, 1e-5),
                dt=1e-8  # dt > tau_c_min / 10
            )
    
    def test_default_values(self):
        """기본값 검증"""
        config = SimulationConfig(
            B_rms=5e-6,
            tau_c_range=(1e-8, 1e-5)
        )
        
        assert config.tau_c_num == 20
        assert config.dt == 0.2e-9
        assert config.T_max == 30e-6
        assert config.M == 1000
        assert config.seed == 42
        assert config.output_dir == 'results'
        assert config.compute_bootstrap is True

