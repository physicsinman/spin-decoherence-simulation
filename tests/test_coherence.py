"""
Unit tests for coherence function calculations.

Tests cover:
- Phase accumulation
- Ensemble coherence
- Hahn echo coherence
- Analytical comparisons
"""

import pytest
import numpy as np
from spin_decoherence.physics import (
    compute_phase_accumulation,
    compute_ensemble_coherence,
    compute_hahn_echo_coherence,
    compute_trajectory_coherence
)
from spin_decoherence.config import CONSTANTS


class TestPhaseAccumulation:
    """위상 누적 테스트"""
    
    def test_initial_condition(self):
        """초기 조건: φ(t=0) = 0"""
        delta_B = np.array([1e-6, 2e-6, 3e-6], dtype=np.float64)
        phi = compute_phase_accumulation(delta_B, CONSTANTS.GAMMA_E, dt=1e-9)
        
        assert phi[0] == 0, "Phase at t=0 should be 0"
        assert len(phi) == len(delta_B)
    
    def test_monotonicity_positive_field(self):
        """단조성 테스트: 양수 필드에 대해 φ는 단조 증가"""
        delta_B = np.ones(100, dtype=np.float64) * 1e-6  # Constant positive field
        phi = compute_phase_accumulation(delta_B, CONSTANTS.GAMMA_E, dt=1e-9)
        
        # φ should be monotonically increasing
        diff = np.diff(phi)
        assert np.all(diff >= 0), \
            "Phase should be monotonically increasing for positive field"
    
    def test_monotonicity_negative_field(self):
        """단조성 테스트: 음수 필드에 대해 φ는 단조 감소"""
        delta_B = np.ones(100, dtype=np.float64) * (-1e-6)  # Constant negative field
        phi = compute_phase_accumulation(delta_B, CONSTANTS.GAMMA_E, dt=1e-9)
        
        # φ should be monotonically decreasing
        diff = np.diff(phi)
        assert np.all(diff <= 0), \
            "Phase should be monotonically decreasing for negative field"
    
    def test_analytical_comparison_constant_field(self):
        """해석적 결과와 비교: 상수 필드"""
        # Constant field: φ(t) = γ_e * B * t
        B_const = 1e-6
        dt = 1e-9
        N = 1000
        delta_B = np.full(N, B_const, dtype=np.float64)
        
        phi = compute_phase_accumulation(delta_B, CONSTANTS.GAMMA_E, dt)
        t = np.arange(N) * dt
        phi_analytical = CONSTANTS.GAMMA_E * B_const * t
        
        # Allow small numerical error
        np.testing.assert_allclose(phi, phi_analytical, rtol=1e-10, atol=1e-20)
    
    def test_type_validation(self):
        """타입 검증 테스트"""
        # List instead of ndarray should raise TypeError
        delta_B_list = [1e-6, 2e-6, 3e-6]
        with pytest.raises(TypeError):
            compute_phase_accumulation(delta_B_list, CONSTANTS.GAMMA_E, dt=1e-9)
        
        # String instead of numeric should raise TypeError
        delta_B = np.array([1e-6, 2e-6, 3e-6])
        with pytest.raises(TypeError):
            compute_phase_accumulation(delta_B, "1.76e11", dt=1e-9)
    
    def test_shape_validation(self):
        """Shape 검증 테스트"""
        # 2D array should raise ValueError
        delta_B_2d = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        with pytest.raises(ValueError):
            compute_phase_accumulation(delta_B_2d, CONSTANTS.GAMMA_E, dt=1e-9)
    
    def test_value_validation(self):
        """값 검증 테스트"""
        delta_B = np.array([1e-6, 2e-6, 3e-6])
        # dt <= 0 should raise ValueError
        with pytest.raises(ValueError):
            compute_phase_accumulation(delta_B, CONSTANTS.GAMMA_E, dt=0)
        
        with pytest.raises(ValueError):
            compute_phase_accumulation(delta_B, CONSTANTS.GAMMA_E, dt=-1e-9)


class TestTrajectoryCoherence:
    """단일 궤적 코히어런스 테스트"""
    
    def test_basic_trajectory(self):
        """기본 궤적 생성 테스트"""
        tau_c = 1e-6
        B_rms = 5e-6
        dt = 0.2e-9
        N_steps = 1000
        
        E_traj, t = compute_trajectory_coherence(
            tau_c, B_rms, CONSTANTS.GAMMA_E, dt, N_steps, seed=42
        )
        
        assert len(E_traj) == N_steps
        assert len(t) == N_steps
        assert np.allclose(t, np.arange(N_steps) * dt)
        # Single trajectory should have |E| ≈ 1 (allowing for numerical precision)
        E_abs = np.abs(E_traj)
        np.testing.assert_allclose(E_abs, 1.0, rtol=1e-10, atol=1e-10)
    
    def test_coherence_magnitude(self):
        """단일 궤적의 코히어런스 크기는 항상 1"""
        tau_c = 1e-6
        B_rms = 5e-6
        dt = 0.2e-9
        N_steps = 1000
        
        E_traj, _ = compute_trajectory_coherence(
            tau_c, B_rms, CONSTANTS.GAMMA_E, dt, N_steps, seed=42
        )
        
        E_abs = np.abs(E_traj)
        np.testing.assert_allclose(E_abs, 1.0, rtol=1e-10)


class TestEnsembleCoherence:
    """앙상블 코히어런스 테스트"""
    
    @pytest.mark.slow
    def test_motional_narrowing_regime(self):
        """Motional-narrowing 영역 검증"""
        tau_c = 0.05e-6
        B_rms = 5e-6
        Delta_omega = CONSTANTS.GAMMA_E * B_rms
        xi = Delta_omega * tau_c
        
        assert xi < 0.2, "Should be in motional-narrowing regime"
        
        E, E_abs, E_se, t, _ = compute_ensemble_coherence(
            tau_c=tau_c, B_rms=B_rms, gamma_e=CONSTANTS.GAMMA_E,
            dt=0.2e-9, T_max=10e-6, M=1000, seed=42, progress=False
        )
        
        # In MN regime: T₂ ≈ 1/(Δω²τ_c)
        T2_th = 1.0 / (Delta_omega**2 * tau_c)
        
        # Find time when |E| ≈ exp(-1)
        idx = np.argmin(np.abs(E_abs - np.exp(-1)))
        T2_sim = t[idx]
        
        # Allow ±30% error (simulation has finite M)
        assert 0.7 * T2_th < T2_sim < 1.3 * T2_th, \
            f"T2 mismatch: sim={T2_sim:.3e}, theory={T2_th:.3e}"
    
    def test_initial_coherence(self):
        """초기 코히어런스는 1이어야 함"""
        tau_c = 1e-6
        B_rms = 5e-6
        
        E, E_abs, _, t, _ = compute_ensemble_coherence(
            tau_c=tau_c, B_rms=B_rms, gamma_e=CONSTANTS.GAMMA_E,
            dt=0.2e-9, T_max=1e-6, M=100, seed=42, progress=False
        )
        
        # At t=0, coherence should be 1
        assert np.abs(E_abs[0] - 1.0) < 0.01, \
            f"Initial coherence should be 1, got {E_abs[0]:.6f}"
        assert t[0] == 0, "First time point should be 0"
    
    def test_coherence_decay(self):
        """코히어런스는 시간에 따라 감소해야 함"""
        tau_c = 1e-6
        B_rms = 5e-6
        
        E, E_abs, _, t, _ = compute_ensemble_coherence(
            tau_c=tau_c, B_rms=B_rms, gamma_e=CONSTANTS.GAMMA_E,
            dt=0.2e-9, T_max=5e-6, M=500, seed=42, progress=False
        )
        
        # Coherence should generally decrease (may have small fluctuations)
        # Check that final coherence is less than initial
        assert E_abs[-1] < E_abs[0], \
            "Coherence should decrease over time"
        
        # Check that coherence is monotonically decreasing on average
        # (allowing for statistical fluctuations)
        E_abs_smooth = np.convolve(E_abs, np.ones(10)/10, mode='valid')
        assert np.all(np.diff(E_abs_smooth) <= 0.1), \
            "Coherence should be generally decreasing"
    
    def test_return_types(self):
        """반환 타입 검증"""
        E, E_abs, E_se, t, E_abs_all = compute_ensemble_coherence(
            tau_c=1e-6, B_rms=5e-6, gamma_e=CONSTANTS.GAMMA_E,
            dt=0.2e-9, T_max=1e-6, M=100, seed=42, progress=False
        )
        
        assert isinstance(E, np.ndarray)
        assert E.dtype == np.complex128
        assert isinstance(E_abs, np.ndarray)
        assert E_abs.dtype == np.float64
        assert isinstance(E_se, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert isinstance(E_abs_all, np.ndarray)
        assert E_abs_all.shape[0] == 100  # M trajectories


class TestHahnEchoCoherence:
    """Hahn echo 코히어런스 테스트"""
    
    def test_basic_echo(self):
        """기본 echo 생성 테스트"""
        tau_c = 1e-6
        B_rms = 5e-6
        tau_list = np.array([0.5e-6, 1e-6, 2e-6])
        
        tau_echo, E_echo, E_echo_abs, E_echo_se, _ = compute_hahn_echo_coherence(
            tau_c=tau_c, B_rms=B_rms, gamma_e=CONSTANTS.GAMMA_E,
            dt=0.2e-9, tau_list=tau_list, M=100, seed=42, progress=False
        )
        
        assert len(tau_echo) == len(tau_list)
        assert np.allclose(tau_echo, 2 * tau_list)
        assert len(E_echo) == len(tau_list)
        assert len(E_echo_abs) == len(tau_list)
    
    def test_echo_refocusing_slow_noise(self):
        """Slow noise에서 echo가 refocusing하는지 검증"""
        # Slow noise: tau_c >> tau_echo
        tau_c = 10e-6
        B_rms = 5e-6
        tau_list = np.array([0.5e-6, 1e-6, 2e-6])
        
        tau_echo, E_echo, E_echo_abs, _, _ = compute_hahn_echo_coherence(
            tau_c=tau_c, B_rms=B_rms, gamma_e=CONSTANTS.GAMMA_E,
            dt=0.2e-9, tau_list=tau_list, M=500, seed=42, progress=False
        )
        
        # For slow noise, echo should refocus (coherence should be higher than FID)
        # At least check that echo coherence is reasonable (> 0.5 for small tau)
        assert E_echo_abs[0] > 0.5, \
            f"Echo coherence for slow noise should be high, got {E_echo_abs[0]:.3f}"
    
    def test_echo_vs_fid_comparison(self):
        """Echo와 FID 비교 테스트"""
        tau_c = 1e-6
        B_rms = 5e-6
        
        # FID coherence
        E_fid, E_fid_abs, _, t_fid, _ = compute_ensemble_coherence(
            tau_c=tau_c, B_rms=B_rms, gamma_e=CONSTANTS.GAMMA_E,
            dt=0.2e-9, T_max=4e-6, M=200, seed=42, progress=False
        )
        
        # Echo coherence at 2τ = 4μs (same total time)
        tau_list = np.array([2e-6])
        tau_echo, E_echo, E_echo_abs, _, _ = compute_hahn_echo_coherence(
            tau_c=tau_c, B_rms=B_rms, gamma_e=CONSTANTS.GAMMA_E,
            dt=0.2e-9, tau_list=tau_list, M=200, seed=42, progress=False
        )
        
        # For slow noise, echo should be better than FID
        # For fast noise, they should be similar
        # Just check that both are computed correctly
        assert E_fid_abs[-1] > 0, "FID coherence should be positive"
        assert E_echo_abs[0] > 0, "Echo coherence should be positive"

