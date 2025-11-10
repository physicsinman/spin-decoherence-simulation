"""
Unit tests for Ornstein-Uhlenbeck noise generation.

Tests cover:
- Basic noise generation
- Statistical properties (mean, variance, autocorrelation)
- Reproducibility
- Error handling
- Edge cases
"""

import pytest
import numpy as np
from spin_decoherence.noise import (
    generate_ou_noise,
    generate_ou_noise_vectorized,
    InvalidParameterError,
    NumericalStabilityError
)


class TestOUNoiseGeneration:
    """OU 노이즈 생성 테스트"""
    
    def test_basic_generation(self):
        """기본 생성 테스트"""
        tau_c = 1e-6
        B_rms = 5e-6
        dt = 0.2e-9
        N_steps = 10000
        
        noise = generate_ou_noise(tau_c, B_rms, dt, N_steps, seed=42)
        
        assert isinstance(noise, np.ndarray)
        assert len(noise) == N_steps
        assert noise.dtype == np.float64
        assert np.all(np.isfinite(noise)), "Noise should contain only finite values"
    
    def test_statistical_properties(self):
        """통계적 성질 검증"""
        tau_c = 1e-6
        B_rms = 5e-6
        dt = 0.2e-9
        N_steps = 100000  # Large for good statistics
        
        noise = generate_ou_noise(tau_c, B_rms, dt, N_steps, seed=42)
        
        # Check mean ≈ 0
        mean_abs = np.abs(np.mean(noise))
        assert mean_abs < 0.01 * B_rms, \
            f"Mean should be close to 0, got {np.mean(noise):.3e}"
        
        # Check variance ≈ B_rms²
        var_ratio = np.var(noise) / B_rms**2
        assert 0.95 < var_ratio < 1.05, \
            f"Variance ratio {var_ratio:.3f} out of range [0.95, 1.05]"
        
        # Check autocorrelation
        if len(noise) > 1:
            rho_emp = np.corrcoef(noise[:-1], noise[1:])[0, 1]
            rho_th = np.exp(-dt / tau_c)
            error = np.abs(rho_emp - rho_th)
            assert error < 0.01, \
                f"Autocorrelation error {error:.6f} too large (emp={rho_emp:.6f}, th={rho_th:.6f})"
    
    def test_reproducibility(self):
        """재현성 테스트"""
        params = dict(tau_c=1e-6, B_rms=5e-6, dt=0.2e-9, N_steps=1000, seed=42)
        
        noise1 = generate_ou_noise(**params)
        noise2 = generate_ou_noise(**params)
        
        np.testing.assert_array_equal(
            noise1, noise2,
            err_msg="Same seed should give same result"
        )
    
    def test_different_seeds_different_results(self):
        """다른 seed는 다른 결과를 생성해야 함"""
        params = dict(tau_c=1e-6, B_rms=5e-6, dt=0.2e-9, N_steps=1000)
        
        noise1 = generate_ou_noise(**params, seed=42)
        noise2 = generate_ou_noise(**params, seed=43)
        
        # Should be different (very unlikely to be identical)
        assert not np.array_equal(noise1, noise2), \
            "Different seeds should produce different results"
    
    def test_invalid_parameters(self):
        """유효하지 않은 파라미터 검증"""
        # tau_c <= 0
        with pytest.raises(InvalidParameterError):
            generate_ou_noise(tau_c=0, B_rms=5e-6, dt=0.2e-9, N_steps=1000)
        
        with pytest.raises(InvalidParameterError):
            generate_ou_noise(tau_c=-1e-6, B_rms=5e-6, dt=0.2e-9, N_steps=1000)
        
        # B_rms <= 0
        with pytest.raises(InvalidParameterError):
            generate_ou_noise(tau_c=1e-6, B_rms=-5e-6, dt=0.2e-9, N_steps=1000)
        
        with pytest.raises(InvalidParameterError):
            generate_ou_noise(tau_c=1e-6, B_rms=0, dt=0.2e-9, N_steps=1000)
        
        # dt <= 0
        with pytest.raises(InvalidParameterError):
            generate_ou_noise(tau_c=1e-6, B_rms=5e-6, dt=0, N_steps=1000)
        
        # N_steps <= 0
        with pytest.raises(InvalidParameterError):
            generate_ou_noise(tau_c=1e-6, B_rms=5e-6, dt=0.2e-9, N_steps=0)
        
        # N_steps not integer
        with pytest.raises(InvalidParameterError):
            generate_ou_noise(tau_c=1e-6, B_rms=5e-6, dt=0.2e-9, N_steps=1000.5)
    
    def test_numerical_stability_errors(self):
        """수치적 안정성 에러 검증"""
        # dt >> tau_c (should raise NumericalStabilityError)
        with pytest.raises(NumericalStabilityError):
            generate_ou_noise(tau_c=1e-6, B_rms=5e-6, dt=1e-3, N_steps=1000)
        
        # dt/tau_c = 0.15 (should warn but not error)
        with pytest.warns(UserWarning):
            generate_ou_noise(tau_c=1e-6, B_rms=5e-6, dt=0.15e-6, N_steps=1000)
    
    def test_burnin_effect(self):
        """Burn-in 효과 검증"""
        tau_c = 1e-6
        B_rms = 5e-6
        dt = 0.2e-9
        N_steps = 10000
        
        # With default burnin_mult=5
        noise = generate_ou_noise(tau_c, B_rms, dt, N_steps, seed=42, burnin_mult=5)
        
        # Variance should be close to B_rms² (burn-in ensures stationarity)
        var_ratio = np.var(noise) / B_rms**2
        assert 0.9 < var_ratio < 1.1, \
            f"Variance ratio {var_ratio:.3f} should be close to 1 after burn-in"
    
    def test_vectorized_version(self):
        """Vectorized 버전이 main 버전과 동일한 결과를 생성하는지"""
        params = dict(tau_c=1e-6, B_rms=5e-6, dt=0.2e-9, N_steps=1000, seed=42)
        
        noise1 = generate_ou_noise(**params)
        noise2 = generate_ou_noise_vectorized(**params)
        
        np.testing.assert_array_equal(noise1, noise2)
    
    def test_numba_availability(self):
        """Numba 사용 가능 여부 확인"""
        from spin_decoherence.noise.ou import NUMBA_AVAILABLE
        
        # Numba가 설치되어 있으면 True, 없으면 False
        # Both cases are valid
        assert isinstance(NUMBA_AVAILABLE, bool)
        
        # If numba is available, verify it's being used
        if NUMBA_AVAILABLE:
            # Generate noise - should use JIT-compiled version
            noise = generate_ou_noise(
                tau_c=1e-6, B_rms=5e-6, dt=0.2e-9, N_steps=1000, seed=42
            )
            # Just verify it works correctly
            assert len(noise) == 1000
            assert np.all(np.isfinite(noise))
    
    @pytest.mark.slow
    def test_long_trajectory(self):
        """긴 궤적 생성 테스트 (느림)"""
        noise = generate_ou_noise(
            tau_c=1e-6, B_rms=5e-6, dt=0.2e-9,
            N_steps=1000000, seed=42
        )
        assert len(noise) == 1000000
        assert np.all(np.isfinite(noise))
    
    def test_memory_warning(self):
        """큰 메모리 할당 시 경고"""
        # This should trigger a warning (but not error) for large allocation
        with pytest.warns(UserWarning):
            generate_ou_noise(
                tau_c=1e-6, B_rms=5e-6, dt=0.2e-9,
                N_steps=10000000, seed=42
            )

