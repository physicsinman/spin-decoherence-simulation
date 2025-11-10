"""
Unit tests for extended noise models (Double-OU).

Tests cover:
- Double-OU noise generation
- Component independence
- Statistical properties
- Error handling
"""

import pytest
import numpy as np
from spin_decoherence.noise import (
    generate_double_OU_noise,
    compute_double_OU_PSD_theory,
    verify_double_OU_statistics,
    InvalidParameterError,
    NumericalStabilityError
)


class TestDoubleOUNoise:
    """Double-OU 노이즈 생성 테스트"""
    
    def test_basic_generation(self):
        """기본 생성 테스트"""
        tau_c1 = 0.05e-6
        tau_c2 = 2.0e-6
        B_rms1 = 4.0e-6
        B_rms2 = 3.0e-6
        dt = 0.2e-9
        N_steps = 10000
        
        delta_B = generate_double_OU_noise(
            tau_c1, tau_c2, B_rms1, B_rms2, dt, N_steps, seed=42
        )
        
        assert isinstance(delta_B, np.ndarray)
        assert len(delta_B) == N_steps
        assert delta_B.dtype == np.float64
        assert np.all(np.isfinite(delta_B))
    
    def test_component_generation(self):
        """컴포넌트 분리 생성 테스트"""
        tau_c1 = 0.05e-6
        tau_c2 = 2.0e-6
        B_rms1 = 4.0e-6
        B_rms2 = 3.0e-6
        dt = 0.2e-9
        N_steps = 10000
        
        delta_B, delta_B1, delta_B2 = generate_double_OU_noise(
            tau_c1, tau_c2, B_rms1, B_rms2, dt, N_steps,
            seed=42, return_components=True
        )
        
        # Total should be sum of components
        np.testing.assert_allclose(delta_B, delta_B1 + delta_B2, rtol=1e-10)
        
        # Components should have correct lengths
        assert len(delta_B1) == N_steps
        assert len(delta_B2) == N_steps
    
    def test_statistical_properties(self):
        """통계적 성질 검증"""
        tau_c1 = 0.05e-6
        tau_c2 = 2.0e-6
        B_rms1 = 4.0e-6
        B_rms2 = 3.0e-6
        dt = 0.2e-9
        N_steps = 100000  # Large for good statistics
        
        delta_B, delta_B1, delta_B2 = generate_double_OU_noise(
            tau_c1, tau_c2, B_rms1, B_rms2, dt, N_steps,
            seed=42, return_components=True
        )
        
        # Check component variances
        var1_ratio = np.var(delta_B1) / B_rms1**2
        var2_ratio = np.var(delta_B2) / B_rms2**2
        
        assert 0.95 < var1_ratio < 1.05, \
            f"Component 1 variance ratio {var1_ratio:.3f} out of range"
        assert 0.95 < var2_ratio < 1.05, \
            f"Component 2 variance ratio {var2_ratio:.3f} out of range"
        
        # Total variance should be approximately sum of component variances
        # (if uncorrelated)
        var_total = np.var(delta_B)
        var_expected = B_rms1**2 + B_rms2**2
        var_ratio = var_total / var_expected
        
        assert 0.9 < var_ratio < 1.1, \
            f"Total variance ratio {var_ratio:.3f} out of range"
    
    def test_component_independence(self):
        """컴포넌트 독립성 검증"""
        tau_c1 = 0.05e-6
        tau_c2 = 2.0e-6
        B_rms1 = 4.0e-6
        B_rms2 = 3.0e-6
        dt = 0.2e-9
        N_steps = 100000
        
        delta_B, delta_B1, delta_B2 = generate_double_OU_noise(
            tau_c1, tau_c2, B_rms1, B_rms2, dt, N_steps,
            seed=42, return_components=True
        )
        
        # Components should be approximately uncorrelated
        correlation = np.corrcoef(delta_B1, delta_B2)[0, 1]
        assert np.abs(correlation) < 0.1, \
            f"Components should be uncorrelated, got correlation {correlation:.3f}"
    
    def test_invalid_parameters(self):
        """유효하지 않은 파라미터 검증"""
        # tau_c1 <= 0
        with pytest.raises(InvalidParameterError):
            generate_double_OU_noise(
                0, 2e-6, 4e-6, 3e-6, 0.2e-9, 1000, seed=42
            )
        
        # tau_c2 <= tau_c1
        with pytest.raises(InvalidParameterError):
            generate_double_OU_noise(
                2e-6, 1e-6, 4e-6, 3e-6, 0.2e-9, 1000, seed=42
            )
        
        # B_rms <= 0
        with pytest.raises(InvalidParameterError):
            generate_double_OU_noise(
                0.05e-6, 2e-6, -4e-6, 3e-6, 0.2e-9, 1000, seed=42
            )
        
        # dt <= 0
        with pytest.raises(InvalidParameterError):
            generate_double_OU_noise(
                0.05e-6, 2e-6, 4e-6, 3e-6, 0, 1000, seed=42
            )
    
    def test_numerical_stability(self):
        """수치적 안정성 검증"""
        # dt too large relative to tau_c1
        with pytest.raises(NumericalStabilityError):
            generate_double_OU_noise(
                0.05e-6, 2e-6, 4e-6, 3e-6, 1e-3, 1000, seed=42
            )
    
    def test_reproducibility(self):
        """재현성 테스트"""
        params = dict(
            tau_c1=0.05e-6, tau_c2=2e-6,
            B_rms1=4e-6, B_rms2=3e-6,
            dt=0.2e-9, N_steps=1000, seed=42
        )
        
        noise1 = generate_double_OU_noise(**params)
        noise2 = generate_double_OU_noise(**params)
        
        np.testing.assert_array_equal(noise1, noise2)


class TestDoubleOUPSD:
    """Double-OU PSD 테스트"""
    
    def test_psd_theory(self):
        """PSD 이론 계산 테스트"""
        tau_c1 = 0.05e-6
        tau_c2 = 2.0e-6
        B_rms1 = 4.0e-6
        B_rms2 = 3.0e-6
        
        f = np.logspace(4, 8, 100)  # 10 kHz to 100 MHz
        
        S_total, S1, S2 = compute_double_OU_PSD_theory(
            f, tau_c1, tau_c2, B_rms1, B_rms2
        )
        
        # Total should be sum of components
        np.testing.assert_allclose(S_total, S1 + S2, rtol=1e-10)
        
        # PSD should be positive
        assert np.all(S_total > 0)
        assert np.all(S1 > 0)
        assert np.all(S2 > 0)
    
    def test_psd_frequency_scaling(self):
        """PSD 주파수 스케일링 검증"""
        tau_c1 = 0.05e-6
        tau_c2 = 2.0e-6
        B_rms1 = 4.0e-6
        B_rms2 = 3.0e-6
        
        f = np.logspace(4, 8, 100)
        
        S_total, S1, S2 = compute_double_OU_PSD_theory(
            f, tau_c1, tau_c2, B_rms1, B_rms2
        )
        
        # At low frequency, PSD should be approximately constant
        # At high frequency, PSD should decay as 1/f²
        f_low = f[f < 1e6]
        f_high = f[f > 1e7]
        
        if len(f_low) > 1:
            S_low = S_total[f < 1e6]
            # Low frequency PSD should be relatively flat
            assert np.std(S_low) / np.mean(S_low) < 0.5
        
        if len(f_high) > 1:
            S_high = S_total[f > 1e7]
            # High frequency PSD should decrease
            assert S_high[-1] < S_high[0]

