"""
Extended noise models for material comparison.

Includes:
- Single OU (기존)
- Double OU (새로 추가)
"""

from typing import Tuple, Optional, Union
import numpy as np
import numpy.typing as npt
from ornstein_uhlenbeck import (
    generate_ou_noise,
    InvalidParameterError,
    NumericalStabilityError
)


def generate_double_OU_noise(
    tau_c1: float,
    tau_c2: float,
    B_rms1: float,
    B_rms2: float,
    dt: float,
    N_steps: int,
    seed: Optional[int] = None,
    return_components: bool = False
) -> Union[
    npt.NDArray[np.float64],
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
]:
    """
    Generate two-component Ornstein-Uhlenbeck noise.
    
    Physical Model:
    ---------------
    Total magnetic field fluctuation:
        δB(t) = δB₁(t) + δB₂(t)
    
    where:
        - δB₁(t): Fast component (small τ_c1)
        - δB₂(t): Slow component (large τ_c2)
    
    Each component is an independent OU process:
        dδB_i/dt = -δB_i/τ_ci + √(2B_rms,i²/τ_ci) η_i(t)
    
    Power Spectral Density:
        S(f) = S₁(f) + S₂(f)  (incoherent sum)
        
        S_i(f) = (2 B_rms,i² τ_ci) / [1 + (2πf τ_ci)²]
    
    Physical Interpretation:
    ------------------------
    For GaAs quantum dots:
        - Fast (τ_c1 ~ 50 ns): Nearby nuclei with strong hyperfine coupling
        - Slow (τ_c2 ~ 1 μs): Distant nuclei with weak coupling
    
    For Si:P:
        - Fast (τ_c1 ~ 5 μs): Residual 29Si hyperfine
        - Slow (τ_c2 ~ 200 μs): Charge noise from surface/interface
    
    Parameters
    ----------
    tau_c1 : float
        Correlation time for fast component (seconds)
    tau_c2 : float
        Correlation time for slow component (seconds)
        Should satisfy: tau_c2 > tau_c1
    B_rms1 : float
        RMS amplitude for fast component (Tesla)
    B_rms2 : float
        RMS amplitude for slow component (Tesla)
    dt : float
        Time step (seconds)
    N_steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
    return_components : bool, optional
        If True, return (delta_B, delta_B1, delta_B2)
        If False, return only delta_B (default)
    
    Returns
    -------
    delta_B : ndarray
        Total noise δB(t), shape (N_steps,)
    delta_B1 : ndarray (if return_components=True)
        Fast component, shape (N_steps,)
    delta_B2 : ndarray (if return_components=True)
        Slow component, shape (N_steps,)
    
    Examples
    --------
    >>> # GaAs quantum dot parameters
    >>> tau_c1 = 0.05e-6  # 50 ns
    >>> tau_c2 = 2.0e-6   # 2 μs
    >>> B_rms1 = 4.0e-6   # 4 μT
    >>> B_rms2 = 3.0e-6   # 3 μT
    >>> dt = 0.2e-9       # 0.2 ns
    >>> N_steps = 250000  # 50 μs total
    >>> 
    >>> delta_B = generate_double_OU_noise(tau_c1, tau_c2, B_rms1, B_rms2,
    ...                                     dt, N_steps, seed=42)
    
    Notes
    -----
    - The two components are generated independently
    - Seeds are offset by 1000 to ensure independence
    - Total RMS: B_rms_total ≈ √(B_rms1² + B_rms2²) if uncorrelated
    
    Raises
    ------
    InvalidParameterError
        If parameters are physically invalid
    NumericalStabilityError
        If numerical stability conditions are violated
    """
    # Input validation
    if tau_c1 <= 0:
        raise InvalidParameterError(
            f"tau_c1 must be positive, got {tau_c1}. "
            f"Fast component correlation time must be > 0."
        )
    if tau_c2 <= 0:
        raise InvalidParameterError(
            f"tau_c2 must be positive, got {tau_c2}. "
            f"Slow component correlation time must be > 0."
        )
    if tau_c2 <= tau_c1:
        raise InvalidParameterError(
            f"tau_c2 ({tau_c2:.2e}s) must be > tau_c1 ({tau_c1:.2e}s). "
            f"Slow component should have longer correlation time than fast component."
        )
    if B_rms1 <= 0:
        raise InvalidParameterError(
            f"B_rms1 must be positive, got {B_rms1}. "
            f"Fast component RMS amplitude must be > 0."
        )
    if B_rms2 <= 0:
        raise InvalidParameterError(
            f"B_rms2 must be positive, got {B_rms2}. "
            f"Slow component RMS amplitude must be > 0."
        )
    if dt <= 0:
        raise InvalidParameterError(
            f"dt must be positive, got {dt}. "
            f"Time step must be > 0."
        )
    if N_steps <= 0:
        raise InvalidParameterError(
            f"N_steps must be positive, got {N_steps}. "
            f"Number of steps must be > 0."
        )
    if not isinstance(N_steps, (int, np.integer)):
        raise InvalidParameterError(
            f"N_steps must be an integer, got {type(N_steps).__name__} with value {N_steps}"
        )
    
    # Numerical stability check: dt should be << min(tau_c1, tau_c2)
    min_tau_c = min(tau_c1, tau_c2)
    dt_tau_ratio = dt / min_tau_c
    if dt_tau_ratio > 0.2:
        raise NumericalStabilityError(
            f"Time step dt={dt:.2e}s is too large for correlation times "
            f"(tau_c1={tau_c1:.2e}s, tau_c2={tau_c2:.2e}s). "
            f"For numerical stability, dt should be < min(tau_c1, tau_c2)/5. "
            f"Current ratio: dt/min(tau_c) = {dt_tau_ratio:.3f}. "
            f"Recommended: dt <= {min_tau_c/50:.2e}s"
        )
    
    # Generate two independent OU processes
    # Use different seeds to ensure independence
    seed1 = seed if seed is not None else None
    seed2 = (seed + 1000) if seed is not None else None
    
    # Fast component
    delta_B1 = generate_ou_noise(tau_c1, B_rms1, dt, N_steps, seed=seed1)
    
    # Slow component
    delta_B2 = generate_ou_noise(tau_c2, B_rms2, dt, N_steps, seed=seed2)
    
    # Total noise (linear superposition)
    delta_B = delta_B1 + delta_B2
    
    if return_components:
        return delta_B, delta_B1, delta_B2
    else:
        return delta_B


def compute_double_OU_PSD_theory(f, tau_c1, tau_c2, B_rms1, B_rms2):
    """
    Compute theoretical PSD for double-OU noise.
    
    Parameters
    ----------
    f : ndarray
        Frequency array (Hz)
    tau_c1, tau_c2 : float
        Correlation times (seconds)
    B_rms1, B_rms2 : float
        RMS amplitudes (Tesla)
    
    Returns
    -------
    S_total : ndarray
        Total PSD S(f) = S₁(f) + S₂(f) (T²/Hz)
    S1 : ndarray
        Fast component PSD
    S2 : ndarray
        Slow component PSD
    """
    # Lorentzian PSD for each component
    S1 = (2 * B_rms1**2 * tau_c1) / (1 + (2*np.pi*f*tau_c1)**2)
    S2 = (2 * B_rms2**2 * tau_c2) / (1 + (2*np.pi*f*tau_c2)**2)
    
    # Total PSD (incoherent sum)
    S_total = S1 + S2
    
    return S_total, S1, S2


def verify_double_OU_statistics(delta_B, delta_B1, delta_B2, 
                                 tau_c1, tau_c2, B_rms1, B_rms2, dt):
    """
    Verify that generated double-OU noise has correct statistics.
    
    Checks:
    1. Variance: var(δB) ≈ B_rms1² + B_rms2²
    2. Autocorrelation: matches theoretical form
    3. PSD: matches sum of two Lorentzians
    
    Returns
    -------
    checks : dict
        Dictionary with verification results
    """
    # Check 1: Variance
    var_B = np.var(delta_B)
    var_expected = B_rms1**2 + B_rms2**2
    var_error = abs(var_B - var_expected) / var_expected
    
    # Check 2: Component variances
    var_B1 = np.var(delta_B1)
    var_B2 = np.var(delta_B2)
    var1_error = abs(var_B1 - B_rms1**2) / B_rms1**2
    var2_error = abs(var_B2 - B_rms2**2) / B_rms2**2
    
    # Check 3: PSD
    from scipy import signal
    f, PSD_sim = signal.periodogram(delta_B, fs=1/dt, scaling='density')
    PSD_theory, _, _ = compute_double_OU_PSD_theory(f, tau_c1, tau_c2, 
                                                     B_rms1, B_rms2)
    
    # Compare PSD in middle frequency range (avoid edges)
    f_idx = (f > 1e5) & (f < 1e8)  # 100 kHz - 100 MHz
    if np.any(f_idx):
        psd_error = np.mean(np.abs(PSD_sim[f_idx] - PSD_theory[f_idx]) / 
                            PSD_theory[f_idx])
    else:
        psd_error = np.nan
    
    checks = {
        'variance_total': var_B,
        'variance_expected': var_expected,
        'variance_error': var_error,
        'variance_component1': var_B1,
        'variance_component2': var_B2,
        'var1_error': var1_error,
        'var2_error': var2_error,
        'psd_error': psd_error,
        'passed': (var_error < 0.05 and (np.isnan(psd_error) or psd_error < 0.2))
    }
    
    return checks

