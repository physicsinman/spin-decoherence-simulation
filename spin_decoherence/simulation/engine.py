"""
Core simulation engine utilities.

This module provides helper functions for simulation parameter estimation
and optimization.
"""

import numpy as np
from spin_decoherence.config.constants import CONSTANTS
from spin_decoherence.physics.analytical import analytical_ou_coherence


def _solve_T2_exact(tau_c, delta_omega):
    """Solve for T2 from the analytical OU coherence by bisection."""
    if delta_omega <= 0:
        return np.inf

    def coherence_argument(t):
        return delta_omega**2 * tau_c**2 * (
            np.exp(-t / tau_c) + t / tau_c - 1.0
        ) - 1.0

    t_low = 0.0
    t_high = max(10.0 * tau_c, 10.0 / delta_omega)

    # Increase upper bound until the coherence argument becomes positive
    while coherence_argument(t_high) < 0:
        t_high *= 2.0
        if t_high > 1e3 / delta_omega:
            break

    # Bisection search for root
    for _ in range(80):
        t_mid = 0.5 * (t_low + t_high)
        value = coherence_argument(t_mid)
        if value > 0:
            t_high = t_mid
        else:
            t_low = t_mid

    return t_high


def estimate_characteristic_T2(tau_c, gamma_e, B_rms):
    """
    Estimate characteristic T2 for OU noise across regimes.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    B_rms : float
        RMS noise amplitude (Tesla)
        
    Returns
    -------
    T2 : float
        Estimated coherence time (seconds)
    """
    delta_omega = abs(gamma_e * B_rms)
    if delta_omega == 0:
        return np.inf

    xi = delta_omega * tau_c
    mn_T2 = 1.0 / (delta_omega**2 * tau_c)
    static_T2 = np.sqrt(2.0) / delta_omega

    if xi < 0.05:
        return mn_T2
    if xi > 20.0:
        return static_T2

    return _solve_T2_exact(tau_c, delta_omega)


def get_dimensionless_tau_range(tau_c, n_points=28, upsilon_min=0.05, upsilon_max=0.8, 
                                 dt=None, T_max=None):
    """
    Get dimensionless tau range for Hahn echo: υ = τ/τ_c.
    
    This ensures consistent scanning across different tau_c values.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    n_points : int
        Number of tau points
    upsilon_min : float
        Minimum dimensionless delay υ_min = τ_min/τ_c
    upsilon_max : float
        Maximum dimensionless delay υ_max = τ_max/τ_c
    dt : float, optional
        Time step (seconds). Used to enforce minimum tau.
    T_max : float, optional
        Maximum simulation time (seconds). Used to cap maximum tau.
        
    Returns
    -------
    tau_list : ndarray
        Optimal tau range (seconds)
    """
    tau_min = upsilon_min * tau_c
    tau_max = upsilon_max * tau_c
    
    # Enforce practical bounds
    if dt is not None:
        tau_min = max(tau_min, 10.0 * dt)
    if T_max is not None:
        # CRITICAL FIX: Hahn echo is measured at time 2*tau
        # So we need 2*tau_max <= T_max, i.e., tau_max <= T_max/2
        # Previous limit of 0.4*T_max was too restrictive
        tau_max = min(tau_max, 0.5 * T_max)  # 2*tau_max <= T_max
    
    if tau_max <= tau_min:
        tau_max = tau_min * 1.5
    
    tau_list = np.logspace(np.log10(tau_min), np.log10(tau_max), n_points)
    
    return tau_list


def get_optimal_tau_range(tau_c, n_points=30, factor_min=0.1, factor_max=10,
                          dt=None, T_max=None, gamma_e=None, B_rms=None):
    """
    Get optimal tau range for Hahn echo based on correlation time.
    
    DEPRECATED: Use get_dimensionless_tau_range instead for consistent
    dimensionless scanning.
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    n_points : int
        Number of tau points
    factor_min : float
        tau_min = factor_min * tau_c
    factor_max : float
        tau_max = factor_max * tau_c
        
    Returns
    -------
    tau_list : ndarray
        Optimal tau range (seconds)
    """
    if gamma_e is None:
        gamma_e = CONSTANTS.GAMMA_E
    if dt is None or T_max is None:
        from spin_decoherence.config.simulation import SimulationConfig
        from spin_decoherence.config.units import Units
        default_config = SimulationConfig(
            B_rms=0.57e-6,  # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration
            tau_c_range=(Units.us_to_s(0.01), Units.us_to_s(10.0)),
        )
        if dt is None:
            dt = default_config.dt
        if T_max is None:
            T_max = default_config.T_max_echo
        if B_rms is None:
            B_rms = default_config.B_rms

    tau_min = factor_min * tau_c
    tau_max = factor_max * tau_c

    T2_est = estimate_characteristic_T2(tau_c, gamma_e, B_rms)

    # Determine practical time window for echo measurement (2τ)
    max_time = min(6.0 * T2_est, T_max)
    min_time = max(0.02 * T2_est, 20.0 * dt)

    tau_min = max(tau_min, 0.5 * min_time)
    tau_max = max(tau_max, 0.5 * max_time)

    # Enforce absolute bounds
    tau_min = max(tau_min, 0.01e-6)
    tau_max = min(tau_max, 0.5 * T_max)

    if tau_max <= tau_min:
        tau_max = tau_min * 1.5

    tau_list = np.logspace(np.log10(tau_min), np.log10(tau_max), n_points)

    return tau_list

