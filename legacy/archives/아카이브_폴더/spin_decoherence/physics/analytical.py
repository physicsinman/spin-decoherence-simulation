"""
Analytical solutions for spin decoherence.

This module provides closed-form analytical expressions for coherence functions
and related quantities, useful for comparison with numerical simulations.
"""

import numpy as np


def analytical_ou_coherence(t, gamma_e, B_rms, tau_c):
    """
    Analytical coherence function for OU noise using cumulant expansion.
    
    E(t) = exp[-Δω²τ_c² (e^(-t/τ_c) + t/τ_c - 1)]
    
    This is exact for Gaussian phase noise from OU magnetic field fluctuations.
    
    Parameters
    ----------
    t : ndarray
        Time array (seconds)
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    B_rms : float
        RMS noise amplitude (Tesla)
    tau_c : float
        Correlation time (seconds)
        
    Returns
    -------
    E : ndarray
        Analytical coherence function |E(t)|
    """
    Delta_omega = gamma_e * B_rms
    Delta_omega_sq = Delta_omega**2
    tau_c_sq = tau_c**2
    
    # Cumulant expansion result: χ(t) = Δω²τ_c² [exp(-t/τ_c) + t/τ_c - 1]
    chi = Delta_omega_sq * tau_c_sq * (
        np.exp(-t / tau_c) + t / tau_c - 1.0
    )
    
    # CRITICAL: Prevent underflow by clipping chi
    # exp(-700) ≈ 10^-304 (near machine epsilon for float64)
    # Clip chi to [0, 700] to ensure exp(-chi) is numerically stable
    chi_clipped = np.clip(chi, 0.0, 700.0)  # chi is always ≥ 0
    
    # Use log-domain calculation for numerical stability
    logE = -chi_clipped
    E = np.exp(logE)
    
    return E


def analytical_hahn_echo_coherence(tau_list, gamma_e, B_rms, tau_c):
    """
    Analytical Hahn echo coherence for OU noise using closed-form expression.
    
    χ_echo(2τ) = Δω²τ_c² [2τ/τ_c - 3 + 4e^(-τ/τ_c) - e^(-2τ/τ_c)]
    E_echo(2τ) = exp[-χ_echo(2τ)]
    
    Parameters
    ----------
    tau_list : ndarray
        Echo delays τ (seconds)
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    B_rms : float
        RMS noise amplitude (Tesla)
    tau_c : float
        Correlation time (seconds)
        
    Returns
    -------
    E_echo : ndarray
        Analytical coherence |E_echo(2τ)|
    """
    Delta_omega = gamma_e * B_rms
    Delta_omega_sq = Delta_omega**2
    tau_c_sq = tau_c**2
    
    tau_list = np.array(tau_list)
    
    # Closed-form expression for χ_echo(2τ)
    chi_echo = Delta_omega_sq * tau_c_sq * (
        2 * tau_list / tau_c - 3 + 
        4 * np.exp(-tau_list / tau_c) - 
        np.exp(-2 * tau_list / tau_c)
    )
    
    # CRITICAL: Prevent underflow by clipping chi_echo
    # exp(-700) ≈ 10^-304 (near machine epsilon for float64)
    chi_echo_clipped = np.clip(chi_echo, 0.0, 700.0)
    
    # Use log-domain calculation for numerical stability
    logE_echo = -chi_echo_clipped
    E_echo = np.exp(logE_echo)
    
    return E_echo


def theoretical_T2_motional_narrowing(gamma_e, B_rms, tau_c):
    """
    Theoretical T_2 in motional narrowing regime.
    
    T_2 = 1 / (Δω² τ_c)
    
    where Δω = γ_e B_rms
    
    This is valid when τ_c << T_2 (fast noise limit).
    
    Parameters
    ----------
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    B_rms : float
        RMS noise amplitude (Tesla)
    tau_c : float
        Correlation time (seconds)
        
    Returns
    -------
    T2 : float
        Coherence time (seconds)
    """
    return 1.0 / ((gamma_e * B_rms)**2 * tau_c)


def analytical_T2_echo(tau_c, gamma_e, B_rms, n_points=1000):
    """
    Calculate analytical T2_echo from Hahn echo coherence.
    
    Finds the time 2τ where E_echo(2τ) = 1/e (i.e., χ_echo = 1).
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    B_rms : float
        RMS noise amplitude (Tesla)
    n_points : int
        Number of points for numerical search
        
    Returns
    -------
    T2_echo : float
        Analytical echo coherence time (seconds)
    """
    Delta_omega = gamma_e * B_rms
    Delta_omega_sq = Delta_omega**2
    tau_c_sq = tau_c**2
    xi = Delta_omega * tau_c
    
    # For MN regime (xi << 1), echo ≈ FID
    if xi < 0.1:
        return 1.0 / (Delta_omega_sq * tau_c)
    
    # For QS regime (xi >> 1), use theoretical echo gain
    # In QS regime, echo gain is theoretically ~2 (T2_echo ≈ 2 × T2_fid)
    # This is because echo refocuses static noise, giving ~2× improvement
    if xi > 20.0:
        # QS regime: T2_fid ≈ sqrt(2) / Delta_omega
        T2_fid = np.sqrt(2.0) / Delta_omega
        # Echo gain in QS regime is approximately 2
        # Use gain = 2 for all QS regime (20 < xi)
        return T2_fid * 2.0
    
    # For intermediate regime (0.1 < xi < 20), solve numerically
    # Find 2τ where χ_echo(2τ) = 1
    # χ_echo(2τ) = Δω²τ_c² [2τ/τ_c - 3 + 4e^(-τ/τ_c) - e^(-2τ/τ_c)]
    
    # Search range: from small tau to large tau
    tau_min = tau_c * 0.01
    tau_max = tau_c * 100.0
    
    # Binary search for solution
    for _ in range(100):
        tau_mid = 0.5 * (tau_min + tau_max)
        chi_echo = Delta_omega_sq * tau_c_sq * (
            2 * tau_mid / tau_c - 3 + 
            4 * np.exp(-tau_mid / tau_c) - 
            np.exp(-2 * tau_mid / tau_c)
        )
        
        if chi_echo > 1.0:
            tau_max = tau_mid
        else:
            tau_min = tau_mid
        
        if tau_max - tau_min < tau_c * 1e-6:
            break
    
    # T2_echo = 2τ (echo time is 2τ)
    T2_echo = 2.0 * tau_max
    
    return T2_echo

