"""
Phase accumulation calculations for spin decoherence.

This module computes the accumulated phase from magnetic field fluctuations.
"""

import numpy as np
import numpy.typing as npt


def compute_phase_accumulation(
    delta_B: npt.NDArray[np.float64],
    gamma_e: float,
    dt: float
) -> npt.NDArray[np.float64]:
    """
    Compute accumulated phase from magnetic field fluctuations.
    
    CRITICAL FIX: Properly handle phi(t=0) = 0 without losing first time step.
    
    Phase at time t[k] = k*dt:
    - phi[0] = 0 (at t=0)
    - phi[1] = ∫₀^dt δω(t') dt' ≈ δω[0] * dt
    - phi[k] = ∫₀^(k*dt) δω(t') dt' = Σᵢ₌₀ᵏ⁻¹ δω[i] * dt
    
    Parameters
    ----------
    delta_B : ndarray of float64
        Array of delta_B_z(t_k) values (length N_steps), units: Tesla
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    dt : float
        Time step (seconds)
        
    Returns
    -------
    phi : ndarray of float64
        Accumulated phase phi(t_k) = int_0^t gamma_e * delta_B_z(t') dt'
        Length: N_steps (phi[0] = 0 at t=0, phi[k] at t=k*dt), units: radians
    
    Raises
    ------
    TypeError
        If inputs have incorrect types
    ValueError
        If inputs have incorrect shapes or values
    """
    # Type validation
    if not isinstance(delta_B, np.ndarray):
        raise TypeError(f"delta_B must be ndarray, got {type(delta_B)}")
    if not isinstance(gamma_e, (int, float, np.number)):
        raise TypeError(f"gamma_e must be numeric, got {type(gamma_e)}")
    if not isinstance(dt, (int, float, np.number)):
        raise TypeError(f"dt must be numeric, got {type(dt)}")
    
    # Shape validation
    if delta_B.ndim != 1:
        raise ValueError(f"delta_B must be 1D array, got shape {delta_B.shape}")
    
    # Value validation
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    
    # Convert to float64 if needed
    delta_B = np.asarray(delta_B, dtype=np.float64)
    
    # Instantaneous frequency fluctuation (ensure float64)
    delta_omega: npt.NDArray[np.float64] = (gamma_e * delta_B).astype(np.float64)
    
    # FIXED: Proper phase accumulation
    # phi[k] = Σᵢ₌₀ᵏ⁻¹ δω[i] * dt for k > 0, phi[0] = 0
    # This correctly includes all time steps from t=0 to t=k*dt
    phi: npt.NDArray[np.float64] = np.zeros(len(delta_omega), dtype=np.float64)
    if len(delta_omega) > 0:
        # phi[1:] = cumulative sum of delta_omega[0:k] * dt for k=1,2,...
        phi[1:] = np.cumsum(delta_omega[:-1] * dt, dtype=np.float64)
        # Now phi[k] = Σᵢ₌₀ᵏ⁻¹ δω[i] * dt, which is correct
    
    return phi

