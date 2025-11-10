"""
Coherence function calculation for spin decoherence.

This module computes the phase accumulation and coherence function E(t)
for a spin subject to stochastic magnetic field fluctuations.
"""

from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
import numpy.typing as npt
from ornstein_uhlenbeck import generate_ou_noise
from noise_models import generate_double_OU_noise
from config import CONSTANTS


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


def compute_trajectory_coherence(
    tau_c: float,
    B_rms: float,
    gamma_e: float,
    dt: float,
    N_steps: int,
    seed: Optional[int] = None
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """
    Compute coherence for a single noise realization (trajectory).
    
    Note: Pure dephasing simulation does not require B_0 (static field).
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    B_rms : float
        RMS amplitude of noise (Tesla)
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    dt : float
        Time step (seconds)
    N_steps : int
        Number of time steps
    seed : int, optional
        Random seed for this trajectory
        
    Returns
    -------
    E_traj : ndarray
        Complex coherence E_traj(t_k) = exp(i * phi_k)
    t : ndarray
        Time array t_k = k * dt
    """
    # Generate OU noise
    delta_B = generate_ou_noise(tau_c, B_rms, dt, N_steps, seed=seed)
    
    # DEBUG: Verify OU noise scaling (for first few trajectories to catch issues)
    # Check only first trajectory to avoid excessive output
    should_check = (seed is not None and seed == 42)  # Only check first trajectory
    if should_check:
        emp_std = np.std(delta_B)
        emp_mean = np.mean(delta_B)
        print(f"[CHECK] OU noise (seed={seed}): emp std = {emp_std:.3e} T "
              f"(expected ~ {B_rms:.3e} T), mean = {emp_mean:.3e} T")
        if len(delta_B) > 1:
            rho_emp = np.corrcoef(delta_B[:-1], delta_B[1:])[0, 1]
            rho_th = np.exp(-dt / tau_c)
            print(f"[CHECK] OU autocorr: rho_emp = {rho_emp:.4f}, "
                  f"rho_th = {rho_th:.4f}, error = {abs(rho_emp - rho_th):.6f}")
    
    # Compute phase accumulation
    phi = compute_phase_accumulation(delta_B, gamma_e, dt)
    
    # Time array
    t = np.arange(N_steps) * dt
    
    # DEBUG: Check phase accumulation (for first trajectory and occasionally others)
    # NOTE: Single trajectory variance != ensemble variance
    # We can only check phase accumulation correctness, not ensemble variance
    if should_check:
        phi_std = np.std(phi)
        phi_var_time = np.var(phi)  # Variance across time (not ensemble variance!)
        
        # Exact formula for OU noise phase variance (ensemble average):
        # var(φ) = (γ_e B_rms)² τ_c² [exp(-t/τ_c) + t/τ_c - 1]
        # This is from the analytical coherence: E(t) = exp(-var(φ)/2)
        Delta_omega = gamma_e * B_rms
        Delta_omega_sq = Delta_omega**2
        tau_c_sq = tau_c**2
        
        # Compute exact ensemble variance at final time
        t_final = t[-1]
        phi_var_exact = Delta_omega_sq * tau_c_sq * (
            np.exp(-t_final / tau_c) + t_final / tau_c - 1.0
        )
        
        # Simplified formulas for comparison
        phi_var_mn = Delta_omega_sq * tau_c * t_final  # Motional narrowing (t >> tau_c)
        
        # For single trajectory, we can check:
        # 1. Phase accumulation is monotonic (phi increases/decreases smoothly)
        # 2. Phase magnitude is reasonable
        # 3. Coherence from single trajectory: |E| = 1 (always, since it's a single realization)
        
        # CRITICAL FIX: Use variance (var) consistently, not std
        # Coherence is related to variance: |E| ≈ exp(-var(φ)/2)
        # Avoid confusion between std and var in logs
        var_emp_time = phi_var_time  # Variance across time (single trajectory)
        var_th_exact = phi_var_exact  # Exact ensemble variance (theory)
        var_th_mn = phi_var_mn  # Motional narrowing approximation
        
        print(f"[CHECK] Phase variance (single traj, across time): var_emp = {var_emp_time:.3e}")
        print(f"[CHECK] Phase variance (ensemble theory, exact OU): var_th = {var_th_exact:.3e}")
        print(f"[CHECK] Phase variance (ensemble theory, MN approx): var_th_mn = {var_th_mn:.3e}")
        print(f"[CHECK] Phase: max|phi| = {np.abs(phi).max():.3e}")
        print(f"[CHECK] Note: Single trajectory variance ≠ ensemble variance")
        print(f"[CHECK]       Ensemble variance will be verified via coherence decay")
        
        # Check coherence: single trajectory always has |E| = 1
        E_final = np.exp(1j * phi[-1])
        # Coherence from variance: |E| = exp(-var(φ)/2)
        coherence_from_var = np.exp(-var_th_exact / 2)
        print(f"[CHECK] Coherence: |E(t_final)| (single traj) = {np.abs(E_final):.4f} (always 1)")
        print(f"[CHECK] Coherence: |⟨E⟩| (ensemble theory) = exp(-var_th/2) = {coherence_from_var:.4f}")
    
    # Coherence: E(t) = exp(i * phi(t))
    E_traj = np.exp(1j * phi)
    
    return E_traj, t


def _generate_noise(
    noise_model: str,
    params: Dict[str, float],
    dt: float,
    N_steps: int,
    seed: Optional[int]
) -> npt.NDArray[np.float64]:
    """
    Generate noise based on noise model type.
    
    Parameters
    ----------
    noise_model : str
        'OU' for single Ornstein-Uhlenbeck, 'Double_OU' for double OU
    params : dict
        For 'OU': {'tau_c': float, 'B_rms': float}
        For 'Double_OU': {'tau_c1': float, 'tau_c2': float, 'B_rms1': float, 'B_rms2': float}
    dt : float
        Time step (seconds)
    N_steps : int
        Number of time steps
    seed : int, optional
        Random seed
        
    Returns
    -------
    delta_B : ndarray
        Generated noise array
    """
    if noise_model == 'OU':
        return generate_ou_noise(params['tau_c'], params['B_rms'], dt, N_steps, seed=seed)
    elif noise_model == 'Double_OU':
        return generate_double_OU_noise(
            params['tau_c1'], params['tau_c2'],
            params['B_rms1'], params['B_rms2'],
            dt, N_steps, seed=seed
        )
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")


def _compute_echo_phase(
    delta_omega: npt.NDArray[np.float64],
    I_cumulative: npt.NDArray[np.float64],
    tau: float,
    dt: float,
    n: int
) -> float:
    """
    Compute echo phase for a given tau using cumulative integral.
    
    Hahn echo sequence: π/2 - τ - π - τ (total time = 2τ)
    Toggling function: y(t) = +1 for t < τ, -1 for τ ≤ t ≤ 2τ
    
    Parameters
    ----------
    delta_omega : ndarray
        Frequency fluctuations δω = γ_e * δB
    I_cumulative : ndarray
        Cumulative integral I[k] = ∫₀^(k*dt) δω dt'
    tau : float
        Echo delay τ (seconds)
    dt : float
        Time step (seconds)
    n : int
        Length of arrays
        
    Returns
    -------
    phase_integral : float
        Echo phase φ_echo(2τ) = pos_phase - neg_phase
    """
    # Calculate fractional indices
    k_tau = tau / dt
    i_tau = int(np.floor(k_tau))  # Integer part
    f_tau = k_tau - i_tau  # Fractional part (0 ≤ f_tau < 1)
    
    k_2tau = 2 * tau / dt
    i_2tau = int(np.floor(k_2tau))
    f_2tau = k_2tau - i_2tau
    
    # Ensure indices are within bounds
    i_tau = min(i_tau, n - 1)
    i_2tau = min(i_2tau, n - 1)
    
    # Positive phase: ∫₀^τ δω(t') dt'
    pos_phase = I_cumulative[i_tau] + (f_tau * delta_omega[i_tau] * dt if i_tau < n else 0.0)
    
    # Negative phase: ∫_τ^(2τ) δω(t') dt'
    neg_phase = 0.0
    if i_tau < n:
        if f_tau > 0:
            neg_phase += (1.0 - f_tau) * delta_omega[i_tau] * dt
        # Sum from i_tau+1 to i_2tau
        if i_tau + 1 < min(i_2tau, n):
            neg_phase += I_cumulative[min(i_2tau, n - 1)] - I_cumulative[i_tau + 1]
        # Fractional contribution at 2τ
        if i_2tau < n and f_2tau > 0:
            neg_phase += f_2tau * delta_omega[i_2tau] * dt
    elif i_2tau < n:
        # If i_tau >= n, just use I_cumulative[i_2tau] + fractional part
        neg_phase = I_cumulative[i_2tau] + (f_2tau * delta_omega[i_2tau] * dt if f_2tau > 0 else 0.0)
    
    # Total phase: φ_echo(2τ) = pos_phase - neg_phase
    return pos_phase - neg_phase


def _compute_coherence_core(
    noise_model: str,
    noise_params: Dict[str, float],
    gamma_e: float,
    dt: float,
    N_steps: int,
    M: int,
    sequence: str = 'FID',
    tau_list: Optional[Union[npt.NDArray[np.float64], List[float], Tuple[float, ...]]] = None,
    seed: Optional[int] = None,
    progress: bool = True,
    use_online: bool = True
) -> npt.NDArray[np.complex128]:
    """
    Core coherence computation logic shared by FID and Echo sequences.
    
    Parameters
    ----------
    noise_model : str
        'OU' or 'Double_OU'
    noise_params : dict
        Parameters for noise generation (see _generate_noise)
    gamma_e : float
        Electron gyromagnetic ratio
    dt : float
        Time step (seconds)
    N_steps : int
        Number of time steps
    M : int
        Number of trajectories
    sequence : str
        'FID' or 'Echo'
    tau_list : array-like, optional
        List of echo delays τ (required for 'Echo')
    seed : int, optional
        Base random seed
    progress : bool
        Whether to show progress bar
    use_online : bool
        If True, use memory-efficient online algorithm (Welford's method).
        If False, store all trajectories in memory (for bootstrap analysis).
        
    Returns
    -------
    For FID:
        E_all : ndarray, shape (M, N_steps)
            Coherence for each trajectory (only if use_online=False)
        OR (E_mean, E_M2) : tuple
            Online statistics (only if use_online=True)
    For Echo:
        E_echo_all : ndarray, shape (M, len(tau_list))
            Echo coherence for each trajectory
    """
    from tqdm import tqdm
    
    # Initialize arrays
    if sequence == 'FID':
        if use_online:
            # Welford's online algorithm: only store mean and M2 (for variance)
            E_mean = np.zeros(N_steps, dtype=complex)
            E_abs_mean = np.zeros(N_steps, dtype=float)  # Mean of |E|
            E_M2 = np.zeros(N_steps, dtype=float)  # For variance of |E|
        else:
            # Store all trajectories (for bootstrap analysis)
            E_all = np.zeros((M, N_steps), dtype=complex)
    elif sequence == 'Echo':
        if tau_list is None:
            raise ValueError("tau_list is required for Echo sequence")
        tau_list = np.array(tau_list)
        E_echo_all = []
    else:
        raise ValueError(f"Unknown sequence: {sequence}")
    
    # Progress bar
    if progress:
        desc = f"Computing {sequence} ({noise_model})"
        iterator = tqdm(range(M), desc=desc)
    else:
        iterator = range(M)
    
    # Iterate over trajectories
    for m in iterator:
        traj_seed = seed + m if seed is not None else None
        
        # Generate noise
        delta_B = _generate_noise(noise_model, noise_params, dt, N_steps, traj_seed)
        delta_omega = (gamma_e * delta_B).astype(np.float64)
        
        if sequence == 'FID':
            # Compute phase accumulation
            phi = compute_phase_accumulation(delta_B, gamma_e, dt)
            # Coherence: E(t) = exp(i * phi(t))
            E_traj = np.exp(1j * phi)
            
            if use_online:
                # Welford's online algorithm for mean and variance
                # Update mean of complex E: E_mean_new = E_mean_old + (E_traj - E_mean_old) / (m + 1)
                delta = E_traj - E_mean
                E_mean += delta / (m + 1)
                
                # Update mean and variance of |E| using Welford's algorithm
                E_abs_traj = np.abs(E_traj)
                delta_abs = E_abs_traj - E_abs_mean
                E_abs_mean += delta_abs / (m + 1)
                delta2_abs = E_abs_traj - E_abs_mean
                E_M2 += delta_abs * delta2_abs
            else:
                E_all[m] = E_traj
            
        elif sequence == 'Echo':
            # OPTIMIZATION: Compute cumulative integral I(t) = ∫₀^t δω(t') dt' once
            # This allows O(1) evaluation for each tau instead of O(N) per tau
            I_cumulative = np.zeros(N_steps, dtype=np.float64)
            if N_steps > 1:
                I_cumulative[1:] = np.cumsum(delta_omega[:-1] * dt, dtype=np.float64)
            
            # Compute echo phase for each tau
            E_echo_traj = []
            n = len(delta_omega)
            for tau in tau_list:
                phase_integral = _compute_echo_phase(delta_omega, I_cumulative, tau, dt, n)
                E_echo_traj.append(np.exp(1j * phase_integral))
            
            E_echo_all.append(E_echo_traj)
    
    if sequence == 'FID':
        if use_online:
            return E_mean, E_abs_mean, E_M2
        else:
            return E_all
    else:  # Echo
        return np.array(E_echo_all)  # Shape: (M, len(tau_list))


def compute_ensemble_coherence(
    tau_c: float,
    B_rms: float,
    gamma_e: float,
    dt: float,
    T_max: float,
    M: int,
    seed: Optional[int] = None,
    progress: bool = True,
    use_online: bool = True
) -> Tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Compute ensemble-averaged coherence function E(t).
    
    Note: Pure dephasing simulation does not require B_0 (static field).
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    B_rms : float
        RMS amplitude of noise (Tesla)
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    dt : float
        Time step (seconds)
    T_max : float
        Maximum simulation time (seconds)
    M : int
        Number of independent realizations
    seed : int, optional
        Base random seed (each trajectory gets seed + m)
    progress : bool
        Whether to show progress bar
    use_online : bool
        If True, use memory-efficient online algorithm (default: True).
        If False, store all trajectories in memory (for bootstrap analysis).
        Memory usage: O(N_steps) vs O(M * N_steps)
        
    Returns
    -------
    E : ndarray
        Ensemble-averaged complex coherence E(t_k)
    E_abs : ndarray
        Magnitude of ensemble-averaged coherence |E(t_k)|
    E_se : ndarray
        Standard error of |E(t_k)|
    t : ndarray
        Time array

    E_abs_all : ndarray
        Array of |E| for each trajectory (for bootstrap analysis).
        If use_online=True, returns empty array (shape (0, N_steps)).
        If use_online=False, returns full array (shape (M, N_steps)).
    """
    N_steps = int(T_max / dt) + 1
    
    # Use core coherence computation
    noise_params = {'tau_c': tau_c, 'B_rms': B_rms}
    
    if use_online:
        # Memory-efficient online algorithm (Welford's method)
        E_mean, E_abs_mean, E_M2 = _compute_coherence_core(
            'OU', noise_params, gamma_e, dt, N_steps, M,
            sequence='FID', seed=seed, progress=progress, use_online=True
        )
        
        # Time array
        t = np.arange(N_steps) * dt
        
        # Ensemble average of complex coherence
        E = E_mean
        
        # Magnitude of ensemble-averaged coherence
        # IMPORTANT: |⟨E⟩| not ⟨|E|⟩
        E_abs = np.abs(E)
        
        # Variance and standard error from Welford's algorithm
        E_var = E_M2 / M if M > 1 else np.zeros(N_steps)
        E_se = np.sqrt(E_var / M)
        
        # Return empty array for E_abs_all when using online algorithm
        E_abs_all = np.zeros((0, N_steps), dtype=float)
    else:
        # Store all trajectories (for bootstrap analysis)
        E_all = _compute_coherence_core(
            'OU', noise_params, gamma_e, dt, N_steps, M,
            sequence='FID', seed=seed, progress=progress, use_online=False
        )
        
        # Time array
        t = np.arange(N_steps) * dt
        
        # Ensemble average of complex coherence
        E = np.mean(E_all, axis=0)
        
        # Magnitude of ensemble-averaged coherence
        # IMPORTANT: |⟨E⟩| not ⟨|E|⟩
        E_abs = np.abs(E)
        
        # Standard error for |E| (magnitude)
        # Calculate |E| for each trajectory for error estimation
        E_abs_all = np.abs(E_all)  # Shape: (M, N_steps)
        
        # Standard error of the mean for |E|
        E_se = np.std(E_abs_all, axis=0, ddof=1) / np.sqrt(M)
    
    return E, E_abs, E_se, t, E_abs_all


def compute_hahn_echo_coherence(
    tau_c: float,
    B_rms: float,
    gamma_e: float,
    dt: float,
    tau_list: Union[npt.NDArray[np.float64], List[float], Tuple[float, ...]],
    M: int,
    seed: Optional[int] = None,
    progress: bool = True
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Compute Hahn echo coherence E_echo(2τ) for a list of echo delays τ.
    
    Hahn echo sequence: π/2 - τ - π - τ (total time = 2τ)
    Toggling function: y(t) = +1 for t < τ, -1 for τ ≤ t ≤ 2τ
    
    Note: Pure dephasing simulation does not require B_0 (static field).
    
    Parameters
    ----------
    tau_c : float
        Correlation time (seconds)
    B_rms : float
        RMS noise amplitude (Tesla)
    gamma_e : float
        Electron gyromagnetic ratio
    dt : float
        Time step (seconds)
    tau_list : array-like
        List of echo delays τ (seconds)
    M : int
        Number of trajectories
    seed : int, optional
        Random seed for reproducibility
    progress : bool
        Whether to show progress bar
        
    Returns
    -------
    tau_echo : ndarray
        Echo times 2τ (seconds)
    E_echo : ndarray
        Complex coherence E_echo(2τ)
    E_echo_abs : ndarray
        Magnitude |E_echo(2τ)|
    E_echo_se : ndarray
        Standard error of |E_echo|
    E_echo_abs_all : ndarray
        Array of |E_echo| for each trajectory (for bootstrap analysis)
    """
    tau_list = np.array(tau_list)
    tau_max = tau_list.max()
    T_max = 2 * tau_max
    N_steps = int(T_max / dt) + 1
    
    if progress:
        print(f"  Computing Hahn echo: {len(tau_list)} tau values, {M} trajectories, {N_steps} time steps")
    
    # Use core coherence computation
    noise_params = {'tau_c': tau_c, 'B_rms': B_rms}
    E_echo_all = _compute_coherence_core(
        'OU', noise_params, gamma_e, dt, N_steps, M,
        sequence='Echo', tau_list=tau_list, seed=seed, progress=progress
    )
    
    tau_echo = 2 * tau_list
    
    # Ensemble average
    E_echo = np.mean(E_echo_all, axis=0)
    E_echo_abs = np.abs(E_echo)
    
    # Standard error of |E_echo|
    E_echo_abs_all = np.abs(E_echo_all)
    E_echo_se = np.std(E_echo_abs_all, axis=0, ddof=1) / np.sqrt(M)
    
    return tau_echo, E_echo, E_echo_abs, E_echo_se, E_echo_abs_all


def compute_ensemble_coherence_double_OU(
    tau_c1: float,
    tau_c2: float,
    B_rms1: float,
    B_rms2: float,
    gamma_e: float,
    dt: float,
    T_max: float,
    M: int,
    seed: Optional[int] = None,
    progress: bool = True,
    use_online: bool = True
) -> Tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Compute ensemble-averaged coherence function E(t) for Double-OU noise.
    
    Same as compute_ensemble_coherence() but with Double-OU noise model.
    
    Physical Model:
    ---------------
    Total magnetic field fluctuation: δB(t) = δB₁(t) + δB₂(t)
    where δB₁ and δB₂ are independent OU processes with correlation times
    tau_c1 (fast) and tau_c2 (slow), respectively.
    
    Note: Pure dephasing simulation does not require B_0 (static field).
    
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
    gamma_e : float
        Electron gyromagnetic ratio (rad·s⁻¹·T⁻¹)
    dt : float
        Time step (seconds)
    T_max : float
        Maximum simulation time (seconds)
    M : int
        Number of independent realizations
    seed : int, optional
        Base random seed (each trajectory gets seed + m)
    progress : bool
        Whether to show progress bar
        
    Returns
    -------
    E : ndarray
        Ensemble-averaged complex coherence E(t_k)
    E_abs : ndarray
        Magnitude of ensemble-averaged coherence |E(t_k)|
    E_se : ndarray
        Standard error of |E(t_k)|
    t : ndarray
        Time array
    E_abs_all : ndarray
        Array of |E| for each trajectory (for bootstrap analysis).
        If use_online=True, returns empty array (shape (0, N_steps)).
        If use_online=False, returns full array (shape (M, N_steps)).
    """
    N_steps = int(T_max / dt) + 1
    
    # Use core coherence computation
    noise_params = {
        'tau_c1': tau_c1, 'tau_c2': tau_c2,
        'B_rms1': B_rms1, 'B_rms2': B_rms2
    }
    
    if use_online:
        # Memory-efficient online algorithm (Welford's method)
        E_mean, E_abs_mean, E_M2 = _compute_coherence_core(
            'Double_OU', noise_params, gamma_e, dt, N_steps, M,
            sequence='FID', seed=seed, progress=progress, use_online=True
        )
        
        # Time array
        t = np.arange(N_steps) * dt
        
        # Ensemble average of complex coherence
        E = E_mean
        
        # Magnitude of ensemble-averaged coherence
        # IMPORTANT: |⟨E⟩| not ⟨|E|⟩
        E_abs = np.abs(E)
        
        # Variance and standard error from Welford's algorithm
        E_var = E_M2 / M if M > 1 else np.zeros(N_steps)
        E_se = np.sqrt(E_var / M)
        
        # Return empty array for E_abs_all when using online algorithm
        E_abs_all = np.zeros((0, N_steps), dtype=float)
    else:
        # Store all trajectories (for bootstrap analysis)
        E_all = _compute_coherence_core(
            'Double_OU', noise_params, gamma_e, dt, N_steps, M,
            sequence='FID', seed=seed, progress=progress, use_online=False
        )
        
        # Time array
        t = np.arange(N_steps) * dt
        
        # Ensemble average of complex coherence
        E = np.mean(E_all, axis=0)
        
        # Magnitude of ensemble-averaged coherence
        # IMPORTANT: |⟨E⟩| not ⟨|E|⟩
        E_abs = np.abs(E)
        
        # Standard error for |E| (magnitude)
        # Calculate |E| for each trajectory for error estimation
        E_abs_all = np.abs(E_all)  # Shape: (M, N_steps)
        
        # Standard error of the mean for |E|
        E_se = np.std(E_abs_all, axis=0, ddof=1) / np.sqrt(M)
    
    return E, E_abs, E_se, t, E_abs_all


def compute_hahn_echo_coherence_double_OU(
    tau_c1: float,
    tau_c2: float,
    B_rms1: float,
    B_rms2: float,
    gamma_e: float,
    dt: float,
    tau_list: Union[npt.NDArray[np.float64], List[float], Tuple[float, ...]],
    M: int,
    seed: Optional[int] = None,
    progress: bool = True
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Compute Hahn echo coherence E_echo(2τ) for Double-OU noise.
    
    Same as compute_hahn_echo_coherence() but with Double-OU noise model.
    
    Hahn echo sequence: π/2 - τ - π - τ (total time = 2τ)
    Toggling function: y(t) = +1 for t < τ, -1 for τ ≤ t ≤ 2τ
    
    Note: Pure dephasing simulation does not require B_0 (static field).
    
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
    gamma_e : float
        Electron gyromagnetic ratio
    dt : float
        Time step (seconds)
    tau_list : array-like
        List of echo delays τ (seconds)
    M : int
        Number of trajectories
    seed : int, optional
        Random seed for reproducibility
    progress : bool
        Whether to show progress bar
        
    Returns
    -------
    tau_echo : ndarray
        Echo times 2τ (seconds)
    E_echo : ndarray
        Complex coherence E_echo(2τ)
    E_echo_abs : ndarray
        Magnitude |E_echo(2τ)|
    E_echo_se : ndarray
        Standard error of |E_echo|
    E_echo_abs_all : ndarray
        Array of |E_echo| for each trajectory (for bootstrap analysis)
    """
    tau_list = np.array(tau_list)
    tau_max = tau_list.max()
    T_max = 2 * tau_max
    N_steps = int(T_max / dt) + 1
    
    if progress:
        print(f"  Computing Hahn echo (Double-OU): {len(tau_list)} tau values, {M} trajectories, {N_steps} time steps")
    
    # Use core coherence computation
    noise_params = {
        'tau_c1': tau_c1, 'tau_c2': tau_c2,
        'B_rms1': B_rms1, 'B_rms2': B_rms2
    }
    E_echo_all = _compute_coherence_core(
        'Double_OU', noise_params, gamma_e, dt, N_steps, M,
        sequence='Echo', tau_list=tau_list, seed=seed, progress=progress
    )
    
    tau_echo = 2 * tau_list
    
    # Ensemble average
    E_echo = np.mean(E_echo_all, axis=0)
    E_echo_abs = np.abs(E_echo)
    
    # Standard error of |E_echo|
    E_echo_abs_all = np.abs(E_echo_all)
    E_echo_se = np.std(E_echo_abs_all, axis=0, ddof=1) / np.sqrt(M)
    
    return tau_echo, E_echo, E_echo_abs, E_echo_se, E_echo_abs_all
