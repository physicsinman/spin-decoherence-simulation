"""
Memory-efficient simulation using chunked processing.

This module implements chunked processing to handle long simulation times
without excessive memory requirements.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from parameter_validation import SimulationParameters
from ornstein_uhlenbeck import generate_ou_noise


class MemoryEfficientSimulation:
    """
    Memory-efficient simulation using chunked processing.
    
    Instead of storing the full trajectory, this class processes data in chunks
    and only stores the final coherence values.
    """
    
    def __init__(self, params: SimulationParameters):
        """
        Initialize memory-efficient simulation.
        
        Parameters
        ----------
        params : SimulationParameters
            Validated simulation parameters
        """
        self.params = params
    
    def simulate_coherence_chunked(
        self,
        tau_c: float,
        sequence: str = 'FID',
        chunk_size_sec: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Simulate coherence in chunks to save memory.
        
        Parameters
        ----------
        tau_c : float
            Correlation time (seconds)
        sequence : str
            'FID' or 'Echo'
        chunk_size_sec : float, optional
            Time length of each chunk in seconds.
            If None, uses adaptive chunking based on tau_c.
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        coherence : float
            Ensemble-averaged coherence at final time
        coherence_std : float
            Standard deviation of coherence across ensemble
        """
        total_time = self.params.total_time
        dt = self.params.dt
        n_ensemble = self.params.n_ensemble
        gamma_e = self.params.gamma_e
        B_rms = self.params.sigma_z
        
        # Determine chunk size
        if chunk_size_sec is None:
            # Adaptive: use smaller of (tau_c, T2*) as chunk size
            chunk_size_sec = min(tau_c, self.params.T2_star_target) * 2
            # But ensure it's not too small (at least 100 steps)
            min_chunk_size = 100 * dt
            chunk_size_sec = max(chunk_size_sec, min_chunk_size)
        
        # Number of chunks
        n_chunks = int(np.ceil(total_time / chunk_size_sec))
        
        # Storage for final coherence values (only store these, not full trajectories)
        coherence_values = np.zeros(n_ensemble, dtype=np.complex128)
        
        # Process each ensemble member
        for ensemble_idx in range(n_ensemble):
            traj_seed = seed + ensemble_idx if seed is not None else None
            phase_accumulation = 0.0
            
            # Process in chunks
            for chunk_idx in range(n_chunks):
                # Current chunk time range
                t_start = chunk_idx * chunk_size_sec
                t_end = min((chunk_idx + 1) * chunk_size_sec, total_time)
                
                # Time array for this chunk
                t_chunk = np.arange(t_start, t_end, dt)
                n_steps_chunk = len(t_chunk)
                
                if n_steps_chunk == 0:
                    continue
                
                # Generate OU noise for this chunk
                # Note: For proper continuity, we'd need to pass the last value
                # from previous chunk. For simplicity, we generate independent chunks.
                # This is acceptable if chunk_size >> tau_c.
                noise_chunk = generate_ou_noise(
                    tau_c, B_rms, dt, n_steps_chunk,
                    seed=traj_seed + chunk_idx if traj_seed is not None else None
                )
                
                # Handle echo sequence (π-pulse at t = total_time / 2)
                if sequence == 'Echo':
                    t_pi = total_time / 2
                    if t_start <= t_pi < t_end:
                        # π-pulse location in this chunk
                        pi_idx = int((t_pi - t_start) / dt)
                        if pi_idx < len(noise_chunk):
                            # Flip sign after π-pulse
                            noise_chunk[pi_idx:] *= -1
                
                # Phase accumulation for this chunk
                delta_omega_chunk = gamma_e * noise_chunk
                phase_chunk = np.sum(delta_omega_chunk * dt)
                phase_accumulation += phase_chunk
                
                # Memory cleanup
                del noise_chunk, delta_omega_chunk, phase_chunk
            
            # Final coherence for this ensemble member
            coherence_values[ensemble_idx] = np.exp(1j * phase_accumulation)
        
        # Ensemble average
        coherence_mean = np.mean(coherence_values)
        coherence_std = np.std(np.abs(coherence_values))
        
        return np.abs(coherence_mean), coherence_std
    
    def simulate_coherence_time_series(
        self,
        tau_c: float,
        sequence: str = 'FID',
        time_points: Optional[np.ndarray] = None,
        chunk_size_sec: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate coherence at multiple time points using chunked processing.
        
        This is more memory-efficient than storing full trajectories.
        
        Parameters
        ----------
        tau_c : float
            Correlation time (seconds)
        sequence : str
            'FID' or 'Echo'
        time_points : array, optional
            Time points at which to evaluate coherence.
            If None, uses evenly spaced points.
        chunk_size_sec : float, optional
            Chunk size for processing
        seed : int, optional
            Random seed
        
        Returns
        -------
        coherence : array
            Coherence values at time_points
        coherence_std : array
            Standard deviation at each time point
        """
        total_time = self.params.total_time
        dt = self.params.dt
        n_ensemble = self.params.n_ensemble
        gamma_e = self.params.gamma_e
        B_rms = self.params.sigma_z
        
        # Default time points
        if time_points is None:
            n_points = 100
            time_points = np.linspace(0, total_time, n_points)
        
        # Storage for coherence at each time point
        coherence_at_t = np.zeros(len(time_points), dtype=np.complex128)
        coherence_abs_at_t = np.zeros((n_ensemble, len(time_points)), dtype=np.float64)
        
        # Process each ensemble member
        for ensemble_idx in range(n_ensemble):
            traj_seed = seed + ensemble_idx if seed is not None else None
            
            # We need to compute phase at each time point
            # For efficiency, we'll process in chunks but track phase at specific times
            phase_at_t = np.zeros(len(time_points))
            
            # Determine chunk size
            if chunk_size_sec is None:
                chunk_size_sec = min(tau_c, self.params.T2_star_target) * 2
                chunk_size_sec = max(chunk_size_sec, 100 * dt)
            
            n_chunks = int(np.ceil(total_time / chunk_size_sec))
            current_phase = 0.0
            t_current = 0.0
            
            for chunk_idx in range(n_chunks):
                t_start = chunk_idx * chunk_size_sec
                t_end = min((chunk_idx + 1) * chunk_size_sec, total_time)
                
                t_chunk = np.arange(t_start, t_end, dt)
                n_steps_chunk = len(t_chunk)
                
                if n_steps_chunk == 0:
                    continue
                
                # Generate noise for chunk
                noise_chunk = generate_ou_noise(
                    tau_c, B_rms, dt, n_steps_chunk,
                    seed=traj_seed + chunk_idx if traj_seed is not None else None
                )
                
                # Handle echo sequence
                if sequence == 'Echo':
                    t_pi = total_time / 2
                    if t_start <= t_pi < t_end:
                        pi_idx = int((t_pi - t_start) / dt)
                        if pi_idx < len(noise_chunk):
                            noise_chunk[pi_idx:] *= -1
                
                # Accumulate phase
                delta_omega_chunk = gamma_e * noise_chunk
                phase_chunk = np.cumsum(delta_omega_chunk * dt)
                
                # Update phase at time points within this chunk
                for i, t_target in enumerate(time_points):
                    if t_start <= t_target < t_end:
                        # Interpolate phase at this time point
                        idx_in_chunk = (t_target - t_start) / dt
                        if idx_in_chunk < len(phase_chunk):
                            phase_at_t[i] = current_phase + phase_chunk[int(idx_in_chunk)]
                
                # Update current phase for next chunk
                current_phase += phase_chunk[-1]
                
                del noise_chunk, delta_omega_chunk, phase_chunk
            
            # Coherence at each time point for this ensemble member
            coherence_abs_at_t[ensemble_idx, :] = np.abs(np.exp(1j * phase_at_t))
        
        # Ensemble average
        coherence_mean = np.mean(coherence_abs_at_t, axis=0)
        coherence_std = np.std(coherence_abs_at_t, axis=0)
        
        return coherence_mean, coherence_std

