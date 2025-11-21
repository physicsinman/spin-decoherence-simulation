"""
Adaptive simulation strategy for different regimes.

This module implements regime-specific optimization strategies to efficiently
handle different correlation time regimes.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from parameter_validation import SimulationParameters
from memory_efficient_sim import MemoryEfficientSimulation


class AdaptiveSimulation:
    """
    Adaptive simulation with regime-specific optimization.
    
    Different strategies for different regimes:
    - Motional-narrowing: Short simulation time, many ensembles
    - Quasi-static: Long simulation time, fewer ensembles
    - Intermediate: Standard simulation
    """
    
    def __init__(self, params: SimulationParameters):
        """
        Initialize adaptive simulation.
        
        Parameters
        ----------
        params : SimulationParameters
            Base simulation parameters
        """
        self.params = params
        self.base_sim = MemoryEfficientSimulation(params)
    
    def determine_regime(self, tau_c: float) -> str:
        """
        Determine regime for given tau_c.
        
        Parameters
        ----------
        tau_c : float
            Correlation time (seconds)
        
        Returns
        -------
        regime : str
            'motional_narrowing', 'quasi_static', or 'intermediate'
        """
        T2_star = self.params.T2_star_target
        
        if tau_c < 0.1 * T2_star:
            return 'motional_narrowing'
        elif tau_c > 10 * T2_star:
            return 'quasi_static'
        else:
            return 'intermediate'
    
    def estimate_T2(self, tau_c: float) -> float:
        """
        Estimate theoretical T2 for given tau_c.
        
        Used to determine appropriate simulation time.
        
        Parameters
        ----------
        tau_c : float
            Correlation time (seconds)
        
        Returns
        -------
        T2_estimate : float
            Estimated T2 value (seconds)
        """
        T2_star = self.params.T2_star_target
        gamma_e = self.params.gamma_e
        B_rms = self.params.sigma_z
        Delta_omega = gamma_e * B_rms
        
        if tau_c < 0.1 * T2_star:
            # Motional narrowing: T2 ∝ tau_c^-1
            # T2 = 1 / (Delta_omega^2 * tau_c)
            return 1.0 / (Delta_omega**2 * tau_c)
        else:
            # Quasi-static or intermediate: T2 ≈ T2*
            return T2_star
    
    def simulate_adaptive(
        self,
        tau_c: float,
        sequence: str = 'FID',
        seed: Optional[int] = None
    ) -> Tuple[float, float, Dict[str, any]]:
        """
        Run optimized simulation for given tau_c based on regime.
        
        Parameters
        ----------
        tau_c : float
            Correlation time (seconds)
        sequence : str
            'FID' or 'Echo'
        seed : int, optional
            Random seed
        
        Returns
        -------
        coherence : float
            Ensemble-averaged coherence
        coherence_std : float
            Standard deviation
        metadata : dict
            Additional information (regime, n_ensemble_used, etc.)
        """
        regime = self.determine_regime(tau_c)
        T2_estimate = self.estimate_T2(tau_c)
        
        # Create regime-specific parameters
        sim_params = self._get_regime_params(regime, tau_c, T2_estimate)
        
        # Create simulation with optimized parameters
        sim = MemoryEfficientSimulation(sim_params)
        
        # Run simulation
        coherence, coherence_std = sim.simulate_coherence_chunked(
            tau_c, sequence=sequence, seed=seed
        )
        
        metadata = {
            'regime': regime,
            'T2_estimate': T2_estimate,
            'n_ensemble_used': sim_params.n_ensemble,
            'sim_time_used': sim_params.total_time,
        }
        
        return coherence, coherence_std, metadata
    
    def _get_regime_params(
        self,
        regime: str,
        tau_c: float,
        T2_estimate: float
    ) -> SimulationParameters:
        """
        Get optimized parameters for specific regime.
        
        Parameters
        ----------
        regime : str
            Regime name
        tau_c : float
            Correlation time
        T2_estimate : float
            Estimated T2
        
        Returns
        -------
        params : SimulationParameters
            Optimized parameters for this regime
        """
        # Start with base parameters
        params = SimulationParameters(
            system=self.params.system,
            target_regime=regime
        )
        
        if regime == 'motional_narrowing':
            # Short simulation time sufficient
            # Use more ensembles for better statistics
            params.total_time = 5 * T2_estimate
            params.n_ensemble = min(self.params.n_ensemble * 2, 500)  # Up to 500
            
        elif regime == 'quasi_static':
            # Long simulation needed but fewer ensembles possible
            # (variance is lower in QS regime)
            params.total_time = 10 * self.params.T2_star_target
            params.n_ensemble = max(self.params.n_ensemble // 2, 50)  # At least 50
            
        else:  # intermediate
            # Standard parameters
            params.total_time = 10 * self.params.T2_star_target
            params.n_ensemble = self.params.n_ensemble
        
        # Ensure dt is appropriate
        params.dt = min(tau_c / 50, self.params.dt)
        
        return params
    
    def simulate_time_series_adaptive(
        self,
        tau_c: float,
        sequence: str = 'FID',
        time_points: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Run adaptive simulation with time series output.
        
        Parameters
        ----------
        tau_c : float
            Correlation time (seconds)
        sequence : str
            'FID' or 'Echo'
        time_points : array, optional
            Time points for evaluation
        seed : int, optional
            Random seed
        
        Returns
        -------
        coherence : array
            Coherence values at time_points
        coherence_std : array
            Standard deviation at each time point
        metadata : dict
            Additional information
        """
        regime = self.determine_regime(tau_c)
        T2_estimate = self.estimate_T2(tau_c)
        
        # Get optimized parameters
        sim_params = self._get_regime_params(regime, tau_c, T2_estimate)
        sim = MemoryEfficientSimulation(sim_params)
        
        # Default time points
        if time_points is None:
            n_points = 100
            time_points = np.linspace(0, sim_params.total_time, n_points)
        
        # Run simulation
        coherence, coherence_std = sim.simulate_coherence_time_series(
            tau_c, sequence=sequence, time_points=time_points, seed=seed
        )
        
        metadata = {
            'regime': regime,
            'T2_estimate': T2_estimate,
            'n_ensemble_used': sim_params.n_ensemble,
            'sim_time_used': sim_params.total_time,
        }
        
        return coherence, coherence_std, metadata

