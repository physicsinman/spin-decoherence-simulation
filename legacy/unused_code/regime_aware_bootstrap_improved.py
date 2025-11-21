"""
Improved regime-aware bootstrap analysis.

This module provides bootstrap methods that are adapted for different
regimes, with special handling for quasi-static regime where standard
bootstrap fails.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from improved_t2_extraction import ImprovedT2Extraction
from parameter_validation import SimulationParameters


class RegimeAwareBootstrap:
    """
    Regime-aware bootstrap with improved handling for quasi-static regime.
    
    Uses different strategies for different regimes:
    - Motional-narrowing: Standard bootstrap
    - Quasi-static: Log-space statistics or analytical CI
    - Intermediate: Hybrid approach
    """
    
    def __init__(self, params: Optional[SimulationParameters] = None):
        """
        Initialize regime-aware bootstrap.
        
        Parameters
        ----------
        params : SimulationParameters, optional
            Simulation parameters (for regime determination)
        """
        self.params = params
        self.extractor = ImprovedT2Extraction()
    
    def determine_regime(self, tau_c: float, T2_star: Optional[float] = None) -> str:
        """
        Determine regime for given tau_c.
        
        Parameters
        ----------
        tau_c : float
            Correlation time (seconds)
        T2_star : float, optional
            T2* value. If None, uses params.T2_star_target
        
        Returns
        -------
        regime : str
            'motional_narrowing', 'quasi_static', or 'intermediate'
        """
        if T2_star is None:
            if self.params is None:
                raise ValueError("Either T2_star or params must be provided")
            T2_star = self.params.T2_star_target
        
        if tau_c < 0.1 * T2_star:
            return 'motional_narrowing'
        elif tau_c > 10 * T2_star:
            return 'quasi_static'
        else:
            return 'intermediate'
    
    def bootstrap_T2(
        self,
        coherence_ensemble: np.ndarray,
        time_points: np.ndarray,
        tau_c: float,
        T2_star: Optional[float] = None,
        n_bootstrap: int = 1000,
        coherence_errors: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, Dict[str, any]]:
        """
        Bootstrap T2 with regime-aware strategy.
        
        Parameters
        ----------
        coherence_ensemble : array
            Coherence values for each ensemble member (M, N_time)
        time_points : array
            Time points (N_time,)
        tau_c : float
            Correlation time (for regime determination)
        T2_star : float, optional
            T2* value (for regime determination)
        n_bootstrap : int
            Number of bootstrap samples
        coherence_errors : array, optional
            Standard errors at each time point
        
        Returns
        -------
        T2_median : float
            Median T2 from bootstrap
        T2_lower : float
            Lower 95% CI bound
        T2_upper : float
            Upper 95% CI bound
        metadata : dict
            Additional information
        """
        regime = self.determine_regime(tau_c, T2_star)
        
        if regime == 'quasi_static':
            return self.bootstrap_quasi_static(
                coherence_ensemble, time_points, n_bootstrap, coherence_errors
            )
        elif regime == 'motional_narrowing':
            return self.bootstrap_standard(
                coherence_ensemble, time_points, n_bootstrap, coherence_errors
            )
        else:  # intermediate
            # Try standard first, fall back to quasi-static if fails
            try:
                return self.bootstrap_standard(
                    coherence_ensemble, time_points, n_bootstrap, coherence_errors
                )
            except:
                return self.bootstrap_quasi_static(
                    coherence_ensemble, time_points, n_bootstrap, coherence_errors
                )
    
    def bootstrap_quasi_static(
        self,
        coherence_ensemble: np.ndarray,
        time_points: np.ndarray,
        n_bootstrap: int,
        coherence_errors: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, Dict[str, any]]:
        """
        Special bootstrap for quasi-static regime.
        
        Uses log-space statistics to handle wide distributions.
        
        Parameters
        ----------
        coherence_ensemble : array
            Coherence values (M, N_time)
        time_points : array
            Time points (N_time,)
        n_bootstrap : int
            Number of bootstrap samples
        coherence_errors : array, optional
            Standard errors
        
        Returns
        -------
        T2_median : float
            Median T2
        T2_lower : float
            Lower CI bound
        T2_upper : float
            Upper CI bound
        metadata : dict
            Additional information
        """
        n_ensemble = coherence_ensemble.shape[0]
        T2_bootstrap = []
        
        # Extract T2 for each bootstrap sample
        for _ in range(n_bootstrap):
            # Resample ensembles with replacement
            indices = np.random.choice(n_ensemble, n_ensemble, replace=True)
            coherence_resampled = coherence_ensemble[indices]
            
            # Average over resampled ensembles
            coherence_mean = np.mean(coherence_resampled, axis=0)
            
            # Extract T2
            T2, _, _ = self.extractor.extract_T2_auto(
                time_points, coherence_mean, coherence_errors
            )
            
            if not np.isnan(T2) and T2 > 0:
                T2_bootstrap.append(T2)
        
        if len(T2_bootstrap) < 10:
            # Too few valid samples
            return np.nan, np.nan, np.nan, {'error': 'Insufficient valid bootstrap samples'}
        
        T2_bootstrap = np.array(T2_bootstrap)
        
        # Calculate statistics in log-space (more robust for wide distributions)
        log_T2 = np.log(T2_bootstrap)
        log_T2_median = np.median(log_T2)
        log_T2_lower = np.percentile(log_T2, 2.5)  # 95% CI
        log_T2_upper = np.percentile(log_T2, 97.5)
        
        # Convert back to linear space
        T2_median = np.exp(log_T2_median)
        T2_lower = np.exp(log_T2_lower)
        T2_upper = np.exp(log_T2_upper)
        
        metadata = {
            'regime': 'quasi_static',
            'method': 'log_space_bootstrap',
            'n_valid_samples': len(T2_bootstrap),
            'log_std': np.std(log_T2),
            'linear_std': np.std(T2_bootstrap),
        }
        
        return T2_median, T2_lower, T2_upper, metadata
    
    def bootstrap_standard(
        self,
        coherence_ensemble: np.ndarray,
        time_points: np.ndarray,
        n_bootstrap: int,
        coherence_errors: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, Dict[str, any]]:
        """
        Standard bootstrap for motional-narrowing regime.
        
        Parameters
        ----------
        coherence_ensemble : array
            Coherence values (M, N_time)
        time_points : array
            Time points (N_time,)
        n_bootstrap : int
            Number of bootstrap samples
        coherence_errors : array, optional
            Standard errors
        
        Returns
        -------
        T2_mean : float
            Mean T2
        T2_lower : float
            Lower CI bound
        T2_upper : float
            Upper CI bound
        metadata : dict
            Additional information
        """
        n_ensemble = coherence_ensemble.shape[0]
        T2_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Resample ensembles
            indices = np.random.choice(n_ensemble, n_ensemble, replace=True)
            coherence_resampled = coherence_ensemble[indices]
            
            # Average
            coherence_mean = np.mean(coherence_resampled, axis=0)
            
            # Extract T2
            T2, _, _ = self.extractor.extract_T2_auto(
                time_points, coherence_mean, coherence_errors
            )
            
            if not np.isnan(T2) and T2 > 0:
                T2_bootstrap.append(T2)
        
        if len(T2_bootstrap) < 10:
            return np.nan, np.nan, np.nan, {'error': 'Insufficient valid bootstrap samples'}
        
        T2_bootstrap = np.array(T2_bootstrap)
        
        # Standard statistics in linear space
        T2_mean = np.mean(T2_bootstrap)
        T2_lower = np.percentile(T2_bootstrap, 2.5)
        T2_upper = np.percentile(T2_bootstrap, 97.5)
        
        metadata = {
            'regime': 'motional_narrowing',
            'method': 'standard_bootstrap',
            'n_valid_samples': len(T2_bootstrap),
            'std': np.std(T2_bootstrap),
        }
        
        return T2_mean, T2_lower, T2_upper, metadata

