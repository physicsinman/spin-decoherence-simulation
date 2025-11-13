"""
Improved T2 extraction methods.

This module provides robust methods for extracting T2 from coherence decay
curves, including multi-point fitting and initial decay rate methods.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy.optimize import curve_fit
import warnings


class ImprovedT2Extraction:
    """
    Improved T2 extraction with multiple methods.
    
    Provides robust fitting methods that are less sensitive to noise
    and can work with partial decay curves.
    """
    
    def __init__(self):
        """Initialize T2 extraction methods."""
        pass
    
    def extract_T2_multipoint(
        self,
        time_points: np.ndarray,
        coherence_values: np.ndarray,
        coherence_errors: Optional[np.ndarray] = None,
        model: str = 'exponential'
    ) -> Tuple[float, float, Dict[str, any]]:
        """
        Extract T2 using multi-point weighted fitting.
        
        Parameters
        ----------
        time_points : array
            Time points (seconds)
        coherence_values : array
            Coherence values at each time point
        coherence_errors : array, optional
            Standard errors for weighted fitting
        model : str
            'exponential' or 'gaussian'
        
        Returns
        -------
        T2 : float
            Extracted T2 value (seconds)
        T2_error : float
            Estimated error in T2
        fit_info : dict
            Additional fitting information (chi-squared, etc.)
        """
        # Remove invalid points
        valid = np.isfinite(time_points) & np.isfinite(coherence_values)
        valid = valid & (coherence_values > 0) & (time_points >= 0)
        
        if np.sum(valid) < 3:
            return np.nan, np.nan, {'error': 'Insufficient valid points'}
        
        t_valid = time_points[valid]
        c_valid = coherence_values[valid]
        
        if coherence_errors is not None:
            sigma = coherence_errors[valid]
            # Remove zero or negative errors
            sigma = np.maximum(sigma, np.abs(c_valid) * 1e-6)
        else:
            sigma = None
        
        # Define model functions
        if model == 'exponential':
            def decay_model(t, T2, amplitude, offset):
                return amplitude * np.exp(-t / T2) + offset
            
            # Initial guess
            # T2: roughly time where coherence drops to 1/e
            c_initial = c_valid[0] if len(c_valid) > 0 else 1.0
            c_final = c_valid[-1] if len(c_valid) > 0 else 0.0
            target = c_initial / np.e
            
            # Find approximate T2
            if np.any(c_valid < target):
                idx = np.where(c_valid < target)[0][0]
                T2_guess = t_valid[idx] if idx > 0 else t_valid[-1] / 3
            else:
                T2_guess = t_valid[-1] / 3
            
            p0 = [T2_guess, c_initial - c_final, c_final]
            bounds = ([0, 0, -0.1], [np.inf, 2.0, 0.1])
            
        elif model == 'gaussian':
            def decay_model(t, T2_star, amplitude, offset):
                return amplitude * np.exp(-(t / T2_star)**2) + offset
            
            # Initial guess for Gaussian
            c_initial = c_valid[0] if len(c_valid) > 0 else 1.0
            c_final = c_valid[-1] if len(c_valid) > 0 else 0.0
            target = c_initial / np.e
            
            if np.any(c_valid < target):
                idx = np.where(c_valid < target)[0][0]
                T2_guess = t_valid[idx] / np.sqrt(2) if idx > 0 else t_valid[-1] / 3
            else:
                T2_guess = t_valid[-1] / 3
            
            p0 = [T2_guess, c_initial - c_final, c_final]
            bounds = ([0, 0, -0.1], [np.inf, 2.0, 0.1])
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Perform fitting
        try:
            popt, pcov = curve_fit(
                decay_model,
                t_valid,
                c_valid,
                p0=p0,
                sigma=sigma,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=10000
            )
            
            T2_fit = popt[0]
            T2_error = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else np.nan
            
            # Goodness of fit
            predicted = decay_model(t_valid, *popt)
            residuals = c_valid - predicted
            
            if sigma is not None:
                chi_squared = np.sum((residuals / sigma)**2)
            else:
                chi_squared = np.sum(residuals**2)
            
            dof = len(t_valid) - len(popt)
            reduced_chi_squared = chi_squared / dof if dof > 0 else np.nan
            
            fit_info = {
                'model': model,
                'chi_squared': chi_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'dof': dof,
                'n_points': len(t_valid),
                'amplitude': popt[1],
                'offset': popt[2],
                'success': True
            }
            
            return T2_fit, T2_error, fit_info
            
        except (RuntimeError, ValueError) as e:
            warnings.warn(f"Fitting failed: {e}")
            return np.nan, np.nan, {'error': str(e), 'success': False}
    
    def extract_T2_initial_decay(
        self,
        time_points: np.ndarray,
        coherence_values: np.ndarray,
        fraction: float = 0.2
    ) -> Tuple[float, float]:
        """
        Extract T2 from initial decay rate.
        
        Advantage: Less sensitive to noise, no need for full decay.
        
        Parameters
        ----------
        time_points : array
            Time points (seconds)
        coherence_values : array
            Coherence values
        fraction : float
            Fraction of data to use (default: 20%)
        
        Returns
        -------
        T2 : float
            Extracted T2 value (seconds)
        T2_error : float
            Estimated error (rough)
        """
        # Use initial fraction of data
        n_points = max(3, int(len(time_points) * fraction))
        t_initial = time_points[:n_points]
        c_initial = coherence_values[:n_points]
        
        # Remove invalid points
        valid = np.isfinite(t_initial) & np.isfinite(c_initial)
        valid = valid & (c_initial > 0) & (t_initial >= 0)
        
        if np.sum(valid) < 3:
            return np.nan, np.nan
        
        t_valid = t_initial[valid]
        c_valid = c_initial[valid]
        
        # Log-linear fitting
        # ln(C(t)) = -t/T2 + const
        log_coherence = np.log(np.abs(c_valid) + 1e-10)
        
        # Linear regression
        try:
            coeffs = np.polyfit(t_valid, log_coherence, 1)
            slope = coeffs[0]
            
            if slope >= 0:
                warnings.warn("Positive slope in log-coherence - coherence not decaying")
                return np.nan, np.nan
            
            T2 = -1.0 / slope
            
            # Error estimation from residuals
            predicted = np.polyval(coeffs, t_valid)
            residuals = log_coherence - predicted
            std_residuals = np.std(residuals)
            
            # Rough error estimate: T2_error ≈ T2 * std_residuals
            T2_error = T2 * std_residuals
            
            return T2, T2_error
            
        except Exception as e:
            warnings.warn(f"Initial decay fitting failed: {e}")
            return np.nan, np.nan
    
    def extract_T2_auto(
        self,
        time_points: np.ndarray,
        coherence_values: np.ndarray,
        coherence_errors: Optional[np.ndarray] = None
    ) -> Tuple[float, float, Dict[str, any]]:
        """
        Automatically select best method and extract T2.
        
        Tries multiple methods and selects the most reliable result.
        
        Parameters
        ----------
        time_points : array
            Time points (seconds)
        coherence_values : array
            Coherence values
        coherence_errors : array, optional
            Standard errors
        
        Returns
        -------
        T2 : float
            Best T2 estimate
        T2_error : float
            Estimated error
        info : dict
            Information about method used
        """
        results = []
        
        # Method 1: Exponential fitting
        T2_exp, T2_err_exp, info_exp = self.extract_T2_multipoint(
            time_points, coherence_values, coherence_errors, model='exponential'
        )
        if not np.isnan(T2_exp):
            results.append({
                'method': 'exponential',
                'T2': T2_exp,
                'error': T2_err_exp,
                'reduced_chi_sq': info_exp.get('reduced_chi_squared', np.nan),
                'info': info_exp
            })
        
        # Method 2: Gaussian fitting (for quasi-static)
        T2_gauss, T2_err_gauss, info_gauss = self.extract_T2_multipoint(
            time_points, coherence_values, coherence_errors, model='gaussian'
        )
        if not np.isnan(T2_gauss):
            results.append({
                'method': 'gaussian',
                'T2': T2_gauss,
                'error': T2_err_gauss,
                'reduced_chi_sq': info_gauss.get('reduced_chi_squared', np.nan),
                'info': info_gauss
            })
        
        # Method 3: Initial decay
        T2_init, T2_err_init = self.extract_T2_initial_decay(
            time_points, coherence_values
        )
        if not np.isnan(T2_init):
            results.append({
                'method': 'initial_decay',
                'T2': T2_init,
                'error': T2_err_init,
                'reduced_chi_sq': np.nan,  # Not available for this method
                'info': {}
            })
        
        if not results:
            return np.nan, np.nan, {'error': 'All methods failed'}
        
        # Select best result based on reduced chi-squared (if available)
        # or smallest error
        best_result = min(
            results,
            key=lambda x: (
                x['reduced_chi_sq'] if not np.isnan(x['reduced_chi_sq']) else np.inf,
                x['error'] if not np.isnan(x['error']) else np.inf
            )
        )
        
        return best_result['T2'], best_result['error'], {
            'method_used': best_result['method'],
            'all_results': results
        }

