"""
Fitting functions for extracting T_2 relaxation times.

This module provides functions to fit coherence decay curves to various
analytical models (Gaussian, exponential, stretched exponential).
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy import integrate

# Import estimate_characteristic_T2 from spin_decoherence package (avoids circular import)
from spin_decoherence.simulation.engine import estimate_characteristic_T2


def gaussian_decay(t, T2_star):
    """
    Gaussian decay model (quasi-static noise limit).
    
    E(t) = exp[-(t/T_2^*)^2]
    """
    return np.exp(-(t / T2_star)**2)


def exponential_decay(t, T2):
    """
    Exponential decay model (motional-narrowing limit).
    
    E(t) = exp[-t/T_2]
    """
    return np.exp(-t / T2)


def stretched_exponential_decay(t, T_beta, beta):
    """
    Stretched exponential decay model (intermediate regime).
    
    E(t) = exp[-(t/T_beta)^beta]
    """
    return np.exp(-(t / T_beta)**beta)


def select_echo_fit_window(t, E_abs, E_se=None, eps=None, min_pts=20, tau_c=None, gamma_e=None, B_rms=None):
    """
    Select fitting window for echo signals.
    
    For fast noise (ξ < 0.5), echo ≈ FID, so use similar threshold as FID.
    For slow noise (ξ >> 1), echo can have longer decay, use more conservative threshold.
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_abs : ndarray
        Magnitude of coherence |E(t)|
    E_se : ndarray, optional
        Standard error of |E(t)|
    eps : float, optional
        Minimum threshold (default: adaptive based on regime)
    min_pts : int
        Minimum number of points required (default: 20)
    tau_c : float, optional
        Correlation time (for regime-aware threshold)
    gamma_e : float, optional
        Electron gyromagnetic ratio (for regime-aware threshold)
    B_rms : float, optional
        RMS noise amplitude (for regime-aware threshold)
        
    Returns
    -------
    t_fit : ndarray
        Selected time points for fitting
    E_fit : ndarray
        Selected |E| values for fitting
    """
    # Adaptive threshold based on noise regime
    if eps is None:
        if tau_c is not None and gamma_e is not None and B_rms is not None:
            Delta_omega = gamma_e * B_rms
            xi = Delta_omega * tau_c
            if xi < 0.5:  # Fast noise: echo ≈ FID, use similar threshold
                eps = 0.05  # Same as FID
            else:  # Slow noise: more conservative
                eps = 0.1
        else:
            eps = 0.1  # Default: conservative
    
    if E_se is not None:
        # Estimate noise floor from late-time data
        noise_floor = np.percentile(E_abs[-len(E_abs)//4:], 10)  # 10th percentile of last quarter
        sigma_noise = np.std(E_abs[-len(E_abs)//4:])  # Standard deviation of noise floor
        # Use adaptive threshold: 3-sigma for fast noise, 5-sigma for slow noise
        if tau_c is not None and gamma_e is not None and B_rms is not None:
            Delta_omega = gamma_e * B_rms
            xi = Delta_omega * tau_c
            sigma_mult = 3.0 if xi < 0.5 else 5.0  # Less conservative for fast noise
        else:
            sigma_mult = 5.0  # Default: conservative
        
        threshold = np.maximum(sigma_mult * sigma_noise, np.maximum(sigma_mult * E_se, eps))
    else:
        # Fallback to SNR-based threshold
        threshold = eps
    
    # Find valid points: exclude all data points with |E| < threshold
    mask = (E_abs > threshold) & np.isfinite(E_abs) & (E_abs > 0)
    idx = np.where(mask)[0]
    
    if len(idx) < min_pts:
        # If too few points, use initial portion
        idx = np.arange(min(min_pts, len(t)))
    
    return t[idx], E_abs[idx]


def select_fit_window(t, E_abs, E_se=None, eps=None, min_pts=20, tau_c=None, gamma_e=None, B_rms=None):
    """
    Select fitting window to avoid late-time bias from noise floor.
    
    Only fits data where |E(t)| > max(3*SE, eps) to ensure signal is
    above noise floor. For static regime (fast decay), uses higher threshold
    to focus on initial decay.
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_abs : ndarray
        Magnitude of coherence |E(t)|
    E_se : ndarray, optional
        Standard error of |E(t)|
    eps : float, optional
        Minimum threshold (default: exp(-3) ≈ 0.05 for SNR-based cutoff)
    min_pts : int
        Minimum number of points required (default: 20)
    tau_c : float, optional
        Correlation time (for regime detection)
    gamma_e : float, optional
        Electron gyromagnetic ratio (for regime detection)
    B_rms : float, optional
        RMS noise amplitude (for regime detection)
        
    Returns
    -------
    t_fit : ndarray
        Selected time points for fitting
    E_fit : ndarray
        Selected |E| values for fitting
    """
    # Default threshold: SNR-based cutoff (|E| > exp(-3) ≈ 0.05)
    if eps is None:
        eps = np.exp(-3.0)  # ~0.05, reasonable SNR cutoff
    
    # Adjust threshold for static regime (fast decay)
    # In static regime, decay is very fast, so we want to focus on initial decay
    # Use higher threshold to exclude noise floor earlier
    if tau_c is not None and gamma_e is not None and B_rms is not None:
        Delta_omega = gamma_e * B_rms
        xi = Delta_omega * tau_c
        if xi > 2.0:  # Static/quasi-static regime
            # Use much higher threshold for static regime: exp(-1) ≈ 0.368
            # Static regime has Gaussian decay: E(t) ≈ exp[-(t/T2)^2]
            # We want to focus on the initial decay (E > 0.37) where the signal is strong
            # This ensures we fit the Gaussian decay shape accurately
            eps = max(eps, np.exp(-1.0))  # ~0.368, much higher threshold for static regime
    
    if E_se is not None:
        # Use 3-sigma threshold: signal must be > max(3*SE, eps)
        # Estimate noise floor from late-time data
        noise_floor = np.percentile(E_abs[-len(E_abs)//4:], 10)  # 10th percentile of last quarter
        sigma_noise = np.std(E_abs[-len(E_abs)//4:])  # Standard deviation of noise floor
        threshold = np.maximum(3 * sigma_noise, np.maximum(3 * E_se, eps))
    else:
        # Fallback to SNR-based threshold
        threshold = eps
    
    # Find valid points: exclude all data points with |E| < 3*sigma_noise
    mask = (E_abs > threshold) & np.isfinite(E_abs) & (E_abs > 0)
    
    # For static regime, also limit to initial decay window (t < 3*T2_th)
    # This ensures we only fit the initial Gaussian decay, not the noise floor
    # Use 3*T2_th instead of 2*T2_th to allow more variation for bootstrap CI
    if tau_c is not None and gamma_e is not None and B_rms is not None:
        Delta_omega = gamma_e * B_rms
        xi = Delta_omega * tau_c
        if xi > 2.0:  # Static/quasi-static regime
            # Estimate T2_th for static regime: T2 ≈ sqrt(2)/Delta_omega
            T2_th = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
            # Limit to initial decay: t < 3*T2_th (relaxed from 2*T2_th for bootstrap CI)
            # This captures the Gaussian decay before it reaches noise floor
            # Relaxed to allow more variation in bootstrap samples
            time_limit = 3.0 * T2_th
            mask = mask & (t < time_limit)
            if len(t[mask]) < min_pts:
                # If too few points after time limit, use initial portion up to time_limit
                mask_initial = (t < time_limit) & np.isfinite(E_abs) & (E_abs > 0)
                idx_initial = np.where(mask_initial)[0]
                if len(idx_initial) >= min_pts:
                    # Take first min_pts points
                    mask = np.zeros(len(t), dtype=bool)
                    mask[idx_initial[:min_pts]] = True
                else:
                    # Use all available points up to time_limit
                    mask = mask_initial
    
    idx = np.where(mask)[0]
    
    if len(idx) < min_pts:
        # If too few points, use initial portion
        idx = np.arange(min(min_pts, len(t)))
    
    return t[idx], E_abs[idx]


def fit_echo_decay(t, E_magnitude, E_se=None, model='auto', use_bic=False,
                   tau_c=None, gamma_e=None, B_rms=None):
    """
    Fit echo coherence decay with echo-optimized window selection.
    
    This is a wrapper around fit_coherence_decay that uses select_echo_fit_window
    for regime-aware fitting suitable for echo signals.
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_magnitude : ndarray
        Magnitude of coherence |E(t)|
    E_se : ndarray, optional
        Standard error of |E(t)|
    model : str
        Model to use: 'gaussian', 'exponential', 'stretched', or 'auto'
    use_bic : bool
        Whether to use BIC instead of AIC for model selection
    tau_c : float, optional
        Correlation time (for regime-aware window selection)
    gamma_e : float, optional
        Electron gyromagnetic ratio (for regime-aware window selection)
    B_rms : float, optional
        RMS noise amplitude (for regime-aware window selection)
        
    Returns
    -------
    result : dict
        Same as fit_coherence_decay
    """
    # Use echo-optimized window selection (regime-aware)
    t_fit, E_fit = select_echo_fit_window(t, E_magnitude, E_se=E_se,
                                          tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms)
    
    if len(t_fit) < 3:
        return None
    
    # Get E_se for the selected window
    E_se_fit = None
    if E_se is not None:
        # Find indices that correspond to t_fit
        # t_fit is a subset of t, so we need to find matching indices
        indices = []
        for t_val in t_fit:
            idx = np.argmin(np.abs(t - t_val))
            indices.append(idx)
        indices = np.array(indices)
        E_se_fit = E_se[indices]
    
    # Call fit_coherence_decay with pre-selected window
    # Since we've already selected the window, we pass E_se_fit=None to prevent
    # fit_coherence_decay from re-selecting the window (it will use all points)
    # Actually, fit_coherence_decay will still select a window, but since our
    # echo window is more conservative, it should include all our points.
    # To be safe, we'll pass the pre-selected data directly to the fitting logic.
    
    # Extract the fitting logic from fit_coherence_decay
    # We'll call it with the pre-selected window, but need to handle E_se properly
    results = {}
    
    # Fit Gaussian model
    try:
        popt_gauss, pcov_gauss = curve_fit(
            gaussian_decay, t_fit, E_fit,
            p0=[t_fit[-1] / 2],
            bounds=(0, np.inf)
        )
        T2_gauss = popt_gauss[0]
        E_gauss = gaussian_decay(t_fit, *popt_gauss)
        ss_res_gauss = np.sum((E_fit - E_gauss)**2)
        ss_tot_gauss = np.sum((E_fit - np.mean(E_fit))**2)
        R2_gauss = 1 - (ss_res_gauss / ss_tot_gauss) if ss_tot_gauss > 0 else 0
        n_params_gauss = 1
        n_points = len(t_fit)
        AIC_gauss = n_points * np.log(ss_res_gauss / n_points) + 2 * n_params_gauss
        BIC_gauss = n_points * np.log(ss_res_gauss / n_points) + n_params_gauss * np.log(n_points)
        RMSE_gauss = np.sqrt(ss_res_gauss / n_points)
        
        results['gaussian'] = {
            'params': popt_gauss,
            'T2': T2_gauss,
            'R2': R2_gauss,
            'AIC': AIC_gauss,
            'BIC': BIC_gauss,
            'RMSE': RMSE_gauss,
            'fit_curve': gaussian_decay(t, *popt_gauss)
        }
    except:
        results['gaussian'] = None
    
    # Fit exponential model
    try:
        popt_exp, pcov_exp = curve_fit(
            exponential_decay, t_fit, E_fit,
            p0=[t_fit[-1] / 2],
            bounds=(0, np.inf)
        )
        T2_exp = popt_exp[0]
        E_exp = exponential_decay(t_fit, *popt_exp)
        ss_res_exp = np.sum((E_fit - E_exp)**2)
        ss_tot_exp = np.sum((E_fit - np.mean(E_fit))**2)
        R2_exp = 1 - (ss_res_exp / ss_tot_exp) if ss_tot_exp > 0 else 0
        n_params_exp = 1
        n_points = len(t_fit)
        AIC_exp = n_points * np.log(ss_res_exp / n_points) + 2 * n_params_exp
        BIC_exp = n_points * np.log(ss_res_exp / n_points) + n_params_exp * np.log(n_points)
        RMSE_exp = np.sqrt(ss_res_exp / n_points)
        
        results['exponential'] = {
            'params': popt_exp,
            'T2': T2_exp,
            'R2': R2_exp,
            'AIC': AIC_exp,
            'BIC': BIC_exp,
            'RMSE': RMSE_exp,
            'fit_curve': exponential_decay(t, *popt_exp)
        }
    except:
        results['exponential'] = None
    
    # Fit stretched exponential model
    try:
        popt_stretch, pcov_stretch = curve_fit(
            stretched_exponential_decay, t_fit, E_fit,
            p0=[t_fit[-1] / 2, 1.0],
            bounds=([0, 0.1], [np.inf, 2.0])
        )
        T_beta, beta = popt_stretch
        E_stretch = stretched_exponential_decay(t_fit, *popt_stretch)
        ss_res_stretch = np.sum((E_fit - E_stretch)**2)
        ss_tot_stretch = np.sum((E_fit - np.mean(E_fit))**2)
        R2_stretch = 1 - (ss_res_stretch / ss_tot_stretch) if ss_tot_stretch > 0 else 0
        n_params_stretch = 2
        n_points = len(t_fit)
        AIC_stretch = n_points * np.log(ss_res_stretch / n_points) + 2 * n_params_stretch
        BIC_stretch = n_points * np.log(ss_res_stretch / n_points) + n_params_stretch * np.log(n_points)
        RMSE_stretch = np.sqrt(ss_res_stretch / n_points)
        
        # Extract beta uncertainty from covariance matrix (beta is 2nd parameter, index 1)
        beta_std = np.sqrt(pcov_stretch[1, 1]) if pcov_stretch is not None and pcov_stretch.size > 0 else None
        
        results['stretched'] = {
            'params': popt_stretch,
            'T2': T_beta,
            'beta': beta,
            'beta_std': beta_std,
            'R2': R2_stretch,
            'AIC': AIC_stretch,
            'BIC': BIC_stretch,
            'RMSE': RMSE_stretch,
            'fit_curve': stretched_exponential_decay(t, *popt_stretch)
        }
    except:
        results['stretched'] = None
    
    # Select best model
    if model == 'auto':
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            return None
        
        criterion = 'BIC' if use_bic else 'AIC'
        best_model = min(valid_results.keys(), 
                        key=lambda k: valid_results[k].get(criterion, 
                                                           valid_results[k].get('RMSE', 
                                                                                valid_results[k]['AIC'])))
        best_result = valid_results[best_model].copy()
        best_result['model'] = best_model
        best_result['selection_criterion'] = criterion
    else:
        if model not in results or results[model] is None:
            return None
        best_result = results[model].copy()
        best_result['model'] = model
    
    return best_result


def fit_coherence_decay(t, E_magnitude, E_se=None, model='auto', use_bic=False, 
                        tau_c=None, gamma_e=None, B_rms=None):
    """
    Fit coherence magnitude |E(t)| to decay models.
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_magnitude : ndarray
        Magnitude of coherence |E(t)|
    E_se : ndarray, optional
        Standard error of |E(t)| (used for window selection)
    model : str
        Model to use: 'gaussian', 'exponential', 'stretched', or 'auto'
    use_bic : bool
        Whether to use BIC instead of AIC for model selection
    tau_c : float, optional
        Correlation time (for regime-aware window selection)
    gamma_e : float, optional
        Electron gyromagnetic ratio (for regime-aware window selection)
    B_rms : float, optional
        RMS noise amplitude (for regime-aware window selection)
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'model': fitted model name
        - 'params': fitted parameters
        - 'T2': extracted T_2 value (in seconds)
        - 'R2': R-squared coefficient of determination
        - 'AIC': Akaike Information Criterion
        - 'RMSE': Root mean square error
        - 'fit_curve': fitted curve values
    """
    # Select fitting window to avoid late-time bias
    t_fit, E_fit = select_fit_window(t, E_magnitude, E_se=E_se, 
                                     tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms)
    
    if len(t_fit) < 3:
        return None
    
    results = {}
    
    # Fit Gaussian model
    try:
        popt_gauss, pcov_gauss = curve_fit(
            gaussian_decay, t_fit, E_fit,
            p0=[t_fit[-1] / 2],  # Initial guess
            bounds=(0, np.inf)
        )
        T2_gauss = popt_gauss[0]
        E_gauss = gaussian_decay(t_fit, *popt_gauss)
        ss_res_gauss = np.sum((E_fit - E_gauss)**2)
        ss_tot_gauss = np.sum((E_fit - np.mean(E_fit))**2)
        R2_gauss = 1 - (ss_res_gauss / ss_tot_gauss) if ss_tot_gauss > 0 else 0
        n_params_gauss = 1
        n_points = len(t_fit)
        # AIC = n*ln(SS_res/n) + 2*k
        # BIC = n*ln(SS_res/n) + k*ln(n)
        AIC_gauss = n_points * np.log(ss_res_gauss / n_points) + 2 * n_params_gauss
        BIC_gauss = n_points * np.log(ss_res_gauss / n_points) + n_params_gauss * np.log(n_points)
        RMSE_gauss = np.sqrt(ss_res_gauss / n_points)
        
        results['gaussian'] = {
            'params': popt_gauss,
            'T2': T2_gauss,
            'R2': R2_gauss,
            'AIC': AIC_gauss,
            'BIC': BIC_gauss,
            'RMSE': RMSE_gauss,
            'fit_curve': gaussian_decay(t, *popt_gauss)
        }
    except:
        results['gaussian'] = None
    
    # Fit exponential model
    try:
        popt_exp, pcov_exp = curve_fit(
            exponential_decay, t_fit, E_fit,
            p0=[t_fit[-1] / 2],
            bounds=(0, np.inf)
        )
        T2_exp = popt_exp[0]
        E_exp = exponential_decay(t_fit, *popt_exp)
        ss_res_exp = np.sum((E_fit - E_exp)**2)
        ss_tot_exp = np.sum((E_fit - np.mean(E_fit))**2)
        R2_exp = 1 - (ss_res_exp / ss_tot_exp) if ss_tot_exp > 0 else 0
        n_params_exp = 1
        n_points = len(t_fit)
        AIC_exp = n_points * np.log(ss_res_exp / n_points) + 2 * n_params_exp
        BIC_exp = n_points * np.log(ss_res_exp / n_points) + n_params_exp * np.log(n_points)
        RMSE_exp = np.sqrt(ss_res_exp / n_points)
        
        results['exponential'] = {
            'params': popt_exp,
            'T2': T2_exp,
            'R2': R2_exp,
            'AIC': AIC_exp,
            'BIC': BIC_exp,
            'RMSE': RMSE_exp,
            'fit_curve': exponential_decay(t, *popt_exp)
        }
    except:
        results['exponential'] = None
    
    # Fit stretched exponential model
    try:
        popt_stretch, pcov_stretch = curve_fit(
            stretched_exponential_decay, t_fit, E_fit,
            p0=[t_fit[-1] / 2, 1.0],
            bounds=([0, 0.1], [np.inf, 2.0])
        )
        T_beta, beta = popt_stretch
        E_stretch = stretched_exponential_decay(t_fit, *popt_stretch)
        ss_res_stretch = np.sum((E_fit - E_stretch)**2)
        ss_tot_stretch = np.sum((E_fit - np.mean(E_fit))**2)
        R2_stretch = 1 - (ss_res_stretch / ss_tot_stretch) if ss_tot_stretch > 0 else 0
        n_params_stretch = 2
        n_points = len(t_fit)
        AIC_stretch = n_points * np.log(ss_res_stretch / n_points) + 2 * n_params_stretch
        BIC_stretch = n_points * np.log(ss_res_stretch / n_points) + n_params_stretch * np.log(n_points)
        RMSE_stretch = np.sqrt(ss_res_stretch / n_points)
        
        results['stretched'] = {
            'params': popt_stretch,
            'T2': T_beta,
            'beta': beta,
            'R2': R2_stretch,
            'AIC': AIC_stretch,
            'BIC': BIC_stretch,
            'RMSE': RMSE_stretch,
            'fit_curve': stretched_exponential_decay(t, *popt_stretch)
        }
    except:
        results['stretched'] = None
    
    # Select best model
    if model == 'auto':
        # Choose model with lowest information criterion (BIC if use_bic, else AIC)
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            return None
        
        # Use BIC if requested (more conservative, penalizes complexity), else AIC
        criterion = 'BIC' if use_bic else 'AIC'
        best_model = min(valid_results.keys(), 
                        key=lambda k: valid_results[k].get(criterion, 
                                                           valid_results[k].get('RMSE', 
                                                                                valid_results[k]['AIC'])))
        best_result = valid_results[best_model].copy()
        best_result['model'] = best_model
        best_result['selection_criterion'] = criterion
    else:
        if model not in results or results[model] is None:
            return None
        best_result = results[model].copy()
        best_result['model'] = model
    
    return best_result


def theoretical_T2_motional_narrowing(gamma_e, B_rms, tau_c):
    """
    Theoretical prediction for T_2 in the motional-narrowing limit.
    
    T_2 ≈ 1 / [(γ_e * B_rms)^2 * τ_c]
    
    Parameters
    ----------
    gamma_e : float
        Electron gyromagnetic ratio
    B_rms : float
        RMS noise amplitude
    tau_c : float
        Correlation time
        
    Returns
    -------
    T2_theory : float
        Theoretical T_2 value
    """
    return 1.0 / ((gamma_e * B_rms)**2 * tau_c)


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


def filter_function_hahn_echo(tau, omega):
    """
    Filter function for Hahn echo sequence.
    
    |F_echo(ω,τ)|² = 8 sin⁴(ωτ/2)
    
    Parameters
    ----------
    tau : float
        Echo delay τ (seconds)
    omega : ndarray
        Angular frequency array (rad/s)
        
    Returns
    -------
    F_squared : ndarray
        |F_echo(ω,τ)|²
    """
    return 8 * np.sin(omega * tau / 2)**4


def ou_psd_omega(omega, Delta_omega, tau_c):
    """
    Power spectral density of OU noise in frequency domain.
    
    S_ω(ω) = 2(Δω)²τ_c / (1 + ω²τ_c²)
    
    Parameters
    ----------
    omega : ndarray
        Angular frequency array (rad/s)
    Delta_omega : float
        Δω = γ_e * B_rms (rad/s)
    tau_c : float
        Correlation time (seconds)
        
    Returns
    -------
    S_omega : ndarray
        Power spectral density S_ω(ω)
    """
    Delta_omega_sq = Delta_omega**2
    return 2 * Delta_omega_sq * tau_c / (1 + (omega * tau_c)**2)


def analytical_hahn_echo_filter_function(tau_list, gamma_e, B_rms, tau_c, omega_max=None, n_points=5000):
    """
    Analytical Hahn echo coherence using exact filter-function integral.
    
    E_echo(2τ) = exp[-1/π ∫₀^∞ S_ω(ω)/ω² |F_echo(ω,τ)|² dω]
    
    where:
    - |F_echo(ω,τ)|² = 8 sin⁴(ωτ/2)
    - S_ω(ω) = 2(Δω)²τ_c / (1 + ω²τ_c²)
    
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
    omega_max : float, optional
        Maximum frequency for integration (default: 100/τ_c)
    n_points : int
        Number of points for numerical integration
        
    Returns
    -------
    E_echo : ndarray
        Analytical coherence |E_echo(2τ)|
    """
    Delta_omega = gamma_e * B_rms
    tau_list = np.array(tau_list)
    
    if omega_max is None:
        # Use a frequency range that captures the full spectrum
        # For OU noise, most power is below ω_c = 1/τ_c
        # Limit to reasonable range for faster computation
        omega_max = min(100.0 / tau_c, 1e10)  # Cap at 10^10 rad/s
    
    E_echo = np.zeros_like(tau_list)
    
    # Show progress for filter-function calculation
    from tqdm import tqdm
    tau_iterator = tqdm(enumerate(tau_list), total=len(tau_list), 
                       desc=f"Filter-function integral (τc={tau_c*1e6:.2f}μs)", 
                       disable=len(tau_list) < 10)
    
    for i, tau in tau_iterator:
        # Define integrand: S_ω(ω)/ω² |F_echo(ω,τ)|²
        def integrand(omega):
            # Avoid division by zero at ω=0
            omega_safe = np.maximum(omega, 1e-10)
            S = ou_psd_omega(omega_safe, Delta_omega, tau_c)
            F_sq = filter_function_hahn_echo(tau, omega_safe)
            return S / (omega_safe**2) * F_sq
        
        # Integrate from 0 to omega_max
        # Use adaptive quadrature for accuracy
        omega_array = np.logspace(-3, np.log10(omega_max), n_points)
        # Handle ω=0 separately (integrand → 0 as ω→0)
        omega_array = np.concatenate([[0], omega_array])
        
        # Numerical integration using Simpson's rule (faster, sufficient accuracy)
        integrand_vals = integrand(omega_array[1:])  # Skip ω=0
        integral = integrate.simpson(integrand_vals, omega_array[1:])
        
        # Optional: Use quad for better accuracy (slower, comment out for speed)
        # For production runs, Simpson's rule with n_points=10000 is sufficient
        # try:
        #     integral_quad, _ = integrate.quad(integrand, 0, omega_max, limit=100, epsabs=1e-6, epsrel=1e-4)
        #     integral = integral_quad
        # except:
        #     pass
        
        # E_echo(2τ) = exp[-1/π * integral]
        # CRITICAL: Prevent underflow by clipping the exponent
        chi_echo = integral / np.pi
        chi_echo_clipped = np.clip(chi_echo, 0.0, 700.0)
        E_echo[i] = np.exp(-chi_echo_clipped)
    
    return E_echo


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


def fit_mn_slope(results, gamma_e, B_rms, xi_threshold=0.2):
    """
    Fit T_2 vs tau_c in motional-narrowing regime (ξ < threshold).
    
    Performs linear regression on log-log scale to extract slope.
    
    Parameters
    ----------
    results : list
        List of simulation result dictionaries
    gamma_e : float
        Electron gyromagnetic ratio
    B_rms : float
        RMS noise amplitude
    xi_threshold : float
        Maximum ξ for MN regime (default: 0.2)
        
    Returns
    -------
    fit_result : dict
        Dictionary containing:
        - 'slope': fitted slope
        - 'slope_std': standard error of slope
        - 'R2': R-squared
        - 'n_points': number of points used
        - 'tau_c_range': (min, max) tau_c range
        - 'T2_range': (min, max) T2 range
    """
    Delta_omega = gamma_e * B_rms
    
    # Extract data in MN regime
    tau_c_mn = []
    T2_mn = []
    
    for r in results:
        if r.get('fit_result') is not None:
            tau_c = r['tau_c']
            T2 = r['fit_result']['T2']
            xi = Delta_omega * tau_c
            
            if xi < xi_threshold and T2 > 0:
                tau_c_mn.append(tau_c)
                T2_mn.append(T2)
    
    if len(tau_c_mn) < 3:
        return None
    
    tau_c_mn = np.array(tau_c_mn)
    T2_mn = np.array(T2_mn)
    
    # Linear regression on log-log scale
    log_tau = np.log10(tau_c_mn)
    log_T2 = np.log10(T2_mn)
    
    # Use np.polyfit with covariance
    coeffs, cov = np.polyfit(log_tau, log_T2, 1, cov=True)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Standard error of slope
    slope_std = np.sqrt(cov[0, 0])
    
    # R-squared
    log_T2_pred = slope * log_tau + intercept
    ss_res = np.sum((log_T2 - log_T2_pred)**2)
    ss_tot = np.sum((log_T2 - np.mean(log_T2))**2)
    R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    fit_result = {
        'slope': slope,
        'slope_std': slope_std,
        'intercept': intercept,
        'R2': R2,
        'n_points': len(tau_c_mn),
        'tau_c_range': (tau_c_mn.min(), tau_c_mn.max()),
        'T2_range': (T2_mn.min(), T2_mn.max()),
        'xi_range': (Delta_omega * tau_c_mn.min(), Delta_omega * tau_c_mn.max())
    }
    
    return fit_result


def weights_from_E(E, M):
    """
    Compute fitting weights from coherence values and ensemble size.
    
    For heteroscedastic data, weights should be inversely proportional to variance.
    For coherence |E|, the variance is approximately: Var(|E|) ≈ (1 - |E|²)/(2M)
    
    Parameters
    ----------
    E : ndarray
        Coherence values |E(t)|
    M : int
        Number of ensemble realizations
        
    Returns
    -------
    weights : ndarray
        Fitting weights w(t) = 1/Var(|E(t)|)
    """
    # Variance: Var(|E|) ≈ (1 - |E|²)/(2M)
    # Avoid division by zero: use maximum with small epsilon
    v = np.maximum(1e-12, (1.0 - np.abs(E)**2) / (2.0 * M))
    weights = 1.0 / v
    # Normalize weights to avoid numerical issues
    weights = weights / np.max(weights)
    return weights


def decay_with_offset(t, E_func, A, B, *params):
    """
    Decay model with scale and offset: y(t) = A * E(t) + B
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_func : callable
        Base decay function E(t, *params)
    A : float
        Scale factor (A ∈ [0.9, 1.1])
    B : float
        Offset (B ∈ [0, 0.05])
    *params : tuple
        Parameters for E_func
        
    Returns
    -------
    y : ndarray
        y(t) = A * E(t, *params) + B
    """
    return A * E_func(t, *params) + B


def fit_coherence_decay_with_offset(t, E_magnitude, E_se=None, model='auto', use_bic=False, 
                                    is_echo=False, tau_c=None, gamma_e=None, B_rms=None, M=None):
    """
    Fit coherence magnitude |E(t)| to decay models with scale and offset.
    
    Fits y(t) = A * E(t) + B where:
    - A ∈ [0.9, 1.1]
    - B ∈ [0, 0.05]
    
    Uses weighted least squares for heteroscedastic data (weights ∝ 1/Var(|E|)).
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_magnitude : ndarray
        Magnitude of coherence |E(t)|
    E_se : ndarray, optional
        Standard error of |E(t)| (used for window selection)
    model : str
        Model to use: 'gaussian', 'exponential', 'stretched', or 'auto'
    use_bic : bool
        Whether to use BIC instead of AIC for model selection
    is_echo : bool
        If True, use echo-optimized window selection (more conservative)
    tau_c : float, optional
        Correlation time (for regime-aware window selection)
    gamma_e : float, optional
        Electron gyromagnetic ratio (for regime-aware window selection)
    B_rms : float, optional
        RMS noise amplitude (for regime-aware window selection)
    M : int, optional
        Number of ensemble realizations (for weighted fitting)
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'model': fitted model name
        - 'params': fitted parameters (including A, B)
        - 'T2': extracted T_2 value (in seconds, from χ(t) = 1)
        - 'R2': R-squared coefficient of determination
        - 'AIC': Akaike Information Criterion
        - 'RMSE': Root mean square error
        - 'fit_curve': fitted curve values
        - 'A': scale factor
        - 'B': offset
    """
    # ===== CRITICAL FIX 1: Detect flat curves before fitting =====
    # If curve shows negligible decay, use analytical estimate immediately
    # FIXED: Use initial-to-final decay ratio instead of range/mean
    # This correctly identifies flat curves even when noise causes small fluctuations
    if len(E_magnitude) > 1:
        E_initial = E_magnitude[0]
        E_final = E_magnitude[-1]
        # Use relative decay: (initial - final) / initial
        # This measures actual decay, not just noise fluctuations
        relative_decay = (E_initial - E_final) / E_initial if E_initial > 0 else 0
    else:
        relative_decay = 0
    
    # RESEARCH NOTE: For dissertation, we use strict threshold (1%) to ensure
    # only truly flat curves use analytical estimates. This allows proper
    # validation of motional narrowing theory through actual fitting.
    # The previous approach (range/mean) was incorrectly classifying decay curves
    # as flat when noise caused small fluctuations around a decaying trend.
    # CRITICAL FIX: For echo, use very strict threshold (0.1%) to catch nearly-flat curves
    # Even small noise can cause relative_decay > 0.5%, but if decay is truly negligible,
    # we should use analytical estimate to avoid underestimating T2_echo
    decay_threshold = 0.001 if is_echo else 0.01  # Echo: 0.1%, FID: 1%
    if relative_decay < decay_threshold:  # Less than threshold decay from initial to final
        # Curve is essentially flat - fitting will fail
        if tau_c is not None and gamma_e is not None and B_rms is not None:
            if is_echo:
                # CRITICAL FIX: For echo, if decay is negligible, T2_echo must be >> t_max
                # E(t_max) ≈ 1.0 means T2 >> t_max
                # Use conservative estimate: T2_echo = 10 * t_max (minimum)
                # This ensures gain calculation is physically reasonable
                t_max = t[-1] if len(t) > 0 else 0
                if t_max > 0:
                    # For echo that doesn't decay, T2 must be much larger than simulation time
                    # Use extrapolation: E(t_max) = exp(-t_max/T2) ≈ 1.0
                    # If E(t_max) > 0.99, then T2 > t_max / (-ln(0.99)) ≈ 100 * t_max
                    # Use conservative estimate: T2 = 50 * t_max (minimum)
                    E_final = E_magnitude[-1] if len(E_magnitude) > 0 else 1.0
                    if E_final > 0.99:
                        # Echo barely decayed - T2 must be very large
                        # 물리학적으로 타당한 추정:
                        # E(t_max) = exp(-t_max/T2) ≈ 1.0
                        # T2 >> t_max
                        # 하지만 T_FID와의 관계도 고려해야 함
                        # 보수적으로: T2_echo >= 100 * t_max (최소)
                        # 하지만 T_FID가 t_max보다 크면, T2_echo >= 50 * T_FID도 고려
                        # 실제로는 echo가 거의 decay하지 않으므로 매우 큰 값이어야 함
                        T2_analytical = max(100.0 * t_max, 50.0e-6)  # 최소 50 μs, 이상적으로 100×t_max
                        # Cap at reasonable maximum (50 ms) to avoid unphysical values
                        T2_analytical = min(T2_analytical, 50.0e-3)
                    else:
                        # Some decay but still flat - use extrapolation
                        T2_analytical = t_max / (-np.log(E_final)) if E_final > 0 else 100.0e-6
                        T2_analytical = max(T2_analytical, 20.0 * t_max)  # Conservative minimum
                        T2_analytical = min(T2_analytical, 50.0e-3)  # Cap at 50 ms
                else:
                    # Fallback to FID estimate if t_max is invalid
                    T2_analytical = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
            else:
                # For FID, use standard analytical estimate
                T2_analytical = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
            
            return {
                'T2': T2_analytical,
                'model': 'analytical_flat_curve',
                'note': f'Decay negligible ({relative_decay*100:.1f}%), using analytical estimate',
                'R2': np.nan,
                'AIC': np.nan,
                'BIC': np.nan,
                'RMSE': np.nan,
                'is_analytical': True,
                'params': {'T2': T2_analytical, 'A': 1.0, 'B': 0.0},
                'A': 1.0,
                'B': 0.0,
                'fit_curve': None
            }
        else:
            # No parameters for analytical estimate, return failure
            return None
    # ===== END CRITICAL FIX 1 =====
    
    # Select fitting window: use echo-optimized window for echo signals
    if is_echo:
        t_fit, E_fit = select_echo_fit_window(t, E_magnitude, E_se=E_se,
                                              tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms)
    else:
        t_fit, E_fit = select_fit_window(t, E_magnitude, E_se=E_se,
                                       tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms)
    
    if len(t_fit) < 3:
        return None
    
    # Compute weights for weighted least squares (if M is provided)
    # For heteroscedastic data, weights w(t) ≈ 1/Var(|E(t)|) ≈ 2M/(1-|E|²)
    weights = None
    if M is not None and M > 0:
        # E_fit is already the values at t_fit from select_fit_window
        # Use E_fit directly (more accurate than interpolation)
        weights = weights_from_E(E_fit, M)
    
    results = {}
    
    # Fit Gaussian model with offset
    try:
        def gaussian_with_offset(t, T2_star, A, B):
            return decay_with_offset(t, gaussian_decay, A, B, T2_star)
        
        # ===== CRITICAL FIX 2: For QS regime, use analytical estimate as initial guess =====
        # For QS regime, use analytical estimate as initial guess
        if tau_c is not None and gamma_e is not None and B_rms is not None:
            Delta_omega = gamma_e * B_rms
            xi = Delta_omega * tau_c
            if xi > 2.0:  # QS regime
                T2_guess_qs = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
                # Use analytical estimate as initial guess
                p0_gauss = [T2_guess_qs, 1.0, 0.0]
            else:
                # Original initial guess
                p0_gauss = [t_fit[-1] / 2, 1.0, 0.0]
        else:
            p0_gauss = [t_fit[-1] / 2, 1.0, 0.0]
        # ===== END CRITICAL FIX 2 =====
        
        # Use weighted least squares if weights are available
        fit_kwargs = {
            'p0': p0_gauss,  # Use modified initial guess
            'bounds': ([0, 0.9, 0], [np.inf, 1.1, 0.05])
        }
        if weights is not None:
            fit_kwargs['sigma'] = 1.0 / np.sqrt(weights)  # curve_fit uses sigma, not weights directly
        
        popt_gauss, pcov_gauss = curve_fit(
            gaussian_with_offset, t_fit, E_fit,
            **fit_kwargs
        )
        T2_gauss, A_gauss, B_gauss = popt_gauss
        E_gauss = gaussian_with_offset(t_fit, *popt_gauss)
        ss_res_gauss = np.sum((E_fit - E_gauss)**2)
        ss_tot_gauss = np.sum((E_fit - np.mean(E_fit))**2)
        R2_gauss = 1 - (ss_res_gauss / ss_tot_gauss) if ss_tot_gauss > 0 else 0
        n_params_gauss = 3
        n_points = len(t_fit)
        AIC_gauss = n_points * np.log(ss_res_gauss / n_points) + 2 * n_params_gauss
        BIC_gauss = n_points * np.log(ss_res_gauss / n_points) + n_params_gauss * np.log(n_points)
        RMSE_gauss = np.sqrt(ss_res_gauss / n_points)
        
        # Extract T2 from χ(t) = 1 (i.e., E = 1/e)
        # For Gaussian: E(t) = exp[-(t/T2)^2], so E = 1/e when t = T2
        # But we need to account for offset: (A*E + B) = 1/e
        # For pure decay: E = (1/e - B)/A, so t where E = (1/e - B)/A
        E_target = (1.0/np.e - B_gauss) / A_gauss if A_gauss > 0 else 1.0/np.e
        if E_target > 0:
            T2_gauss = T2_gauss * np.sqrt(-np.log(E_target))
        else:
            T2_gauss = T2_gauss  # Fallback to fitted value
        
        results['gaussian'] = {
            'params': popt_gauss,
            'T2': T2_gauss,
            'R2': R2_gauss,
            'AIC': AIC_gauss,
            'BIC': BIC_gauss,
            'RMSE': RMSE_gauss,
            'fit_curve': gaussian_with_offset(t, *popt_gauss),
            'A': A_gauss,
            'B': B_gauss
        }
    except:
        results['gaussian'] = None
    
    # Fit exponential model with offset
    try:
        def exponential_with_offset(t, T2, A, B):
            return decay_with_offset(t, exponential_decay, A, B, T2)
        
        # Use weighted least squares if weights are available
        fit_kwargs_exp = {
            'p0': [t_fit[-1] / 2, 1.0, 0.0],
            'bounds': ([0, 0.9, 0], [np.inf, 1.1, 0.05])
        }
        if weights is not None:
            fit_kwargs_exp['sigma'] = 1.0 / np.sqrt(weights)
        
        popt_exp, pcov_exp = curve_fit(
            exponential_with_offset, t_fit, E_fit,
            **fit_kwargs_exp
        )
        T2_exp, A_exp, B_exp = popt_exp
        E_exp = exponential_with_offset(t_fit, *popt_exp)
        ss_res_exp = np.sum((E_fit - E_exp)**2)
        ss_tot_exp = np.sum((E_fit - np.mean(E_fit))**2)
        R2_exp = 1 - (ss_res_exp / ss_tot_exp) if ss_tot_exp > 0 else 0
        n_params_exp = 3
        n_points = len(t_fit)
        AIC_exp = n_points * np.log(ss_res_exp / n_points) + 2 * n_params_exp
        BIC_exp = n_points * np.log(ss_res_exp / n_points) + n_params_exp * np.log(n_points)
        RMSE_exp = np.sqrt(ss_res_exp / n_points)
        
        # Extract T2 from χ(t) = 1 (i.e., E = 1/e)
        E_target = (1.0/np.e - B_exp) / A_exp if A_exp > 0 else 1.0/np.e
        if E_target > 0:
            T2_exp = -T2_exp * np.log(E_target)
        else:
            T2_exp = T2_exp  # Fallback to fitted value
        
        results['exponential'] = {
            'params': popt_exp,
            'T2': T2_exp,
            'R2': R2_exp,
            'AIC': AIC_exp,
            'BIC': BIC_exp,
            'RMSE': RMSE_exp,
            'fit_curve': exponential_with_offset(t, *popt_exp),
            'A': A_exp,
            'B': B_exp
        }
    except:
        results['exponential'] = None
    
    # Fit stretched exponential model with offset
    try:
        def stretched_with_offset(t, T_beta, beta, A, B):
            return decay_with_offset(t, stretched_exponential_decay, A, B, T_beta, beta)
        
        # Use weighted least squares if weights are available
        fit_kwargs_stretch = {
            'p0': [t_fit[-1] / 2, 1.0, 1.0, 0.0],
            'bounds': ([0, 0.1, 0.9, 0], [np.inf, 2.0, 1.1, 0.05])
        }
        if weights is not None:
            fit_kwargs_stretch['sigma'] = 1.0 / np.sqrt(weights)
        
        popt_stretch, pcov_stretch = curve_fit(
            stretched_with_offset, t_fit, E_fit,
            **fit_kwargs_stretch
        )
        T_beta, beta, A_stretch, B_stretch = popt_stretch
        E_stretch = stretched_with_offset(t_fit, *popt_stretch)
        ss_res_stretch = np.sum((E_fit - E_stretch)**2)
        ss_tot_stretch = np.sum((E_fit - np.mean(E_fit))**2)
        R2_stretch = 1 - (ss_res_stretch / ss_tot_stretch) if ss_tot_stretch > 0 else 0
        n_params_stretch = 4
        n_points = len(t_fit)
        AIC_stretch = n_points * np.log(ss_res_stretch / n_points) + 2 * n_params_stretch
        BIC_stretch = n_points * np.log(ss_res_stretch / n_points) + n_params_stretch * np.log(n_points)
        RMSE_stretch = np.sqrt(ss_res_stretch / n_points)
        
        # Extract beta uncertainty from covariance matrix (beta is 2nd parameter, index 1)
        beta_std = np.sqrt(pcov_stretch[1, 1]) if pcov_stretch is not None and pcov_stretch.size > 0 else None
        
        # Extract T2 from χ(t) = 1 (i.e., E = 1/e)
        E_target = (1.0/np.e - B_stretch) / A_stretch if A_stretch > 0 else 1.0/np.e
        if E_target > 0:
            T2_stretch = T_beta * (-np.log(E_target))**(1.0/beta) if beta > 0 else T_beta
        else:
            T2_stretch = T_beta  # Fallback to fitted value
        
        results['stretched'] = {
            'params': popt_stretch,
            'T2': T2_stretch,
            'beta': beta,
            'beta_std': beta_std,
            'R2': R2_stretch,
            'AIC': AIC_stretch,
            'BIC': BIC_stretch,
            'RMSE': RMSE_stretch,
            'fit_curve': stretched_with_offset(t, *popt_stretch),
            'A': A_stretch,
            'B': B_stretch
        }
    except:
        results['stretched'] = None
    
    # Select best model
    if model == 'auto':
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            return None
        
        # DISSERTATION FIX: Regime-aware model selection
        # - MN regime (ξ < 1): Prefer exponential (theoretically correct: T2 ∝ 1/τ_c)
        # - Static regime (ξ > 2): Prefer Gaussian (theoretically correct: E(t) ≈ exp[-(t/T2)^2])
        criterion = 'BIC' if use_bic else 'AIC'
        
        if tau_c is not None and gamma_e is not None and B_rms is not None:
            Delta_omega = gamma_e * B_rms
            xi = Delta_omega * tau_c
            
            if xi < 1.0 and 'exponential' in valid_results:
                # Motional narrowing regime: exponential is theoretically correct
                # T2 = 1/(Δω²τ_c), so log(T2) = -log(τ_c) + const (slope = -1)
                exponential_result = valid_results['exponential']
                
                # Find best model by criterion
                best_by_criterion = min(valid_results.keys(), 
                                       key=lambda k: valid_results[k].get(criterion, 
                                                                        valid_results[k].get('RMSE', 
                                                                                             valid_results[k]['AIC'])))
                best_criterion_value = valid_results[best_by_criterion].get(criterion, 
                                                                           valid_results[best_by_criterion].get('RMSE', 
                                                                                                                valid_results[best_by_criterion]['AIC']))
                exponential_criterion_value = exponential_result.get(criterion, 
                                                                     exponential_result.get('RMSE', 
                                                                                            exponential_result['AIC']))
                
                # Use exponential if it's within 20% of best, or if R2 > 0.9
                # More lenient than Gaussian because MN regime can have noise
                if (exponential_result.get('R2', 0) > 0.9 or 
                    (best_criterion_value != 0 and abs(exponential_criterion_value - best_criterion_value) / abs(best_criterion_value) < 0.2)):
                    best_model = 'exponential'
                    best_result = exponential_result.copy()
                    best_result['model'] = 'exponential'
                    best_result['selection_criterion'] = f'{criterion} (MN regime: preferred exponential)'
                else:
                    # Use best by criterion
                    best_model = best_by_criterion
                    best_result = valid_results[best_model].copy()
                    best_result['model'] = best_model
                    best_result['selection_criterion'] = criterion
            elif xi > 2.0 and 'gaussian' in valid_results:
                # Static regime: Gaussian is theoretically correct
                # Only use Gaussian if it's reasonable (R2 > 0.9 or close to best)
                gaussian_result = valid_results['gaussian']
                
                # Find best model by criterion
                best_by_criterion = min(valid_results.keys(), 
                                       key=lambda k: valid_results[k].get(criterion, 
                                                                        valid_results[k].get('RMSE', 
                                                                                             valid_results[k]['AIC'])))
                best_criterion_value = valid_results[best_by_criterion].get(criterion, 
                                                                           valid_results[best_by_criterion].get('RMSE', 
                                                                                                                valid_results[best_by_criterion]['AIC']))
                gaussian_criterion_value = gaussian_result.get(criterion, 
                                                               gaussian_result.get('RMSE', 
                                                                                   gaussian_result['AIC']))
                
                # Use Gaussian if it's within 10% of best, or if R2 > 0.9
                if (gaussian_result.get('R2', 0) > 0.9 or 
                    (best_criterion_value != 0 and abs(gaussian_criterion_value - best_criterion_value) / abs(best_criterion_value) < 0.1)):
                    best_model = 'gaussian'
                    best_result = gaussian_result.copy()
                    best_result['model'] = 'gaussian'
                    best_result['selection_criterion'] = f'{criterion} (static regime: preferred Gaussian)'
                else:
                    # Use best by criterion
                    best_model = best_by_criterion
                    best_result = valid_results[best_model].copy()
                    best_result['model'] = best_model
                    best_result['selection_criterion'] = criterion
            else:
                # Normal model selection (crossover regime or no regime info)
                best_model = min(valid_results.keys(), 
                                key=lambda k: valid_results[k].get(criterion, 
                                                                   valid_results[k].get('RMSE', 
                                                                                        valid_results[k]['AIC'])))
                best_result = valid_results[best_model].copy()
                best_result['model'] = best_model
                best_result['selection_criterion'] = criterion
        else:
            # Normal model selection (no regime info)
            best_model = min(valid_results.keys(), 
                            key=lambda k: valid_results[k].get(criterion, 
                                                               valid_results[k].get('RMSE', 
                                                                                    valid_results[k]['AIC'])))
            best_result = valid_results[best_model].copy()
            best_result['model'] = best_model
            best_result['selection_criterion'] = criterion
    else:
        if model not in results or results[model] is None:
            return None
        best_result = results[model].copy()
        best_result['model'] = model
    
    # CRITICAL FIX: Check for negative R² (fitting failure)
    # Negative R² means fit is worse than horizontal line (complete failure)
    if 'R2' in best_result and best_result['R2'] is not None:
        R2_value = best_result['R2']
        if R2_value < 0:
            # Fitting completely failed - use analytical estimate
            if tau_c is not None and gamma_e is not None and B_rms is not None:
                T2_analytical = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
                best_result['T2'] = T2_analytical
                best_result['model'] = f"{best_result.get('model', 'unknown')}_analytical"
                best_result['note'] = f'Fitting failed (R² = {R2_value:.4f}), using analytical estimate'
                best_result['R2'] = np.nan  # Mark as analytical (no fit quality)
                best_result['is_analytical'] = True
            else:
                # No parameters for analytical estimate - return None to indicate failure
                return None
    
    # CRITICAL FIX: Sanity check on T2 value to prevent extremely large values
    # T2 values > 1 second (1e6 μs) are unphysical for spin coherence
    # If T2 is unreasonably large, use analytical estimate for QS regime
    if 'T2' in best_result and best_result['T2'] is not None:
        T2_value = best_result['T2']
        # Reasonable maximum: 1 second = 1e6 μs (extremely long for spin coherence)
        T2_max_reasonable = 1.0  # seconds
        if T2_value > T2_max_reasonable:
            # T2 is unreasonably large, likely due to fitting failure
            # Use analytical estimate if in QS regime
            if tau_c is not None and gamma_e is not None and B_rms is not None:
                Delta_omega = gamma_e * B_rms
                xi = Delta_omega * tau_c
                if xi > 2.0:  # QS regime
                    T2_analytical = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
                    
                    # Replace with analytical estimate
                    best_result['T2'] = T2_analytical
                    best_result['model'] = 'gaussian_analytical'
                    best_result['note'] = f'Fitted T2 ({T2_value*1e6:.2e} μs) unreasonably large, using analytical QS estimate ({T2_analytical*1e6:.2f} μs)'
                    best_result['is_analytical'] = True
                    best_result['params'] = {'T2_star': T2_analytical, 'A': best_result.get('A', 1.0), 'B': best_result.get('B', 0.0)}
                else:
                    # Not QS regime, but T2 is still too large - cap it
                    best_result['T2'] = T2_max_reasonable
                    best_result['note'] = f'Fitted T2 ({T2_value*1e6:.2e} μs) capped at {T2_max_reasonable*1e6:.2e} μs'
            else:
                # Missing parameters, just cap it
                best_result['T2'] = T2_max_reasonable
                best_result['note'] = f'Fitted T2 ({T2_value*1e6:.2e} μs) capped at {T2_max_reasonable*1e6:.2e} μs'
    
    return best_result


def extract_T2_from_chi(t, E_abs, E_fit_curve=None):
    """
    Extract T₂ as the time where χ(t) = 1 (i.e., E = 1/e).
    
    This provides a consistent definition of T₂ across all decay models.
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_abs : ndarray
        Magnitude of coherence |E(t)|
    E_fit_curve : ndarray, optional
        Fitted coherence curve (if None, uses E_abs)
        
    Returns
    -------
    T2 : float
        T₂ value (seconds), or None if not found
    """
    if E_fit_curve is None:
        E_fit_curve = E_abs
    
    E_target = 1.0 / np.e  # E = 1/e when χ = 1
    
    # Find where E_fit_curve crosses E_target
    # Interpolate to find exact crossing point
    if np.any(E_fit_curve > E_target) and np.any(E_fit_curve < E_target):
        # Find crossing point
        idx = np.where(E_fit_curve <= E_target)[0]
        if len(idx) > 0:
            idx_cross = idx[0]
            if idx_cross > 0:
                # Linear interpolation
                t1, t2 = t[idx_cross - 1], t[idx_cross]
                E1, E2 = E_fit_curve[idx_cross - 1], E_fit_curve[idx_cross]
                if E2 != E1:
                    T2 = t1 + (E_target - E1) * (t2 - t1) / (E2 - E1)
                else:
                    T2 = t[idx_cross]
            else:
                T2 = t[0]
        else:
            # Extrapolate from last point
            if len(E_fit_curve) > 1:
                # Use exponential extrapolation
                E_last = E_fit_curve[-1]
                t_last = t[-1]
                if E_last > 0:
                    # Assume exponential decay: E = E0 * exp(-t/T2)
                    # E_target = E_last * exp(-(T2 - t_last)/T2_est)
                    # Solve for T2
                    T2_est = t_last / (-np.log(E_last))
                    T2 = T2_est * (-np.log(E_target / E_last))
                else:
                    T2 = None
            else:
                T2 = None
    elif np.all(E_fit_curve > E_target):
        # All values above target, extrapolate forward
        if len(E_fit_curve) > 1:
            E_last = E_fit_curve[-1]
            t_last = t[-1]
            if E_last > 0:
                T2_est = t_last / (-np.log(E_last))
                T2 = T2_est * (-np.log(E_target / E_last))
            else:
                T2 = None
        else:
            T2 = None
    else:
        # All values below target, use first point
        T2 = t[0] if len(t) > 0 else None
    
    return T2


