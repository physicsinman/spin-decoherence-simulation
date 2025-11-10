"""
Fitting functions for extracting T_2 relaxation times.

This module provides functions to fit coherence decay curves to various
analytical models (Gaussian, exponential, stretched exponential).
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy import integrate


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
            from simulate import estimate_characteristic_T2
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
        
        # Use weighted least squares if weights are available
        fit_kwargs = {
            'p0': [t_fit[-1] / 2, 1.0, 0.0],
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
        
        # For static regime, prefer Gaussian model (theoretically correct)
        # Static regime: ξ >> 1, decay is Gaussian: E(t) ≈ exp[-(t/T2)^2]
        criterion = 'BIC' if use_bic else 'AIC'
        
        if tau_c is not None and gamma_e is not None and B_rms is not None:
            Delta_omega = gamma_e * B_rms
            xi = Delta_omega * tau_c
            if xi > 2.0 and 'gaussian' in valid_results:
                # In static regime, Gaussian is theoretically correct
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
                # Normal model selection
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


def bootstrap_T2(t, E_abs_all, E_se=None, B=500, rng=None, verbose=False, 
                 tau_c=None, gamma_e=None, B_rms=None):
    """
    Bootstrap resampling to estimate T_2 confidence intervals.
    
    CRITICAL FIX: Use fixed fitting window and scalar T2 values only.
    - Each bootstrap sample uses the same fit_window_idx to ensure consistency
    - Only scalar T2 values are stored (not arrays)
    
    Parameters
    ----------
    t : ndarray
        Time array
    E_abs_all : ndarray
        Array of |E| trajectories, shape (M, N_steps)
    E_se : ndarray, optional
        Standard error (used for fitting window selection)
    B : int
        Number of bootstrap samples (default: 500)
    rng : numpy.random.Generator, optional
        Random number generator
    verbose : bool
        Whether to print diagnostic information
    tau_c : float, optional
        Correlation time (for regime-aware window selection)
    gamma_e : float, optional
        Electron gyromagnetic ratio (for regime-aware window selection)
    B_rms : float, optional
        RMS noise amplitude (for regime-aware window selection)
        
    Returns
    -------
    T2_mean : float
        Mean T_2 from bootstrap samples
    T2_ci : tuple
        (lower, upper) 95% confidence interval
    T2_samples : ndarray
        All bootstrap T_2 values (scalar array)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    M = E_abs_all.shape[0]
    
    # CRITICAL FIX: For static regime, use per-sample fitting window to avoid degenerate CI
    # In static regime, fixed window causes all bootstrap samples to produce identical T2
    # Solution: Allow each bootstrap sample to select its own fitting window
    # This introduces natural variation in bootstrap samples
    use_per_sample_window = False
    if tau_c is not None and gamma_e is not None and B_rms is not None:
        Delta_omega = gamma_e * B_rms
        xi = Delta_omega * tau_c
        if xi > 2.0:  # Static regime
            use_per_sample_window = True
            if verbose:
                print(f"  [FIX] Static regime detected: Using per-sample fitting window "
                      f"to avoid degenerate bootstrap CI")
    
    if use_per_sample_window:
        # For static regime: Don't pre-determine window, let each sample select its own
        # This allows natural variation in bootstrap samples
        fit_window_idx = None  # Will be determined per sample
    else:
        # For MN/crossover regime: Use fixed window (original behavior)
        # CRITICAL: Determine fitting window ONCE using original mean
        # This ensures all bootstrap samples use the same window indices
        E_mean_original = np.mean(E_abs_all, axis=0)
        t_fit, E_fit = select_fit_window(t, E_mean_original, E_se=E_se,
                                         tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms)
        
        # Get indices of fitting window
        # Use tolerance-based matching for floating point safety
        fit_window_idx = []
        for t_val in t_fit:
            idx = np.argmin(np.abs(t - t_val))
            if abs(t[idx] - t_val) < 1e-12 * max(abs(t_val), 1.0):
                fit_window_idx.append(idx)
        
        fit_window_idx = np.array(fit_window_idx)
        if len(fit_window_idx) == 0:
            # Fallback: use all indices if matching fails
            fit_window_idx = np.arange(len(t))
        
        # Ensure indices are unique and sorted
        fit_window_idx = np.unique(fit_window_idx)
        fit_window_idx = np.sort(fit_window_idx)
    
    # Pre-allocate array for T2 values (scalars only)
    vals = np.empty(B, dtype=np.float64)
    failed_fits = 0
    
    # Show progress for bootstrap if verbose
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(range(B), desc="Bootstrap", disable=False)
    else:
        iterator = range(B)
    
    for b in iterator:
        # Resample trajectories with replacement
        idx = rng.integers(0, M, size=M)
        
        # Bootstrap mean |E|
        E_boot = np.mean(E_abs_all[idx], axis=0)
        
        # CRITICAL FIX: For static regime, select fitting window per sample
        # This allows natural variation in bootstrap samples
        if use_per_sample_window:
            # Select fitting window for this bootstrap sample
            E_se_boot = np.std(E_abs_all[idx], axis=0, ddof=1) / np.sqrt(M) if E_se is None else E_se
            t_fit_boot, E_fit_boot = select_fit_window(
                t, E_boot, E_se=E_se_boot,
                tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
            )
            E_se_fit_boot = None  # Don't use SE for per-sample window (already incorporated)
        else:
            # Fit using FIXED window indices (original behavior for MN/crossover)
            # Extract data at fixed window indices
            t_fit_boot = t[fit_window_idx]
            E_fit_boot = E_boot[fit_window_idx]
            E_se_fit_boot = E_se[fit_window_idx] if E_se is not None else None
        
        # Fit to get T_2 (use fitting with offset)
        # M is the number of trajectories (for weighted fitting)
        fit_result = fit_coherence_decay_with_offset(
            t_fit_boot, E_fit_boot, E_se=E_se_fit_boot, model='auto',
            tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms, M=M
        )
        
        if fit_result is not None:
            # CRITICAL: Extract scalar T2 value only
            T2_hat = fit_result['T2']
            # Ensure it's a scalar (not array)
            if np.isscalar(T2_hat):
                vals[b] = float(T2_hat)
            else:
                # If somehow it's an array, take first element
                vals[b] = float(T2_hat[0] if hasattr(T2_hat, '__len__') else T2_hat)
        else:
            failed_fits += 1
            # Use NaN for failed fits (will be filtered out)
            vals[b] = np.nan
    
    # Filter out NaN values (failed fits)
    valid_mask = ~np.isnan(vals)
    T2_samples = vals[valid_mask]
    
    if len(T2_samples) == 0:
        if verbose:
            print(f"  WARNING: All {B} bootstrap fits failed")
        return None, None, None
    
    T2_mean = np.mean(T2_samples)
    T2_std = np.std(T2_samples, ddof=1)
    
    # Check for suspiciously low variance (potential bug indicator)
    T2_unique = len(np.unique(T2_samples))
    if T2_std < T2_mean * 1e-10:  # Relative std < 1e-10
        if verbose:
            print(f"  ⚠️  WARNING: Bootstrap T2 samples have very low variance "
                  f"(std={T2_std:.3e}, mean={T2_mean:.3e})")
            print(f"  Unique T2 values: {T2_unique}/{len(T2_samples)}")
            print(f"  This may indicate:")
            print(f"    1. All trajectories are identical (bug in noise generation)")
            print(f"    2. Fitting window selection too restrictive (all samples use same points)")
            print(f"    3. Numerical precision issue")
            # Additional diagnostic
            if T2_unique == 1:
                print(f"  🔴 CRITICAL: All bootstrap samples produced identical T2!")
                print(f"     This is a serious bug. Check noise generation and fitting.")
    
    # 95% confidence interval using percentile method
    T2_ci = np.percentile(T2_samples, [2.5, 97.5])
    
    # Check if CI is degenerate (all bootstrap samples produced identical T2)
    # In static regime, fitting window may be too narrow, making CI meaningless
    ci_width = abs(T2_ci[1] - T2_ci[0])
    is_degenerate = (ci_width < T2_mean * 1e-10) or (T2_unique == 1)
    
    if is_degenerate:
        if verbose:
            print(f"  ⚠️  WARNING: Bootstrap CI is degenerate (all samples produced identical T2)")
            print(f"     CI width: {ci_width:.3e}, Unique values: {T2_unique}/{len(T2_samples)}")
            print(f"     This occurs in static regime when fitting window is too narrow.")
            print(f"     Returning None for CI (fitted T2 value is still accurate).")
        # Return None for CI when it's truly degenerate (meaningless)
        # The fitted T2 value itself is still accurate
        T2_ci = None
    elif ci_width < T2_mean * 1e-6:  # Very narrow but not completely degenerate
        if verbose:
            print(f"  WARNING: Percentile CI is very narrow [{T2_ci[0]:.6e}, {T2_ci[1]:.6e}]")
            print(f"  Using std-based CI as fallback (1.96 * std)")
        # Use normal approximation: CI = mean ± 1.96 * std
        T2_ci = (T2_mean - 1.96 * T2_std, T2_mean + 1.96 * T2_std)
    
    if verbose:
        print(f"  Bootstrap: {len(T2_samples)}/{B} successful fits, "
              f"{failed_fits} failed")
        print(f"  T2_mean = {T2_mean:.6e}, T2_std = {T2_std:.6e}")
        if T2_ci is not None:
            print(f"  T2_ci = [{T2_ci[0]:.6e}, {T2_ci[1]:.6e}]")
        else:
            print(f"  T2_ci = None (degenerate - all samples produced identical T2)")
    
    return T2_mean, T2_ci, T2_samples

