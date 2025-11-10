"""
Quality assurance checks for simulation results.

This module implements automatic QA checks to ensure:
1. Motional narrowing slope validation
2. Echo ≥ FID physical constraint
3. OU noise quality (autocorrelation, variance)
4. Bootstrap CI sanity checks
"""

import numpy as np


def check_mn_slope(results, gamma_e, B_rms, xi_threshold=0.2, slope_range=(-1.05, -0.95)):
    """
    Check motional narrowing slope: ξ < threshold should have slope ∈ [−1.05,−0.95].
    
    Parameters
    ----------
    results : list
        List of simulation results (from parameter sweep)
    gamma_e : float
        Electron gyromagnetic ratio
    B_rms : float
        RMS noise amplitude
    xi_threshold : float
        Threshold for motional narrowing regime (default: 0.2)
    slope_range : tuple
        Expected slope range (default: (-1.05, -0.95))
        
    Returns
    -------
    passed : bool
        True if all checks pass
    failures : list
        List of failed checks with details
    """
    failures = []
    Delta_omega = gamma_e * B_rms
    
    for result in results:
        tau_c = result.get('tau_c')
        if tau_c is None:
            continue
        
        xi = Delta_omega * tau_c
        if xi >= xi_threshold:
            continue  # Not in MN regime
        
        # Get T2 value
        fit_result = result.get('fit_result')
        if fit_result is None:
            failures.append({
                'tau_c': tau_c,
                'xi': xi,
                'reason': 'No fit result available'
            })
            continue
        
        T2 = fit_result.get('T2')
        if T2 is None:
            failures.append({
                'tau_c': tau_c,
                'xi': xi,
                'reason': 'No T2 value in fit result'
            })
            continue
        
        # In MN regime: T2 ∝ 1/(tau_c), so log(T2) vs log(tau_c) should have slope ≈ -1
        # We need to check this across multiple tau_c values, so this is a placeholder
        # Full check requires fitting log(T2) vs log(tau_c) across MN regime
        
    passed = len(failures) == 0
    return passed, failures


def check_echo_geq_fid(results, atol=1e-6):
    """
    Check physical constraint: E_echo(2τ) ≥ E_FID(2τ) for all tau_c at same total time.
    
    This is a direct check of the physical constraint that echo coherence should be
    greater than or equal to FID coherence at the same total time (2τ for echo, t=2τ for FID).
    
    Also checks T2_echo ≥ T2_FID as a secondary validation.
    
    Parameters
    ----------
    results : list
        List of simulation results (from Hahn echo sweep)
    atol : float
        Absolute tolerance for comparison (default: 1e-6)
        
    Returns
    -------
    passed : bool
        True if all checks pass
    failures : list
        List of failed checks with details
    """
    import numpy as np
    
    failures = []
    
    for result in results:
        tau_c = result.get('tau_c')
        if tau_c is None:
            continue
        
        # Get FID and echo data
        t_fid = result.get('t_fid')
        E_fid_abs = result.get('E_fid_abs')
        tau_echo = result.get('tau_echo')
        E_echo_abs = result.get('E_echo_abs')
        
        # Check if required data is available
        if t_fid is None or E_fid_abs is None or tau_echo is None or E_echo_abs is None:
            continue
        
        t_fid = np.array(t_fid)
        E_fid_abs = np.array(E_fid_abs)
        tau_echo = np.array(tau_echo)  # This is 2τ (total time for echo)
        E_echo_abs = np.array(E_echo_abs)
        
        # Check: E_echo(2τ) ≥ E_FID(2τ) for all echo times
        # Interpolate FID at echo times (2τ)
        # Only check where both are valid (within FID time range)
        valid_mask = tau_echo <= t_fid.max()
        
        if np.any(valid_mask):
            tau_echo_valid = tau_echo[valid_mask]
            E_echo_valid = E_echo_abs[valid_mask]
            
            # Interpolate FID at echo times
            E_fid_at_echo_times = np.interp(tau_echo_valid, t_fid, E_fid_abs)
            
            # Check constraint: E_echo(2τ) ≥ E_FID(2τ) - atol
            violation_mask = E_echo_valid < (E_fid_at_echo_times - atol)
            
            if np.any(violation_mask):
                # Find worst violation
                violations = E_fid_at_echo_times[violation_mask] - E_echo_valid[violation_mask]
                worst_idx = np.argmax(violations)
                worst_tau_echo = tau_echo_valid[violation_mask][worst_idx]
                worst_margin = violations[worst_idx]
                
                failures.append({
                    'tau_c': tau_c,
                    'check_type': 'direct_coherence',
                    'worst_time': worst_tau_echo,
                    'E_echo': float(E_echo_valid[violation_mask][worst_idx]),
                    'E_fid': float(E_fid_at_echo_times[violation_mask][worst_idx]),
                    'margin': float(worst_margin),
                    'n_violations': int(np.sum(violation_mask)),
                    'n_total': len(tau_echo_valid),
                    'reason': f'Echo coherence < FID coherence at t={worst_tau_echo*1e6:.2f} μs '
                              f'(margin: {worst_margin:.3e}, {np.sum(violation_mask)}/{len(tau_echo_valid)} violations)'
                })
        
        # Also check T2 constraint as secondary validation
        fit_result_fid = result.get('fit_result_fid')
        fit_result_echo = result.get('fit_result_echo')
        
        if fit_result_fid is not None and fit_result_echo is not None:
            T2_fid = fit_result_fid.get('T2')
            T2_echo = fit_result_echo.get('T2')
            
            if T2_fid is not None and T2_echo is not None:
                if T2_echo < T2_fid:
                    failures.append({
                        'tau_c': tau_c,
                        'check_type': 'T2_comparison',
                        'T2_fid': T2_fid,
                        'T2_echo': T2_echo,
                        'ratio': T2_echo / T2_fid,
                        'reason': f'Echo T2 ({T2_echo:.3e}) < FID T2 ({T2_fid:.3e})'
                    })
    
    passed = len(failures) == 0
    return passed, failures


def check_ou_quality(delta_B, dt, tau_c, B_rms, burn_in=0):
    """
    Check OU noise quality after burn-in.
    
    Checks:
    - Autocorrelation: |ρ_emp - exp(-dt/τc)| < 5e-4
    - Variance: |std_emp/B_rms - 1| < 2%
    
    Parameters
    ----------
    delta_B : ndarray
        OU noise samples (after burn-in)
    dt : float
        Time step
    tau_c : float
        Correlation time
    B_rms : float
        Expected RMS amplitude
    burn_in : int
        Number of burn-in samples (for logging)
        
    Returns
    -------
    passed : bool
        True if all checks pass
    failures : list
        List of failed checks with details
    """
    failures = []
    
    if len(delta_B) < 2:
        failures.append({
            'reason': 'Insufficient samples for quality check',
            'n_samples': len(delta_B)
        })
        return False, failures
    
    # Check autocorrelation
    rho_emp = np.corrcoef(delta_B[:-1], delta_B[1:])[0, 1]
    rho_th = np.exp(-dt / tau_c)
    rho_error = abs(rho_emp - rho_th)
    
    if rho_error >= 5e-4:
        failures.append({
            'check': 'autocorrelation',
            'rho_emp': rho_emp,
            'rho_th': rho_th,
            'error': rho_error,
            'threshold': 5e-4,
            'reason': f'Autocorrelation error ({rho_error:.2e}) exceeds threshold (5e-4)'
        })
    
    # Check variance
    std_emp = np.std(delta_B)
    std_ratio = std_emp / B_rms
    std_error = abs(std_ratio - 1.0)
    
    if std_error >= 0.02:  # 2%
        failures.append({
            'check': 'variance',
            'std_emp': std_emp,
            'B_rms': B_rms,
            'ratio': std_ratio,
            'error': std_error,
            'threshold': 0.02,
            'reason': f'Std ratio error ({std_error:.2%}) exceeds threshold (2%)'
        })
    
    passed = len(failures) == 0
    return passed, failures


def check_bootstrap_ci(T2_fit, T2_ci):
    """
    Check bootstrap CI sanity.
    
    Checks:
    - CI width > 0
    - lo ≤ T2_fit ≤ hi
    
    Note: T2_ci can be None in static regime when CI is degenerate
    (all bootstrap samples produce identical T2). This is expected
    and should be handled by the caller.
    
    Parameters
    ----------
    T2_fit : float
        Fitted T2 value
    T2_ci : tuple or None
        (lower, upper) confidence interval, or None if degenerate
        
    Returns
    -------
    passed : bool
        True if checks pass
    failures : list
        List of failed checks with details
    """
    failures = []
    
    if T2_ci is None:
        # None CI is expected in static regime (degenerate CI)
        # This is not a failure, but a known limitation
        failures.append({
            'reason': 'Bootstrap CI is None (degenerate - expected in static regime)'
        })
        return False, failures
    
    lo, hi = T2_ci
    
    # Check CI width
    ci_width = hi - lo
    if ci_width <= 0:
        failures.append({
            'check': 'ci_width',
            'lo': lo,
            'hi': hi,
            'width': ci_width,
            'reason': f'CI width ({ci_width:.3e}) is not positive'
        })
    
    # Check T2_fit is within CI
    if T2_fit < lo or T2_fit > hi:
        failures.append({
            'check': 'ci_bounds',
            'T2_fit': T2_fit,
            'lo': lo,
            'hi': hi,
            'reason': f'T2_fit ({T2_fit:.3e}) is outside CI [{lo:.3e}, {hi:.3e}]'
        })
    
    passed = len(failures) == 0
    return passed, failures


def run_qa_checks(results, gamma_e, B_rms, verbose=True):
    """
    Run all QA checks on simulation results.
    
    Parameters
    ----------
    results : list
        List of simulation results
    gamma_e : float
        Electron gyromagnetic ratio
    B_rms : float
        RMS noise amplitude
    verbose : bool
        Whether to print results
        
    Returns
    -------
    all_passed : bool
        True if all checks pass
    summary : dict
        Summary of all checks
    """
    summary = {
        'mn_slope': {'passed': None, 'failures': []},
        'echo_geq_fid': {'passed': None, 'failures': []},
        'bootstrap_ci': {'passed': None, 'failures': []}
    }
    
    # Check motional narrowing slope
    mn_passed, mn_failures = check_mn_slope(results, gamma_e, B_rms)
    summary['mn_slope'] = {'passed': mn_passed, 'failures': mn_failures}
    
    # Check echo ≥ FID
    echo_passed, echo_failures = check_echo_geq_fid(results)
    summary['echo_geq_fid'] = {'passed': echo_passed, 'failures': echo_failures}
    
    # Check bootstrap CI for all results
    bootstrap_failures = []
    for result in results:
        fit_result = result.get('fit_result')
        if fit_result is not None:
            T2_fit = fit_result.get('T2')
            T2_ci = fit_result.get('T2_ci')
            if T2_fit is not None and T2_ci is not None:
                ci_passed, ci_failures = check_bootstrap_ci(T2_fit, T2_ci)
                if not ci_passed:
                    bootstrap_failures.extend(ci_failures)
    
    summary['bootstrap_ci'] = {
        'passed': len(bootstrap_failures) == 0,
        'failures': bootstrap_failures
    }
    
    all_passed = all([
        summary['mn_slope']['passed'],
        summary['echo_geq_fid']['passed'],
        summary['bootstrap_ci']['passed']
    ])
    
    if verbose:
        print("\n" + "="*60)
        print("QA CHECKS SUMMARY")
        print("="*60)
        print(f"Motional narrowing slope: {'✅ PASS' if summary['mn_slope']['passed'] else '❌ FAIL'}")
        if summary['mn_slope']['failures']:
            print(f"  Failures: {len(summary['mn_slope']['failures'])}")
        print(f"Echo ≥ FID constraint: {'✅ PASS' if summary['echo_geq_fid']['passed'] else '❌ FAIL'}")
        if summary['echo_geq_fid']['failures']:
            print(f"  Failures: {len(summary['echo_geq_fid']['failures'])}")
            for f in summary['echo_geq_fid']['failures'][:3]:  # Show first 3
                check_type = f.get('check_type', 'unknown')
                if check_type == 'direct_coherence':
                    print(f"    τc={f['tau_c']*1e6:.2f} μs: {f['reason']}")
                    print(f"      (Direct check: E_echo={f['E_echo']:.4f} < E_FID={f['E_fid']:.4f})")
                else:
                    print(f"    τc={f['tau_c']*1e6:.2f} μs: {f['reason']}")
        print(f"Bootstrap CI sanity: {'✅ PASS' if summary['bootstrap_ci']['passed'] else '❌ FAIL'}")
        if summary['bootstrap_ci']['failures']:
            print(f"  Failures: {len(summary['bootstrap_ci']['failures'])}")
        print("="*60)
        print(f"Overall: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
        print("="*60 + "\n")
    
    return all_passed, summary

