"""
Real-time validation and monitoring during simulation.

This module provides classes for validating simulation parameters and
monitoring convergence during execution.
"""

import numpy as np
from typing import List, Optional, Dict
from parameter_validation import SimulationParameters


class SimulationMonitor:
    """
    Real-time validation and monitoring during simulation.
    
    This class checks parameter consistency, convergence, and compares
    results with literature values.
    """
    
    def __init__(self, params: SimulationParameters):
        """
        Initialize simulation monitor.
        
        Parameters
        ----------
        params : SimulationParameters
            Validated simulation parameters
        """
        self.params = params
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.checks_passed = 0
        self.checks_total = 0
    
    def check_noise_amplitude(self) -> bool:
        """
        Check if noise amplitude is appropriate for target T2*.
        
        Returns
        -------
        valid : bool
            True if noise amplitude is consistent with target T2*
        """
        self.checks_total += 1
        
        sigma_z = self.params.sigma_z
        gamma_e = self.params.gamma_e
        
        # Expected T2* from current sigma_z
        T2_expected = np.sqrt(2.0) / (gamma_e * sigma_z)
        T2_target = self.params.T2_star_target
        
        ratio = T2_expected / T2_target
        
        if not (0.5 < ratio < 2.0):
            warning = (
                f"Noise amplitude gives T2* = {T2_expected*1e6:.3f} µs, "
                f"target is {T2_target*1e6:.3f} µs (ratio = {ratio:.2f})"
            )
            self.warnings.append(warning)
            return False
        
        self.checks_passed += 1
        return True
    
    def check_simulation_time(self) -> bool:
        """
        Check if simulation time is sufficient for target T2*.
        
        Returns
        -------
        valid : bool
            True if simulation time is sufficient
        """
        self.checks_total += 1
        
        required_T_max = 5 * self.params.T2_star_target
        actual_T_max = self.params.total_time
        
        if actual_T_max < required_T_max:
            warning = (
                f"Simulation time ({actual_T_max*1e6:.1f} µs) is insufficient "
                f"for T2* = {self.params.T2_star_target*1e6:.3f} µs. "
                f"Required: ≥ {required_T_max*1e6:.1f} µs"
            )
            self.warnings.append(warning)
            return False
        
        self.checks_passed += 1
        return True
    
    def check_time_step(self, tau_c: float) -> bool:
        """
        Check if time step is appropriate for given tau_c.
        
        Parameters
        ----------
        tau_c : float
            Correlation time (seconds)
        
        Returns
        -------
        valid : bool
            True if time step is appropriate
        """
        self.checks_total += 1
        
        dt = self.params.dt
        ratio = dt / tau_c
        
        if ratio > 0.2:
            warning = (
                f"Time step dt={dt*1e9:.3f} ns is too large for tau_c={tau_c*1e6:.3f} µs. "
                f"Ratio dt/tau_c = {ratio:.3f} (should be < 0.2)"
            )
            self.warnings.append(warning)
            return False
        
        self.checks_passed += 1
        return True
    
    def check_convergence(
        self,
        coherence_trajectory: np.ndarray,
        time_points: np.ndarray,
        threshold: float = 0.01
    ) -> bool:
        """
        Check if coherence decay has converged.
        
        Parameters
        ----------
        coherence_trajectory : array
            Coherence values over time
        time_points : array
            Time points corresponding to coherence values
        threshold : float
            Relative change threshold (default: 1%)
        
        Returns
        -------
        converged : bool
            True if coherence has converged
        """
        self.checks_total += 1
        
        if len(coherence_trajectory) < 10:
            return True  # Not enough data to check
        
        # Check last 20% of trajectory
        n_points = len(coherence_trajectory)
        last_20pct = coherence_trajectory[int(0.8 * n_points):]
        
        if len(last_20pct) < 2:
            return True
        
        # Calculate relative change
        relative_change = np.abs(np.diff(last_20pct)) / (np.abs(last_20pct[:-1]) + 1e-10)
        mean_change = np.mean(relative_change)
        
        if mean_change > threshold:
            warning = (
                f"Coherence not converged - relative change in last 20% = {mean_change:.3f} "
                f"(threshold = {threshold}). Simulation time may be too short."
            )
            self.warnings.append(warning)
            return False
        
        self.checks_passed += 1
        return True
    
    def check_T2_vs_literature(
        self,
        T2_measured: float,
        system: Optional[str] = None
    ) -> bool:
        """
        Check if measured T2 is within reasonable range of literature.
        
        Parameters
        ----------
        T2_measured : float
            Measured T2 value (seconds)
        system : str, optional
            System name ('Si_P' or 'GaAs'). If None, uses self.params.system
        
        Returns
        -------
        valid : bool
            True if T2 is within reasonable range
        """
        self.checks_total += 1
        
        if system is None:
            system = self.params.system
        
        T2_star_lit = self.params.T2_star_target
        
        # Allow 10× deviation (very lenient)
        if T2_measured < T2_star_lit / 10 or T2_measured > T2_star_lit * 10:
            warning = (
                f"T2 = {T2_measured*1e6:.3f} µs is far from "
                f"literature value {T2_star_lit*1e6:.3f} µs "
                f"(ratio = {T2_measured/T2_star_lit:.2f})"
            )
            self.warnings.append(warning)
            return False
        
        self.checks_passed += 1
        return True
    
    def check_memory_requirement(self) -> bool:
        """
        Check if memory requirement is reasonable.
        
        Returns
        -------
        valid : bool
            True if memory requirement is acceptable
        """
        self.checks_total += 1
        
        n_steps = int(self.params.total_time / self.params.dt)
        memory_per_ensemble = n_steps * 8 / 1e9  # GB (float64)
        total_memory = memory_per_ensemble * self.params.n_ensemble
        
        if total_memory > 16:  # 16 GB threshold
            warning = (
                f"Memory requirement ({total_memory:.2f} GB) exceeds typical limits. "
                f"Consider: 1) Shorter total_time, 2) Larger dt, 3) Fewer ensembles"
            )
            self.warnings.append(warning)
            return False
        
        self.checks_passed += 1
        return True
    
    def report(self) -> Dict[str, any]:
        """
        Print all warnings and return summary report.
        
        Returns
        -------
        report : dict
            Summary report with warnings, errors, and statistics
        """
        report = {
            'warnings': self.warnings.copy(),
            'errors': self.errors.copy(),
            'checks_passed': self.checks_passed,
            'checks_total': self.checks_total,
            'all_passed': len(self.warnings) == 0 and len(self.errors) == 0
        }
        
        if self.warnings or self.errors:
            print("\n" + "="*60)
            print("SIMULATION VALIDATION REPORT:")
            print("="*60)
            
            if self.errors:
                print("\n❌ ERRORS:")
                for error in self.errors:
                    print(f"   - {error}")
            
            if self.warnings:
                print("\n⚠️  WARNINGS:")
                for warning in self.warnings:
                    print(f"   - {warning}")
            
            print(f"\nChecks: {self.checks_passed}/{self.checks_total} passed")
            print("="*60 + "\n")
        else:
            print(f"\n✓ All validation checks passed ({self.checks_passed}/{self.checks_total})\n")
        
        return report
    
    def reset(self):
        """Reset all warnings and counters."""
        self.warnings.clear()
        self.errors.clear()
        self.checks_passed = 0
        self.checks_total = 0

