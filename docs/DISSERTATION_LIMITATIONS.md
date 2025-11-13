# Limitations Section for Dissertation

## Suggested Text for Dissertation

---

## 5. Limitations and Future Directions

### 5.1 Limitations of Current Study

While the simulations successfully reproduce the motional-narrowing scaling relationship (T₂ ∝ τ_c⁻¹) with excellent quantitative agreement (slope = -1.043 ± 0.006), several fundamental limitations must be clearly acknowledged:

#### 5.1.1 Parameter Selection and Validation

**B_rms Parameter Mismatch:**

The RMS noise amplitude B_rms = 5 μT used for Si:P simulations was not directly validated against literature values. Analysis reveals that to match the experimental T₂* range of 1-10 ms reported in the literature [Kane 1998, Morello 2010, Tyryshkin 2012], a B_rms value of approximately 3.21 nT would be required, which is ~1500× smaller than the value used. This parameter mismatch results in:

- Simulated T₂* values (~2-3 μs) that are ~1000× smaller than experimental values (1-10 ms)
- Quasi-static regime results that do not quantitatively match theoretical predictions
- Bootstrap confidence intervals spanning 11 orders of magnitude, indicating measurement failure rather than large uncertainty

**Implication:** The absolute T₂ values in quasi-static regime cannot be directly compared to experimental measurements without parameter recalibration.

#### 5.1.2 Computational Constraints

**Simulation Time Limitations:**

For accurate measurement of T₂* in quasi-static regime, the simulation must run for at least 5 × T₂* to capture the full decoherence dynamics. For Si:P with T₂* ≈ 2.5 ms, this would require T_max ≥ 12.5 ms. While the current T_max = 30 μs is sufficient for the (incorrectly) short T₂* values produced by the parameter mismatch, it would be insufficient for the true T₂* values.

**Memory Requirements:**

Simulating the full quasi-static regime for Si:P with correct parameters would require:
- T_max ≥ 12.5 ms
- Number of time steps: N ≥ 62.5 million (at dt = 0.2 ns)
- Memory: ~375 GB for M = 750 trajectories

This is computationally infeasible with current resources, necessitating either:
- Adaptive time stepping algorithms
- Analytical corrections for quasi-static regime
- Regime-specific simulation strategies

#### 5.1.3 Bootstrap Confidence Intervals in Quasi-Static Regime

In the quasi-static regime, bootstrap confidence intervals span 11 orders of magnitude (10³ ms to 10¹⁴ ms), indicating that:

1. **Ensemble variance is extreme**: Some trajectories show almost no decoherence, others show rapid decoherence
2. **Measurement is unreliable**: The large CI reflects fundamental instability, not genuine uncertainty
3. **Bootstrap method fails**: The resampling amplifies outliers, making the CI meaningless

This failure is a direct consequence of the parameter mismatch and insufficient simulation time relative to the true T₂*.

#### 5.1.4 Material Asymmetry

The apparent success with GaAs (T₂ ≈ 1-2 μs matching theory) is likely due to fortuitous parameter matching rather than superior simulation accuracy. The B_rms = 8 μT value used for GaAs may be closer to the true experimental value, or GaAs's shorter T₂* (~1 μs) happens to align with the simulation time scale.

**This asymmetry highlights the importance of:**
- Literature-based parameter validation for each material
- Material-specific optimization of simulation parameters
- Careful interpretation of "successful" results

#### 5.1.5 Scope of Validated Results

**What This Study Successfully Validates:**

✅ Motional-narrowing regime physics (T₂ ∝ τ_c⁻¹)
✅ Monte Carlo framework for regime-dependent analysis
✅ Echo enhancement mechanism (qualitative)
✅ Relative timescale differences between materials

**What This Study Does NOT Validate:**

❌ Absolute T₂ values in quasi-static regime
❌ Quantitative agreement with experimental T₂* measurements
❌ Bootstrap CI reliability in quasi-static regime
❌ Parameter selection methodology

### 5.2 Future Directions

To address the limitations identified above, the following directions are recommended:

#### 5.2.1 Parameter Validation and Calibration

1. **Literature Survey:** Conduct comprehensive literature review to establish experimentally-validated B_rms values for each material
2. **Parameter Fitting:** Use experimental T₂* values to back-calculate B_rms, then validate against independent measurements
3. **Uncertainty Quantification:** Propagate experimental uncertainties in T₂* to B_rms parameter space

#### 5.2.2 Computational Improvements

1. **Adaptive Time Stepping:** Implement variable time steps to efficiently handle long T₂* values without excessive memory requirements
2. **Regime-Specific Algorithms:** Develop separate simulation strategies optimized for each regime (MN, crossover, QS)
3. **Analytical Corrections:** Apply post-processing corrections for quasi-static regime based on analytical limits

#### 5.2.3 Statistical Methods

1. **Regime-Aware Bootstrap:** Develop bootstrap methods that account for regime-dependent variance
2. **Alternative CI Methods:** Explore analytical CI methods for quasi-static regime (e.g., based on S(0) uncertainty)
3. **Robust Statistics:** Use median-based statistics instead of mean-based for highly skewed distributions

#### 5.2.4 Validation Strategy

1. **Hierarchical Validation:** Validate at multiple levels:
   - Noise generation (PSD) ✓ (current study)
   - Motional-narrowing regime ✓ (current study)
   - Quasi-static regime ✗ (future work)
   - Absolute T₂ values ✗ (future work)

2. **Cross-Validation:** Compare results across different:
   - Parameter sets
   - Simulation methods
   - Numerical implementations

### 5.3 Honest Assessment

**For Examiners:**

This study demonstrates that:
1. The Monte Carlo framework successfully captures motional-narrowing physics
2. Computational challenges in quasi-static regime are significant and quantifiable
3. Parameter selection is critical and must be literature-validated
4. Bootstrap methods require regime-specific adaptation

**The main contribution is not "accurate simulation of all regimes" but rather:**
- Successful validation in motional-narrowing regime
- Identification and quantification of fundamental limitations
- Framework for regime-dependent analysis
- Clear roadmap for future improvements

**This is a valuable contribution if honestly presented as an investigation that reveals both successes and challenges, rather than a complete solution.**

---

## Key Takeaways for Discussion Section

When discussing results, emphasize:

1. **Motional-narrowing success is genuine** - not just "easy case"
2. **Quasi-static failure is instructive** - reveals computational challenges
3. **Parameter validation is essential** - cannot be overlooked
4. **Framework is sound** - limitations are in parameters, not methodology
5. **Future work is clear** - specific, actionable improvements identified

---

## References for Limitations Section

- Kane, B. E. (1998). A silicon-based nuclear spin quantum computer. *Nature* 393, 133-137.
- Morello, A., et al. (2010). Single-shot readout of an electron spin in silicon. *Nature* 467, 687-691.
- Tyryshkin, A. M., et al. (2012). Electron spin coherence exceeding seconds in high-purity silicon. *Nature Materials* 11, 143-147.

