# Convergence Test Guide

## 🔴 Critical Issue: Convergence Tests

### Why It Matters

**Examiner Question**: "How do you know N=2000 is enough?"

**Without convergence tests**: No answer → Embarrassing!

**With convergence tests**: "We tested N=500, 1000, and 2000. Results converged within uncertainty at N=2000, with CI width decreasing from 5% to 2%."

### Cost-Benefit Analysis

| Test | Time | Impact | Recommendation |
|------|------|--------|----------------|
| **N_traj** | 4 hours | +3 marks | ✅ **DO IT** |
| dt | 4 hours | +2 marks | If time |
| T_sim | 2 hours | +1 mark | Optional |

**Total for N_traj only**: 0.5 days → +3 marks (Excellent ROI!)

---

## How to Run

### 1. N_traj Convergence Test (CRITICAL)

```bash
python3 run_convergence_test.py
```

**What it does**:
- Tests N_traj = [500, 1000, 2000] for 2 representative tau_c values
- Checks if T₂ values converge within uncertainty
- Verifies CI width decreases with increasing N_traj

**Expected output**:
```
N=500:  T₂ = 1.26 ± 0.05 μs (CI width: 5.0%)
N=1000: T₂ = 1.26 ± 0.03 μs (CI width: 3.0%)
N=2000: T₂ = 1.26 ± 0.02 μs (CI width: 2.0%) ← Converged!
```

**Time**: ~4 hours (3 simulations × 2 tau_c values)

---

### 2. dt Convergence Test (OPTIONAL)

Uncomment the dt test section in `run_convergence_test.py` if time permits.

**What it does**:
- Tests dt = [0.5×, 1.0×, 2.0×] base dt
- Checks if T₂ is stable (not sensitive to dt)

**Time**: ~4 hours

---

### 3. T_sim Adequacy Test (OPTIONAL)

Can be added if needed.

**What it does**:
- Tests T_sim = [1×, 2×, 3×] T₂
- Checks if fitted T₂ converges

**Time**: ~2 hours

---

## Results

### Output Files

1. **`convergence_N_traj_*.csv`**: Detailed results for each tau_c
2. **`convergence_test_summary.txt`**: Summary with conclusions

### What to Look For

✅ **Good convergence**:
- T₂ values within 2σ uncertainty
- CI width decreasing with N_traj
- Final T₂ stable

⚠️ **Poor convergence**:
- T₂ values differ by >2σ
- CI width not decreasing
- May need N_traj > 2000

---

## Dissertation Writing

### Methods Section

> "To ensure statistical reliability, we performed convergence tests for the ensemble size N_traj. We tested N_traj = 500, 1000, and 2000 for representative correlation times in the MN and crossover regimes. Results converged within uncertainty at N_traj = 2000, with the 95% confidence interval width decreasing from 5.0% to 2.0% (see Figure X). This confirms that N_traj = 2000 is sufficient for our analysis."

### Results Section

Include a figure showing:
- T₂ vs N_traj (with error bars)
- CI width vs N_traj (decreasing)
- Convergence criterion (within 2σ)

---

## Priority

1. ✅ **N_traj test**: **DO IT** (essential for credibility)
2. 🟡 dt test: If time permits
3. 🟡 T_sim test: Optional

**Minimum**: Run N_traj test before submitting dissertation!

