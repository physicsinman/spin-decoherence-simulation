# Spin Decoherence Simulation

Monte Carlo simulation of electron spin decoherence under stochastic magnetic fields using Ornstein-Uhlenbeck noise.

## 📋 Overview

This simulation package models electron spin decoherence in silicon (Si:P) under stochastic magnetic field fluctuations. It implements:

- **FID (Free Induction Decay)** and **Hahn Echo** sequences
- **Three physical regimes**: Motional Narrowing (MN), Crossover, and Quasi-Static (QS)
- **Bootstrap confidence intervals** for statistical accuracy
- **Publication-quality figures** for dissertation

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run All Simulations

```bash
python3 run_all.py
```

This automatically runs:
1. FID parameter sweep → `results/t2_vs_tau_c.csv`
2. FID representative curves → `results/fid_tau_c_*.csv`
3. Motional narrowing analysis → `results/motional_narrowing_fit.txt`
4. Echo parameter sweep → `results/t2_echo_vs_tau_c.csv`
5. Echo representative curves → `results/echo_tau_c_*.csv`
6. Echo gain analysis → `results/echo_gain.csv`
7. Noise trajectory examples → `results/noise_trajectory_*.csv`

**Expected time:** ~4-8 hours (depending on system)

### Generate All Figures

```bash
python3 plot_all_figures.py
```

This generates 5 publication-quality figures:
1. `fig1_T2_vs_tau_c.png` - Main result (T₂ vs τc with regime boundaries)
2. `fig2_MN_regime_slope.png` - Motional narrowing validation (slope = -1)
3. `fig3_echo_gain.png` - Echo gain vs correlation time
4. `fig4_representative_curves.png` - FID vs Echo comparison
5. `fig5_convergence_test.png` - Convergence test

All figures are saved to `results/figures/`.

## 📁 Project Structure

```
simulation/
├── README.md                    # This file
├── QUICK_START.md              # Detailed quick start guide
├── SIMULATION_PARAMETERS.md    # Physical parameters and settings
├── CODE_STRUCTURE.md           # Code architecture documentation
│
├── run_all.py                  # Main entry point (runs all simulations)
├── plot_all_figures.py         # Generate all publication figures
│
├── sim_fid_sweep.py            # FID parameter sweep
├── sim_fid_curves.py           # FID representative curves
├── sim_echo_sweep.py           # Echo parameter sweep
├── sim_echo_curves.py          # Echo representative curves
│
├── analyze_mn.py                # Motional narrowing analysis
├── analyze_echo_gain.py         # Echo gain analysis
├── check_slope.py              # Slope consistency check
│
├── generate_noise_data.py       # Generate noise trajectory examples
│
├── spin_decoherence/           # Core simulation package
│   ├── noise/                  # Noise generation (OU process)
│   ├── physics/                # Physics calculations (coherence, phase)
│   ├── simulation/             # Simulation engine (FID, Echo)
│   ├── analysis/               # Data analysis (fitting, bootstrap)
│   ├── config/                 # Configuration (constants, units)
│   ├── visualization/          # Plotting utilities
│   └── utils/                  # Utilities (IO, logging, validation)
│
├── results/                    # Output directory
│   ├── t2_vs_tau_c.csv        # FID main results
│   ├── t2_echo_vs_tau_c.csv   # Echo main results
│   ├── echo_gain.csv          # Echo gain results
│   ├── fid_tau_c_*.csv        # FID representative curves
│   ├── echo_tau_c_*.csv       # Echo representative curves
│   └── figures/               # Generated figures
│
├── tests/                      # Unit tests
├── docs/                       # Additional documentation
└── legacy/                     # Legacy code (archived)
```

## 🔬 Physics

### Physical Parameters (Si:P)

- **γₑ** (electron gyromagnetic ratio): `1.76 × 10¹¹` rad/(s·T)
- **B_rms** (RMS magnetic field): `0.57 μT` (800 ppm ²⁹Si concentration)
- **Δω** = γₑ × B_rms: `0.10 MHz`

### Three Regimes

1. **Motional Narrowing (MN)**: ξ < 0.5
   - T₂ ∝ τc⁻¹
   - Fast noise averaging

2. **Crossover**: 0.5 ≤ ξ < 2.0
   - No analytical solution
   - Transition between MN and QS

3. **Quasi-Static (QS)**: ξ ≥ 2.0
   - T₂ ≈ T₂* = √2/Δω ≈ 14.1 μs
   - Slow noise fluctuations

Where **ξ** = γₑ × B_rms × τc (dimensionless parameter)

### Simulation Methods

- **FID**: Single π/2 pulse, measure coherence decay
- **Hahn Echo**: π/2 - τ - π - τ sequence, refocuses static dephasing
- **Bootstrap**: 800 iterations for 95% confidence intervals

## 📊 Output Files

### Main Results

- `results/t2_vs_tau_c.csv` - FID T₂ vs τc (67 points)
- `results/t2_echo_vs_tau_c.csv` - Echo T₂ vs τc (67 points)
- `results/echo_gain.csv` - Echo gain = T₂_echo / T₂_fid
- `results/motional_narrowing_fit.txt` - MN regime slope analysis

### Representative Curves

- `results/fid_tau_c_*.csv` - FID coherence decay curves
- `results/echo_tau_c_*.csv` - Echo coherence decay curves

### Figures

- `results/figures/fig*.png` - All publication figures
- `results/figures/supplementary/` - Additional figures

## ⚙️ Configuration

### Key Parameters (in `sim_fid_sweep.py`)

```python
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.57e-6            # T (0.57 μT)
N_traj = 2000              # Monte Carlo trajectories
B_bootstrap = 800          # Bootstrap iterations
```

### Adaptive Parameters

- **dt**: Automatically adjusted based on τc (dt < τc/5 for stability)
- **T_max**: Regime-dependent (10×T₂ for MN, 100-200×T₂ for QS)
- **Memory limit**: 8 GB (automatic dt adjustment)

## 📚 Documentation

- **QUICK_START.md** - Detailed step-by-step guide
- **SIMULATION_PARAMETERS.md** - Physical parameters and regime definitions
- **CODE_STRUCTURE.md** - Code architecture and module organization
- **COMMANDS.md** - Command reference
- **PAPER_CODE_COMPARISON.md** - Comparison with paper results

## 🧪 Testing

```bash
pytest tests/
```

## 📝 Key Features

- ✅ **Physical accuracy**: Validated against analytical theory
- ✅ **Statistical rigor**: Bootstrap confidence intervals
- ✅ **Regime-aware**: Adaptive parameters for each regime
- ✅ **Memory efficient**: Automatic memory management
- ✅ **Publication ready**: High-quality figures with proper error bars

## 🔧 Troubleshooting

### Memory Issues

- Reduce `N_traj` (e.g., 2000 → 1000)
- Reduce `tau_c_npoints` in sweep
- System will automatically adjust `dt` if memory limit exceeded

### Slow Execution

- Bootstrap iterations: 800 (can reduce to 200 for faster runs)
- Reduce number of trajectories: `N_traj = 1000`
- Reduce tau_c grid points

### Zero Error Bars

- Increase bootstrap iterations: `B_bootstrap = 800` (default)
- Check that sufficient trajectories are used: `N_traj ≥ 1000`

## 📄 License

This code is part of a physics dissertation project.

## 👤 Author

Physics dissertation simulation code for electron spin decoherence in Si:P.

---

**Last Updated:** 2025-01-XX
