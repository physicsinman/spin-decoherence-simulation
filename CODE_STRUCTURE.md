# Code Structure Documentation

## 📁 Project Architecture

```
simulation/
├── README.md                    # Main documentation
├── QUICK_START.md              # Quick start guide
├── SIMULATION_PARAMETERS.md    # Physical parameters
├── CODE_STRUCTURE.md           # This file
│
├── run_all.py                  # Main entry point
├── plot_all_figures.py         # Generate all figures
│
├── sim_fid_sweep.py            # FID parameter sweep
├── sim_fid_curves.py           # FID representative curves
├── sim_echo_sweep.py           # Echo parameter sweep
├── sim_echo_curves.py          # Echo representative curves
│
├── analyze_mn.py               # Motional narrowing analysis
├── analyze_echo_gain.py        # Echo gain analysis
├── check_slope.py             # Slope consistency check
├── generate_noise_data.py     # Noise trajectory examples
│
├── spin_decoherence/           # Core simulation package
│   ├── noise/                  # Noise generation
│   │   ├── ou.py              # Ornstein-Uhlenbeck process
│   │   ├── double_ou.py       # Double-OU process
│   │   └── base.py            # Base noise class
│   │
│   ├── physics/               # Physics calculations
│   │   ├── coherence.py       # Coherence function
│   │   ├── phase.py           # Phase accumulation
│   │   ├── analytical.py      # Analytical solutions
│   │   └── coherence_temp.py  # Temporary coherence utilities
│   │
│   ├── simulation/             # Simulation engine
│   │   ├── fid.py             # FID simulation
│   │   ├── echo.py            # Hahn Echo simulation
│   │   └── engine.py          # Common simulation engine
│   │
│   ├── analysis/              # Data analysis
│   │   ├── fitting.py         # Curve fitting (T2 extraction)
│   │   ├── bootstrap.py       # Bootstrap confidence intervals
│   │   └── statistics.py      # Statistical utilities
│   │
│   ├── config/                # Configuration
│   │   ├── constants.py       # Physical constants
│   │   ├── simulation.py       # Simulation configuration
│   │   └── units.py           # Unit conversions
│   │
│   ├── visualization/         # Plotting utilities
│   │   ├── plots.py           # Plot functions
│   │   ├── styles.py          # Plot styles
│   │   └── comparison.py      # Comparison plots
│   │
│   └── utils/                 # Utilities
│       ├── io.py              # Input/output
│       ├── logging.py          # Logging utilities
│       └── validation.py      # Parameter validation
│
├── results/                   # Output directory
│   ├── t2_vs_tau_c.csv        # FID main results
│   ├── t2_echo_vs_tau_c.csv   # Echo main results
│   ├── echo_gain.csv          # Echo gain results
│   ├── fid_tau_c_*.csv        # FID curves
│   ├── echo_tau_c_*.csv       # Echo curves
│   └── figures/               # Generated figures
│
├── tests/                     # Unit tests
├── docs/                       # Additional documentation
└── legacy/                     # Legacy/archived code
```

## 🔄 Execution Flow

### 1. Complete Simulation Workflow

```
run_all.py
  ├─> sim_fid_sweep.py
  │     └─> spin_decoherence/simulation/fid.py
  │           ├─> spin_decoherence/noise/ou.py (noise generation)
  │           ├─> spin_decoherence/physics/coherence.py (coherence calc)
  │           └─> spin_decoherence/analysis/fitting.py (T2 extraction)
  │                 └─> spin_decoherence/analysis/bootstrap.py (CI)
  │
  ├─> sim_fid_curves.py (representative curves)
  ├─> analyze_mn.py (MN regime analysis)
  │
  ├─> sim_echo_sweep.py
  │     └─> spin_decoherence/simulation/echo.py
  │
  ├─> sim_echo_curves.py (representative curves)
  ├─> analyze_echo_gain.py (gain calculation)
  └─> generate_noise_data.py (noise examples)
```

### 2. Figure Generation Workflow

```
plot_all_figures.py
  ├─> Load results/t2_vs_tau_c.csv
  ├─> Load results/t2_echo_vs_tau_c.csv
  ├─> Load results/echo_gain.csv
  ├─> Load results/fid_tau_c_*.csv
  ├─> Load results/echo_tau_c_*.csv
  │
  ├─> fig1_T2_vs_tau_c.png (main result)
  ├─> fig2_MN_regime_slope.png (MN validation)
  ├─> fig3_echo_gain.png (echo gain)
  ├─> fig4_representative_curves.png (FID vs Echo)
  └─> fig5_convergence_test.png (convergence)
```

## 🎯 Core Modules

### 1. Noise Generation (`spin_decoherence/noise/ou.py`)

**Purpose**: Generate Ornstein-Uhlenbeck stochastic noise

**Key Algorithm**:
```python
# AR(1) recursive relation
δB_{k+1} = ρ·δB_k + σ_η·η_k
where:
  ρ = exp(-dt/τc)           # Autocorrelation
  σ_η = B_rms·√(1-ρ²)      # Noise amplitude
  η_k ~ N(0,1)              # White noise
```

**Features**:
- Memory-efficient recursive generation
- Correct autocorrelation function
- Validated PSD

### 2. Physics Calculations (`spin_decoherence/physics/`)

#### `coherence.py` - Coherence Function
```python
# Phase accumulation
φ(t) = ∫₀^t γ_e·δB(t') dt'

# Ensemble coherence
E(t) = ⟨exp(i·φ(t))⟩
```

#### `analytical.py` - Analytical Solutions
- **MN regime**: T₂ = 1/(Δω²·τc)
- **QS regime**: T₂* = √2/Δω
- **Crossover**: No analytical solution

### 3. Simulation Engine (`spin_decoherence/simulation/`)

#### `fid.py` - FID Simulation
- Single π/2 pulse
- Direct phase accumulation
- Coherence decay measurement

#### `echo.py` - Hahn Echo Simulation
- π/2 - τ - π - τ sequence
- Toggling function: y(t) = +1 (t < τ), -1 (τ ≤ t ≤ 2τ)
- Refocuses static dephasing

### 4. Data Analysis (`spin_decoherence/analysis/`)

#### `fitting.py` - T2 Extraction
```python
# Fit with scale and offset
|E(t)| = A·exp(-t/T₂) + B

# Extract T2 where |E(T2)| = 1/e
```

**Features**:
- Regime-aware window selection
- Robust fitting with offset
- Error estimation

#### `bootstrap.py` - Confidence Intervals
- 800 bootstrap iterations (configurable)
- 95% confidence intervals
- Handles degenerate cases

### 5. Configuration (`spin_decoherence/config/`)

#### `constants.py` - Physical Constants
```python
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.57e-6            # T (0.57 μT for Si:P)
```

#### `simulation.py` - Simulation Config
- Adaptive parameters (dt, T_max)
- Memory limits
- Bootstrap settings

## 📊 Data Flow

### Input → Simulation → Analysis → Output

```
1. Parameters (tau_c, B_rms, gamma_e)
   ↓
2. Noise Generation (OU process)
   ↓
3. Coherence Calculation (FID/Echo)
   ↓
4. Fitting (T2 extraction)
   ↓
5. Bootstrap (confidence intervals)
   ↓
6. Results (CSV files)
   ↓
7. Figures (PNG files)
```

## 🔑 Key Scripts

### Main Entry Points

1. **`run_all.py`**
   - Runs all simulations in sequence
   - Generates all required data files
   - Expected time: ~4-8 hours

2. **`plot_all_figures.py`**
   - Generates all publication figures
   - Reads from `results/` directory
   - Outputs to `results/figures/`

### Simulation Scripts

1. **`sim_fid_sweep.py`**
   - FID parameter sweep (67 tau_c values)
   - Output: `results/t2_vs_tau_c.csv`
   - Bootstrap: 800 iterations

2. **`sim_echo_sweep.py`**
   - Echo parameter sweep (67 tau_c values)
   - Output: `results/t2_echo_vs_tau_c.csv`
   - Bootstrap: 800 iterations

3. **`sim_fid_curves.py`**
   - Representative FID curves (4 tau_c values)
   - Output: `results/fid_tau_c_*.csv`

4. **`sim_echo_curves.py`**
   - Representative Echo curves (4 tau_c values)
   - Output: `results/echo_tau_c_*.csv`

### Analysis Scripts

1. **`analyze_mn.py`**
   - Motional narrowing regime analysis
   - Slope = -1 validation
   - Output: `results/motional_narrowing_fit.txt`

2. **`analyze_echo_gain.py`**
   - Echo gain calculation
   - Output: `results/echo_gain.csv`

## 🛠️ Adaptive Parameters

### Time Step (dt)
- **Constraint**: dt < τc/5 (numerical stability)
- **Target**: dt = τc/100 (precision)
- **Adjustment**: Automatic based on memory limit

### Simulation Time (T_max)
- **MN regime**: T_max = 10×T₂
- **Crossover**: T_max = 20×T₂
- **QS regime**: T_max = 100-200×T₂ (depending on ξ)

### Memory Management
- **Limit**: 8 GB
- **Strategy**: Automatic dt adjustment
- **Fallback**: Reduce T_max if needed

## 📝 Configuration Files

### `sim_fid_sweep.py` / `sim_echo_sweep.py`
```python
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.57e-6            # T (0.57 μT)
N_traj = 2000              # Trajectories
B_bootstrap = 800          # Bootstrap iterations
```

### Adaptive Parameters
- Automatically calculated based on regime
- Memory-aware adjustments
- Stability constraints enforced

## 🧪 Testing

```bash
pytest tests/
```

**Test Coverage**:
- `test_noise_models.py` - Noise generation
- `test_coherence.py` - Coherence calculations
- `test_config.py` - Configuration validation
- `test_ornstein_uhlenbeck.py` - OU process
- `test_units.py` - Unit conversions

## 📚 Documentation Files

- **README.md** - Main documentation
- **QUICK_START.md** - Step-by-step guide
- **SIMULATION_PARAMETERS.md** - Physical parameters
- **CODE_STRUCTURE.md** - This file
- **COMMANDS.md** - Command reference
- **PAPER_CODE_COMPARISON.md** - Paper-code comparison

## 🔍 Legacy Code

The `legacy/` directory contains:
- Old simulation code (archived)
- One-time scripts
- Unused code
- Old documentation

**Note**: Legacy code is kept for reference but should not be used for new simulations.

---

**Last Updated**: 2025-01-XX
