# Spin Decoherence Simulation

Monte Carlo simulation of electron spin decoherence under stochastic magnetic fields.

## 🚀 Quick Start

### Run All Simulations
```bash
python run_all.py
```

This will automatically run:
1. FID sweep → `t2_vs_tau_c.csv`
2. FID curves → `fid_tau_c_*.csv`
3. MN analysis → `motional_narrowing_fit.txt`
4. Echo sweep → `t2_echo_vs_tau_c.csv`
5. Echo curves → `echo_tau_c_*.csv`
6. Echo gain analysis → `echo_gain.csv`
7. Noise data → `noise_trajectory_*.csv`

### Generate All Figures
```bash
python plot_all_figures.py
```

This generates 5 publication-quality figures (in order):
1. `fig1_T2_vs_tau_c.png` - Main result (T2 vs tau_c)
2. `fig2_MN_regime_slope.png` - MN validation (slope = -1)
3. `fig3_echo_gain.png` - Echo gain vs tau_c
4. `fig4_representative_curves.png` - FID vs Echo comparison
5. `fig5_convergence_test.png` - Convergence test

All figures are saved to `results/figures/`.
Supplemental figures are saved to `results/figures/supplementary/`.

---

## 📁 File Structure

### **Simulation Scripts**
- `run_all.py` - Run all simulations automatically
- `sim_fid_sweep.py` - FID parameter sweep
- `sim_echo_sweep.py` - Echo parameter sweep
- `sim_fid_curves.py` - FID representative curves
- `sim_echo_curves.py` - Echo representative curves

### **Analysis Scripts**
- `analyze_mn.py` - Motional narrowing analysis
- `analyze_echo_gain.py` - Echo gain analysis

### **Plotting Scripts**
- `plot_all_figures.py` - Generate all figures
- `generate_noise_data.py` - Generate noise trajectory data

### **Utility Scripts**
- `check_slope.py` - Check slope consistency

### **Core Package**
- `spin_decoherence/` - Core simulation modules
  - `noise/` - Noise generation
  - `physics/` - Physics calculations
  - `simulation/` - Simulation engine
  - `analysis/` - Data analysis
  - `config/` - Configuration

---

## 📊 Output Files

### **Main Results**
- `results/t2_vs_tau_c.csv` - FID main results
- `results/t2_echo_vs_tau_c.csv` - Echo main results
- `results/echo_gain.csv` - Echo gain results
- `results/motional_narrowing_fit.txt` - MN analysis

### **Representative Curves**
- `results/fid_tau_c_*.csv` - FID curves
- `results/echo_tau_c_*.csv` - Echo curves

### **Figures**
- `results/figures/fig*.png` - All publication figures

---

## 📚 Documentation

- `QUICK_START.md` - Detailed quick start guide
- `COMMANDS.md` - Command reference
- `CODE_STRUCTURE.md` - Code structure explanation
- `PAPER_CODE_COMPARISON.md` - Paper-code comparison analysis

---

## ⚙️ Configuration

Edit `profiles.yaml` to change material parameters:
- `gamma_e`: Electron gyromagnetic ratio
- `B_rms`: RMS noise amplitude
- `tau_c_range`: Correlation time range
- `M`: Number of trajectories

---

## 🔬 Physics

This simulation models:
- **OU noise**: Ornstein-Uhlenbeck stochastic process
- **Three regimes**: Motional Narrowing, Crossover, Quasi-Static
- **FID & Echo**: Free Induction Decay and Hahn Echo sequences
- **T2 extraction**: Coherence time from decay curves

---

## 📝 Requirements

See `requirements.txt` for Python package dependencies.

---

## ✅ Status

All core functionality is implemented and tested.
Codebase is clean and well-organized.
