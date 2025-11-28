#!/usr/bin/env python3
"""
Generate Sample OU Noise Trajectories
Creates noise_trajectory_fast.csv and noise_trajectory_slow.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.noise.ou import generate_ou_noise

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.57e-6            # T (0.57 μT) - Physical value for 800 ppm ²⁹Si concentration

# Fast noise example (MN regime)
tau_c_fast = 1e-8          # s (10 ns) - Fast fluctuation
t_max_fast = 1e-6          # s (1 μs, ~100 correlation times)

# Slow noise example (QS regime)
tau_c_slow = 1e-4          # s (100 μs) - Slow fluctuation (더 명확한 대비)
t_max_slow = 1e-2          # s (10 ms, ~100 correlation times)

def main():
    print("="*80)
    print("Generate Sample OU Noise Trajectories")
    print("="*80)
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # ===== Fast Noise =====
    print(f"\n[1/2] Generating fast noise trajectory...")
    print(f"  tau_c = {tau_c_fast*1e9:.1f} ns")
    print(f"  t_max = {t_max_fast*1e6:.1f} μs")
    
    dt_fast = tau_c_fast / 100  # 100 steps per correlation time
    N_steps_fast = int(t_max_fast / dt_fast)
    
    B_z_fast = generate_ou_noise(
        tau_c=tau_c_fast,
        B_rms=B_rms,
        dt=dt_fast,
        N_steps=N_steps_fast,
        seed=42,
        burnin_mult=20.0  # Increased for better stationarity
    )
    
    t_fast = np.arange(N_steps_fast) * dt_fast
    
    df_fast = pd.DataFrame({
        'time (s)': t_fast,
        'B_z (T)': B_z_fast
    })
    
    output_file_fast = output_dir / "noise_trajectory_fast.csv"
    df_fast.to_csv(output_file_fast, index=False)
    
    print(f"  ✅ Saved to: {output_file_fast}")
    print(f"  Points: {len(df_fast)}, Time range: {t_fast[0]*1e9:.2f} to {t_fast[-1]*1e9:.2f} ns")
    print(f"  B_z range: [{B_z_fast.min()*1e6:.3f}, {B_z_fast.max()*1e6:.3f}] μT")
    print(f"  B_z RMS: {np.std(B_z_fast)*1e6:.3f} μT (expected: {B_rms*1e6:.3f} μT)")
    
    # ===== Slow Noise =====
    print(f"\n[2/2] Generating slow noise trajectory...")
    print(f"  tau_c = {tau_c_slow*1e6:.1f} μs")
    print(f"  t_max = {t_max_slow*1e3:.1f} ms")
    
    dt_slow = tau_c_slow / 100
    N_steps_slow = int(t_max_slow / dt_slow)
    
    B_z_slow = generate_ou_noise(
        tau_c=tau_c_slow,
        B_rms=B_rms,
        dt=dt_slow,
        N_steps=N_steps_slow,
        seed=43,
        burnin_mult=20.0  # Increased for better stationarity
    )
    
    t_slow = np.arange(N_steps_slow) * dt_slow
    
    df_slow = pd.DataFrame({
        'time (s)': t_slow,
        'B_z (T)': B_z_slow
    })
    
    output_file_slow = output_dir / "noise_trajectory_slow.csv"
    df_slow.to_csv(output_file_slow, index=False)
    
    print(f"  ✅ Saved to: {output_file_slow}")
    print(f"  Points: {len(df_slow)}, Time range: {t_slow[0]*1e6:.2f} to {t_slow[-1]*1e6:.2f} μs")
    print(f"  B_z range: [{B_z_slow.min()*1e6:.3f}, {B_z_slow.max()*1e6:.3f}] μT")
    print(f"  B_z RMS: {np.std(B_z_slow)*1e6:.3f} μT (expected: {B_rms*1e6:.3f} μT)")
    
    print(f"\n{'='*80}")
    print(f"✅ All noise trajectories saved!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

