"""
Standalone script to generate Figure 4: OU Noise PSD Verification.

This script generates the PSD verification plot showing theoretical vs simulated
power spectral density for Ornstein-Uhlenbeck noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from ornstein_uhlenbeck import generate_ou_noise
from visualize import plot_ou_psd_verification


def main():
    """Generate OU Noise PSD Verification figure."""
    
    # Parameters (using τ_c = 0.01 μs as mentioned in the description)
    tau_c = 0.01e-6  # 0.01 μs in seconds
    B_rms = 5e-6     # 5 μT in Tesla
    dt = 0.2e-9      # 0.2 ns in seconds
    T_max = 30e-6    # 30 μs in seconds
    
    # Calculate number of steps
    N_steps = int(T_max / dt)
    
    print("=" * 70)
    print("Generating Figure 4: OU Noise PSD Verification")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  τ_c = {tau_c*1e6:.2f} μs")
    print(f"  B_rms = {B_rms*1e6:.2f} μT")
    print(f"  dt = {dt*1e9:.2f} ns")
    print(f"  T_max = {T_max*1e6:.2f} μs")
    print(f"  N_steps = {N_steps}")
    print("=" * 70)
    
    # Generate OU noise
    print("\nGenerating OU noise...")
    delta_B = generate_ou_noise(tau_c, B_rms, dt, N_steps, seed=42)
    
    # Verify noise properties
    print(f"Generated noise statistics:")
    print(f"  Mean: {np.mean(delta_B):.3e} T")
    print(f"  Std: {np.std(delta_B):.3e} T (expected: {B_rms:.3e} T)")
    print(f"  RMS: {np.sqrt(np.mean(delta_B**2)):.3e} T")
    
    # Calculate corner frequency
    f_c = 1.0 / (2 * np.pi * tau_c)
    print(f"\nCorner frequency: ω_c = 1/τ_c ≈ {f_c:.2e} Hz")
    
    # Create figure
    print("\nCreating PSD verification plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_ou_psd_verification(delta_B, tau_c, B_rms, dt, ax=ax)
    
    # Save figure
    output_path = 'results/ou_psd_verification.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    print("=" * 70)
    
    # Show figure
    plt.show()


if __name__ == "__main__":
    main()

