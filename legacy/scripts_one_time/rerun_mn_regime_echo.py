#!/usr/bin/env python3
"""
MN Regime Echo Sweep 재실행
MN regime (xi < 0.2) 포인트만 선택적으로 재실행하여 echo gain 개선
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.echo import run_simulation_with_hahn_echo
from spin_decoherence.simulation.engine import get_dimensionless_tau_range
import json
from datetime import datetime

# ============================================
# SI:P SIMULATION PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (0.05 mT)
N_traj = 2000              # Monte Carlo trajectories

# MN regime 포인트만 로드
fid_file = Path("results_comparison/T2_vs_tau_c.csv")
df_fid = pd.read_csv(fid_file)
df_fid['xi'] = gamma_e * B_rms * df_fid['tau_c']

# MN regime 포인트만 선택
mn_mask = df_fid['xi'] < 0.2
mn_tau_c_list = df_fid[mn_mask]['tau_c'].values

print("="*80)
print("MN Regime Echo Sweep 재실행")
print("="*80)
print(f"\nParameters:")
print(f"  gamma_e = {gamma_e:.3e} rad/(s·T)")
print(f"  B_rms = {B_rms*1e3:.3f} mT")
print(f"  MN regime points: {len(mn_tau_c_list)}")
print(f"  N_traj = {N_traj}")
print(f"\n개선 사항:")
print(f"  - MN regime: T_max_echo를 매우 길게 설정 (50-100×T2_echo)")
print(f"  - Echo decay를 충분히 관측하여 정확한 T2_echo 추출")
print(f"\nExpected time: ~30-60 minutes")
print("\nStarting simulation...\n")

# Adaptive parameters
def get_dt(tau_c, T_max=None, max_memory_gb=8.0):
    """Adaptive timestep selection"""
    dt_target = tau_c / 100
    if T_max is not None:
        N_steps = int(T_max / dt_target) + 1
        memory_gb = (N_steps * N_traj * 8) / (1024**3)
        if memory_gb > max_memory_gb:
            max_N_steps = int((max_memory_gb * 1024**3) / (N_traj * 8))
            dt_required = T_max / max_N_steps if max_N_steps > 0 else dt_target
            dt_min = tau_c / 50
            dt = max(dt_required, dt_min)
            return dt
    return dt_target

def get_tmax(tau_c, B_rms, gamma_e):
    """Calculate appropriate simulation duration"""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est
    elif xi > 3:  # QS regime
        T2_est = 1.0 / (gamma_e * B_rms)
        if xi < 10:
            multiplier = 100
        elif xi < 50:
            multiplier = 150
        else:
            multiplier = 200
        T_max_from_T2 = multiplier * T2_est
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        return min(T_max_final, 100e-3)
    else:  # Crossover
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 20 * T2_est

# Create output directory
output_dir = Path("results_comparison")
output_dir.mkdir(exist_ok=True)

# Load existing echo data
echo_file = output_dir / "t2_echo_vs_tau_c.csv"
if echo_file.exists():
    df_echo_existing = pd.read_csv(echo_file)
    print(f"✅ 기존 echo 데이터 로드: {len(df_echo_existing)} points")
else:
    df_echo_existing = pd.DataFrame()
    print("⚠️  기존 echo 데이터 없음, 새로 생성합니다")

# Storage for results
results_data = []

for i, tau_c in enumerate(mn_tau_c_list):
    print(f"\n[{i+1}/{len(mn_tau_c_list)}] Processing τ_c = {tau_c*1e6:.3f} μs (MN regime)")
    
    # Calculate adaptive parameters
    T_max = get_tmax(tau_c, B_rms, gamma_e)
    dt = get_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
    xi = gamma_e * B_rms * tau_c
    
    print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.4f}")
    
    # CRITICAL: MN regime에서 매우 긴 T_max_echo
    # Echo는 거의 decay하지 않으므로 매우 긴 시간 필요
    # Hybrid method를 사용하려면 T_FID까지 echo curve가 도달해야 함
    T2_fid_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)  # MN regime FID T2
    T2_echo_est = 20.0 * T2_fid_est  # MN regime echo T2 (conservative estimate)
    
    # T_FID ≈ T2_fid_est이므로, echo curve가 최소 T_FID까지 도달해야 함
    # 더 안전하게 T_FID의 2배까지 확보
    T_FID_est = T2_fid_est  # T_FID ≈ T2_FID for exponential decay
    T_max_echo = max(2.0 * T_FID_est, 50.0 * T2_echo_est, T_max * 20.0)
    T_max_echo = min(T_max_echo, 200e-3)  # 최대 200 ms로 제한
    
    print(f"  T_max_echo = {T_max_echo*1e6:.2f} μs ({T_max_echo/T_max:.1f}× T_max)")
    print(f"  T2_echo_est = {T2_echo_est*1e6:.2f} μs")
    print(f"  T_max_echo / T2_echo_est = {T_max_echo/T2_echo_est:.1f}×")
    
    params = {
        'B_rms': B_rms,
        'tau_c_range': (tau_c, tau_c),
        'tau_c_num': 1,
        'gamma_e': gamma_e,
        'dt': dt,
        'T_max': T_max,
        'T_max_echo': T_max_echo,
        'M': N_traj,
        'seed': 42 + i + 10000,  # Different seed from original
        'output_dir': str(output_dir),
        'compute_bootstrap': False,
        'save_delta_B_sample': False,
    }
    
    # Generate tau_list with very large upsilon_max for MN regime
    # MN regime에서 echo는 매우 느리게 decay하므로 긴 범위 필요
    upsilon_max = min(2.0 * T_max_echo / tau_c, 30.0)  # 최대 30까지
    
    tau_list = get_dimensionless_tau_range(
        tau_c, n_points=150, upsilon_min=0.05, upsilon_max=upsilon_max,  # More points
        dt=dt, T_max=T_max_echo
    )
    
    print(f"  tau_list: {len(tau_list)} points, range: {tau_list[0]*1e6:.3f} - {tau_list[-1]*1e6:.3f} μs")
    print(f"  upsilon_max = {upsilon_max:.2f}")
    
    # Run simulation
    try:
        result = run_simulation_with_hahn_echo(
            tau_c, params=params, tau_list=tau_list, verbose=True
        )
        
        # Extract results
        fit_result_echo = result.get('fit_result_echo')
        fit_result_fid = result.get('fit_result_fid')
        
        if fit_result_echo is not None and fit_result_fid is not None:
            T2_echo = fit_result_echo['T2']
            T2_fid = fit_result_fid['T2']
            gain = T2_echo / T2_fid
            
            print(f"  ✅ T2_FID = {T2_fid*1e6:.3f} μs")
            print(f"  ✅ T2_echo = {T2_echo*1e6:.3f} μs")
            print(f"  ✅ Gain = {gain:.2f}")
            
            if gain > 5.0:
                print(f"  🎉 Gain이 크게 개선됨!")
            elif gain > 2.0:
                print(f"  ✅ Gain이 개선됨 (이전: ~1)")
            else:
                print(f"  ⚠️  Gain이 여전히 작음")
            
            results_data.append({
                'tau_c': tau_c,
                'T2_echo': T2_echo,
                'T2_echo_lower': fit_result_echo.get('T2_lower', np.nan),
                'T2_echo_upper': fit_result_echo.get('T2_upper', np.nan),
                'R2_echo': fit_result_echo.get('R2', np.nan),
                'T2': T2_fid,
                'xi': xi,
            })
        else:
            print(f"  ❌ Fitting 실패")
            results_data.append({
                'tau_c': tau_c,
                'T2_echo': np.nan,
                'T2_echo_lower': np.nan,
                'T2_echo_upper': np.nan,
                'R2_echo': np.nan,
                'T2': fit_result_fid['T2'] if fit_result_fid else np.nan,
                'xi': xi,
            })
    except Exception as e:
        print(f"  ❌ Simulation failed: {e}")
        results_data.append({
            'tau_c': tau_c,
            'T2_echo': np.nan,
            'T2_echo_lower': np.nan,
            'T2_echo_upper': np.nan,
            'R2_echo': np.nan,
            'T2': np.nan,
            'xi': xi,
        })

# Save results
if results_data:
    df_new = pd.DataFrame(results_data)
    
    # Update existing echo data
    if len(df_echo_existing) > 0:
        # Remove old MN regime points
        df_echo_existing['xi'] = gamma_e * B_rms * df_echo_existing['tau_c']
        df_echo_existing = df_echo_existing[df_echo_existing['xi'] >= 0.2].copy()
        
        # Combine with new data
        df_echo_updated = pd.concat([df_echo_existing, df_new], ignore_index=True)
        df_echo_updated = df_echo_updated.sort_values('tau_c')
        df_echo_updated = df_echo_updated.drop('xi', axis=1)  # Remove temporary column
    else:
        df_echo_updated = df_new
    
    # Save
    output_file = output_dir / "t2_echo_vs_tau_c.csv"
    df_echo_updated.to_csv(output_file, index=False)
    print(f"\n✅ Updated: {output_file}")
    print(f"   Total points: {len(df_echo_updated)}")
    
    # Recalculate echo gain
    print(f"\n🔄 Recalculating echo gain...")
    import subprocess
    subprocess.run(['python3', 'analyze_echo_gain.py'], check=False)
    
    print(f"\n{'='*80}")
    print(f"✅ MN regime echo sweep 완료!")
    print(f"{'='*80}")

if __name__ == '__main__':
    pass

