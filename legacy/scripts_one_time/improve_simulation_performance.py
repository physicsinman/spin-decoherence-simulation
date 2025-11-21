#!/usr/bin/env python3
"""
시뮬레이션 성능 개선 스크립트

개선 사항:
1. QS regime에서 T_max를 더 길게 (50-100배 T2)
2. Crossover regime 포인트 밀도 증가
3. 더 정확한 fitting을 위한 파라미터 조정
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single
from spin_decoherence.analysis.bootstrap import bootstrap_T2
from spin_decoherence.analysis.fitting import fit_coherence_decay_with_offset
import json
from datetime import datetime

# ============================================
# PARAMETERS
# ============================================
gamma_e = 1.76e11          # rad/(s·T)
B_rms = 0.05e-3            # T (50 μT)
tau_c_min = 3e-9           # s
tau_c_max = 1e-3           # s
N_traj = 2000              # trajectories

# ============================================
# IMPROVED ADAPTIVE PARAMETERS
# ============================================

def get_dt_improved(tau_c):
    """개선된 timestep 선택"""
    # 더 정밀한 timestep (더 많은 샘플)
    return tau_c / 150  # 100 → 150 steps per tau_c

def get_tmax_improved(tau_c, B_rms, gamma_e):
    """개선된 T_max 계산"""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est  # 기존과 동일
    elif xi > 3:  # QS regime - 개선!
        T2_est = 1.0 / (gamma_e * B_rms)
        # 개선: 30배 → 50-100배로 증가
        # QS regime에서는 완전한 Gaussian decay를 포착해야 함
        multiplier = 50 if xi < 10 else 100  # 매우 큰 xi에서는 더 길게
        T_max_from_T2 = multiplier * T2_est
        
        # OU noise burn-in도 고려
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        
        # 메모리 제한 (100 ms)
        return min(T_max_final, 100e-3)
    else:  # Crossover - 개선!
        # Crossover에서는 더 보수적으로
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 15 * T2_est  # 10 → 15배

def build_tau_c_sweep_improved():
    """
    개선된 tau_c grid
    - Crossover regime 포인트 밀도 증가
    - QS regime에서도 더 많은 포인트
    """
    # MN regime: 기존과 동일
    mn = np.logspace(np.log10(3e-9), np.log10(3e-8), 18, endpoint=False)
    
    # Crossover regime: 포인트 증가 (24 → 30)
    crossover = np.logspace(np.log10(3e-8), np.log10(3e-6), 30, endpoint=False)
    
    # QS regime: 포인트 증가 (20 → 25)
    qs = np.logspace(np.log10(3e-6), np.log10(1e-3), 25)
    
    tau_vals = np.unique(np.concatenate([mn, crossover, qs]))
    return tau_vals

# ============================================
# MAIN FUNCTION
# ============================================

def run_improved_simulation():
    """개선된 파라미터로 시뮬레이션 실행"""
    print("="*80)
    print("개선된 FID 시뮬레이션")
    print("="*80)
    print("\n개선 사항:")
    print("  1. QS regime: T_max = 50-100×T2 (기존: 30×)")
    print("  2. Crossover regime: 포인트 밀도 증가 (24 → 30)")
    print("  3. QS regime: 포인트 증가 (20 → 25)")
    print("  4. Timestep: 더 정밀 (100 → 150 steps/tau_c)")
    print("  5. Crossover: T_max = 15×T2 (기존: 10×)")
    
    tau_c_sweep = build_tau_c_sweep_improved()
    print(f"\n총 포인트: {len(tau_c_sweep)}")
    
    output_dir = Path("results_comparison")
    output_dir.mkdir(exist_ok=True)
    
    results_data = []
    
    for i, tau_c in enumerate(tau_c_sweep):
        print(f"\n[{i+1}/{len(tau_c_sweep)}] τ_c = {tau_c*1e6:.3f} μs")
        
        # 개선된 파라미터 사용
        dt = get_dt_improved(tau_c)
        T_max = get_tmax_improved(tau_c, B_rms, gamma_e)
        xi = gamma_e * B_rms * tau_c
        
        print(f"  dt = {dt*1e9:.2f} ns, T_max = {T_max*1e6:.2f} μs, ξ = {xi:.3f}")
        
        # 메모리 체크
        N_steps = int(T_max / dt) + 1
        memory_gb = (N_steps * N_traj * 8) / (1024**3)
        if memory_gb > 10:
            print(f"  ⚠️  메모리 경고: {memory_gb:.1f} GB 예상")
        
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj,
            'seed': 42 + i,
            'output_dir': str(output_dir),
            'compute_bootstrap': True,
            'save_delta_B_sample': False,
        }
        
        try:
            result = run_simulation_single(tau_c, params=params, verbose=False)
        except Exception as e:
            print(f"  ❌ 시뮬레이션 실패: {e}")
            results_data.append({
                'tau_c': tau_c,
                'T2': np.nan,
                'T2_lower': np.nan,
                'T2_upper': np.nan,
                'R2': np.nan,
                'xi': xi
            })
            continue
        
        # 결과 추출
        fit_result = result['fit_result']
        if fit_result is not None:
            T2 = fit_result.get('T2', np.nan)
            T2_ci = result.get('T2_ci', None)
            R2 = fit_result.get('R2', np.nan)
            
            # CI 추출 (기존 로직과 동일)
            if T2_ci is not None:
                T2_lower = T2_ci[0]
                T2_upper = T2_ci[1]
            else:
                T2_error = fit_result.get('T2_error', np.nan)
                if not np.isnan(T2_error) and T2_error > 0:
                    T2_lower = T2 - 1.96 * T2_error
                    T2_upper = T2 + 1.96 * T2_error
                else:
                    T2_lower = T2 * 0.95
                    T2_upper = T2 * 1.05
            
            results_data.append({
                'tau_c': tau_c,
                'T2': T2,
                'T2_lower': T2_lower,
                'T2_upper': T2_upper,
                'R2': R2,
                'xi': xi
            })
            
            print(f"  T2 = {T2*1e6:.3f} μs (R² = {R2:.4f})")
        else:
            print(f"  ⚠️  Fit 실패!")
            results_data.append({
                'tau_c': tau_c,
                'T2': np.nan,
                'T2_lower': np.nan,
                'T2_upper': np.nan,
                'R2': np.nan,
                'xi': xi
            })
    
    # 저장
    df = pd.DataFrame(results_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"t2_vs_tau_c_improved_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ 결과 저장: {output_file}")
    print(f"{'='*80}")
    print(f"\n요약:")
    print(f"  총 포인트: {len(results_data)}")
    print(f"  성공한 fit: {df['T2'].notna().sum()}")
    if df['R2'].notna().sum() > 0:
        print(f"  평균 R²: {df['R2'].mean():.4f}")
        print(f"  R² < 0.7인 포인트: {(df['R2'] < 0.7).sum()}")
        print(f"  R² < 0.5인 포인트: {(df['R2'] < 0.5).sum()}")
    
    return df

if __name__ == '__main__':
    df = run_improved_simulation()

