#!/usr/bin/env python3
"""
R²가 낮은 포인트만 선택적으로 재시뮬레이션
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single

# 파라미터
gamma_e = 1.76e11
B_rms = 0.05e-3
N_traj = 2000

def get_dt(tau_c, T_max=None, max_memory_gb=8.0):
    """Adaptive timestep with memory limit"""
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
    """Calculate T_max with improved parameters"""
    xi = gamma_e * B_rms * tau_c
    if xi < 0.3:
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est
    elif xi > 3:
        T2_est = 1.0 / (gamma_e * B_rms)
        if xi < 10:
            multiplier = 80
        elif xi < 50:
            multiplier = 120
        else:
            multiplier = 150
        T_max_from_T2 = multiplier * T2_est
        burnin_time = 5.0 * tau_c
        T_max_final = max(T_max_from_T2, burnin_time)
        return min(T_max_final, 100e-3)
    else:
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 20 * T2_est

def main():
    print("="*80)
    print("R²가 낮은 포인트 재시뮬레이션")
    print("="*80)
    
    # 기존 데이터 로드
    df = pd.read_csv("results_comparison/t2_vs_tau_c.csv")
    
    # R² < 0.9인 포인트 찾기
    low_R2 = df[(df['R2'] < 0.9) & (df['T2'].notna())].copy()
    
    if len(low_R2) == 0:
        print("\n✅ R² < 0.9인 포인트가 없습니다!")
        return
    
    print(f"\nR² < 0.9인 포인트: {len(low_R2)}개")
    print(f"재시뮬레이션 시작...\n")
    
    output_dir = Path("results_comparison")
    updated_count = 0
    
    for idx, row in low_R2.iterrows():
        tau_c = row['tau_c']
        xi = row['xi']
        old_R2 = row['R2']
        
        print(f"[{updated_count+1}/{len(low_R2)}] τ_c = {tau_c*1e6:.3f} μs (ξ = {xi:.3f}, R² = {old_R2:.4f})")
        
        # 개선된 파라미터로 재시뮬레이션
        T_max = get_tmax(tau_c, B_rms, gamma_e)
        dt = get_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        
        params = {
            'B_rms': B_rms,
            'tau_c_range': (tau_c, tau_c),
            'tau_c_num': 1,
            'gamma_e': gamma_e,
            'dt': dt,
            'T_max': T_max,
            'M': N_traj,
            'seed': 42 + int(tau_c * 1e12),  # Unique seed
            'output_dir': str(output_dir),
            'compute_bootstrap': True,
            'save_delta_B_sample': False,
        }
        
        try:
            result = run_simulation_single(tau_c, params=params, verbose=False)
            fit_result = result['fit_result']
            
            if fit_result is not None:
                new_R2 = fit_result.get('R2', np.nan)
                new_T2 = fit_result.get('T2', np.nan)
                
                # 업데이트
                df.loc[idx, 'T2'] = new_T2
                df.loc[idx, 'R2'] = new_R2
                
                # CI 업데이트
                T2_ci = result.get('T2_ci', None)
                if T2_ci is not None:
                    df.loc[idx, 'T2_lower'] = T2_ci[0]
                    df.loc[idx, 'T2_upper'] = T2_ci[1]
                
                print(f"  ✅ R²: {old_R2:.4f} → {new_R2:.4f}")
                updated_count += 1
            else:
                print(f"  ❌ Fit 실패")
        except Exception as e:
            print(f"  ❌ 시뮬레이션 실패: {e}")
    
    # 저장
    df.to_csv("results_comparison/t2_vs_tau_c.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ {updated_count}/{len(low_R2)}개 포인트 업데이트 완료")
    print(f"📁 결과 저장: results_comparison/t2_vs_tau_c.csv")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

