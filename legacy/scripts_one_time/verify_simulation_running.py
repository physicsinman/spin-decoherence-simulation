#!/usr/bin/env python3
"""
Verify that simulations are actually running
시뮬레이션이 실제로 실행되는지 확인
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from spin_decoherence.simulation.fid import run_simulation_single

print("="*80)
print("시뮬레이션 실행 여부 확인")
print("="*80)

# Load data
fid = pd.read_csv('results_comparison/t2_vs_tau_c.csv')

# Find points that need improvement
poor_fit = fid[fid['R2'] < 0.98].sort_values('tau_c')

print(f"\nR² < 0.98인 포인트: {len(poor_fit)}개")

if len(poor_fit) == 0:
    print("\n✅ 모든 포인트가 R² ≥ 0.98입니다!")
    print("   시뮬레이션이 스킵되는 것이 정상입니다.")
else:
    print(f"\n⚠️  {len(poor_fit)}개 포인트 재시뮬레이션 필요")
    print(f"   예상 시간: ~{len(poor_fit) * 25 / 60:.1f}시간")
    
    # Test with first point
    first = poor_fit.iloc[0]
    tau_c = first['tau_c']
    
    print(f"\n{'='*80}")
    print("실제 시뮬레이션 실행 테스트 (첫 번째 포인트)")
    print("="*80)
    print(f"\n테스트 포인트:")
    print(f"  τc = {tau_c*1e6:.3f} μs")
    print(f"  이전 R² = {first['R2']:.4f}")
    print(f"  N_traj = 5000")
    print(f"\n⚠️  실제 시뮬레이션이 실행됩니다!")
    print(f"   진행 상황을 확인할 수 있습니다.")
    print(f"   이것이 실제 시뮬레이션 시간입니다.\n")
    
    response = input("테스트 실행? (yes/no): ")
    if response.lower() == 'yes':
        from comprehensive_improvement import get_tmax_improved, get_adaptive_dt
        
        T_max = get_tmax_improved(tau_c, 0.05e-3, 1.76e11)
        dt = get_adaptive_dt(tau_c, T_max=T_max, max_memory_gb=8.0)
        
        params = {
            'gamma_e': 1.76e11,
            'B_rms': 0.05e-3,
            'dt': dt,
            'T_max': T_max,
            'M': 5000,
            'seed': 42,
            'compute_bootstrap': True,
        }
        
        print(f"\n실행 중... (T_max = {T_max*1e6:.2f} μs, dt = {dt*1e9:.2f} ns)")
        print("진행 상황을 확인하세요:\n")
        
        start = time.time()
        
        try:
            result = run_simulation_single(tau_c, params=params, verbose=True)
            elapsed = time.time() - start
            
            fit_result = result.get('fit_result', {})
            new_r2 = fit_result.get('R2', np.nan)
            
            print(f"\n{'='*80}")
            print(f"✅ 테스트 완료!")
            print(f"{'='*80}")
            print(f"실제 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
            print(f"이전 R² = {first['R2']:.4f}")
            print(f"새로운 R² = {new_r2:.4f}")
            print(f"개선 = {new_r2 - first['R2']:+.4f}")
            print(f"\n💡 이것이 실제 시뮬레이션 시간입니다!")
            print(f"   {len(poor_fit)}개 포인트 × {elapsed/60:.1f}분 = ~{len(poor_fit) * elapsed / 3600:.1f}시간")
            print(f"\n✅ 시뮬레이션이 정상적으로 실행됩니다!")
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("취소되었습니다.")

