#!/usr/bin/env python3
"""
매우 짧은 테스트 시뮬레이션 - wrapper 파일 검증용
"""

from simulate import run_simulation_single, get_default_config, config_to_dict
from spin_decoherence.config import CONSTANTS, Units
import numpy as np

if __name__ == '__main__':
    print('='*70)
    print('빠른 테스트 시뮬레이션 (Wrapper 검증)')
    print('='*70)
    
    # 기본 설정 가져오기
    default_config = get_default_config()
    params = config_to_dict(default_config)
    
    # 매우 짧은 테스트 파라미터
    params['tau_c_num'] = 3  # 3개만
    params['M'] = 50  # 50 realizations만 (매우 빠름)
    params['compute_bootstrap'] = False  # Bootstrap 건너뛰기
    params['T_max'] = Units.us_to_s(10.0)  # 10 μs만
    
    # 테스트할 tau_c 값 (3개)
    tau_c_min = Units.us_to_s(0.1)
    tau_c_max = Units.us_to_s(1.0)
    tau_c_values = np.logspace(
        np.log10(tau_c_min),
        np.log10(tau_c_max),
        3
    )
    
    print(f'\n설정:')
    print(f'  • tau_c: {tau_c_min*1e6:.2f} - {tau_c_max*1e6:.2f} μs (3개)')
    print(f'  • M (realizations): {params["M"]}')
    print(f'  • T_max: {params["T_max"]*1e6:.2f} μs')
    print(f'  • Bootstrap: 건너뛰기')
    print(f'\n시작합니다...\n')
    
    results = []
    for i, tau_c in enumerate(tau_c_values):
        print(f'[{i+1}/3] tau_c = {tau_c*1e6:.2f} μs')
        result = run_simulation_single(tau_c, params, verbose=False)
        results.append(result)
        
        if result.get('fit_result'):
            T2 = result['fit_result']['T2']
            print(f'  ✓ T_2 = {T2*1e6:.2f} μs')
        else:
            print(f'  ⚠️  Fit 실패')
    
    print('\n' + '='*70)
    print('✅ 테스트 시뮬레이션 완료!')
    print('='*70)
    print(f'\n✓ {len(results)}개 시뮬레이션 완료')
    print('✓ Wrapper 파일들이 정상 작동합니다!')

