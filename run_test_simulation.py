#!/usr/bin/env python3
"""
짧은 테스트 시뮬레이션 - 수정된 파라미터 검증
"""

from simulate_materials_improved import run_full_comparison_improved

if __name__ == '__main__':
    print('='*70)
    print('테스트 시뮬레이션 (수정된 파라미터 검증)')
    print('='*70)
    print('\n설정:')
    print('  • GaAs만 실행 (빠름)')
    print('  • Single OU만 (간단)')
    print('  • FID만 (가장 기본)')
    print('  • tau_c_num: 10 (빠른 테스트)')
    print('\n시작합니다...\n')
    
    # 짧은 테스트: GaAs만, OU만, FID만
    results = run_full_comparison_improved(
        materials=['GaAs'],  # GaAs만 (빠름)
        noise_models=['OU'],  # Single OU만
        sequences=['FID'],    # FID만
        use_validation=True,
        use_adaptive=True,
        use_improved_t2=True,
        save_curves=False
    )
    
    print('\n' + '='*70)
    print('✅ 테스트 시뮬레이션 완료!')
    print('='*70)
    print('\n결과 확인 후 전체 시뮬레이션을 실행하세요.')
    print('전체 시뮬: python3 run_full_simulation_chunked.py')

