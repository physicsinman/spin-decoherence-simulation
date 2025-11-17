#!/usr/bin/env python3
"""
청크 처리로 전체 시뮬레이션 실행
"""

from simulate_materials_improved import run_full_comparison_improved

if __name__ == '__main__':
    print('='*70)
    print('청크 처리로 전체 시뮬레이션 시작')
    print('='*70)
    print('\n설정:')
    print('  • Parameter validation: ✓')
    print('  • Memory-efficient (chunked): ✓')
    print('  • Adaptive simulation: ✓')
    print('  • Improved T2 extraction: ✓')
    print('\n시작합니다...\n')
    
    # 전체 시뮬레이션 실행
    results = run_full_comparison_improved(
        materials=['Si_P', 'GaAs'],
        noise_models=['OU', 'Double_OU'],
        sequences=['FID', 'Hahn'],
        use_validation=True,
        use_adaptive=True,
        use_improved_t2=True,
        save_curves=False
    )
    
    print('\n' + '='*70)
    print('✅ 시뮬레이션 완료!')
    print('='*70)
