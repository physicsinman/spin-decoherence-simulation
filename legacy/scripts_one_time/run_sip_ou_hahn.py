#!/usr/bin/env python3
"""
Si_P | OU | Hahn 시뮬레이션만 실행
"""

from simulate_materials_improved import run_full_comparison_improved

if __name__ == '__main__':
    print('='*80)
    print('Si_P | OU | Hahn 시뮬레이션')
    print('='*80)
    print('\n설정:')
    print('  • Material: Si_P')
    print('  • Noise model: OU')
    print('  • Sequence: Hahn')
    print('  • 포인트: 30개')
    print('\n예상 시간: ~7시간')
    print('\n시작합니다...\n')
    
    results = run_full_comparison_improved(
        materials=['Si_P'],
        noise_models=['OU'],
        sequences=['Hahn'],
        use_validation=True,
        use_adaptive=True,
        use_improved_t2=True,
        save_curves=False
    )
    
    print('\n' + '='*80)
    print('✅ Si_P | OU | Hahn 완료!')
    print('='*80)

