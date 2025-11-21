#!/usr/bin/env python3
"""
Si_P | Double_OU | FID 시뮬레이션만 실행
"""

from simulate_materials_improved import run_full_comparison_improved

if __name__ == '__main__':
    print('='*80)
    print('Si_P | Double_OU | FID 시뮬레이션')
    print('='*80)
    print('\n설정:')
    print('  • Material: Si_P')
    print('  • Noise model: Double_OU')
    print('  • Sequence: FID')
    print('  • 포인트: 20개')
    print('\n예상 시간: ~0.9시간')
    print('\n시작합니다...\n')
    
    results = run_full_comparison_improved(
        materials=['Si_P'],
        noise_models=['Double_OU'],
        sequences=['FID'],
        use_validation=True,
        use_adaptive=True,
        use_improved_t2=True,
        save_curves=False
    )
    
    print('\n' + '='*80)
    print('✅ Si_P | Double_OU | FID 완료!')
    print('='*80)

