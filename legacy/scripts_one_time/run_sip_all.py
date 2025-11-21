#!/usr/bin/env python3
"""
Si_P 전체 시뮬레이션 실행 (OU + Double_OU, FID + Hahn)
"""

from simulate_materials_improved import run_full_comparison_improved

if __name__ == '__main__':
    print('='*80)
    print('Si_P 전체 시뮬레이션 실행')
    print('='*80)
    print('\n설정:')
    print('  • Material: Si_P')
    print('  • Noise models: OU, Double_OU')
    print('  • Sequences: FID, Hahn')
    print('  • Parameter validation: ✓')
    print('  • Memory-efficient: ✓')
    print('  • Improved T2 extraction: ✓')
    print('\n예상 시간:')
    print('  • Si_P OU: ~14시간')
    print('  • Si_P Double_OU: ~1.8시간')
    print('  • 총합: ~16시간')
    print('\n시작합니다...\n')
    
    results = run_full_comparison_improved(
        materials=['Si_P'],
        noise_models=['OU', 'Double_OU'],
        sequences=['FID', 'Hahn'],
        use_validation=True,
        use_adaptive=True,
        use_improved_t2=True,
        save_curves=False
    )
    
    print('\n' + '='*80)
    print('✅ Si_P 전체 시뮬레이션 완료!')
    print('='*80)
