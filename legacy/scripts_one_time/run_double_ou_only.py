#!/usr/bin/env python3
"""
Double-OU 케이스만 실행하는 스크립트
"""

from simulate_materials_improved import run_full_comparison_improved

if __name__ == '__main__':
    print('='*70)
    print('Double-OU 시뮬레이션 실행')
    print('='*70)
    print('\n설정:')
    print('  • Materials: Si_P, GaAs')
    print('  • Noise model: Double_OU만')
    print('  • Sequences: FID, Hahn')
    print('  • Parameter validation: ✓')
    print('  • Memory-efficient: ✓')
    print('  • Improved T2 extraction: ✓')
    print('\n시작합니다...\n')
    
    # Double-OU만 실행
    results = run_full_comparison_improved(
        materials=['Si_P', 'GaAs'],
        noise_models=['Double_OU'],  # Double_OU만
        sequences=['FID', 'Hahn'],
        use_validation=True,
        use_adaptive=True,
        use_improved_t2=True,
        save_curves=False
    )
    
    print('\n' + '='*70)
    print('✅ Double-OU 시뮬레이션 완료!')
    print('='*70)

