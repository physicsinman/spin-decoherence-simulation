#!/usr/bin/env python3
"""
Si:P Double_OU 빠른 테스트 - 메모리 문제 해결 확인
"""

import yaml
from simulate_materials_improved import run_single_case_improved
import json
from pathlib import Path

def test_sip_double_ou_quick():
    """Si:P Double_OU 빠른 테스트 (3개 포인트만)"""
    print('='*70)
    print('Si:P Double_OU 빠른 테스트')
    print('='*70)
    
    # profiles.yaml 로드
    with open('profiles.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    profile = data['materials']['Si_P']
    
    # 테스트용: tau_c2_num을 3으로 임시 변경
    original_tau_c2_num = profile['Double_OU']['tau_c2_num']
    profile['Double_OU']['tau_c2_num'] = 3  # 3개만 테스트
    
    print('\n테스트 설정:')
    print(f'  • Material: Si_P')
    print(f'  • Noise model: Double_OU')
    print(f'  • Sequence: FID (먼저 테스트)')
    print(f'  • tau_c2_num: 3 (빠른 테스트)')
    print(f'  • M: {profile["M"]}')
    print(f'  • T_max: {profile["T_max"]*1e3:.1f} ms')
    print(f'  • dt: {profile["dt"]*1e9:.1f} ns')
    print('\n시작합니다...\n')
    
    try:
        # FID만 테스트
        result = run_single_case_improved(
            material_name='Si_P',
            profile=profile,
            noise_model='Double_OU',
            sequence_type='FID',
            use_validation=True,
            use_adaptive=False,  # Adaptive 비활성화 (논문 설정)
            use_improved_t2=True,
            verbose=True
        )
        
        # 결과 확인
        print('\n' + '='*70)
        print('✅ 테스트 완료!')
        print('='*70)
        
        data_points = len(result.get('data', []))
        print(f'\n결과:')
        print(f'  • 데이터 포인트: {data_points}/3')
        
        if data_points > 0:
            print(f'  • 첫 번째 T2: {result["data"][0].get("T2", "N/A")}')
            print(f'  • 메모리 오류 없음 ✓')
            print(f'\n✅ 파라미터가 정상 작동합니다!')
            print(f'   전체 시뮬레이션을 실행하세요.')
        else:
            print(f'  ⚠️  데이터 포인트가 없습니다. 로그를 확인하세요.')
        
        # 원래 값 복원
        profile['Double_OU']['tau_c2_num'] = original_tau_c2_num
        
        return result
        
    except MemoryError as e:
        print(f'\n❌ 메모리 오류 발생: {e}')
        print(f'   파라미터를 더 줄여야 할 수 있습니다.')
        profile['Double_OU']['tau_c2_num'] = original_tau_c2_num
        raise
    except Exception as e:
        print(f'\n❌ 오류 발생: {e}')
        import traceback
        traceback.print_exc()
        profile['Double_OU']['tau_c2_num'] = original_tau_c2_num
        raise

if __name__ == '__main__':
    test_sip_double_ou_quick()

