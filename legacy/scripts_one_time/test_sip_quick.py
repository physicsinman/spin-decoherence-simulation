#!/usr/bin/env python3
"""
Si_P 빠른 테스트 - 메모리 문제 해결 확인
Single OU와 Double OU 모두 테스트
"""

import yaml
from simulate_materials_improved import run_single_case_improved
import json
from pathlib import Path

def test_sip_quick():
    """Si_P 빠른 테스트 (3개 포인트만, FID만)"""
    print('='*80)
    print('Si_P 빠른 테스트 - 메모리 문제 해결 확인')
    print('='*80)
    
    # profiles.yaml 로드
    with open('profiles.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    profile = data['materials']['Si_P']
    
    # 원래 값 저장
    original_tau_c_num = profile['OU']['tau_c_num']
    original_tau_c2_num = profile['Double_OU']['tau_c2_num']
    
    # 테스트용: 포인트 수 줄이기
    profile['OU']['tau_c_num'] = 3
    profile['Double_OU']['tau_c2_num'] = 3
    
    print('\n📋 테스트 설정:')
    print(f'  • Material: Si_P')
    print(f'  • 포인트 수: 3개 (빠른 테스트)')
    print(f'  • Sequence: FID만')
    print(f'  • M: {profile["M"]}')
    print(f'  • T_max: {profile["T_max"]*1e3:.1f} ms')
    print(f'  • dt: {profile["dt"]*1e9:.1f} ns')
    
    results = {}
    
    # Test 1: Single OU
    print('\n' + '='*80)
    print('테스트 1: Si_P | OU | FID')
    print('='*80)
    try:
        result_ou = run_single_case_improved(
            material_name='Si_P',
            profile=profile,
            noise_model='OU',
            sequence_type='FID',
            use_validation=True,
            use_adaptive=True,
            use_improved_t2=True,
            verbose=True
        )
        
        data_points = len(result_ou.get('data', []))
        print(f'\n✅ Single OU 테스트 성공!')
        print(f'   데이터 포인트: {data_points}/3')
        if data_points > 0:
            print(f'   첫 번째 T2: {result_ou["data"][0].get("T2", "N/A")}')
        results['OU'] = 'SUCCESS'
        
    except MemoryError as e:
        print(f'\n❌ Single OU 메모리 오류: {e}')
        results['OU'] = 'MEMORY_ERROR'
    except Exception as e:
        print(f'\n❌ Single OU 오류: {e}')
        import traceback
        traceback.print_exc()
        results['OU'] = 'ERROR'
    
    # Test 2: Double OU
    print('\n' + '='*80)
    print('테스트 2: Si_P | Double_OU | FID')
    print('='*80)
    try:
        result_double = run_single_case_improved(
            material_name='Si_P',
            profile=profile,
            noise_model='Double_OU',
            sequence_type='FID',
            use_validation=True,
            use_adaptive=True,
            use_improved_t2=True,
            verbose=True
        )
        
        data_points = len(result_double.get('data', []))
        print(f'\n✅ Double OU 테스트 성공!')
        print(f'   데이터 포인트: {data_points}/3')
        if data_points > 0:
            print(f'   첫 번째 T2: {result_double["data"][0].get("T2", "N/A")}')
        results['Double_OU'] = 'SUCCESS'
        
    except MemoryError as e:
        print(f'\n❌ Double OU 메모리 오류: {e}')
        results['Double_OU'] = 'MEMORY_ERROR'
    except Exception as e:
        print(f'\n❌ Double OU 오류: {e}')
        import traceback
        traceback.print_exc()
        results['Double_OU'] = 'ERROR'
    
    # 원래 값 복원
    profile['OU']['tau_c_num'] = original_tau_c_num
    profile['Double_OU']['tau_c2_num'] = original_tau_c2_num
    
    # 최종 결과
    print('\n' + '='*80)
    print('📊 최종 결과')
    print('='*80)
    print(f'  Single OU: {results.get("OU", "NOT_TESTED")}')
    print(f'  Double OU: {results.get("Double_OU", "NOT_TESTED")}')
    
    if results.get('OU') == 'SUCCESS' and results.get('Double_OU') == 'SUCCESS':
        print('\n✅ 모든 테스트 성공!')
        print('   전체 시뮬레이션을 실행할 수 있습니다.')
        return True
    else:
        print('\n⚠️  일부 테스트 실패')
        print('   코드를 다시 확인하거나 파라미터를 조정해야 합니다.')
        return False

if __name__ == '__main__':
    success = test_sip_quick()
    exit(0 if success else 1)

