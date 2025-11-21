#!/usr/bin/env python3
"""
논문용 시뮬레이션 결과 검증 스크립트

이 스크립트는 논문에 포함할 시뮬레이션 결과의 품질을 검증합니다:
- 사용된 모델 분포 확인
- Regime별 fitting 성공률 확인
- Analytical estimate 사용 빈도 확인
"""

import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_results(json_file):
    """시뮬레이션 결과 파일 분석"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 통계 수집
    models_used = defaultdict(int)
    regime_stats = defaultdict(lambda: {'total': 0, 'analytical': 0, 'fitted': 0})
    tau_c_values = []
    T2_values = []
    xi_values = []
    
    for point in data.get('data', []):
        model = point.get('model', 'unknown')
        models_used[model] += 1
        
        # Regime 정보 추출
        xi = point.get('xi', None)
        tau_c = point.get('tau_c', None)
        T2 = point.get('T2', None)
        
        if xi is not None:
            xi_values.append(xi)
            if xi < 1.0:
                regime = 'MN'  # Motional Narrowing
            elif xi < 2.0:
                regime = 'Crossover'
            else:
                regime = 'QS'  # Quasi-Static
            
            regime_stats[regime]['total'] += 1
            if model == 'analytical_flat_curve':
                regime_stats[regime]['analytical'] += 1
            else:
                regime_stats[regime]['fitted'] += 1
        
        if tau_c is not None:
            tau_c_values.append(tau_c)
        if T2 is not None:
            T2_values.append(T2)
    
    return {
        'models_used': dict(models_used),
        'regime_stats': dict(regime_stats),
        'tau_c_values': tau_c_values,
        'T2_values': T2_values,
        'xi_values': xi_values,
        'total_points': len(data.get('data', []))
    }

def print_analysis(results, json_file):
    """분석 결과 출력"""
    print("="*80)
    print(f"📊 결과 분석: {Path(json_file).name}")
    print("="*80)
    
    total = results['total_points']
    print(f"\n총 데이터 포인트: {total}")
    
    # 모델 사용 분포
    print("\n📈 사용된 모델 분포:")
    models = results['models_used']
    for model, count in sorted(models.items(), key=lambda x: -x[1]):
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {model:30s}: {count:4d} points ({percentage:5.1f}%)")
    
    # Regime별 통계
    print("\n📊 Regime별 통계:")
    regime_stats = results['regime_stats']
    for regime in ['MN', 'Crossover', 'QS']:
        if regime in regime_stats:
            stats = regime_stats[regime]
            total_regime = stats['total']
            analytical = stats['analytical']
            fitted = stats['fitted']
            
            print(f"\n  {regime} Regime (ξ {'< 1' if regime == 'MN' else '1-2' if regime == 'Crossover' else '> 2'}):")
            print(f"    총 포인트: {total_regime}")
            print(f"    Fitted: {fitted} ({fitted/total_regime*100:.1f}%)")
            print(f"    Analytical: {analytical} ({analytical/total_regime*100:.1f}%)")
    
    # 논문용 품질 체크
    print("\n✅ 논문용 품질 체크:")
    analytical_total = models.get('analytical_flat_curve', 0)
    analytical_percentage = analytical_total / total * 100 if total > 0 else 0
    
    checks = {
        "Analytical estimate 사용 < 5%": analytical_percentage < 5.0,
        "Exponential model 사용 (MN regime)": 'exponential' in models or 'exponential_offset' in models,
        "Gaussian model 사용 (QS regime)": 'gaussian' in models or 'gaussian_offset' in models,
        "Fitted models > 95%": (total - analytical_total) / total * 100 > 95.0 if total > 0 else False
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    # T2 값 범위 확인
    if results['T2_values']:
        T2_array = np.array(results['T2_values'])
        T2_array = T2_array[T2_array > 0]  # 양수만
        T2_array = T2_array[T2_array < 1.0]  # 1초 미만만 (비정상 값 제외)
        
        if len(T2_array) > 0:
            print(f"\n📏 T2 값 범위:")
            print(f"  최소: {np.min(T2_array)*1e6:.2f} μs")
            print(f"  최대: {np.max(T2_array)*1e6:.2f} μs")
            print(f"  평균: {np.mean(T2_array)*1e6:.2f} μs")
            print(f"  중앙값: {np.median(T2_array)*1e6:.2f} μs")
    
    print("\n" + "="*80)

def main():
    """메인 함수"""
    # 최신 결과 파일 찾기
    result_files = glob.glob("results_comparison/*.json") + glob.glob("results_test/*.json")
    
    if not result_files:
        print("❌ 결과 파일을 찾을 수 없습니다.")
        print("   다음 디렉토리에서 검색: results_comparison/, results_test/")
        return
    
    # 최신 파일 선택
    latest_file = max(result_files, key=lambda p: Path(p).stat().st_mtime)
    
    print(f"🔍 분석 중: {latest_file}\n")
    
    # 분석 실행
    results = analyze_results(latest_file)
    print_analysis(results, latest_file)
    
    # 추가 파일이 있으면 모두 분석
    if len(result_files) > 1:
        print("\n\n다른 결과 파일들:")
        for f in sorted(result_files, key=lambda p: Path(p).stat().st_mtime, reverse=True)[:5]:
            if f != latest_file:
                print(f"  - {Path(f).name}")

if __name__ == '__main__':
    main()

