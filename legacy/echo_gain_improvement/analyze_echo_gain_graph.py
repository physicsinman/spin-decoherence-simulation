#!/usr/bin/env python3
"""Analyze echo gain graph issues in detail."""

import pandas as pd
import numpy as np

# Load data
echo_gain = pd.read_csv('results_comparison/echo_gain.csv')
echo_t2 = pd.read_csv('results_comparison/t2_echo_vs_tau_c.csv')
fid_t2 = pd.read_csv('results_comparison/t2_vs_tau_c.csv')

# Merge
merged = pd.merge(echo_gain, echo_t2[['tau_c', 'R2_echo', 'T2_echo']], on='tau_c', how='left', suffixes=('', '_echo_file'))
merged = pd.merge(merged, fid_t2[['tau_c', 'T2', 'R2']], on='tau_c', how='left', suffixes=('', '_fid_file'))

# Filter valid points
valid = merged[merged['R2_echo'].notna()].copy()
valid = valid[valid['echo_gain'] >= 0.95].copy()
valid = valid.sort_values('tau_c')

print('=== 문제 구간 상세 분석 ===\n')

# 1. 첫 번째 급격한 하락 (1.5 → 1.0)
print('1. 첫 번째 급격한 하락 구간 (1.5 → 1.0):')
drop1 = valid[(valid['tau_c'] >= 2.3e-8) & (valid['tau_c'] <= 3.5e-8)]
for idx, row in drop1.iterrows():
    print(f"τc={row['tau_c']*1e6:.3f}μs, ξ={row['xi']:.3f}, T2={row['T2']:.2e}, T2_echo={row['T2_echo']:.2e}, gain={row['echo_gain']:.3f}, method={row['method_used']}, R²={row['R2_echo']:.3f}")
print()

# 2. 평탄 구간 1.0 (0.03-0.08 μs)
print('2. 평탄 구간 1.0 (0.03-0.08 μs):')
flat1 = valid[(valid['tau_c'] >= 3e-8) & (valid['tau_c'] <= 8e-8)]
print(f'포인트 수: {len(flat1)}')
print('처음 5개:')
for idx, row in flat1.head(5).iterrows():
    print(f"τc={row['tau_c']*1e6:.3f}μs, ξ={row['xi']:.3f}, T2={row['T2']:.2e}, T2_echo={row['T2_echo']:.2e}, gain={row['echo_gain']:.3f}, method={row['method_used']}, R²={row['R2_echo']:.3f}")
all_one = all(flat1['echo_gain'] == 1.0)
print(f'모두 gain=1.0? {all_one}')
print()

# 3. 급격한 상승 (1.0 → 5.0)
print('3. 급격한 상승 구간 (1.0 → 5.0):')
rise = valid[(valid['tau_c'] >= 7.5e-8) & (valid['tau_c'] <= 9e-8)]
for idx, row in rise.iterrows():
    print(f"τc={row['tau_c']*1e6:.3f}μs, ξ={row['xi']:.3f}, T2={row['T2']:.2e}, T2_echo={row['T2_echo']:.2e}, gain={row['echo_gain']:.3f}, method={row['method_used']}, R²={row['R2_echo']:.3f}")
print()

# 4. 평탄 구간 5.0
print('4. 평탄 구간 5.0 (0.09-0.18 μs):')
flat5 = valid[(valid['tau_c'] >= 8.7e-8) & (valid['tau_c'] <= 1.9e-7)]
print(f'포인트 수: {len(flat5)}')
print('처음 5개:')
for idx, row in flat5.head(5).iterrows():
    print(f"τc={row['tau_c']*1e6:.3f}μs, ξ={row['xi']:.3f}, T2={row['T2']:.2e}, T2_echo={row['T2_echo']:.2e}, gain={row['echo_gain']:.3f}, method={row['method_used']}, R²={row['R2_echo']:.3f}")
all_five = all(flat5['echo_gain'] == 5.0)
print(f'모두 gain=5.0? {all_five}')
print()

# 5. 두 번째 급격한 하락 (5.0 → 2.6)
print('5. 두 번째 급격한 하락 구간 (5.0 → 2.6):')
drop2 = valid[(valid['tau_c'] >= 1.9e-7) & (valid['tau_c'] <= 2.3e-7)]
for idx, row in drop2.iterrows():
    print(f"τc={row['tau_c']*1e6:.3f}μs, ξ={row['xi']:.3f}, T2={row['T2']:.2e}, T2_echo={row['T2_echo']:.2e}, gain={row['echo_gain']:.3f}, method={row['method_used']}, R²={row['R2_echo']:.3f}")
print()

# 6. 평탄 구간 3.0
print('6. 평탄 구간 3.0 (0.25-1.0 μs):')
flat3 = valid[(valid['tau_c'] >= 2.5e-7) & (valid['tau_c'] <= 1e-3)]
print(f'포인트 수: {len(flat3)}')
print('처음 5개:')
for idx, row in flat3.head(5).iterrows():
    print(f"τc={row['tau_c']*1e6:.3f}μs, ξ={row['xi']:.3f}, T2={row['T2']:.2e}, T2_echo={row['T2_echo']:.2e}, gain={row['echo_gain']:.3f}, method={row['method_used']}, R²={row['R2_echo']:.3f}")
all_three = all(flat3['echo_gain'] == 3.0)
print(f'모두 gain=3.0? {all_three}')
print()

# Method별 통계
print('=== Method별 통계 ===')
print(valid['method_used'].value_counts())
print()
print('Method별 gain 분포:')
for method in valid['method_used'].unique():
    method_data = valid[valid['method_used'] == method]
    print(f"{method}: gain 범위 [{method_data['echo_gain'].min():.3f}, {method_data['echo_gain'].max():.3f}], 평균={method_data['echo_gain'].mean():.3f}")
print()

# T2_echo와 T2_echo_used 비교
print('=== T2_echo 값 비교 ===')
print('T2_echo (from echo_t2 file) vs T2_echo_used (from echo_gain file):')
comparison = valid[['tau_c', 'T2_echo', 'T2_echo_used', 'echo_gain', 'method_used']].copy()
comparison['diff'] = (comparison['T2_echo'] - comparison['T2_echo_used']) / comparison['T2_echo'] * 100
print(comparison.head(20).to_string())
print()

# Check for suspicious patterns
print('=== 의심스러운 패턴 ===')
print('1. gain이 정확히 1.0, 1.5, 2.0, 3.0, 5.0인 포인트:')
suspicious_gains = [1.0, 1.5, 2.0, 3.0, 5.0]
for gain_val in suspicious_gains:
    count = (valid['echo_gain'] == gain_val).sum()
    if count > 0:
        print(f"   gain={gain_val}: {count}개 포인트")
        method_dist = valid[valid['echo_gain'] == gain_val]['method_used'].value_counts()
        print(f"      Method 분포: {dict(method_dist)}")

