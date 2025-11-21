#!/usr/bin/env python3
"""
Diagnose Echo Gain issues in crossover regime (τc ~ 0.3-0.5 μs)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('='*80)
print('Echo Gain 문제 구간 상세 진단')
print('='*80)

# 데이터 로드
fid = pd.read_csv('results_comparison/t2_vs_tau_c.csv')
echo = pd.read_csv('results_comparison/t2_echo_vs_tau_c.csv')
gain = pd.read_csv('results_comparison/echo_gain.csv')

# 문제 구간: τc ~ 0.3-0.5 μs
mask = (gain['tau_c'] >= 0.2e-6) & (gain['tau_c'] <= 0.6e-6)
problem = gain[mask].sort_values('tau_c').copy()

print(f'\n문제 구간 (τc = 0.2-0.6 μs): {len(problem)}개 포인트\n')

# Merge with echo data for R²
problem_merged = pd.merge(
    problem,
    echo[['tau_c', 'R2_echo', 'T2_echo_lower', 'T2_echo_upper']],
    on='tau_c',
    how='left'
)

print('상세 분석:')
print('-'*80)
for idx, row in problem_merged.iterrows():
    tau_c_us = row['tau_c'] * 1e6
    gain_val = row['echo_gain']
    t2_fid = row['T2'] * 1e6
    t2_echo = row['T2_echo'] * 1e6
    xi = row['xi']
    r2_echo = row['R2_echo'] if pd.notna(row['R2_echo']) else np.nan
    
    # CI 확인
    t2_echo_lower = row['T2_echo_lower'] * 1e6 if pd.notna(row['T2_echo_lower']) else np.nan
    t2_echo_upper = row['T2_echo_upper'] * 1e6 if pd.notna(row['T2_echo_upper']) else np.nan
    
    print(f'τc={tau_c_us:.3f}μs, ξ={xi:.3f}:')
    print(f'  T2_FID={t2_fid:.4f}μs, T2_echo={t2_echo:.4f}μs')
    print(f'  gain={gain_val:.4f}, R²_echo={r2_echo:.4f}')
    
    if pd.notna(t2_echo_lower) and pd.notna(t2_echo_upper):
        ci_width = (t2_echo_upper - t2_echo_lower) / t2_echo * 100
        print(f'  CI: [{t2_echo_lower:.4f}, {t2_echo_upper:.4f}] μs (width={ci_width:.2f}%)')
    
    # 급격한 변화 확인
    if idx > problem_merged.index[0]:
        prev_idx = problem_merged.index[list(problem_merged.index).index(idx) - 1]
        prev_row = problem_merged.loc[prev_idx]
        gain_diff = gain_val - prev_row['echo_gain']
        tau_c_diff = (row['tau_c'] - prev_row['tau_c']) * 1e6
        
        if abs(gain_diff) > 0.5:
            print(f'  ⚠️  급격한 변화:')
            print(f'     τc: {prev_row["tau_c"]*1e6:.3f} → {tau_c_us:.3f} μs (diff={tau_c_diff:.3f}μs)')
            print(f'     gain: {prev_row["echo_gain"]:.4f} → {gain_val:.4f} (diff={gain_diff:+.4f})')
            print(f'     R² 변화: {prev_row["R2_echo"]:.4f} → {r2_echo:.4f}')
    
    print()

# 통계 요약
print('='*80)
print('통계 요약:')
print('-'*80)
print(f'평균 gain: {problem["echo_gain"].mean():.4f}')
print(f'gain 표준편차: {problem["echo_gain"].std():.4f}')
print(f'gain 범위: [{problem["echo_gain"].min():.4f}, {problem["echo_gain"].max():.4f}]')

# R² 확인
r2_valid = problem_merged['R2_echo'].dropna()
if len(r2_valid) > 0:
    print(f'\nR²_echo 통계:')
    print(f'  평균: {r2_valid.mean():.4f}')
    print(f'  최소: {r2_valid.min():.4f}')
    print(f'  R² < 0.9인 포인트: {(r2_valid < 0.9).sum()}개')
    print(f'  R² < 0.95인 포인트: {(r2_valid < 0.95).sum()}개')

# 추천 사항
print('\n' + '='*80)
print('추천 사항:')
print('-'*80)

# R²가 낮은 포인트 확인
low_r2 = problem_merged[problem_merged['R2_echo'] < 0.95]
if len(low_r2) > 0:
    print(f'⚠️  R² < 0.95인 포인트 {len(low_r2)}개 발견:')
    for idx, row in low_r2.iterrows():
        print(f'  τc={row["tau_c"]*1e6:.3f}μs: R²={row["R2_echo"]:.4f}')
    print('  → 이 포인트들을 재시뮬레이션 권장')

# 급격한 변화 확인
gain_diff = problem_merged['echo_gain'].diff().abs()
large_changes = gain_diff > 0.5
if large_changes.sum() > 0:
    print(f'\n⚠️  급격한 gain 변화 (|diff| > 0.5) {large_changes.sum()}개 발견:')
    for idx in problem_merged[large_changes].index:
        row = problem_merged.loc[idx]
        prev_idx = problem_merged.index[list(problem_merged.index).index(idx) - 1]
        prev_row = problem_merged.loc[prev_idx]
        print(f'  τc={prev_row["tau_c"]*1e6:.3f} → {row["tau_c"]*1e6:.3f}μs: '
              f'gain={prev_row["echo_gain"]:.4f} → {row["echo_gain"]:.4f}')
    print('  → 이 구간의 echo 시뮬레이션 재확인 권장')

print('='*80)

