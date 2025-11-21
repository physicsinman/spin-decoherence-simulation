#!/usr/bin/env python3
"""
냉정한 이론 일치도 검증
실제 측정값과 이론값을 직접 비교하여 정확히 평가
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# 실제 시뮬레이션에서 사용한 파라미터 확인
# run_fid_sweep.py에서 사용한 값
gamma_e = 1.76e11  # rad/(s·T)
B_rms = 0.05e-3    # Tesla (50 μT, Si:P 파라미터로 보임)
Delta_omega = gamma_e * B_rms  # 8.8e6 rad/s

def theoretical_T2_MN(tau_c):
    """Motional Narrowing 이론값: T2 = 1/(Delta_omega^2 * tau_c)"""
    return 1.0 / (Delta_omega**2 * tau_c)

def theoretical_T2_QS(tau_c):
    """Quasi-Static 이론값: T2 ≈ sqrt(2)/Delta_omega (tau_c와 무관)"""
    return np.sqrt(2.0) / Delta_omega

def main():
    print("="*80)
    print("냉정한 이론 일치도 검증")
    print("="*80)
    
    # 데이터 로드
    df = pd.read_csv("results_comparison/t2_vs_tau_c.csv")
    print(f"\n총 데이터 포인트: {len(df)}")
    
    # 유효 데이터 필터링
    valid_mask = df['T2'].notna() & (df['T2'] > 0) & (df['R2'] > 0.9)
    df_valid = df[valid_mask].copy()
    print(f"유효 데이터 (R² > 0.9): {len(df_valid)}")
    
    # 이론값 계산
    df_valid['T2_theory_MN'] = theoretical_T2_MN(df_valid['tau_c'].values)
    df_valid['T2_theory_QS'] = theoretical_T2_QS(df_valid['tau_c'].values)
    
    # Regime별 분석
    print("\n" + "="*80)
    print("1. MOTIONAL NARROWING REGIME (ξ < 0.2)")
    print("="*80)
    
    mn_mask = df_valid['xi'] < 0.2
    df_mn = df_valid[mn_mask].copy()
    
    if len(df_mn) > 0:
        # 이론값과 비교
        df_mn['ratio'] = df_mn['T2'] / df_mn['T2_theory_MN']
        df_mn['deviation_pct'] = (df_mn['T2'] - df_mn['T2_theory_MN']) / df_mn['T2_theory_MN'] * 100
        
        print(f"\nMN regime 포인트: {len(df_mn)}")
        print(f"τc 범위: {df_mn['tau_c'].min():.2e} ~ {df_mn['tau_c'].max():.2e} s")
        print(f"ξ 범위: {df_mn['xi'].min():.3f} ~ {df_mn['xi'].max():.3f}")
        
        print("\n측정값 vs 이론값 비교:")
        print(f"  평균 비율 (T2_measured/T2_theory): {df_mn['ratio'].mean():.4f}")
        print(f"  표준편차: {df_mn['ratio'].std():.4f}")
        print(f"  최소 비율: {df_mn['ratio'].min():.4f}")
        print(f"  최대 비율: {df_mn['ratio'].max():.4f}")
        
        print(f"\n  평균 편차: {df_mn['deviation_pct'].mean():.2f}%")
        print(f"  RMS 편차: {np.sqrt((df_mn['deviation_pct']**2).mean()):.2f}%")
        print(f"  최대 편차: {df_mn['deviation_pct'].abs().max():.2f}%")
        
        # Slope 검증
        log_tau_c = np.log10(df_mn['tau_c'].values)
        log_T2_measured = np.log10(df_mn['T2'].values)
        log_T2_theory = np.log10(df_mn['T2_theory_MN'].values)
        
        slope_measured, _, r_measured, _, _ = stats.linregress(log_tau_c, log_T2_measured)
        slope_theory, _, r_theory, _, _ = stats.linregress(log_tau_c, log_T2_theory)
        
        print(f"\nSlope 비교:")
        print(f"  측정된 slope: {slope_measured:.4f} (R² = {r_measured**2:.4f})")
        print(f"  이론 slope: {slope_theory:.4f} (R² = {r_theory**2:.4f})")
        print(f"  이론 기대값: -1.0000")
        print(f"  Slope 편차: {abs(slope_measured - (-1.0)) / 1.0 * 100:.2f}%")
        
        # 개별 포인트 상세
        print("\n개별 포인트 상세:")
        for idx, row in df_mn.iterrows():
            print(f"  τc={row['tau_c']*1e9:.2f}ns, ξ={row['xi']:.3f}: "
                  f"T2_meas={row['T2']*1e6:.3f}μs, T2_theory={row['T2_theory_MN']*1e6:.3f}μs, "
                  f"비율={row['ratio']:.4f}, 편차={row['deviation_pct']:.2f}%")
    else:
        print("❌ MN regime 데이터 없음!")
    
    print("\n" + "="*80)
    print("2. QUASI-STATIC REGIME (ξ > 3)")
    print("="*80)
    
    qs_mask = df_valid['xi'] > 3.0
    df_qs = df_valid[qs_mask].copy()
    
    if len(df_qs) > 0:
        T2_QS_theory = theoretical_T2_QS(df_qs['tau_c'].values[0])  # 상수값
        
        df_qs['ratio'] = df_qs['T2'] / T2_QS_theory
        df_qs['deviation_pct'] = (df_qs['T2'] - T2_QS_theory) / T2_QS_theory * 100
        
        print(f"\nQS regime 포인트: {len(df_qs)}")
        print(f"τc 범위: {df_qs['tau_c'].min():.2e} ~ {df_qs['tau_c'].max():.2e} s")
        print(f"ξ 범위: {df_qs['xi'].min():.3f} ~ {df_qs['xi'].max():.3f}")
        print(f"이론값 (상수): T2 = {T2_QS_theory*1e6:.3f} μs")
        
        print("\n측정값 vs 이론값 비교:")
        print(f"  평균 비율: {df_qs['ratio'].mean():.4f}")
        print(f"  표준편차: {df_qs['ratio'].std():.4f}")
        print(f"  최소 비율: {df_qs['ratio'].min():.4f}")
        print(f"  최대 비율: {df_qs['ratio'].max():.4f}")
        
        print(f"\n  평균 편차: {df_qs['deviation_pct'].mean():.2f}%")
        print(f"  RMS 편차: {np.sqrt((df_qs['deviation_pct']**2).mean()):.2f}%")
        print(f"  최대 편차: {df_qs['deviation_pct'].abs().max():.2f}%")
        
        # T2가 tau_c와 무관한지 검증 (slope ≈ 0)
        log_tau_c = np.log10(df_qs['tau_c'].values)
        log_T2 = np.log10(df_qs['T2'].values)
        slope_qs, _, r_qs, _, _ = stats.linregress(log_tau_c, log_T2)
        
        print(f"\nQS regime slope (이론적으로 ≈ 0):")
        print(f"  측정된 slope: {slope_qs:.4f} (R² = {r_qs**2:.4f})")
        print(f"  이론 기대값: 0.0000")
        print(f"  Slope 편차: {abs(slope_qs):.4f}")
        
        # R² 값들 확인
        print(f"\nR² 값 분포:")
        print(f"  평균 R²: {df_qs['R2'].mean():.4f}")
        print(f"  최소 R²: {df_qs['R2'].min():.4f}")
        print(f"  최대 R²: {df_qs['R2'].max():.4f}")
        print(f"  R² < 0.7인 포인트: {(df_qs['R2'] < 0.7).sum()}개")
        print(f"  R² < 0.5인 포인트: {(df_qs['R2'] < 0.5).sum()}개")
    else:
        print("❌ QS regime 데이터 없음!")
    
    print("\n" + "="*80)
    print("3. CROSSOVER REGIME (0.1 < ξ < 3)")
    print("="*80)
    
    crossover_mask = (df_valid['xi'] > 0.1) & (df_valid['xi'] < 3.0)
    df_crossover = df_valid[crossover_mask].copy()
    
    if len(df_crossover) > 0:
        print(f"\nCrossover regime 포인트: {len(df_crossover)}")
        print(f"τc 범위: {df_crossover['tau_c'].min():.2e} ~ {df_crossover['tau_c'].max():.2e} s")
        print(f"ξ 범위: {df_crossover['xi'].min():.3f} ~ {df_crossover['xi'].max():.3f}")
        
        # Slope 측정
        log_tau_c = np.log10(df_crossover['tau_c'].values)
        log_T2 = np.log10(df_crossover['T2'].values)
        slope_crossover, _, r_crossover, _, std_err = stats.linregress(log_tau_c, log_T2)
        
        print(f"\nCrossover regime slope:")
        print(f"  측정된 slope: {slope_crossover:.4f} ± {std_err:.4f}")
        print(f"  R²: {r_crossover**2:.4f}")
        print(f"  이론 기대값: -0.5 ~ -0.6 (중간값, 정확한 공식 없음)")
        print(f"  Literature: ≈ -0.49")
    else:
        print("❌ Crossover regime 데이터 없음!")
    
    print("\n" + "="*80)
    print("종합 평가")
    print("="*80)
    
    print("\n✅ 양호한 점:")
    if len(df_mn) > 0:
        mn_rms = np.sqrt((df_mn['deviation_pct']**2).mean())
        if mn_rms < 5.0:
            print(f"  - MN regime: RMS 편차 {mn_rms:.2f}% < 5% (양호)")
        if abs(slope_measured - (-1.0)) < 0.02:
            print(f"  - MN slope: 편차 {abs(slope_measured - (-1.0)):.4f} < 0.02 (양호)")
    
    print("\n⚠️  문제점:")
    if len(df_mn) > 0:
        if df_mn['deviation_pct'].abs().max() > 10.0:
            print(f"  - MN regime: 최대 편차 {df_mn['deviation_pct'].abs().max():.2f}% > 10%")
        if abs(slope_measured - (-1.0)) > 0.02:
            print(f"  - MN slope: 편차 {abs(slope_measured - (-1.0)):.4f} > 0.02")
    
    if len(df_qs) > 0:
        qs_rms = np.sqrt((df_qs['deviation_pct']**2).mean())
        if qs_rms > 20.0:
            print(f"  - QS regime: RMS 편차 {qs_rms:.2f}% > 20% (큰 편차)")
        if (df_qs['R2'] < 0.7).sum() > len(df_qs) * 0.3:
            print(f"  - QS regime: R² < 0.7인 포인트가 {(df_qs['R2'] < 0.7).sum()}/{len(df_qs)}개 (fitting 문제)")
        if abs(slope_qs) > 0.1:
            print(f"  - QS slope: {slope_qs:.4f} (이론적으로 ≈ 0이어야 함)")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

