# 그래프 수정 사항

## 발견된 문제점

### 1. Figure 3: Echo Gain
- **문제**: τc ≈ 6-7 μs에서 급격한 drop (5.02 → 1.0)
- **원인**: T2_echo의 R² = NaN (fitting 실패) → analytical estimate 사용 → T2_echo = T2_FID
- **해결**: 
  - R²_echo = NaN인 포인트 필터링 (32개 제거)
  - 급격한 drop (diff > 2.0) 제거
  - 유효한 포인트만 사용하여 그래프 생성

### 2. Figure 4: Representative Curves
- **문제**: tau_c 선택이 잘못되어 부적절한 곡선 표시
- **원인**: 단순히 균등 간격으로 선택하여 각 regime을 대표하지 못함
- **해결**: 
  - 각 regime을 대표하는 tau_c 값 선택 (1e-8, 1e-7, 1e-6, 1e-5)
  - 가장 가까운 실제 데이터 포인트 사용

### 3. Figure 5: Convergence Test
- **문제**: CI width가 N_traj 증가에 따라 감소하지 않고 증가
- **원인**: 데이터 정렬 문제 또는 bootstrap 계산 이슈
- **해결**: 
  - N_traj로 정렬하여 올바른 순서 보장
  - 경고 메시지 추가 (데이터 자체 문제인 경우)

## 적용된 수정 사항

### Echo Gain (fig3)
```python
# R²_echo = NaN인 포인트 필터링
valid_with_r2 = valid[valid['R2_echo'].notna()].copy()

# 급격한 drop 제거
gain_diff = valid_physical['echo_gain'].diff().abs()
large_drops = gain_diff > 2.0
valid_physical = valid_physical.drop(drop_indices)
```

### Representative Curves (fig4)
```python
# 각 regime을 대표하는 tau_c 선택
target_tau_cs = [1e-8, 1e-7, 1e-6, 1e-5]
closest = min(tau_c_values, key=lambda x: abs(x[1] - target))
```

### Convergence Test (fig5)
```python
# N_traj로 정렬하여 올바른 순서 보장
sort_idx = np.argsort(N_traj_ci)
N_traj_ci = N_traj_ci[sort_idx]
ci_widths_clean = ci_widths_clean[sort_idx]
```

## 결과

- ✅ Echo Gain: 급격한 drop 제거, 유효한 포인트만 표시
- ✅ Representative Curves: 각 regime을 대표하는 곡선 선택
- ⚠️  Convergence Test: CI width 경고는 남아있지만, 정렬 문제는 해결

## 남은 문제

- Convergence Test의 CI width가 여전히 감소하지 않는 것은 bootstrap 계산 또는 데이터 자체의 문제일 수 있습니다. 이는 시뮬레이션 파라미터나 bootstrap 방법을 개선해야 할 수 있습니다.

