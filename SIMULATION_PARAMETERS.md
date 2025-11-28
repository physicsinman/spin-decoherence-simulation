# 시뮬레이션 파라미터 정리

## 1. 물리 상수 (Si:P)

- **γ_e** (electron gyromagnetic ratio): `1.76 × 10¹¹` rad/(s·T)
- **B_rms** (RMS magnetic field): `0.57 μT` (5.7 × 10⁻⁷ T) - Physical value for 800 ppm ²⁹Si concentration
- **Δω** = γ_e × B_rms: `1.00 × 10⁵` rad/s

## 2. tau_c 범위 및 그리드

- **tau_c_min**: `3 × 10⁻⁹` s (0.003 μs)
- **tau_c_max**: `1 × 10⁻³` s (1 ms)
- **총 포인트 수**: 73개

### 그리드 구성:
- **MN regime** (3 × 10⁻⁹ ~ 3 × 10⁻⁸ s): 18 points
- **Crossover** (3 × 10⁻⁸ ~ 3 × 10⁻⁶ s): 35 points
- **QS regime** (3 × 10⁻⁶ ~ 1 × 10⁻³ s): 30 points

## 3. 시뮬레이션 파라미터

### FID 시뮬레이션
- **M** (trajectories): `2000`
- **메모리 제한**: `8 GB`

### Echo 시뮬레이션
- **M_echo** (trajectories): `10,000` (FID의 5배)
- **dt_echo**: `dt / 2` (더 정밀한 phase alignment)

## 4. 적응형 파라미터 (Regime별)

### FID 시뮬레이션

#### MN regime (ξ < 0.3)
- **dt**: `tau_c / 100` (최소: `tau_c / 50`)
- **T_max**: `10 × T2_est`
  - T2_est = 1 / (γ_e² × B_rms² × tau_c)

#### Crossover (0.3 ≤ ξ < 3)
- **dt**: `tau_c / 100` (최소: `tau_c / 50`)
- **T_max**: `20 × T2_est`
  - T2_est = 1 / (γ_e² × B_rms² × tau_c)

#### QS regime (ξ > 3)
- **dt**: `tau_c / 100` (최소: `tau_c / 50`)
- **T_max**: `100-200 × T2_est` (ξ에 따라)
  - ξ < 10: `100×`
  - 10 ≤ ξ < 50: `150×`
  - ξ ≥ 50: `200×`
  - T2_est = 1 / (γ_e × B_rms)
- **최대 T_max**: `100 ms`
- **Burn-in time**: `5 × tau_c` (최소값으로 고려)

### Echo 시뮬레이션

#### MN regime (ξ < 0.3)
- **T_max_echo**: `max(3×T_FID, 100×T2_echo, 30×T_max)`
- **최대**: `500 ms`

#### Crossover (0.3 ≤ ξ < 3)
- **T_max_echo**: `8 × T_max`
- **최대**: `300 ms`

#### QS regime (ξ > 3)
- **ξ > 50**: `T_max_echo = 10 × T_max` (최대: `1000 ms`)
- **20 < ξ ≤ 50**: `T_max_echo = 8 × T_max` (최대: `800 ms`)
- **3 < ξ ≤ 20**: `T_max_echo = 5 × T_max` (최대: `500 ms`)

## 5. Echo tau_list 파라미터

- **upsilon_min**: `0.05`
- **upsilon_max**: Regime별 적응형
  - MN: 최대 `15.0`
  - Crossover: 최대 `10.0`
  - QS_shallow (3 < ξ < 20): 최대 `20.0`
  - QS_intermediate (20 ≤ ξ < 50): 최대 `30.0`
  - QS_deep (ξ ≥ 50): 최대 `50.0`
- **n_points_echo**: Regime별
  - QS_deep (ξ > 50): `150 points`
  - QS_intermediate (20 < ξ ≤ 50): `120 points`
  - 그 외: `100 points`

## 6. Fitting 파라미터

### FID
- **Model**: `auto` (gaussian/exponential/stretched 자동 선택)
- **Window selection**: Regime-aware
- **Min points**: `20`

### Echo
- **Model**: `auto` (gaussian/exponential/stretched 자동 선택)
- **Window selection**: Echo-optimized (더 보수적)
- **Min points**: `10` (QS regime: `15`)
- **Threshold (eps)**: Regime별
  - MN (ξ < 0.5): `0.05`
  - Crossover (0.5 ≤ ξ < 5): `0.04`
  - QS_shallow (5 ≤ ξ < 20): `0.03`
  - QS_intermediate (20 ≤ ξ < 50): `0.02`
  - QS_deep (ξ ≥ 50): `0.015`

## 7. Analytical Fallback (QS regime)

- **조건**: QS regime (ξ > 20)에서 fitting 실패 시
- **방법**: `analytical_T2_echo()` 사용
- **계산**: 
  - QS regime: `T2_echo = 2 × T2_fid` (echo gain = 2)
  - T2_fid = √2 / Δω (QS regime)
- **불확실도**: `10%` (T2_echo × 0.9 ~ T2_echo × 1.1)

## 8. Bootstrap

- **Bootstrap samples**: `500`
- **CI 계산**: `95% confidence interval`

## 9. Regime 분류

- **MN (Motional Narrowing)**: ξ < 0.3
- **Crossover**: 0.3 ≤ ξ < 3
- **QS (Quasi-Static)**: ξ ≥ 3
  - QS_shallow: 3 ≤ ξ < 20
  - QS_intermediate: 20 ≤ ξ < 50
  - QS_deep: ξ ≥ 50

여기서 **ξ = Δω × tau_c** = γ_e × B_rms × tau_c

