# 물리학적 타당성 문제

## 발견된 문제

### 1. Crossover Regime에서 급격한 감소
- **위치**: τc = 0.257 → 0.300 μs (ξ = 2.264 → 2.640)
- **증상**: gain = 2.761 → 1.507 (diff = -1.254)
- **문제**: 물리학적으로 불가능
  - Echo gain은 ξ가 증가할수록 증가해야 함
  - Crossover regime에서 급격한 감소는 fitting error를 나타냄

### 2. 이론적 예측 vs 측정값

**이론적 기대**:
- **MN regime (ξ < 0.2)**: Echo ≈ FID, gain ≈ 1.0 ✅
- **Crossover (0.2 < ξ < 3)**: Echo > FID, gain은 점진적으로 증가해야 함 ❌
- **QS regime (ξ ≥ 3)**: Echo >> FID, gain >> 1 ✅

**측정값**:
- MN regime: gain ≈ 1.0 ✅
- Crossover: 평균 gain = 1.441, 하지만 급격한 감소 발생 ❌
- QS regime: 평균 gain = 3.591 ✅

## 원인 분석

### 가능한 원인들:

1. **Echo Fitting 실패**
   - R²_echo가 낮은 포인트에서 발생
   - T2_echo 측정이 부정확

2. **T_max_echo 부족**
   - Echo decay를 충분히 관측하지 못함
   - Fitting window가 부적절

3. **Bootstrap CI 문제**
   - CI가 너무 넓거나 좁음
   - Bootstrap resampling이 부정확

## 해결 방안

### 즉시 조치:
1. **문제 구간 재시뮬레이션**
   - τc = 0.257, 0.300 μs 재실행
   - T_max_echo 증가
   - 더 많은 N_traj 사용

2. **Echo Fitting 개선**
   - Window selection 개선
   - 더 robust한 fitting 방법

3. **데이터 필터링**
   - R²_echo < 0.9인 포인트 제외
   - CI width가 너무 큰 포인트 제외

### 장기 개선:
1. **T_max_echo 추가 증가**
   - Crossover regime에서 더 긴 시간 필요
   - 현재 2.5× → 3.5×로 증가

2. **Echo fitting window 개선**
   - Regime-aware window selection
   - 더 보수적인 threshold

## 결론

**현재 결과는 부분적으로만 물리학적으로 타당합니다:**
- ✅ MN regime: 정상
- ❌ Crossover: 문제 있음 (급격한 감소)
- ✅ QS regime: 정상

**논문에 포함하기 전에:**
- 문제 구간 재시뮬레이션 필수
- Echo fitting 개선 필요
- 물리학적으로 타당하지 않은 포인트 제외

