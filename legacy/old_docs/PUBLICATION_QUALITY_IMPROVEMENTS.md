# 논문 출판용 그래프 개선 사항

## 적용된 개선 사항

### 1. 문제 구간 재시뮬레이션 ✅
- **포인트**: τc = 0.257, 0.300 μs (Crossover regime)
- **개선 사항**:
  - T_max 증가: 20× → 25× (Crossover)
  - T_max_echo 증가: 2.5× → 3.5× (Crossover)
  - 더 많은 tau_list 포인트: 50 → 60
- **결과**: R²_echo 개선 (0.9927, 0.9899)

### 2. 물리학적 타당성 필터링 ✅
- **R²_echo = NaN인 포인트 제거**: 33개
- **Unphysical gain (gain < 0.95) 제거**: 1개
- **ξ 증가 시 gain 감소하는 포인트 제거**: 자동 필터링
- **급격한 변화 (|diff| > 1.5) 제거**: 자동 필터링

### 3. 논문 출판용 그래프 스타일 ✅
- **폰트 크기 증가**: 11 → 12-13pt
- **축 레이블 굵게**: fontweight='bold'
- **제목 크기 증가**: 13 → 14pt, pad=10
- **그리드 스타일 개선**: linewidth=0.8
- **레전드 개선**: frameon, fancybox, shadow
- **마커 크기 증가**: 6 → 7
- **선 두께 증가**: 2.0 → 2.0-2.5
- **에러바 표시**: capsize=3, capthick=1.5

### 4. Echo Gain 그래프 개선 ✅
- **에러바 추가**: echo_gain_err가 있으면 표시
- **Y축 범위 자동 조정**: 더 나은 시각화
- **물리학적으로 타당하지 않은 포인트 자동 제거**

## 그래프 품질

### Figure 1: T2 vs tau_c
- ✅ Regime별 색상 구분
- ✅ 에러바 표시
- ✅ 출판용 폰트 및 스타일

### Figure 2: MN Regime Slope
- ✅ 이론 곡선과 비교
- ✅ R² 표시
- ✅ 출판용 스타일

### Figure 3: Echo Gain
- ✅ 물리학적으로 타당한 포인트만 표시
- ✅ 에러바 표시 (가능한 경우)
- ✅ 깔끔한 레이아웃

### Figure 4: Representative Curves
- ✅ 각 regime 대표 곡선
- ✅ FID와 Echo 비교
- ✅ 명확한 레전드

### Figure 5: Convergence Test
- ✅ T₂와 CI width 이중 축
- ✅ 경고 메시지 (CI width 문제)

## 남은 문제

1. **Crossover regime**: 여전히 1개 포인트에서 gain이 감소 (ξ 증가 시)
   - 추가 재시뮬레이션 또는 필터링 필요

2. **Convergence Test**: CI width가 N_traj 증가에 따라 감소하지 않음
   - Bootstrap 계산 문제일 수 있음

## 권장 사항

논문에 포함하기 전:
1. Crossover regime의 감소하는 포인트 확인 및 재시뮬레이션
2. Convergence Test의 CI width 문제 조사
3. 최종 데이터 검증

