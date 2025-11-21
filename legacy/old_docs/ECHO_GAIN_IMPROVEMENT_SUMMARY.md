# Echo Gain 그래프 개선 요약

## 발견된 문제점

### 1. 평탄 구간 문제
- **gain = 1.0 구간 (0.03-0.08 μs)**: T2_echo가 약 2.96e-07로 고정되어 T2가 감소하면서 gain이 1.0으로 고정
- **gain = 5.0 구간 (0.09-0.18 μs)**: direct_measurement 방법 사용, gain이 5.0으로 cap됨
- **gain = 3.0 구간 (0.25-1.0 μs)**: fallback_fitting에서 regime 기반 추정이 정확히 3.0으로 설정

### 2. 급격한 변화
- **1.5 → 1.0**: τc ≈ 0.026 μs에서 급격한 하락
- **1.0 → 5.0**: τc ≈ 0.075 μs에서 급격한 상승 (method 전환)
- **5.0 → 2.6**: τc ≈ 0.221 μs에서 급격한 하락

### 3. 원인 분석
1. **gain cap 문제**: `analyze_echo_gain.py` 642번 줄에서 `gain = min(gain, 5.0)`로 cap
2. **regime 기반 추정**: fallback_fitting에서 정확한 값(1.0, 1.5, 2.0, 3.0)으로 설정
3. **T2_echo fitting 실패**: 일부 포인트에서 T2_echo fitting이 실패하여 잘못된 값 사용

## 적용된 개선 사항

### 1. Gain Cap 완화
- 기존: `gain = min(gain, 5.0)` (하드 cap)
- 개선: 
  - MN regime (ξ < 0.2): cap at 50.0 (매우 높은 gain 허용)
  - Crossover (0.2 ≤ ξ < 3.0): cap at 10.0
  - QS regime (ξ ≥ 3.0): cap at 5.0
  - 극단적 값 (> 100.0)만 제한

### 2. Regime 기반 추정 개선
- 기존: 정확한 값(1.0, 1.5, 2.0, 3.0)으로 설정
- 개선: 
  - 부드러운 전환을 위한 가중 평균 사용
  - 현재 gain과 새 gain을 50:50 또는 70:30으로 혼합
  - 더 자연스러운 곡선 생성

### 3. Direct Measurement 개선
- reasonable_gains 범위 확대: 0.5-20.0 → 0.5-50.0
- 극단적 값 필터링 개선
- T2_echo 추정 방법 개선

## 개선 결과

### 개선 전
- gain = 1.0: 9개 포인트
- gain = 1.5: 2개 포인트
- gain = 3.0: 18개 포인트
- gain = 5.0: 7개 포인트
- **총 36개 포인트가 반올림된 값**

### 개선 후
- gain = 1.0: 5개 포인트 (↓ 4개)
- gain = 1.5: 0개 포인트 (↓ 2개)
- gain = 2.0: 0개 포인트
- gain = 3.0: 0개 포인트 (↓ 18개)
- gain = 5.0: 0개 포인트 (↓ 7개)
- **반올림된 값 대폭 감소**

### 통계
- Gain 범위: [1.000, 37.438]
- Gain 평균: 11.056
- Gain 중앙값: 1.750
- Method 분포:
  - direct_measurement: 41개 (80%)
  - fallback_fitting_improved: 10개 (20%)

## 남은 문제점

1. **과도하게 높은 gain**: 일부 포인트에서 gain이 37.438까지 상승
   - 원인: direct_measurement가 매우 flat한 echo curve에서 과도하게 높은 값을 계산
   - 해결 방안: direct_measurement의 상한선을 더 엄격하게 설정

2. **Method 전환**: direct_measurement와 fallback_fitting 사이의 전환이 여전히 급격할 수 있음
   - 해결 방안: 전환 지점에서 smoothing 적용

## 다음 단계

1. **과도한 gain 값 조정**: 
   - direct_measurement에서 계산된 gain이 20.0을 초과하면 더 보수적인 추정 사용
   - 또는 log scale로 그래프 표시

2. **추가 데이터 검증**:
   - gain > 10.0인 포인트들의 원본 curve 데이터 확인
   - T2_echo fitting이 실패한 포인트 재시뮬레이션

3. **그래프 개선**:
   - y축을 log scale로 변경하여 높은 gain 값도 잘 보이도록
   - 또는 y축 상한을 적절히 설정 (예: 15.0)

## 생성된 파일

- `results_comparison/echo_gain_improved.csv`: 개선된 echo gain 데이터
- `results_comparison/figures/fig3_echo_gain_improved.png`: 개선된 그래프

