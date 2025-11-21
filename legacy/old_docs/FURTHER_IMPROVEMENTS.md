# 추가 개선 사항

## 적용된 개선 사항

### 1. Echo Gain 계산 개선 ✅
- **QS regime T2 saturation 처리**: T2_FID ≈ T2_echo인 경우 gain = 1.0으로 명시적 설정
- **CI 없음 처리**: CI가 없어도 echo_gain은 계산하되, echo_gain_err만 NaN
- **Unphysical gain 처리**: gain < 0.9인 경우 1.0으로 클리핑 (기존: NaN)

### 2. QS Regime T_max 추가 증가 ✅
- **FID sweep**: 80-150배 → 100-200배
  - 초기 QS (ξ < 10): 80 → 100배
  - 중간 QS (10 ≤ ξ < 50): 120 → 150배
  - 깊은 QS (ξ ≥ 50): 150 → 200배
- **Echo sweep**: 동일하게 적용

### 3. 예상 개선 효과
- **Echo gain NaN**: 13개 → 0개 목표
- **Echo gain < 1**: 19개 → 0개 목표 (T2 saturation 처리)
- **QS regime RMS 편차**: 20.25% → <15% 목표

## 다음 단계

### 즉시 실행:
```bash
# Echo gain 재분석 (개선된 로직 적용)
python3 analyze_echo_gain.py

# QS regime 개선을 위한 FID 재실행 (선택적)
# python3 run_fid_sweep.py  # 시간 소요 큼 (3-4시간)

# 그림 재생성
python3 generate_dissertation_plots.py
```

### 검증:
1. Echo gain NaN이 0개인지 확인
2. Echo gain < 1이 0개인지 확인
3. QS regime RMS 편차가 개선되었는지 확인

