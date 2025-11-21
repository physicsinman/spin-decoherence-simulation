# 시뮬레이션 성능 개선 사항

## 문제점 분석

### 1. QS Regime 문제
- **RMS 편차**: 10.86% (목표: <5%)
- **최대 편차**: 29.50%
- **원인**: T_max가 충분하지 않아 완전한 Gaussian decay를 포착하지 못함

### 2. Crossover Regime 문제
- **Slope**: -0.7430 (이론 기대값: -0.5 ~ -0.6)
- **원인**: 포인트 밀도 부족 및 T_max 부족

## 개선 사항

### 1. QS Regime T_max 증가 ✅
```python
# 기존
return 30 * T2_est

# 개선
multiplier = 50 if xi < 10 else 100  # xi에 따라 조정
T_max_from_T2 = multiplier * T2_est
burnin_time = 5.0 * tau_c
T_max_final = max(T_max_from_T2, burnin_time)
return min(T_max_final, 100e-3)  # 메모리 제한
```

**효과**:
- 완전한 Gaussian decay 포착
- 더 정확한 T2 fitting
- R² 값 개선 예상

### 2. Crossover Regime 개선 ✅
- **포인트 증가**: 24 → 30개
- **T_max 증가**: 10× → 15× T2
- **Timestep 정밀도**: 100 → 150 steps/tau_c

**효과**:
- Slope 측정 정확도 향상
- 이론값(-0.5~-0.6)에 더 가까워질 것으로 예상

### 3. QS Regime 포인트 증가 ✅
- **포인트 증가**: 20 → 25개
- 더 넓은 범위에서 T2 상수성 검증

### 4. 전체 포인트 수 증가
- **기존**: 62개
- **개선**: 73개 (+11개)

## 예상 개선 효과

### QS Regime
- **RMS 편차**: 10.86% → **<5%** (목표)
- **최대 편차**: 29.50% → **<10%** (목표)
- **R² < 0.7 포인트**: 감소 예상

### Crossover Regime
- **Slope**: -0.7430 → **-0.55 ~ -0.65** (목표)
- **R²**: 0.9960 → 유지 또는 개선

### MN Regime
- **현재 상태 유지** (이미 양호: RMS 1.95%)

## 실행 방법

### 전체 재시뮬레이션
```bash
python3 run_fid_sweep.py
```

**예상 시간**: 2-3시간 (기존 1-2시간에서 증가)

### 특정 regime만 재시뮬레이션
개선된 스크립트를 사용하여 특정 regime만 선택적으로 실행 가능:
```bash
python3 improve_simulation_performance.py
```

## 검증 방법

개선 후 다음 스크립트로 재검증:
```bash
python3 validate_theory_agreement.py
```

**기대 결과**:
- QS regime RMS 편차 < 5%
- Crossover slope: -0.55 ~ -0.65
- 전체적으로 이론과의 일치도 향상

## 주의사항

1. **메모리 사용량**: T_max 증가로 메모리 사용량 증가
   - QS regime에서 최대 10GB까지 사용 가능
   - 메모리 부족 시 N_traj 감소 고려

2. **실행 시간**: 전체 시뮬레이션 시간 증가
   - 기존: 1-2시간
   - 개선: 2-3시간

3. **디스크 공간**: 결과 파일 크기 증가
   - 기존보다 약 20% 증가 예상

## 다음 단계

1. ✅ 개선된 파라미터 적용 완료
2. ⏳ 전체 시뮬레이션 재실행
3. ⏳ 결과 검증 및 비교
4. ⏳ 필요시 추가 조정

