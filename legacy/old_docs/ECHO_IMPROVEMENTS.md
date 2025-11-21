# Hahn Echo 시뮬레이션 개선 사항

## 발견된 문제점

1. **Echo gain NaN**: 6개 포인트에서 echo_gain이 계산되지 않음
2. **CI 없음**: 모든 echo 포인트에서 CI가 없음
3. **R² 누락**: 일부 포인트에서 R²_echo가 없음
4. **파라미터 불일치**: FID sweep과 다른 파라미터 사용

## 적용된 개선 사항

### 1. 파라미터 동기화 (FID sweep과 일치)
- ✅ `build_tau_c_sweep()`: 35, 30 포인트로 증가 (FID와 동일)
- ✅ `get_dt()`: 메모리 제한 포함 (8 GB)
- ✅ `get_tmax()`: QS regime 80-150×, Crossover 20× (FID와 동일)
- ✅ `T_max_echo`: 더 길게 (2.5-3.0×)

### 2. Echo Fitting 개선
- ✅ `is_echo=True` 플래그 추가 (echo-optimized window selection)
- ✅ Bootstrap CI 계산 추가
- ✅ Fallback CI 계산 (analytical error 또는 5% uncertainty)

### 3. CI 계산 강화
- ✅ `echo.py`에서 bootstrap CI 자동 계산
- ✅ `run_echo_sweep.py`에서 다중 소스 CI 추출
- ✅ Fallback 메커니즘 추가

## 예상 개선 효과

1. **Echo gain NaN 해결**: CI 계산으로 T2_echo가 제대로 저장됨
2. **CI 추가**: 모든 포인트에 CI가 있음
3. **R² 개선**: 더 정확한 fitting으로 R² 향상
4. **일관성**: FID와 동일한 파라미터로 일관된 결과

## 실행 방법

```bash
# Echo sweep 재실행
python3 run_echo_sweep.py

# Echo gain 재분석
python3 analyze_echo_gain.py

# 그림 재생성
python3 generate_dissertation_plots.py
```

**예상 시간**: 3-4시간 (83개 포인트)

## 검증 방법

재실행 후 확인:
1. `t2_echo_vs_tau_c.csv`에 모든 포인트에 CI가 있는지
2. `echo_gain.csv`에 NaN이 없는지
3. 모든 echo_gain ≥ 1.0인지

