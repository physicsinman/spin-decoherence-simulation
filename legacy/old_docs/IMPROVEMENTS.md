# 🔧 시뮬레이션 결과 개선 방안

## 📊 현재 문제점 요약

1. **Echo Fit 품질 낮음**: 평균 R² = 0.58 (FID는 0.99)
2. **Echo Gain ≤ 1**: 5개 포인트에서 물리적으로 비정상
3. **T2 Saturation**: 9개 포인트에서 동일한 T2 값 (0.161 μs)
4. **MN Regime 포인트 부족**: 3개만 있어서 slope 측정 정확도 낮음
5. **Echo Fit 실패**: 5개 포인트에서 R² < 0.50

---

## ✅ 개선 방안

### **1. Echo T_max 증가**

**문제**: Echo decay가 충분히 관측되지 않음

**해결책**:
```python
# run_echo_sweep.py, run_echo_curves.py에서
T_max_echo = T_max * 1.5  # 또는 2.0 (현재는 T_max와 동일)
```

**예상 효과**:
- Echo fit 품질 향상 (R² 증가)
- Echo gain ≤ 1 문제 해결

---

### **2. QS Regime T_max 증가**

**문제**: QS regime에서 T2 saturation 발생

**해결책**:
```python
# get_tmax 함수 수정
if xi > 3:  # QS regime
    T2_est = 1.0 / (gamma_e * B_rms)
    return 15 * T2_est  # 또는 20 * T2_est (현재는 10 * T2_est)
```

**예상 효과**:
- T2 saturation 문제 해결
- QS regime에서 더 정확한 T2 측정

---

### **3. MN Regime 포인트 증가**

**문제**: MN regime 포인트가 3개만 있음

**해결책 A**: tau_c_min 감소
```python
tau_c_min = 5e-9  # 현재 1e-8에서 감소
```

**해결책 B**: tau_c 범위 내에서 더 많은 포인트
```python
tau_c_npoints = 30  # 현재 20에서 증가
```

**예상 효과**:
- MN regime slope 측정 정확도 향상
- 더 많은 포인트로 fit 신뢰도 증가

---

### **4. Echo Fit 방법 개선**

**문제**: 일부 포인트에서 fit 실패 (R² < 0.50)

**해결책**:
- Fit window 조정
- 더 robust한 fit 방법 사용
- Outlier 제거

---

## 📋 우선순위별 개선 계획

### **우선순위 1 (즉시 적용 권장)**
1. ✅ T_max_echo 증가 (1.5-2배)
2. ✅ QS regime T_max 증가 (15-20×T2)

**예상 효과**: Echo fit 품질 크게 향상, Echo gain 문제 해결

**예상 시간 증가**: ~20-30%

---

### **우선순위 2 (중요하지만 선택적)**
3. ✅ MN regime 포인트 증가

**예상 효과**: Slope 측정 정확도 향상

**예상 시간 증가**: ~10-20%

---

### **우선순위 3 (선택적)**
4. ✅ Echo fit 방법 개선

**예상 효과**: 일부 포인트 fit 품질 향상

---

## 🔧 코드 수정 예시

### **run_echo_sweep.py 수정**

```python
def get_tmax(tau_c, B_rms, gamma_e):
    """Calculate appropriate simulation duration"""
    xi = gamma_e * B_rms * tau_c
    
    if xi < 0.3:  # MN regime
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est
    elif xi > 3:  # QS regime
        T2_est = 1.0 / (gamma_e * B_rms)
        return 20 * T2_est  # 증가: 10 → 20
    else:  # Crossover
        T2_est = 1.0 / (gamma_e**2 * B_rms**2 * tau_c)
        return 10 * T2_est

# params 설정 부분
params = {
    ...
    'T_max': T_max,
    'T_max_echo': T_max * 1.5,  # 증가: T_max → T_max * 1.5
    ...
}
```

### **run_fid_sweep.py 수정 (MN regime 포인트 증가)**

```python
# Option 1: tau_c_min 감소
tau_c_min = 5e-9  # 1e-8 → 5e-9

# Option 2: 포인트 수 증가
tau_c_npoints = 30  # 20 → 30
```

---

## 📊 예상 개선 효과

| 개선 사항 | 현재 | 개선 후 | 예상 효과 |
|----------|------|---------|----------|
| Echo 평균 R² | 0.58 | 0.75-0.85 | ⬆️ 30-50% |
| Echo gain ≤ 1 | 5개 | 0-1개 | ⬇️ 80-100% |
| T2 saturation | 9개 | 0-2개 | ⬇️ 80-100% |
| MN slope 정확도 | ±0.034 | ±0.020 | ⬆️ 40% |

---

## ⚠️ 주의사항

1. **시뮬레이션 시간 증가**
   - T_max 증가 → 시간 증가
   - 예상: ~20-40% 시간 증가

2. **메모리 사용량 증가**
   - 더 긴 시뮬레이션 → 더 많은 메모리
   - 필요시 use_online=True 사용

3. **단계적 적용 권장**
   - 먼저 우선순위 1만 적용
   - 결과 확인 후 우선순위 2 적용

---

## 🚀 실행 방법

개선된 코드로 재시뮬레이션:

```bash
python3 run_all_simulations.py
```

또는 개별 실행:

```bash
python3 run_echo_sweep.py      # Echo 개선
python3 run_echo_curves.py     # Echo curves 개선
python3 run_fid_sweep.py       # MN regime 포인트 증가
```

---

**마지막 업데이트**: 2025-11-19

