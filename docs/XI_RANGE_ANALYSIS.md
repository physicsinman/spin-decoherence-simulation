# ξ 범위 분석: 코드 vs 논문

## 🔍 코드에서 실제 사용하는 ξ 범위

### 1. **그래프/시각화** (`plot_all_figures.py`)

```python
# Line 123-125
mn_mask = xi_valid < 0.5
crossover_mask = (xi_valid >= 0.5) & (xi_valid < 2.0)
qs_mask = xi_valid >= 2.0
```

**사용 기준:**
- **MN**: ξ < **0.5**
- **Crossover**: 0.5 ≤ ξ < **2.0**
- **QS**: ξ ≥ **2.0**

**용도**: Figure 1 (T₂ vs τc) 그래프 색상 구분

---

### 2. **Fitting 모델 선택** (`spin_decoherence/analysis/fitting.py`)

```python
# Line 1288-1336
if xi < 0.15:
    # MN regime: Force exponential
elif xi >= 4.0:
    # QS regime: Force Gaussian
elif 0.15 <= xi < 4.0:
    # Crossover: Force stretched exponential
```

**사용 기준:**
- **MN**: ξ < **0.15**
- **Crossover**: 0.15 ≤ ξ < **4.0**
- **QS**: ξ ≥ **4.0**

**용도**: T₂ 추출 시 모델 자동 선택

---

### 3. **MN Slope Fitting** (`analyze_mn.py`, `check_slope.py`)

```python
# analyze_mn.py Line 40
mn_mask = df_valid['xi'] < 0.2
```

**사용 기준:**
- **MN**: ξ < **0.2**

**용도**: Motional narrowing 기울기 검증 (slope = -1)

---

### 4. **시뮬레이션 파라미터 문서** (`SIMULATION_PARAMETERS.md`)

```markdown
## 9. Regime 분류
- **MN (Motional Narrowing)**: ξ < 0.3
- **Crossover**: 0.3 ≤ ξ < 3
- **QS (Quasi-Static)**: ξ ≥ 3
```

**사용 기준:**
- **MN**: ξ < **0.3**
- **Crossover**: 0.3 ≤ ξ < **3**
- **QS**: ξ ≥ **3**

**용도**: 문서화/설명용

---

### 5. **FID 스윕** (`sim_fid_sweep.py`)

```python
# Line 110-117
if xi < 0.5:  # MN regime (그래프 기준: ξ < 0.5)
    ...
elif xi >= 2.0:  # QS regime (그래프 기준: ξ >= 2.0)
    ...
else:  # Crossover (0.5 <= xi < 2.0)
```

**사용 기준:**
- **MN**: ξ < **0.5**
- **Crossover**: 0.5 ≤ ξ < **2.0**
- **QS**: ξ ≥ **2.0**

**용도**: 적응형 T_max 계산

---

### 6. **Echo 관련** (`sim_echo_sweep.py`, `sim_echo_curves.py`)

```python
# sim_echo_sweep.py Line 89-94
if xi < 0.1:  # MN regime (논문 기준: ξ < 0.1)
    ...
elif xi > 10:  # QS regime (논문 기준: ξ > 10)
```

**사용 기준:**
- **MN**: ξ < **0.1** (주석: "논문 기준")
- **QS**: ξ > **10** (주석: "논문 기준")

**용도**: Echo 시뮬레이션 파라미터 조정

---

## 📊 종합 비교표

| 용도 | MN | Crossover | QS | 위치 |
|------|----|-----------|----|----|
| **그래프 색상** | < 0.5 | 0.5 - 2.0 | ≥ 2.0 | `plot_all_figures.py` |
| **Fitting 모델** | < 0.15 | 0.15 - 4.0 | ≥ 4.0 | `fitting.py` |
| **MN slope** | < 0.2 | - | - | `analyze_mn.py` |
| **문서** | < 0.3 | 0.3 - 3 | ≥ 3 | `SIMULATION_PARAMETERS.md` |
| **FID 스윕** | < 0.5 | 0.5 - 2.0 | ≥ 2.0 | `sim_fid_sweep.py` |
| **Echo** | < 0.1 | - | > 10 | `sim_echo_sweep.py` |

---

## ⚠️ 문제점

### 1. **일관성 부족**
- 같은 "MN regime"인데 0.1, 0.15, 0.2, 0.3, 0.5 등 다양한 기준 사용
- 같은 "QS regime"인데 2.0, 3.0, 4.0, 10.0 등 다양한 기준 사용

### 2. **논문과의 불일치**
- 논문 Methods 3.1: MN < 0.5, QS ≥ 2.0
- 논문 Methods 3.7: MN < 0.2 (slope fitting)
- 코드는 용도별로 다른 기준 사용

### 3. **혼란스러운 주석**
- Echo 코드에 "논문 기준: ξ < 0.1" 주석
- 하지만 실제 논문 Methods 3.1은 "ξ < 0.5"

---

## ✅ 권장사항

### **표준화된 기준 (논문과 일치)**

**그래프/시각화 기준** (가장 많이 사용됨):
- **MN**: ξ < **0.5**
- **Crossover**: 0.5 ≤ ξ < **2.0**
- **QS**: ξ ≥ **2.0**

이 기준이 `plot_all_figures.py`와 `sim_fid_sweep.py`에서 사용되며, 논문 Methods 3.1과도 일치합니다.

### **Fitting 모델 선택은 더 엄격하게**

Fitting은 더 보수적으로:
- **MN**: ξ < **0.15** (확실한 MN만)
- **Crossover**: 0.15 ≤ ξ < **4.0**
- **QS**: ξ ≥ **4.0** (확실한 QS만)

이유: 모델 선택은 정확도가 중요하므로 더 엄격한 기준 사용

### **MN Slope Fitting**

- **MN**: ξ < **0.2** (현재 사용 중)

이유: 이론 T₂ ∝ τc⁻¹이 엄격히 성립하는 범위

---

## 📝 논문 Methods 섹션 수정 제안

### **3.1 Simulation Parameters**

현재 논문:
```
Motional Narrowing: ξ < 0.5
Crossover: 0.5 ≤ ξ < 2.0
Quasi-Static: ξ ≥ 2.0
```

✅ **이대로 유지** (코드의 그래프 기준과 일치)

### **3.5 T₂ Extraction**

현재 논문:
```
Model selection based on regime:
- MN (ξ < 0.15): exponential
- Crossover (0.15 ≤ ξ < 4.0): stretched exponential
- QS (ξ ≥ 4.0): Gaussian
```

✅ **이대로 유지** (코드의 fitting 기준과 일치)

### **3.7 Power-Law Fitting**

현재 논문:
```
Restricted to ξ < 0.2
```

✅ **이대로 유지** (코드의 MN slope 기준과 일치)

---

## 🎯 결론

**코드에서 가장 많이 사용하는 기준:**
- **그래프/시각화**: MN < 0.5, QS ≥ 2.0
- **Fitting**: MN < 0.15, QS ≥ 4.0
- **MN slope**: MN < 0.2

**논문 Methods 섹션:**
- ✅ 3.1: MN < 0.5, QS ≥ 2.0 (코드와 일치)
- ✅ 3.5: MN < 0.15, QS ≥ 4.0 (코드와 일치)
- ✅ 3.7: MN < 0.2 (코드와 일치)

**문제는 코드 내부 일관성 부족이지, 논문과의 불일치는 아님!**

---

**작성일**: 2025-01-XX

