# 📊 시뮬레이션 결과물 체크리스트 실행 가이드

이 문서는 체크리스트에 따라 모든 시뮬레이션을 실행하는 방법을 설명합니다.

## 🎯 필수 결과물 (Must Have)

### 생성될 파일 목록

#### **필수 파일:**

```
✅ fid_tau_c_1e-8.csv
✅ fid_tau_c_1e-7.csv
✅ fid_tau_c_1e-6.csv
✅ fid_tau_c_1e-5.csv
✅ t2_vs_tau_c.csv (20 points)
✅ motional_narrowing_fit.txt
✅ echo_tau_c_1e-8.csv
✅ echo_tau_c_1e-7.csv
✅ echo_tau_c_1e-6.csv
✅ echo_tau_c_1e-5.csv
✅ t2_echo_vs_tau_c.csv (20 points)
✅ echo_gain.csv
✅ noise_trajectory_fast.csv
✅ noise_trajectory_slow.csv
```

#### **선택적 파일 (Optional):**

```
⚠️ bootstrap_distribution.csv
⚠️ convergence_test.csv
```

---

## 🚀 실행 방법

### **방법 1: 자동 실행 (권장)**

모든 시뮬레이션을 순서대로 자동 실행:

```bash
python run_all_simulations.py
```

**예상 시간:** ~3-4 시간

---

### **방법 2: 단계별 실행**

각 스크립트를 개별적으로 실행:

#### **Step 1: FID Full Sweep**

```bash
python run_fid_sweep.py
```

**출력:** `results_comparison/t2_vs_tau_c.csv`  
**예상 시간:** ~1-2 시간

---

#### **Step 2: FID Representative Curves**

```bash
python run_fid_curves.py
```

**출력:** 
- `results_comparison/fid_tau_c_1e-8.csv`
- `results_comparison/fid_tau_c_1e-7.csv`
- `results_comparison/fid_tau_c_1e-6.csv`
- `results_comparison/fid_tau_c_1e-5.csv`

**예상 시간:** ~10 분

---

#### **Step 3: Motional Narrowing Fit Analysis**

```bash
python analyze_motional_narrowing.py
```

**입력:** `results_comparison/t2_vs_tau_c.csv`  
**출력:** `results_comparison/motional_narrowing_fit.txt`  
**예상 시간:** 즉시

---

#### **Step 4: Hahn Echo Full Sweep**

```bash
python run_echo_sweep.py
```

**출력:** `results_comparison/t2_echo_vs_tau_c.csv`  
**예상 시간:** ~1-2 시간

---

#### **Step 5: Hahn Echo Representative Curves**

```bash
python run_echo_curves.py
```

**출력:**
- `results_comparison/echo_tau_c_1e-8.csv`
- `results_comparison/echo_tau_c_1e-7.csv`
- `results_comparison/echo_tau_c_1e-6.csv`
- `results_comparison/echo_tau_c_1e-5.csv`

**예상 시간:** ~10 분

---

#### **Step 6: Echo Gain Analysis**

```bash
python analyze_echo_gain.py
```

**입력:** 
- `results_comparison/t2_vs_tau_c.csv`
- `results_comparison/t2_echo_vs_tau_c.csv`

**출력:** `results_comparison/echo_gain.csv`  
**예상 시간:** 즉시

---

#### **Step 7: Noise Trajectory Examples**

```bash
python generate_noise_examples.py
```

**출력:**
- `results_comparison/noise_trajectory_fast.csv`
- `results_comparison/noise_trajectory_slow.csv`

**예상 시간:** 즉시

---

#### **Step 8: Bootstrap Distribution (Optional)**

```bash
python run_bootstrap.py
```

**출력:** `results_comparison/bootstrap_distribution.csv`  
**예상 시간:** ~30 분

---

#### **Step 9: Convergence Test (Optional)**

```bash
python run_convergence_test.py
```

**출력:** `results_comparison/convergence_test.csv`  
**예상 시간:** ~30-60 분

---

## 📋 시뮬레이션 파라미터

### **고정 파라미터:**

```python
gamma_e = 1.76e11          # rad/(s·T) - electron gyromagnetic ratio
B_rms = 0.05e-3            # T (0.05 mT for purified Si-28)
N_traj = 1000              # Monte Carlo trajectories per point
```

### **Swept 파라미터:**

```python
# Full sweep (for T2 vs tau_c)
tau_c_sweep = np.logspace(-8, -3, 20)  # 20 points, 5 decades

# Representative points (for FID/Echo curves)
tau_c_repr = [1e-8, 1e-7, 1e-6, 1e-5]  # 4 points
```

---

## 📊 예상 데이터 크기

| File | Rows | Columns | Size |
|------|------|---------|------|
| `fid_tau_c_*.csv` | ~1000 | 2-3 | ~50 KB each |
| `t2_vs_tau_c.csv` | 20 | 6 | <1 KB |
| `echo_tau_c_*.csv` | ~1000 | 2-3 | ~50 KB each |
| `t2_echo_vs_tau_c.csv` | 20 | 5 | <1 KB |
| `echo_gain.csv` | 20 | 6 | <1 KB |
| `noise_trajectory_*.csv` | ~10000 | 2 | ~200 KB each |

**총 예상 크기:** ~2-3 MB

---

## ✅ 품질 체크리스트

### **시뮬레이션 전:**

1. ✅ **Regime coverage check:**
   ```python
   xi = gamma_e * B_rms * tau_c_sweep
   print(f"xi range: {xi.min():.3e} to {xi.max():.3e}")
   # Should cover: ~0.01 to ~100
   ```

2. ✅ **Simulation time adequacy:**
   - 각 tau_c에 대해 T_max ≥ 10 × T2_expected 확인

3. ✅ **Timestep check:**
   - dt < 0.01 × tau_c (Rule of thumb)

### **시뮬레이션 후:**

1. ✅ **FID decay quality:**
   - P(t) starts at ~1.0
   - Decays smoothly
   - R² > 0.95 for fits

2. ✅ **Bootstrap errors reasonable:**
   - Relative error < 10%

3. ✅ **Echo gain physical:**
   - Echo_gain > 1 (always)
   - Echo_gain increases with τc

---

## 🔧 문제 해결

### **문제: ImportError**

```bash
# Ensure you're in the project root directory
cd /path/to/simulation

# Install dependencies if needed
pip install -r requirements.txt
```

### **문제: 메모리 부족**

- `N_traj`를 줄이거나
- `use_online=True` 옵션 사용 (일부 스크립트에서)

### **문제: 시뮬레이션 시간이 너무 길다**

- `tau_c_npoints`를 줄이거나 (예: 20 → 15)
- `N_traj`를 줄이거나 (예: 1000 → 500)

---

## 📝 출력 파일 형식

### **t2_vs_tau_c.csv**

```csv
tau_c,T2,T2_lower,T2_upper,R2,xi
1.000000e-08,1.234e-04,1.200e-04,1.268e-04,0.9987,8.800e-02
...
```

### **fid_tau_c_1e-8.csv**

```csv
time (s),P(t),P_std
0.000000e+00,1.000000,0.000000
5.000000e-11,0.998765,0.001234
...
```

### **motional_narrowing_fit.txt**

```
Motional Narrowing Regime Fit Results
========================================

Slope: -1.043 ± 0.006
R²: 0.9998
Number of points: 8
...
```

---

## 💬 문의

문제가 발생하면:
1. 에러 메시지 확인
2. 로그 파일 확인
3. 파라미터 설정 확인

---

**마지막 업데이트:** 2025-01-XX

