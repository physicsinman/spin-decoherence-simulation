# 🚀 빠른 시작 가이드 (Quick Start Guide)

## 📋 실행 명령어 정리

### **방법 1: 자동 실행 (권장) ⭐**

모든 시뮬레이션을 순서대로 자동 실행:

```bash
python run_all.py
```

**예상 시간:** ~3-4시간  
**출력:** `results/` 디렉토리에 모든 결과 파일 생성

---

### **방법 2: 단계별 실행**

각 스크립트를 개별적으로 실행:

#### **Step 1: FID Full Sweep**
```bash
python sim_fid_sweep.py
```
- **출력:** `t2_vs_tau_c.csv`
- **예상 시간:** ~1-2시간
- **설명:** 20개 tau_c 값에 대해 FID 시뮬레이션 실행

#### **Step 2: FID Representative Curves**
```bash
python sim_fid_curves.py
```
- **출력:** `fid_tau_c_1e-8.csv`, `fid_tau_c_1e-7.csv`, `fid_tau_c_1e-6.csv`, `fid_tau_c_1e-5.csv`
- **예상 시간:** ~10분
- **설명:** 대표적인 4개 tau_c 값에 대한 상세 FID 곡선

#### **Step 3: Motional Narrowing Fit Analysis**
```bash
python analyze_mn.py
```
- **입력:** `t2_vs_tau_c.csv` (Step 1 결과 필요)
- **출력:** `motional_narrowing_fit.txt`
- **예상 시간:** 즉시
- **설명:** MN regime에서 slope=-1 검증

#### **Step 4: Hahn Echo Full Sweep**
```bash
python sim_echo_sweep.py
```
- **출력:** `t2_echo_vs_tau_c.csv`
- **예상 시간:** ~1-2시간
- **설명:** 20개 tau_c 값에 대해 Hahn Echo 시뮬레이션 실행

#### **Step 5: Hahn Echo Representative Curves**
```bash
python sim_echo_curves.py
```
- **출력:** `echo_tau_c_1e-8.csv`, `echo_tau_c_1e-7.csv`, `echo_tau_c_1e-6.csv`, `echo_tau_c_1e-5.csv`
- **예상 시간:** ~10분
- **설명:** 대표적인 4개 tau_c 값에 대한 상세 Echo 곡선

#### **Step 6: Echo Gain Analysis**
```bash
python3 analyze_echo_gain.py
```
- **입력:** `t2_vs_tau_c.csv`, `t2_echo_vs_tau_c.csv` (Step 1, 4 결과 필요)
- **출력:** `echo_gain.csv`
- **예상 시간:** 즉시
- **설명:** Echo gain = T2_echo / T2_fid 계산

#### **Step 7: Noise Trajectory Examples**
```bash
python generate_noise_data.py
```
- **출력:** `noise_trajectory_fast.csv`, `noise_trajectory_slow.csv`
- **예상 시간:** 즉시
- **설명:** Fast/Slow noise 예제 생성

---

### **선택적 스크립트 (Optional)**

#### **Bootstrap Distribution Analysis**
```bash
python3 run_bootstrap.py
```
- **출력:** `bootstrap_distribution.csv`
- **예상 시간:** ~30분
- **설명:** MN regime에서 bootstrap 분포 분석

#### **Convergence Test**
```bash
python3 run_convergence_test.py
```
- **출력:** `convergence_test.csv`
- **예상 시간:** ~30-60분
- **설명:** N_traj에 따른 T2 수렴 테스트

---

## 📊 생성될 파일 목록

### **필수 파일 (14개):**

1. `t2_vs_tau_c.csv` - FID T2 vs tau_c (20 points)
2. `fid_tau_c_1e-8.csv` - FID curve (tau_c=1e-8)
3. `fid_tau_c_1e-7.csv` - FID curve (tau_c=1e-7)
4. `fid_tau_c_1e-6.csv` - FID curve (tau_c=1e-6)
5. `fid_tau_c_1e-5.csv` - FID curve (tau_c=1e-5)
6. `motional_narrowing_fit.txt` - MN regime fit results
7. `t2_echo_vs_tau_c.csv` - Echo T2 vs tau_c (20 points)
8. `echo_tau_c_1e-8.csv` - Echo curve (tau_c=1e-8)
9. `echo_tau_c_1e-7.csv` - Echo curve (tau_c=1e-7)
10. `echo_tau_c_1e-6.csv` - Echo curve (tau_c=1e-6)
11. `echo_tau_c_1e-5.csv` - Echo curve (tau_c=1e-5)
12. `echo_gain.csv` - Echo gain analysis
13. `noise_trajectory_fast.csv` - Fast noise example
14. `noise_trajectory_slow.csv` - Slow noise example

---

## ⚙️ 설정 변경

### **Representative Points 개수 변경**

`run_fid_curves.py`와 `run_echo_curves.py`에서:

```python
# 현재 (4개 포인트):
tau_c_representative = np.array([1e-8, 1e-7, 1e-6, 1e-5])

# 권장 (7개 포인트) - 주석 해제:
# tau_c_representative = np.array([1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5])
```

자세한 내용은 `POINT_COUNT_RECOMMENDATION.md` 참고

---

## 🔍 결과 확인

### **생성된 파일 확인:**
```bash
ls -lh results/*.csv
ls -lh results/*.txt
```

### **결과 요약 확인:**
```bash
# FID 결과 확인
python3 -c "import pandas as pd; df = pd.read_csv('results/t2_vs_tau_c.csv'); print(df.head(10))"

# Echo 결과 확인
python3 -c "import pandas as pd; df = pd.read_csv('results/t2_echo_vs_tau_c.csv'); print(df.head(10))"
```

---

## ⚠️ 문제 해결

### **메모리 부족 오류:**
- `N_traj`를 줄이세요 (예: 1000 → 500)
- 또는 `use_online=True` 옵션 사용

### **시뮬레이션 시간이 너무 길다:**
- `tau_c_npoints`를 줄이세요 (예: 20 → 15)
- `N_traj`를 줄이세요 (예: 1000 → 500)

### **Fit 실패:**
- T_max가 충분한지 확인
- dt가 충분히 작은지 확인 (dt < 0.01 × tau_c)

---

## 📝 체크리스트

실행 후 확인:

- [ ] `t2_vs_tau_c.csv` 생성됨 (20 points)
- [ ] `fid_tau_c_*.csv` 4개 파일 생성됨
- [ ] `motional_narrowing_fit.txt` 생성됨
- [ ] `t2_echo_vs_tau_c.csv` 생성됨 (20 points)
- [ ] `echo_tau_c_*.csv` 4개 파일 생성됨
- [ ] `echo_gain.csv` 생성됨
- [ ] `noise_trajectory_*.csv` 2개 파일 생성됨

---

**마지막 업데이트:** 2025-01-XX

