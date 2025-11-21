# 🎯 핵심 파일만 남긴 정리된 코드베이스

## ✅ 핵심 실행 스크립트 (5개 Figure 생성용)

### **1. 시뮬레이션 실행**
```
run_all_simulations.py          # ⭐ 전체 자동 실행
run_fid_sweep.py                # FID 전체 sweep → t2_vs_tau_c.csv
run_echo_sweep.py               # Echo 전체 sweep → t2_echo_vs_tau_c.csv
run_fid_curves.py               # FID 대표 곡선 → fid_tau_c_*.csv
run_echo_curves.py              # Echo 대표 곡선 → echo_tau_c_*.csv
```

### **2. 분석 스크립트**
```
analyze_motional_narrowing.py   # MN regime 분석 → motional_narrowing_fit.txt
analyze_echo_gain.py            # Echo gain 분석 → echo_gain.csv
```

### **3. Figure 생성**
```
generate_dissertation_plots.py  # ⭐ 모든 논문용 Figure 생성
generate_noise_examples.py      # Noise trajectory 데이터 생성
check_slope_consistency.py     # Slope 값 일관성 확인
```

---

## 📊 생성되는 핵심 Figure (5개)

### **Figure 1: T2 vs tau_c (Main Result)**
- 파일: `results_comparison/figures/fig6_T2_vs_tau_c.png`
- 데이터: `t2_vs_tau_c.csv`
- 설명: FID coherence time vs correlation time (3개 regime 구분)

### **Figure 2: Motional Narrowing Validation**
- 파일: `results_comparison/figures/fig7_MN_regime_slope.png`
- 데이터: `t2_vs_tau_c.csv` (MN regime만)
- 설명: Slope = -1 검증

### **Figure 3: Echo Gain**
- 파일: `results_comparison/figures/fig8_echo_gain.png`
- 데이터: `echo_gain.csv`
- 설명: Echo gain vs tau_c

### **Figure 4: Representative Curves**
- 파일: `results_comparison/figures/fig5_representative_curves.png`
- 데이터: `fid_tau_c_*.csv`, `echo_tau_c_*.csv`
- 설명: FID vs Echo 비교 (4개 tau_c 값)

### **Figure 5: Convergence Test**
- 파일: `results_comparison/figures/fig9_convergence_test.png`
- 데이터: `convergence_N_traj_*.csv`
- 설명: N_traj에 따른 수렴성 검증

---

## 🚀 사용 방법

### **전체 실행 (권장)**
```bash
# 1. 전체 시뮬레이션 실행
python run_all_simulations.py

# 2. 모든 Figure 생성
python generate_dissertation_plots.py
```

### **단계별 실행**
```bash
# Step 1: FID 시뮬레이션
python run_fid_sweep.py
python run_fid_curves.py
python analyze_motional_narrowing.py

# Step 2: Echo 시뮬레이션
python run_echo_sweep.py
python run_echo_curves.py
python analyze_echo_gain.py

# Step 3: Figure 생성
python generate_dissertation_plots.py
```

---

## 📁 핵심 패키지

```
spin_decoherence/               # 모든 핵심 모듈
  ├── noise/                    # 노이즈 생성
  ├── physics/                  # 물리 계산
  ├── simulation/                # 시뮬레이션 엔진
  ├── analysis/                  # 데이터 분석
  └── config/                    # 설정
```

---

## 📊 결과 파일

### **핵심 결과**
- `results_comparison/t2_vs_tau_c.csv` - FID 메인 결과
- `results_comparison/t2_echo_vs_tau_c.csv` - Echo 메인 결과
- `results_comparison/echo_gain.csv` - Echo gain 결과
- `results_comparison/motional_narrowing_fit.txt` - MN 분석 결과

### **대표 곡선**
- `results_comparison/fid_tau_c_*.csv` - FID 대표 곡선
- `results_comparison/echo_tau_c_*.csv` - Echo 대표 곡선

### **Figure**
- `results_comparison/figures/fig*.png` - 모든 논문용 Figure

---

## ❌ 제거된 항목

### **Material 비교 관련**
- `analyze_results.py` - Material 비교용
- `scripts/run_material_comparison.py` - Material 비교 스크립트

### **개선 모듈 (사용 안 함)**
- `adaptive_simulation.py`
- `memory_efficient_sim.py`
- `improved_t2_extraction.py`
- `parameter_validation.py`
- `regime_aware_bootstrap_improved.py`
- `simulation_monitor.py`

### **선택적 분석**
- `analyze_crossover_regime.py`
- `analyze_systematic_error.py`
- `run_bootstrap.py`
- `run_convergence_test.py`

### **오래된 문서**
- 모든 `*SUMMARY.md`, `*ISSUES.md`, `*FIXES.md` 등
- `legacy/old_docs/`로 이동

---

## ✅ 정리 완료

이제 **핵심 파일만 남아 있어** 사용하기 훨씬 쉬워졌습니다!

**핵심 스크립트**: 10개
**핵심 Figure**: 5개
**핵심 패키지**: `spin_decoherence/`

