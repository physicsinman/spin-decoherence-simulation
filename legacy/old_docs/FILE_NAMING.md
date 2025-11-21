# 📝 파일 이름 규칙

## ✅ 명확하고 일관된 이름 규칙

### **1. 시뮬레이션 실행 스크립트**
- `run_all.py` - 전체 자동 실행
- `sim_fid_sweep.py` - FID parameter sweep
- `sim_echo_sweep.py` - Echo parameter sweep
- `sim_fid_curves.py` - FID representative curves
- `sim_echo_curves.py` - Echo representative curves

**규칙**: `sim_` prefix + 목적

### **2. 분석 스크립트**
- `analyze_mn.py` - Motional narrowing 분석
- `analyze_echo_gain.py` - Echo gain 분석

**규칙**: `analyze_` prefix + 분석 대상

### **3. Figure 생성 스크립트**
- `plot_all_figures.py` - 모든 Figure 생성
- `generate_noise_data.py` - Noise trajectory 데이터 생성

**규칙**: `plot_` 또는 `generate_` prefix + 목적

### **4. 유틸리티 스크립트**
- `check_slope.py` - Slope 값 일관성 확인

**규칙**: `check_` prefix + 확인 대상

---

## 📊 Figure 파일 이름

### **논문 구조에 맞춘 이름**
- `fig1_T2_vs_tau_c.png` - Main result
- `fig2_MN_regime_slope.png` - MN validation
- `fig3_echo_gain.png` - Echo gain
- `fig4_representative_curves.png` - FID vs Echo
- `fig5_convergence_test.png` - Convergence test

**규칙**: `fig{번호}_{내용}.png`

---

## 🎯 이름 변경 요약

### **Before → After**

| Before | After | 이유 |
|--------|-------|------|
| `run_all_simulations.py` | `run_all.py` | 간결함 |
| `run_fid_sweep.py` | `sim_fid_sweep.py` | 일관성 (`sim_` prefix) |
| `run_echo_sweep.py` | `sim_echo_sweep.py` | 일관성 |
| `run_fid_curves.py` | `sim_fid_curves.py` | 일관성 |
| `run_echo_curves.py` | `sim_echo_curves.py` | 일관성 |
| `analyze_motional_narrowing.py` | `analyze_mn.py` | 간결함 |
| `generate_dissertation_plots.py` | `plot_all_figures.py` | 명확함 |
| `generate_noise_examples.py` | `generate_noise_data.py` | 명확함 |
| `check_slope_consistency.py` | `check_slope.py` | 간결함 |

---

## ✅ 장점

1. **일관성**: 같은 종류의 파일은 같은 prefix 사용
2. **명확성**: 파일 이름만 봐도 용도 파악 가능
3. **간결성**: 불필요한 단어 제거
4. **논문 구조**: Figure 번호가 논문 구조와 일치

---

## 📁 최종 구조

```
simulation/
├── run_all.py                  # 전체 실행
├── sim_*.py                    # 시뮬레이션 스크립트
├── analyze_*.py                # 분석 스크립트
├── plot_*.py                   # Figure 생성
├── generate_*.py               # 데이터 생성
├── check_*.py                  # 검증 스크립트
└── spin_decoherence/           # 핵심 패키지
```

