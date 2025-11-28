# Figure 파일 구조

## 📊 핵심 Figure (논문 구조 순서)

### **Figure 1: T2 vs tau_c (Main Result)**
- 파일: `fig1_T2_vs_tau_c.png`
- 내용: FID coherence time vs correlation time
- 3개 regime 구분 (MN, Crossover, QS)

### **Figure 2: Motional Narrowing Validation**
- 파일: `fig2_MN_regime_slope.png`
- 내용: MN regime에서 slope = -1 검증
- Log-log plot with linear fit

### **Figure 3: Echo Gain**
- 파일: `fig3_echo_gain.png`
- 내용: Echo gain vs tau_c
- Regime별 다른 동작

### **Figure 4: Representative Curves**
- 파일: `fig4_representative_curves.png`
- 내용: FID vs Echo 비교 (4개 tau_c 값)
- 2x2 패널 구성

### **Figure 5: Convergence Test**
- 파일: `fig5_convergence_test.png`
- 내용: N_traj에 따른 수렴성 검증
- 3개 tau_c 값 비교

---

## 📁 Supplementary Figures

`supplementary/` 폴더에 보조 Figure들이 보관되어 있습니다:
- `noise_trajectories.png` - Noise trajectory 예제
- `conceptual_noise.png` - Conceptual diagram
- `three_regimes.png` - Three regime schematic
- `simulation_flowchart.png` - Simulation flowchart
- 기타 보조 분석 Figure들

---

## 🎯 생성 방법

```bash
# 모든 핵심 Figure 생성
python plot_all_figures.py
```

생성되는 파일:
- `fig1_T2_vs_tau_c.png`
- `fig2_MN_regime_slope.png`
- `fig3_echo_gain.png`
- `fig4_representative_curves.png`
- `fig5_convergence_test.png`

