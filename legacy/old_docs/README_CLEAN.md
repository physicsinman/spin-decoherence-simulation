# 🎯 정리된 코드베이스 사용 가이드

## 📁 핵심 파일 구조

### **메인 실행 스크립트**
```
run_all_simulations.py          # ⭐ 전체 시뮬레이션 자동 실행 (권장)
run_fid_sweep.py                # FID 전체 sweep
run_echo_sweep.py               # Echo 전체 sweep
run_fid_curves.py               # FID 대표 곡선
run_echo_curves.py              # Echo 대표 곡선
```

### **분석 스크립트**
```
analyze_motional_narrowing.py   # MN regime 분석
analyze_echo_gain.py            # Echo gain 분석
analyze_crossover_regime.py     # Crossover regime 분석
analyze_systematic_error.py     # Systematic error 분석
check_slope_consistency.py     # Slope 값 일관성 확인
```

### **Figure 생성**
```
generate_dissertation_plots.py   # ⭐ 모든 논문용 Figure 생성
generate_noise_examples.py      # Noise trajectory 데이터 생성
```

### **핵심 패키지**
```
spin_decoherence/                # 모든 핵심 모듈
  ├── noise/                    # 노이즈 생성
  ├── physics/                  # 물리 계산
  ├── simulation/               # 시뮬레이션 엔진
  ├── analysis/                 # 데이터 분석
  └── config/                   # 설정
```

---

## 🚀 빠른 시작

### **1. 전체 시뮬레이션 실행 (권장)**
```bash
python run_all_simulations.py
```

이 명령어 하나로 모든 시뮬레이션이 자동으로 실행됩니다:
1. FID Full Sweep
2. FID Representative Curves
3. Motional Narrowing 분석
4. Hahn Echo Full Sweep
5. Hahn Echo Representative Curves
6. Echo Gain 분석
7. Noise Trajectory 예제 생성

### **2. Figure 생성**
```bash
python generate_dissertation_plots.py
```

모든 논문용 Figure가 `results_comparison/figures/`에 생성됩니다.

### **3. Slope 값 확인**
```bash
python check_slope_consistency.py
```

논문 값과 실제 결과를 비교합니다.

---

## 📊 결과 파일 위치

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

## 📚 문서

- `QUICK_START.md` - 빠른 시작 가이드
- `COMMANDS.md` - 실행 명령어 정리
- `CODE_STRUCTURE.md` - 코드 구조 설명
- `PAPER_CODE_COMPARISON.md` - 논문-코드 비교
- `CLEANUP_SUMMARY.md` - 정리 요약

---

## ⚠️ 주의사항

1. **레거시 파일**: `legacy/` 폴더의 파일들은 더 이상 사용되지 않습니다.
2. **Archive**: `results_comparison/archive/`의 파일들은 참고용입니다.
3. **핵심 파일만 사용**: 루트 디렉토리의 파일들만 사용하세요.

---

## ✅ 정리 완료

코드베이스가 깔끔하게 정리되었습니다!
이제 핵심 파일들만 남아 있어 사용하기 쉽습니다.

