# 시뮬레이션 실행 명령어 가이드

## 📊 현재 설정
- **N_traj**: 2000 (Monte Carlo trajectories per point)
- **Total points**: 62 points (MN: 18, Crossover: 24, QS: 20)
- **Material**: Si:P
- **Noise model**: OU (Ornstein-Uhlenbeck)

## ⏱️ 예상 소요 시간
- **FID Sweep**: ~24-30 hours (62 points × ~25-30 min/point)
- **Echo Sweep**: ~30-36 hours (62 points × ~30-35 min/point)
- **Representative Curves**: ~3-4 hours (7 points × ~30 min/point)
- **Total**: ~60-70 hours (~2.5-3 days)

## 🚀 전체 시뮬레이션 실행 순서

### 1단계: FID Sweep (가장 중요, 먼저 실행)
```bash
cd "/Users/physicsinman/Library/Mobile Documents/com~apple~CloudDocs/Documents/Physics/Physics_3rd_Year/5th Semester/Dissertation/simulation"
python3 run_fid_sweep.py
```
**출력 파일**: `results_comparison/t2_vs_tau_c.csv`
**예상 시간**: 24-30 hours

### 2단계: Hahn Echo Sweep (FID 완료 후 실행)
```bash
python3 run_echo_sweep.py
```
**출력 파일**: `results_comparison/t2_echo_vs_tau_c.csv`
**예상 시간**: 30-36 hours

### 3단계: Representative Curves (선택적, 시간 있을 때)
```bash
# FID representative curves
python3 run_fid_curves.py

# Echo representative curves
python3 run_echo_curves.py
```
**출력 파일**: 
- `results_comparison/fid_tau_c_*.csv` (7 files)
- `results_comparison/echo_tau_c_*.csv` (7 files)
**예상 시간**: 3-4 hours

## 📈 분석 스크립트 (시뮬레이션 완료 후)

### Motional Narrowing 분석
```bash
python3 analyze_motional_narrowing.py
```
**출력 파일**: `results_comparison/motional_narrowing_fit.txt`

### Echo Gain 분석
```bash
python3 analyze_echo_gain.py
```
**출력 파일**: `results_comparison/echo_gain.csv`

### Crossover Regime 분석
```bash
python3 analyze_crossover_regime.py
```
**출력 파일**: `results_comparison/crossover_regime_analysis.txt`

### Systematic Error Budget
```bash
python3 analyze_systematic_error.py
```
**출력 파일**: `results_comparison/systematic_error_budget.txt`

### 논문용 그래프 생성
```bash
python3 generate_dissertation_plots.py
```
**출력 파일**: `results_comparison/figures/fig*.png`

## 🔍 검증 스크립트 (선택적)

### OU Noise 검증
```bash
python3 validate_ou_noise.py
```
**출력 파일**: `results_comparison/ou_noise_validation.txt`

### Convergence Test (이미 완료)
```bash
python3 run_convergence_test.py
```
**출력 파일**: `results_comparison/convergence_test_summary.txt`

## 💡 실행 팁

### 백그라운드 실행 (권장)
```bash
# FID Sweep을 백그라운드로 실행
nohup python3 run_fid_sweep.py > fid_sweep.log 2>&1 &

# 진행 상황 확인
tail -f fid_sweep.log
```

### 진행 상황 모니터링
```bash
# CSV 파일의 행 수 확인 (완료된 포인트 수)
wc -l results_comparison/t2_vs_tau_c.csv

# 마지막 포인트 확인
tail -1 results_comparison/t2_vs_tau_c.csv
```

### 중단 후 재개
- 시뮬레이션은 각 포인트마다 CSV에 저장되므로 중단해도 이미 완료된 포인트는 유지됩니다.
- 스크립트를 다시 실행하면 이미 완료된 포인트는 건너뛰고 계속 진행합니다.

## ⚠️ 주의사항

1. **디스크 공간**: 각 포인트당 ~1-2 MB, 총 ~100-150 MB 필요
2. **메모리**: 각 포인트당 ~500 MB - 1 GB 필요
3. **전원**: 노트북인 경우 전원 연결 권장
4. **네트워크**: iCloud 동기화 중이면 느려질 수 있음

## 📝 체크리스트

- [ ] FID Sweep 완료 (`t2_vs_tau_c.csv`에 62개 포인트)
- [ ] Echo Sweep 완료 (`t2_echo_vs_tau_c.csv`에 62개 포인트)
- [ ] Motional Narrowing 분석 완료
- [ ] Echo Gain 분석 완료
- [ ] 논문용 그래프 생성 완료

## 🎯 최소 필수 실행

논문 작성에 필요한 최소 실행:
1. ✅ FID Sweep (필수)
2. ✅ Echo Sweep (필수)
3. ✅ 분석 스크립트 실행
4. ✅ 그래프 생성

Representative Curves는 선택적입니다.

