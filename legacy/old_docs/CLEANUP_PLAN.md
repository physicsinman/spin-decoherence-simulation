# 🧹 코드베이스 정리 계획

## ✅ 실제 사용 중인 핵심 파일들

### **메인 실행 스크립트**
- `run_all_simulations.py` - 전체 시뮬레이션 자동 실행
- `run_fid_sweep.py` - FID 전체 sweep
- `run_echo_sweep.py` - Echo 전체 sweep
- `run_fid_curves.py` - FID 대표 곡선
- `run_echo_curves.py` - Echo 대표 곡선

### **분석 스크립트**
- `analyze_motional_narrowing.py` - MN regime 분석
- `analyze_echo_gain.py` - Echo gain 분석
- `analyze_crossover_regime.py` - Crossover regime 분석
- `analyze_systematic_error.py` - Systematic error 분석

### **Figure 생성**
- `generate_dissertation_plots.py` - 모든 논문용 Figure 생성
- `generate_noise_examples.py` - Noise trajectory 데이터 생성
- `check_slope_consistency.py` - Slope 값 일관성 확인

### **핵심 패키지**
- `spin_decoherence/` - 모든 핵심 모듈
- `profiles.yaml` - Material 파라미터 설정

### **설정 파일**
- `requirements.txt` - Python 패키지 의존성
- `pytest.ini` - 테스트 설정

---

## ❌ 사용되지 않는/레거시 파일들

### **레거시 메인 스크립트**
- `main.py` - 레거시 진입점 (현재 사용 안 함)
- `main_comparison.py` - Material 비교용 (현재 사용 안 함)
- `simulate.py` - 레거시 시뮬레이션 (spin_decoherence 패키지로 대체)
- `visualize.py` - 레거시 시각화 (generate_dissertation_plots.py로 대체)

### **레거시 모듈 (spin_decoherence 패키지로 이동됨)**
- `coherence.py` - `spin_decoherence/physics/coherence.py`로 이동
- `fitting.py` - `spin_decoherence/analysis/fitting.py`로 이동
- `ornstein_uhlenbeck.py` - `spin_decoherence/noise/ou.py`로 이동
- `noise_models.py` - 중복 (spin_decoherence 패키지 사용)
- `config.py` - `spin_decoherence/config/`로 이동
- `units.py` - `spin_decoherence/config/units.py`로 이동

### **Material 비교 관련 (현재 사용 안 함)**
- `simulate_materials.py` - Material 비교용
- `simulate_materials_improved.py` - Material 비교용 (개선 버전)

### **일회성 개선/리런 스크립트**
- `force_improve_all.py` - 일회성 개선 작업
- `force_rerun_fid.py` - 일회성 리런
- `rerun_problem_points.py` - 일회성 리런
- `rerun_poor_fid_points.py` - 일회성 리런
- `rerun_echo_problem_points.py` - 일회성 리런
- `rerun_mn_regime_echo.py` - 일회성 리런
- `improve_echo_gain_calculation.py` - 일회성 개선
- `improve_low_R2_points.py` - 일회성 개선
- `improve_convergence_test.py` - 일회성 개선
- `improve_simulation_performance.py` - 일회성 개선
- `comprehensive_improvement.py` - 일회성 개선
- `run_all_improvements.py` - 일회성 개선
- `calculate_echo_gain_hybrid.py` - analyze_echo_gain.py에 통합됨
- `generate_improved_echo_gain_plot.py` - generate_dissertation_plots.py에 통합됨
- `generate_all_curves.py` - run_fid_curves.py, run_echo_curves.py로 대체

### **진단/검증 스크립트 (선택적)**
- `diagnose_echo_gain_issues.py` - 디버깅용
- `final_validation.py` - 검증용
- `validate_dissertation_results.py` - 검증용
- `validate_theory_agreement.py` - 검증용
- `validate_ou_noise.py` - 검증용
- `verify_simulation_running.py` - 검증용
- `quick_test.py` - 테스트용
- `run_test_simulation.py` - 테스트용

### **SIP 관련 (사용 안 함)**
- `run_sip_*.py` - SIP 관련 스크립트들
- `test_sip_*.py` - SIP 테스트

### **Double OU 관련 (현재 사용 안 함)**
- `run_double_ou_only.py` - Double OU 전용
- `run_full_simulation_chunked.py` - 청크 처리용

### **중복/구버전**
- `figure_generation/generate_dissertation_plots.py` - 루트의 것과 중복
- `echo_gain_improvement/` - 개선 작업 폴더 (완료됨)
- `analyze_echo_gain_graph.py` - analyze_echo_gain.py와 중복

### **로그 파일들**
- `*.log` - 모든 로그 파일들
- `simulation_log.txt`
- `simulation.log`

### **압축 파일**
- `spin_decoherence.zip`
- `아카이브.zip`
- `아카이브 2.zip`

---

## 📁 결과물 정리 계획

### **results_comparison/ 디렉토리**

#### ✅ 유지할 파일들
- `t2_vs_tau_c.csv` - FID 메인 결과
- `t2_echo_vs_tau_c.csv` - Echo 메인 결과
- `echo_gain.csv` - Echo gain 결과
- `motional_narrowing_fit.txt` - MN 분석 결과
- `crossover_regime_analysis.txt` - Crossover 분석
- `systematic_error_budget.txt` - Systematic error
- `fid_tau_c_*.csv` - FID 대표 곡선 (최신 버전만)
- `echo_tau_c_*.csv` - Echo 대표 곡선 (최신 버전만)
- `noise_trajectory_fast.csv` - Fast noise 예제
- `noise_trajectory_slow.csv` - Slow noise 예제
- `figures/` - 모든 Figure 파일들

#### ❌ 정리할 파일들
- `all_results_*.json` - 중간 결과 파일들 (최신 것만 유지)
- `*_20251119_*.json` - 타임스탬프가 있는 중간 결과
- `convergence_N_traj_*.csv` - Convergence 테스트 (최신 것만 유지)
- `echo_gain_improved.csv` - 개선 버전 (echo_gain.csv로 통합됨)
- `ou_noise_validation.txt` - 검증 결과 (선택적)
- `convergence_test_summary.txt` - Convergence 요약 (선택적)
- `아카이브.zip` - 압축 파일

---

## 🗂️ 정리 전략

### **1단계: 레거시 파일 이동**
- `legacy/` 폴더에 레거시 파일들 이동
- 또는 `archive/` 폴더 생성

### **2단계: 일회성 스크립트 정리**
- `scripts/one_time/` 폴더에 일회성 스크립트들 이동
- 또는 삭제 (Git 히스토리에 보존)

### **3단계: 결과물 정리**
- `results_comparison/archive/` 폴더에 오래된 결과 이동
- 최신 결과만 유지

### **4단계: 문서 정리**
- 중복 문서 통합
- 최신 정보만 유지

---

## 📋 실행 계획

1. **레거시 폴더 생성 및 이동**
2. **일회성 스크립트 정리**
3. **결과물 정리**
4. **문서 업데이트**

