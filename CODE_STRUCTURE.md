# 코드베이스 구조 설명

## 📁 전체 구조

```
simulation/
├── spin_decoherence/          # 핵심 패키지 (모듈화된 코드)
│   ├── noise/                 # 노이즈 생성
│   │   ├── ou.py             # Ornstein-Uhlenbeck 노이즈
│   │   └── double_ou.py      # Double-OU 노이즈
│   ├── physics/               # 물리 계산
│   │   ├── coherence.py      # Coherence 함수 계산
│   │   ├── phase.py          # Phase accumulation
│   │   └── analytical.py     # 이론적 해
│   ├── simulation/            # 시뮬레이션 엔진
│   │   ├── fid.py            # FID 시뮬레이션
│   │   ├── echo.py           # Hahn Echo 시뮬레이션
│   │   └── engine.py         # 공통 엔진
│   ├── analysis/             # 데이터 분석
│   │   ├── fitting.py        # 곡선 피팅
│   │   └── bootstrap.py      # Bootstrap 통계
│   └── config/                # 설정
│       └── constants.py       # 물리 상수
│
├── run_all.py                 # ⭐ 전체 자동 실행
├── sim_*.py                   # 시뮬레이션 스크립트
│   ├── sim_fid_sweep.py      # FID parameter sweep
│   ├── sim_echo_sweep.py     # Echo parameter sweep
│   ├── sim_fid_curves.py     # FID representative curves
│   └── sim_echo_curves.py    # Echo representative curves
│
├── analyze_*.py               # 분석 스크립트
│   ├── analyze_mn.py         # Motional narrowing 분석
│   └── analyze_echo_gain.py # Echo gain 분석
│
├── plot_all_figures.py        # ⭐ 모든 논문용 Figure 생성
├── generate_noise_data.py     # Noise trajectory 데이터 생성
└── check_slope.py             # Slope 값 일관성 확인
```

## 🔄 실행 흐름 (Workflow)

### 1. 전체 시뮬레이션 실행

```
run_all.py
  ├─> sim_fid_sweep.py
  ├─> sim_fid_curves.py
  ├─> analyze_mn.py
  ├─> sim_echo_sweep.py
  ├─> sim_echo_curves.py
  ├─> analyze_echo_gain.py
  └─> generate_noise_data.py
       ├─> spin_decoherence/noise/ou.py (노이즈 생성)
       ├─> spin_decoherence/physics/coherence.py (Coherence 계산)
       ├─> spin_decoherence/simulation/fid.py 또는 echo.py
       ├─> spin_decoherence/analysis/fitting.py (T2 추출)
       └─> 결과 저장 (JSON)
```

### 2. 결과 분석 및 그래프 생성

```
analyze_echo_gain.py
  └─> results/echo_gain.csv 생성

plot_all_figures.py
  ├─> results/t2_vs_tau_c.csv 읽기
  ├─> results/echo_gain.csv 읽기
  └─> results/figures/fig*.png 생성
```

## 🎯 핵심 모듈 설명

### 1. **spin_decoherence/noise/ou.py** - 노이즈 생성
```python
# AR(1) 재귀 관계로 OU 노이즈 생성
δB_{k+1} = ρ·δB_k + σ_η·η_k
where ρ = exp(-dt/τ_c), σ_η = B_rms·√(1-ρ²)
```

**역할**: Ornstein-Uhlenbeck 프로세스로 자기장 노이즈 생성

### 2. **spin_decoherence/physics/coherence.py** - Coherence 계산
```python
# Phase accumulation
φ(t) = ∫₀^t γ_e·δB(t') dt'

# Ensemble coherence
E(t) = ⟨exp(i·φ(t))⟩
```

**역할**: 
- Phase accumulation 계산
- Ensemble average로 coherence 함수 계산
- FID와 Hahn Echo 모두 지원

### 3. **spin_decoherence/simulation/fid.py & echo.py** - 시뮬레이션
```python
# FID: 단순 phase accumulation
# Echo: Toggling function 적용
y(t) = +1 (t < τ), -1 (τ ≤ t ≤ 2τ)
```

**역할**: 
- FID: Free Induction Decay 시뮬레이션
- Echo: Hahn Echo 시뮬레이션 (π pulse 효과)

### 4. **spin_decoherence/analysis/fitting.py** - T2 추출
```python
# Fitting with scale and offset
y(t) = A·E(t) + B

# T2 extraction: E(T2) = 1/e
```

**역할**: 
- Coherence decay curve 피팅
- T2 값 추출
- Bootstrap confidence interval 계산

### 5. **simulate_materials_improved.py** - 메인 시뮬레이션 로직
```python
def run_single_case_improved():
    # 1. Parameter validation
    # 2. Noise generation
    # 3. Coherence calculation
    # 4. Fitting
    # 5. Bootstrap CI
    # 6. Save results
```

**역할**: 
- Material별 시뮬레이션 실행
- 파라미터 검증 및 적응형 전략
- 결과 저장 및 통합

### 6. **analyze_echo_gain.py** - Echo Gain 계산
```python
# Hybrid method
- Direct measurement: E_echo(T_FID) 직접 측정
- Fitting method: T2_echo / T2_fid
```

**역할**: 
- FID와 Echo 결과 결합
- Echo gain = T2_echo / T2_fid 계산
- `echo_gain.csv` 생성

### 7. **generate_dissertation_plots.py** - 그래프 생성
```python
def plot_echo_gain():
    # Load echo_gain.csv
    # Filter problematic points
    # Generate publication-quality plot
```

**역할**: 
- 논문용 고품질 그래프 생성
- 모든 주요 결과 시각화
- `figures/fig*.png` 생성

## 🔑 주요 데이터 흐름

### 시뮬레이션 → 분석 → 그래프

```
1. 시뮬레이션 실행
   simulate_materials_improved.py
   └─> results_comparison/*.json

2. 결과 추출 및 정리
   analyze_echo_gain.py
   └─> results_comparison/echo_gain.csv
   └─> results_comparison/t2_vs_tau_c.csv
   └─> results_comparison/t2_echo_vs_tau_c.csv

3. 그래프 생성
   generate_dissertation_plots.py
   └─> results_comparison/figures/fig*.png
```

## 📊 현재 중심 코드

### 1. **Material 비교 시뮬레이션**
- `main_comparison.py`: 진입점
- `simulate_materials_improved.py`: 실제 시뮬레이션 로직
- `profiles.yaml`: Material 파라미터 설정

### 2. **Echo Gain 분석**
- `analyze_echo_gain.py`: Echo gain 계산
- `improve_echo_gain_calculation.py`: 개선된 계산 (최근 추가)
- `generate_improved_echo_gain_plot.py`: 개선된 그래프

### 3. **논문용 그래프**
- `generate_dissertation_plots.py`: 모든 주요 그래프 생성
  - fig1: T2 vs tau_c
  - fig2: MN regime slope
  - fig3: Echo gain
  - fig4: Representative curves
  - fig5: Convergence test

## 🛠️ 주요 개선 사항 (최근)

### 1. Echo Gain 개선
- **문제**: gain이 1.0, 1.5, 3.0, 5.0으로 고정
- **해결**: 
  - `improve_echo_gain_calculation.py`: gain cap 완화
  - Regime별 다른 cap 적용
  - 부드러운 전환 구현

### 2. 파라미터 검증
- `parameter_validation.py`: Material 파라미터 검증
- 적응형 시뮬레이션 전략

### 3. 메모리 효율성
- `memory_efficient_sim.py`: 대용량 시뮬레이션 지원
- 청크 단위 처리

## 🎓 사용 예시

### 전체 시뮬레이션 실행
```bash
python main_comparison.py --full
```

### Echo Gain 분석
```bash
python analyze_echo_gain.py
python improve_echo_gain_calculation.py
```

### 그래프 생성
```bash
python generate_dissertation_plots.py
```

## 📝 주요 설정 파일

- `profiles.yaml`: Material 파라미터 (Si:P, GaAs)
- `config.py`: 전역 설정
- `requirements.txt`: Python 패키지 의존성

## 🔍 디버깅 및 검증

- `validate_*.py`: 검증 스크립트들
- `test_*.py`: 테스트 파일들
- `analyze_*.py`: 결과 분석 스크립트들

