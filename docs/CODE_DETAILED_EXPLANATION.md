# 코드 상세 설명 문서

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [물리학적 배경](#물리학적-배경)
3. [코드 아키텍처](#코드-아키텍처)
4. [핵심 알고리즘](#핵심-알고리즘)
5. [주요 모듈 상세 설명](#주요-모듈-상세-설명)
6. [데이터 흐름](#데이터-흐름)
7. [수치적 고려사항](#수치적-고려사항)
8. [사용 예제](#사용-예제)

---

## 프로젝트 개요

이 프로젝트는 **실리콘(Si:P)에서 전자 스핀의 결맞음 상실(decoherence) 현상**을 Monte Carlo 시뮬레이션으로 연구하는 코드입니다. Ornstein-Uhlenbeck (OU) 확률 과정을 사용하여 확률적 자기장 변동을 모델링하고, FID (Free Induction Decay)와 Hahn Echo 시퀀스를 구현하여 다양한 노이즈 체제에서의 결맞음 시간 T₂를 계산합니다.

### 주요 특징

- **물리학적 정확성**: 이론적 해석과 검증
- **통계적 엄밀성**: Bootstrap 신뢰구간 계산
- **체제 인식 알고리즘**: Motional Narrowing, Crossover, Quasi-Static 체제별 적응형 파라미터
- **메모리 효율성**: 자동 메모리 관리 및 온라인 알고리즘
- **출판 품질**: 고품질 그래프 생성

---

## 물리학적 배경

### 물리적 파라미터 (Si:P)

- **γₑ** (전자 자이로자기비): `1.76 × 10¹¹` rad/(s·T)
- **B_rms** (RMS 자기장): `0.57 μT` (800 ppm ²⁹Si 농도에 대한 물리적 값)
- **Δω** = γₑ × B_rms: `1.00 × 10⁵` rad/s ≈ 0.10 MHz

### 세 가지 물리적 체제

시뮬레이션은 **무차원 파라미터 ξ = γₑ × B_rms × τc**에 따라 세 가지 체제로 구분됩니다:

#### 1. Motional Narrowing (MN) 체제: ξ < 0.5

- **특징**: 빠른 노이즈 변동으로 인한 평균화 효과
- **이론적 예측**: T₂ = 1 / (Δω² × τc)
- **특징**: T₂ ∝ τc⁻¹ (로그 스케일에서 기울기 = -1)

#### 2. Crossover 체제: 0.5 ≤ ξ < 2.0

- **특징**: MN과 QS 사이의 전이 영역
- **이론적 해석**: 없음 (수치 시뮬레이션 필요)
- **특징**: 복잡한 동역학

#### 3. Quasi-Static (QS) 체제: ξ ≥ 2.0

- **특징**: 느린 노이즈 변동 (거의 정적)
- **이론적 예측**: T₂* ≈ √2 / Δω ≈ 14.1 μs
- **특징**: 가우시안 감쇠: E(t) ≈ exp[-(t/T₂*)²]

### 시퀀스 타입

#### FID (Free Induction Decay)
- 단일 π/2 펄스
- 직접적인 위상 누적 측정
- 결맞음 감쇠 관찰

#### Hahn Echo
- π/2 - τ - π - τ 시퀀스
- 토글 함수: y(t) = +1 (t < τ), -1 (τ ≤ t ≤ 2τ)
- 정적 디페징(refocusing) 효과
- Echo gain = T₂_echo / T₂_fid (일반적으로 1-2)

---

## 코드 아키텍처

### 디렉토리 구조

```
simulation/
├── spin_decoherence/          # 핵심 시뮬레이션 패키지
│   ├── noise/                # 노이즈 생성
│   │   ├── ou.py            # Ornstein-Uhlenbeck 프로세스
│   │   ├── double_ou.py     # 이중 OU 프로세스
│   │   └── base.py          # 기본 노이즈 클래스
│   │
│   ├── physics/              # 물리 계산
│   │   ├── coherence.py     # 결맞음 함수 계산
│   │   ├── phase.py         # 위상 누적
│   │   └── analytical.py   # 해석적 해
│   │
│   ├── simulation/           # 시뮬레이션 엔진
│   │   ├── fid.py           # FID 시뮬레이션
│   │   ├── echo.py          # Hahn Echo 시뮬레이션
│   │   └── engine.py        # 공통 시뮬레이션 엔진
│   │
│   ├── analysis/             # 데이터 분석
│   │   ├── fitting.py       # 곡선 피팅 (T2 추출)
│   │   ├── bootstrap.py     # Bootstrap 신뢰구간
│   │   └── statistics.py    # 통계 유틸리티
│   │
│   ├── config/               # 설정
│   │   ├── constants.py     # 물리 상수
│   │   ├── simulation.py    # 시뮬레이션 설정
│   │   └── units.py         # 단위 변환
│   │
│   ├── visualization/        # 플롯 유틸리티
│   │   ├── plots.py         # 플롯 함수
│   │   ├── styles.py        # 플롯 스타일
│   │   └── comparison.py    # 비교 플롯
│   │
│   └── utils/                # 유틸리티
│       ├── io.py            # 입출력
│       ├── logging.py       # 로깅
│       └── validation.py    # 파라미터 검증
│
├── sim_fid_sweep.py          # FID 파라미터 스윕
├── sim_echo_sweep.py         # Echo 파라미터 스윕
├── plot_all_figures.py       # 모든 그래프 생성
└── run_all.py                # 전체 시뮬레이션 실행
```

### 모듈 의존성

```
sim_fid_sweep.py / sim_echo_sweep.py
  └─> spin_decoherence/simulation/fid.py / echo.py
        ├─> spin_decoherence/noise/ou.py (노이즈 생성)
        ├─> spin_decoherence/physics/coherence.py (결맞음 계산)
        ├─> spin_decoherence/physics/phase.py (위상 누적)
        └─> spin_decoherence/analysis/fitting.py (T2 추출)
              └─> spin_decoherence/analysis/bootstrap.py (신뢰구간)
```

---

## 핵심 알고리즘

### 1. Ornstein-Uhlenbeck 노이즈 생성

**위치**: `spin_decoherence/noise/ou.py`

OU 프로세스는 다음 확률 미분방정식으로 정의됩니다:

```
dδB(t) = -(1/τc) × δB(t) dt + σ × dW(t)
```

여기서:
- **τc**: 상관 시간 (correlation time)
- **σ**: 노이즈 진폭
- **dW(t)**: 위너 프로세스 (Wiener process)

#### 이산화 알고리즘 (AR(1) 재귀)

고정 시간 간격 `dt`에서 OU 프로세스는 AR(1) 재귀 관계로 정확히 구현됩니다:

```python
δB[k+1] = ρ × δB[k] + σ_η × η[k]
```

여기서:
- **ρ** = exp(-dt/τc): 자기상관 계수
- **σ_η** = B_rms × √(1 - ρ²): 노이즈 스케일링 인자
- **η[k]** ~ N(0,1): 표준 정규 분포 화이트 노이즈

#### 구현 세부사항

1. **초기값**: δB[0] ~ N(0, B_rms²) (정상 분포에서 샘플링)
2. **Burn-in**: 10 × τc 시간 동안 버닝하여 과도 효과 제거
3. **검증**: 경험적 분산과 자기상관 함수 검증
4. **최적화**: Numba JIT 컴파일로 10-100배 속도 향상

#### 코드 예제

```python
def generate_ou_noise(tau_c, B_rms, dt, N_steps, seed=None):
    """
    OU 노이즈 생성
    
    Parameters
    ----------
    tau_c : float
        상관 시간 (초)
    B_rms : float
        RMS 진폭 (Tesla)
    dt : float
        시간 간격 (초)
    N_steps : int
        시간 스텝 수
    seed : int, optional
        랜덤 시드
        
    Returns
    -------
    delta_B : ndarray
        생성된 노이즈 배열
    """
    # AR(1) 계수 계산
    rho = np.exp(-dt / tau_c)
    sigma = B_rms * np.sqrt(1 - rho**2)
    
    # 초기값 (정상 분포)
    rng = np.random.default_rng(seed)
    delta_B = np.empty(N_steps, dtype=np.float64)
    delta_B[0] = rng.normal(0.0, B_rms)
    
    # 화이트 노이즈 생성
    eta = rng.normal(0.0, 1.0, size=N_steps - 1)
    
    # AR(1) 재귀 (JIT 컴파일됨)
    if NUMBA_AVAILABLE:
        delta_B = _ar1_recursion(delta_B, rho, sigma, eta)
    else:
        for k in range(N_steps - 1):
            delta_B[k + 1] = rho * delta_B[k] + sigma * eta[k]
    
    return delta_B
```

### 2. 위상 누적 계산

**위치**: `spin_decoherence/physics/phase.py`

위상 누적은 자기장 변동의 시간 적분으로 계산됩니다:

```
φ(t) = ∫₀^t γₑ × δB(t') dt'
```

#### 이산화 구현

```python
def compute_phase_accumulation(delta_B, gamma_e, dt):
    """
    위상 누적 계산
    
    Parameters
    ----------
    delta_B : ndarray
        자기장 변동 배열
    gamma_e : float
        전자 자이로자기비
    dt : float
        시간 간격
        
    Returns
    -------
    phi : ndarray
        누적 위상 배열
    """
    # 순간 주파수 변동
    delta_omega = gamma_e * delta_B
    
    # 위상 누적: φ[k] = Σᵢ₌₀ᵏ⁻¹ δω[i] × dt
    phi = np.zeros(len(delta_omega), dtype=np.float64)
    if len(delta_omega) > 0:
        phi[1:] = np.cumsum(delta_omega[:-1] * dt, dtype=np.float64)
    
    return phi
```

**중요**: φ[0] = 0 (초기 위상은 0)

### 3. 결맞음 함수 계산

**위치**: `spin_decoherence/physics/coherence.py`

결맞음 함수는 앙상블 평균으로 계산됩니다:

```
E(t) = ⟨exp(i × φ(t))⟩
```

여기서 괄호는 여러 독립적인 노이즈 실현에 대한 평균을 의미합니다.

#### FID 시퀀스

```python
def compute_trajectory_coherence(tau_c, B_rms, gamma_e, dt, N_steps, seed=None):
    """
    단일 궤적에 대한 결맞음 계산
    
    Returns
    -------
    E_traj : ndarray
        복소수 결맞음 E_traj(t) = exp(i × φ(t))
    t : ndarray
        시간 배열
    """
    # OU 노이즈 생성
    delta_B = generate_ou_noise(tau_c, B_rms, dt, N_steps, seed=seed)
    
    # 위상 누적
    phi = compute_phase_accumulation(delta_B, gamma_e, dt)
    
    # 결맞음: E(t) = exp(i × φ(t))
    E_traj = np.exp(1j * phi)
    
    return E_traj, t
```

#### 앙상블 평균

```python
def compute_ensemble_coherence(tau_c, B_rms, gamma_e, dt, T_max, M, seed=None):
    """
    앙상블 평균 결맞음 계산
    
    Parameters
    ----------
    M : int
        앙상블 크기 (독립적인 실현 수)
    use_online : bool
        True: 메모리 효율적인 온라인 알고리즘 (Welford's method)
        False: 모든 궤적 저장 (Bootstrap 분석용)
        
    Returns
    -------
    E : ndarray
        앙상블 평균 복소수 결맞음
    E_abs : ndarray
        |E(t)| (결맞음 크기)
    E_se : ndarray
        표준 오차
    """
    # 온라인 알고리즘 (메모리 효율적)
    if use_online:
        E_mean = np.zeros(N_steps, dtype=complex)
        E_abs_mean = np.zeros(N_steps, dtype=float)
        E_M2 = np.zeros(N_steps, dtype=float)  # 분산용
        
        for m in range(M):
            E_traj, _ = compute_trajectory_coherence(...)
            
            # Welford's 온라인 알고리즘
            delta = E_traj - E_mean
            E_mean += delta / (m + 1)
            
            E_abs_traj = np.abs(E_traj)
            delta_abs = E_abs_traj - E_abs_mean
            E_abs_mean += delta_abs / (m + 1)
            delta2_abs = E_abs_traj - E_abs_mean
            E_M2 += delta_abs * delta2_abs
        
        E = E_mean
        E_abs = np.abs(E)
        E_var = E_M2 / M
        E_se = np.sqrt(E_var / M)
    else:
        # 모든 궤적 저장 (Bootstrap용)
        E_all = np.zeros((M, N_steps), dtype=complex)
        for m in range(M):
            E_all[m], _ = compute_trajectory_coherence(...)
        
        E = np.mean(E_all, axis=0)
        E_abs = np.abs(E)
        E_abs_all = np.abs(E_all)
        E_se = np.std(E_abs_all, axis=0, ddof=1) / np.sqrt(M)
    
    return E, E_abs, E_se, t, E_abs_all
```

### 4. Hahn Echo 시퀀스

**위치**: `spin_decoherence/physics/coherence.py`

Hahn Echo는 토글 함수를 사용하여 정적 디페징을 제거합니다:

```
φ_echo(2τ) = ∫₀^τ δω(t') dt' - ∫_τ^(2τ) δω(t') dt'
```

#### 구현

```python
def _compute_echo_phase(delta_omega, I_cumulative, tau, dt, n):
    """
    Echo 위상 계산 (최적화됨)
    
    최적화: 누적 적분 I[k] = ∫₀^(k×dt) δω dt'를 한 번 계산하여
    각 τ에 대해 O(1) 평가 (O(N) 대신)
    """
    # τ와 2τ의 인덱스 계산
    k_tau = tau / dt
    i_tau = int(np.floor(k_tau))
    f_tau = k_tau - i_tau  # 분수 부분
    
    k_2tau = 2 * tau / dt
    i_2tau = int(np.floor(k_2tau))
    f_2tau = k_2tau - i_2tau
    
    # 양의 위상: ∫₀^τ δω dt'
    pos_phase = I_cumulative[i_tau] + (f_tau * delta_omega[i_tau] * dt if i_tau < n else 0.0)
    
    # 음의 위상: ∫_τ^(2τ) δω dt'
    neg_phase = I_cumulative[i_2tau] - I_cumulative[i_tau] + ...
    
    # Echo 위상: φ_echo(2τ) = pos_phase - neg_phase
    return pos_phase - neg_phase
```

### 5. T₂ 추출 (곡선 피팅)

**위치**: `spin_decoherence/analysis/fitting.py`

T₂는 결맞음 감쇠 곡선을 피팅하여 추출됩니다. 체제에 따라 다른 모델을 사용합니다:

#### 모델 선택

1. **MN 체제 (ξ < 0.15)**: 지수 감쇠
   ```
   E(t) = A × exp(-t/T₂) + B
   ```

2. **Crossover (0.15 ≤ ξ < 4.0)**: 스트레치드 지수
   ```
   E(t) = A × exp[-(t/T_β)^β] + B
   ```

3. **QS 체제 (ξ ≥ 4.0)**: 가우시안 감쇠
   ```
   E(t) = A × exp[-(t/T₂*)^²] + B
   ```

#### 피팅 윈도우 선택

노이즈 플로어를 피하기 위해 적응형 윈도우 선택:

```python
def select_fit_window(t, E_abs, E_se=None, tau_c=None, gamma_e=None, B_rms=None):
    """
    피팅 윈도우 선택 (노이즈 플로어 제외)
    
    - 기본 임계값: |E| > exp(-3) ≈ 0.05
    - QS 체제: 더 높은 임계값 (exp(-1) ≈ 0.37)
    - 3-시그마 규칙: |E| > max(3×SE, eps)
    """
    # 체제 인식 임계값
    if tau_c and gamma_e and B_rms:
        Delta_omega = gamma_e * B_rms
        xi = Delta_omega * tau_c
        if xi > 2.0:  # QS 체제
            eps = max(eps, np.exp(-1.0))  # ~0.37
    
    # 3-시그마 임계값
    if E_se is not None:
        threshold = np.maximum(3 * E_se, eps)
    else:
        threshold = eps
    
    # 유효한 포인트 선택
    mask = (E_abs > threshold) & np.isfinite(E_abs) & (E_abs > 0)
    
    return t[mask], E_abs[mask]
```

#### 가중 최소제곱법

이분산성(heteroscedasticity)을 고려한 가중 피팅:

```python
def weights_from_E(E, M):
    """
    결맞음 값으로부터 가중치 계산
    
    Var(|E|) ≈ (1 - |E|²) / (2M)
    weights = 1 / Var(|E|)
    """
    v = np.maximum(1e-12, (1.0 - np.abs(E)**2) / (2.0 * M))
    weights = 1.0 / v
    weights = weights / np.max(weights)  # 정규화
    return weights
```

### 6. Bootstrap 신뢰구간

**위치**: `spin_decoherence/analysis/bootstrap.py`

Bootstrap 방법으로 T₂의 95% 신뢰구간을 계산합니다:

```python
def bootstrap_T2(t, E_abs_all, E_se, B=800, tau_c=None, gamma_e=None, B_rms=None):
    """
    Bootstrap T₂ 신뢰구간 계산
    
    Parameters
    ----------
    B : int
        Bootstrap 반복 횟수 (기본값: 800)
        
    Returns
    -------
    T2_mean : float
        평균 T₂
    T2_ci : tuple
        95% 신뢰구간 (lower, upper)
    T2_samples : ndarray
        Bootstrap 샘플
    """
    M, N = E_abs_all.shape
    
    T2_samples = []
    for b in range(B):
        # 부트스트랩 샘플링 (복원 추출)
        indices = np.random.choice(M, size=M, replace=True)
        E_bootstrap = E_abs_all[indices]
        
        # 앙상블 평균
        E_mean = np.mean(E_bootstrap, axis=0)
        E_abs_mean = np.abs(E_mean)
        
        # 피팅
        fit_result = fit_coherence_decay_with_offset(
            t, E_abs_mean, tau_c=tau_c, gamma_e=gamma_e, B_rms=B_rms
        )
        
        if fit_result:
            T2_samples.append(fit_result['T2'])
    
    # 95% 신뢰구간 (2.5%와 97.5% 백분위수)
    T2_samples = np.array(T2_samples)
    T2_mean = np.mean(T2_samples)
    T2_ci = (np.percentile(T2_samples, 2.5), np.percentile(T2_samples, 97.5))
    
    return T2_mean, T2_ci, T2_samples
```

---

## 주요 모듈 상세 설명

### 1. `spin_decoherence/noise/ou.py`

**역할**: Ornstein-Uhlenbeck 노이즈 생성

**주요 함수**:
- `generate_ou_noise()`: OU 노이즈 생성
- `_ar1_recursion()`: JIT 컴파일된 AR(1) 재귀

**특징**:
- 수치적 안정성 검증
- Burn-in 시간 자동 계산
- PSD 검증

### 2. `spin_decoherence/physics/coherence.py`

**역할**: 결맞음 함수 계산

**주요 함수**:
- `compute_trajectory_coherence()`: 단일 궤적 결맞음
- `compute_ensemble_coherence()`: 앙상블 평균 결맞음
- `compute_hahn_echo_coherence()`: Hahn Echo 결맞음

**특징**:
- 메모리 효율적인 온라인 알고리즘 (Welford's method)
- Echo 최적화 (누적 적분 사용)

### 3. `spin_decoherence/physics/analytical.py`

**역할**: 해석적 해 제공

**주요 함수**:
- `analytical_ou_coherence()`: OU 노이즈에 대한 정확한 해
- `analytical_hahn_echo_coherence()`: Hahn Echo 해석적 해
- `theoretical_T2_motional_narrowing()`: MN 체제 이론값

**수식**:
- FID: `E(t) = exp[-Δω²τc² (e^(-t/τc) + t/τc - 1)]`
- Echo: `E_echo(2τ) = exp[-Δω²τc² (2τ/τc - 3 + 4e^(-τ/τc) - e^(-2τ/τc))]`

### 4. `spin_decoherence/analysis/fitting.py`

**역할**: T₂ 추출 및 곡선 피팅

**주요 함수**:
- `fit_coherence_decay_with_offset()`: 오프셋 포함 피팅
- `select_fit_window()`: 적응형 윈도우 선택
- `fit_mn_slope()`: MN 체제 기울기 피팅

**특징**:
- 체제 인식 모델 선택
- 가중 최소제곱법
- AIC/BIC 기반 모델 선택

### 5. `spin_decoherence/simulation/fid.py`

**역할**: FID 시뮬레이션 실행

**주요 함수**:
- `run_simulation_single()`: 단일 τc 값에 대한 시뮬레이션
- `run_simulation_sweep()`: τc 스윕

**특징**:
- 적응형 T_max (체제별)
- Bootstrap CI 자동 계산

### 6. `spin_decoherence/simulation/echo.py`

**역할**: Hahn Echo 시뮬레이션 실행

**주요 함수**:
- `run_simulation_with_hahn_echo()`: FID + Echo 동시 실행
- `run_hahn_echo_sweep()`: Echo 스윕

**특징**:
- Echo 전용 파라미터 (M_echo, dt_echo)
- Echo 최적화 윈도우 선택

### 7. `spin_decoherence/simulation/engine.py`

**역할**: 공통 시뮬레이션 유틸리티

**주요 함수**:
- `estimate_characteristic_T2()`: T₂ 추정 (체제별)
- `get_dimensionless_tau_range()`: Echo τ 범위 생성

---

## 데이터 흐름

### 전체 시뮬레이션 워크플로우

```
1. 파라미터 입력
   ├─> tau_c (상관 시간)
   ├─> B_rms (RMS 자기장)
   └─> gamma_e (자이로자기비)

2. 노이즈 생성
   └─> generate_ou_noise()
       ├─> AR(1) 재귀
       └─> 검증

3. 위상 누적
   └─> compute_phase_accumulation()
       └─> φ(t) = ∫ γₑ × δB dt'

4. 결맞음 계산
   ├─> FID: E(t) = ⟨exp(i × φ(t))⟩
   └─> Echo: E_echo(2τ) = ⟨exp(i × φ_echo(2τ))⟩

5. 곡선 피팅
   └─> fit_coherence_decay_with_offset()
       ├─> 윈도우 선택
       ├─> 모델 선택 (지수/가우시안/스트레치드)
       └─> T₂ 추출

6. Bootstrap CI
   └─> bootstrap_T2()
       ├─> 부트스트랩 샘플링
       └─> 95% 신뢰구간

7. 결과 저장
   └─> CSV 파일 출력
```

### 메모리 효율성

**온라인 알고리즘** (기본):
- 메모리: O(N_steps) (단일 궤적만 저장)
- 속도: 빠름
- 제한: Bootstrap 불가

**전체 저장** (Bootstrap용):
- 메모리: O(M × N_steps) (모든 궤적 저장)
- 속도: 느림
- 장점: Bootstrap 가능

---

## 수치적 고려사항

### 1. 시간 간격 제약

**안정성 조건**: `dt < τc / 5`

이 조건은 OU 프로세스의 정확한 이산화를 보장합니다. 위반 시 수치적 불안정성이 발생할 수 있습니다.

**구현**:
```python
def get_dt(tau_c, T_max=None, max_memory_gb=8.0):
    # 안정성 제약
    dt_max_stable = tau_c / 6.0  # 안전 마진
    
    # 목표: 100 스텝/τc
    dt_target = tau_c / 100
    
    # 메모리 제약 확인
    if T_max is not None:
        N_steps = int(T_max / dt_target) + 1
        memory_gb = (N_steps * N_traj * 8) / (1024**3)
        
        if memory_gb > max_memory_gb:
            # dt 증가 (안정성 제약 내에서)
            dt_required = ...
            dt = min(dt_required, dt_max_stable)
    
    return min(dt_target, dt_max_stable)
```

### 2. 수치적 언더플로우 방지

결맞음 함수는 지수 감쇠를 포함하므로 언더플로우가 발생할 수 있습니다:

```python
# 해석적 해에서
chi = Delta_omega_sq * tau_c_sq * (np.exp(-t / tau_c) + t / tau_c - 1.0)

# 클리핑: exp(-700) ≈ 10^-304 (float64 머신 엡실론 근처)
chi_clipped = np.clip(chi, 0.0, 700.0)
E = np.exp(-chi_clipped)
```

### 3. Burn-in 시간

OU 프로세스는 정상 분포에서 시작하지만, 과도 효과를 제거하기 위해 burn-in을 사용합니다:

```python
burnin_mult = 10.0  # 10 × τc
burn_in = int(burnin_mult * tau_c / dt)
total_steps = N_steps + burn_in

# 전체 생성 후 burn-in 제거
delta_B = delta_B_full[burn_in:]
```

### 4. 적응형 시뮬레이션 시간

체제에 따라 T_max를 적응적으로 조정:

```python
def get_tmax(tau_c, B_rms, gamma_e):
    Delta_omega = gamma_e * B_rms
    xi = Delta_omega * tau_c
    T2_est = estimate_characteristic_T2(tau_c, gamma_e, B_rms)
    
    if xi < 0.1:  # MN 체제
        T_max = 10 * T2_est
    elif xi < 2.0:  # Crossover
        T_max = 20 * T2_est
    else:  # QS 체제
        T_max = 100 * T2_est  # 또는 200 × T2_est (xi에 따라)
    
    return min(T_max, 100e-3)  # 최대 100 ms
```

### 5. 정밀도 검증

코드는 여러 단계에서 검증을 수행합니다:

1. **OU 노이즈 검증**:
   - 경험적 분산 ≈ B_rms²
   - 자기상관 함수 ≈ exp(-dt/τc)

2. **위상 누적 검증**:
   - 이론적 분산과 비교
   - 단일 궤적 vs 앙상블 분산

3. **결맞음 검증**:
   - 해석적 해와 비교
   - MN/QS 체제에서 이론값과 일치 확인

---

## 사용 예제

### 예제 1: 단일 FID 시뮬레이션

```python
from spin_decoherence.simulation.fid import run_simulation_single
from spin_decoherence.config.constants import CONSTANTS

# 파라미터 설정
tau_c = 1e-6  # 1 μs
params = {
    'B_rms': 0.57e-6,  # 0.57 μT
    'gamma_e': CONSTANTS.GAMMA_E,
    'dt': 0.2e-9,  # 0.2 ns
    'T_max': 30e-6,  # 30 μs
    'M': 2000,  # 2000 궤적
    'seed': 42
}

# 시뮬레이션 실행
result = run_simulation_single(tau_c, params=params, verbose=True)

# 결과
print(f"T₂ = {result['fit_result']['T2']*1e6:.2f} μs")
print(f"95% CI: [{result['T2_ci'][0]*1e6:.2f}, {result['T2_ci'][1]*1e6:.2f}] μs")
```

### 예제 2: FID 파라미터 스윕

```python
from spin_decoherence.simulation.fid import run_simulation_sweep

# tau_c 범위 설정
params = {
    'tau_c_range': (3e-9, 1e-3),  # 3 ns ~ 1 ms
    'tau_c_num': 67,  # 67개 포인트
    'B_rms': 0.57e-6,
    'gamma_e': CONSTANTS.GAMMA_E,
    'M': 2000,
    'B_bootstrap': 800
}

# 스윕 실행
results = run_simulation_sweep(params=params, verbose=True)

# 결과 저장
import pandas as pd
df = pd.DataFrame([{
    'tau_c': r['tau_c'],
    'T2': r['fit_result']['T2'],
    'T2_ci_lower': r['T2_ci'][0],
    'T2_ci_upper': r['T2_ci'][1]
} for r in results])
df.to_csv('results/t2_vs_tau_c.csv', index=False)
```

### 예제 3: Hahn Echo 시뮬레이션

```python
from spin_decoherence.simulation.echo import run_simulation_with_hahn_echo

# Echo 전용 파라미터
params = {
    'B_rms': 0.57e-6,
    'gamma_e': CONSTANTS.GAMMA_E,
    'dt': 0.2e-9,
    'dt_echo': 0.1e-9,  # Echo는 더 작은 dt
    'M': 2000,
    'M_echo': 10000,  # Echo는 더 많은 궤적
    'T_max_echo': 500e-6  # Echo는 더 긴 시간
}

# Echo 시뮬레이션
result = run_simulation_with_hahn_echo(tau_c=1e-6, params=params)

# Echo gain 계산
T2_fid = result['fit_result_fid']['T2']
T2_echo = result['fit_result_echo']['T2']
echo_gain = T2_echo / T2_fid

print(f"Echo gain = {echo_gain:.2f}")
```

### 예제 4: 해석적 해와 비교

```python
from spin_decoherence.physics.analytical import analytical_ou_coherence
import numpy as np

# 시뮬레이션 결과
t = result['t']
E_sim = result['E_abs']

# 해석적 해
E_analytical = analytical_ou_coherence(t, CONSTANTS.GAMMA_E, 0.57e-6, tau_c)

# 비교
error = np.abs(E_sim - E_analytical)
print(f"최대 오차: {np.max(error):.6f}")
print(f"평균 오차: {np.mean(error):.6f}")
```

### 예제 5: 커스텀 노이즈 모델

```python
from spin_decoherence.noise.double_ou import generate_double_OU_noise

# 이중 OU 노이즈 (빠른 + 느린 성분)
delta_B = generate_double_OU_noise(
    tau_c1=1e-9,  # 빠른 성분: 1 ns
    tau_c2=1e-6,  # 느린 성분: 1 μs
    B_rms1=0.3e-6,
    B_rms2=0.4e-6,
    dt=0.2e-9,
    N_steps=10000,
    seed=42
)
```

---

## 결론

이 코드는 전자 스핀 결맞음 상실을 Monte Carlo 시뮬레이션으로 연구하는 포괄적인 패키지입니다. 주요 특징:

1. **물리학적 정확성**: OU 노이즈의 정확한 구현 및 해석적 해와의 검증
2. **수치적 안정성**: 적응형 파라미터 및 언더플로우 방지
3. **통계적 엄밀성**: Bootstrap 신뢰구간 및 가중 피팅
4. **효율성**: 메모리 효율적인 온라인 알고리즘 및 JIT 컴파일
5. **유연성**: 다양한 노이즈 모델 및 시퀀스 지원

이 문서는 코드의 핵심 구조와 알고리즘을 이해하는 데 도움이 되기를 바랍니다. 추가 질문이나 세부 사항이 필요하면 각 모듈의 독스트링을 참조하세요.

---

**작성일**: 2025-01-XX  
**버전**: 1.0.0

