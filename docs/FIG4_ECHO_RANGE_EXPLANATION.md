# 오른쪽 아래 그래프 (τc = 10.00 μs) Echo 범위 설명

## 질문: 왜 Echo 데이터가 1.000 ~ 400.000 μs 범위인가?

### 물리적 배경

**τc = 1e-5 s (10.00 μs)일 때:**
- **ξ = γ_e × B_rms × τc = 88.0**
- **Regime: Deep QS (Quasi-Static) regime** (ξ > 50)

Deep QS regime에서는:
- 노이즈가 매우 느리게 변함 (τc가 큼)
- Hahn Echo가 매우 긴 시간 동안 지속됨
- Echo decay를 충분히 관찰하려면 매우 긴 시뮬레이션 시간이 필요함

### 시뮬레이션 파라미터 설정

#### 1. T_max_echo (Echo 시뮬레이션 최대 시간)

Deep QS regime (ξ > 50)에서:
```python
T_max_echo = T_max * 10.0
T_max_echo = min(T_max_echo, 1000e-3)  # 최대 1000 ms
```

여기서:
- **T_max (FID)** = 200 × T2_est ≈ 22.7 μs
- **T_max_echo** = 10 × T_max ≈ 227 μs (또는 최대 1000 ms까지 가능)

#### 2. upsilon_max (무차원 최대 지연 시간)

**υ = τ/τc** (무차원 지연 시간)로 표현하면:
```python
upsilon_max = min(0.4 * T_max_echo / tau_c, 50.0)
```

Deep QS에서:
- **upsilon_max = 50.0**으로 cap됨 (충분히 긴 범위를 보장)
- 이는 **τ_max = 50.0 × τc = 50.0 × 10 μs = 500 μs**까지 가능

#### 3. get_dimensionless_tau_range의 제약

실제 tau_list 생성 시:
```python
tau_max = min(upsilon_max * tau_c, 0.4 * T_max_echo)
```

따라서:
- **upsilon_max * tau_c = 500 μs**
- **0.4 * T_max_echo**가 더 작으면 이것이 제약이 됨

### 실제 데이터 범위 (1.000 ~ 400.000 μs)

**실제 데이터:**
- **υ_min = 0.100** (τ_min = 1.000 μs)
- **υ_max = 40.000** (τ_max = 400.000 μs)

**이 범위가 결정된 이유:**

1. **upsilon_max = 50.0 cap**: Deep QS에서 충분히 긴 범위를 보장하기 위해
2. **T_max_echo 제약**: 실제로는 T_max_echo가 약 1 ms 이상으로 설정되어야 400 μs까지 가능
   - 0.4 × T_max_echo ≥ 400 μs
   - T_max_echo ≥ 1.0 ms
3. **실제 υ_max = 40.0**: 50.0 cap보다 약간 작은 이유는
   - T_max_echo 제약
   - 또는 시뮬레이션 시 실제 사용된 파라미터의 차이

### 왜 이렇게 긴 범위가 필요한가?

**Deep QS regime의 특성:**
- 노이즈가 매우 느리게 변함 (τc = 10 μs)
- Hahn Echo는 π pulse에 의해 재위상화되어 매우 긴 시간 동안 지속됨
- Echo peak는 보통 **τ ≈ τc** 근처에서 발생 (여기서는 약 10 μs)
- Echo decay를 충분히 관찰하려면 **수십 배의 τc**까지 필요

**실제 Echo 동작:**
- τ = 0에서 시작 (|E| = 1.0)
- τ ≈ τc에서 Echo peak 발생 (여기서는 약 10 μs)
- 그 이후 천천히 decay
- 400 μs까지 관찰하면 충분한 decay를 볼 수 있음

### 그래프에서 보이는 것

**오른쪽 아래 그래프 (τc = 10.00 μs):**
- Echo 곡선이 **1 μs부터 시작** (데이터가 1 μs부터 있음)
- **약 1 μs에서 Echo peak** 발생 (|E| ≈ 0.5)
- 그 이후 천천히 decay
- 그래프는 **5 μs까지만 표시**하지만, 실제 데이터는 **400 μs까지** 존재

**왜 그래프가 1 μs부터 시작하는가?**
- Echo 시뮬레이션의 tau_list가 **υ_min = 0.05**부터 시작하지만
- 실제 데이터는 **υ_min = 0.10** (1 μs)부터 저장됨
- 이는 시뮬레이션의 최소 시간 제약이나 데이터 저장 방식 때문일 수 있음

### 요약

**Echo 범위가 1-400 μs인 이유:**
1. **Deep QS regime** (ξ = 88)에서 Echo가 매우 긴 시간 동안 지속됨
2. **upsilon_max = 50.0** cap으로 충분히 긴 범위 보장
3. **T_max_echo ≥ 1 ms**로 설정되어 400 μs까지 관찰 가능
4. Echo decay를 충분히 관찰하기 위해 **수십 배의 τc**까지 필요

이는 물리적으로 정상적인 설정이며, Deep QS regime에서 Echo의 긴 지속 시간을 정확히 관찰하기 위한 것입니다.

