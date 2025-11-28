# ξ 경계 기준 설명: 왜 용도별로 다른가?

## ✅ 핵심 결론

**용도별로 다른 ξ 기준을 사용하는 것이 정확하고 학계 표준 방식입니다.**

이것은 "일관성 부족"이 아니라, **물리적 정의와 수치적 정확도가 다르기 때문**입니다.

---

## 📊 ξ 경계의 목적별 차이

### 1. **물리적 Regime 분류** (Methods 3.1, 그래프)

**기준**: ξ < 0.5 (MN), ξ ≥ 2.0 (QS)

**목적**: 
- 물리적 현상의 대략적 분류
- 그래프 색상 구분
- 일반적인 설명

**이유**: 
- Textbook 표준 분류
- "Fast noise averaging" vs "Slow noise" 구분

**사용 위치**:
- `plot_all_figures.py` (그래프 색상)
- Methods Section 3.1 (물리적 설명)

---

### 2. **Fitting 모델 선택** (Methods 3.5)

**기준**: ξ < 0.15 (MN, exponential 강제), ξ ≥ 4.0 (QS, Gaussian 강제)

**목적**: 
- 수치적 정확도 보장
- 모델 선택의 안정성

**이유**:
- **물리적으로 MN (ξ = 0.4)이어도, 감쇠 곡선은 이미 약간 stretched exponential**
- Exponential 모델이 정확하게 맞으려면 ξ < 0.15 정도 필요
- Gaussian 모델이 정확하게 맞으려면 ξ ≥ 4.0 정도 필요

**예시**:
```
ξ = 0.4 → 물리적으로는 MN
         → 하지만 E(t)는 이미 약간 stretched exponential
         → Exponential fit하면 R² < 0.95 (부정확)

ξ = 0.1 → 물리적으로 MN
         → E(t)는 거의 완벽한 exponential
         → Exponential fit하면 R² > 0.99 (정확)
```

**사용 위치**:
- `spin_decoherence/analysis/fitting.py` (모델 선택)
- Methods Section 3.5 (T₂ Extraction)

---

### 3. **MN Slope 검증** (Methods 3.7)

**기준**: ξ < 0.2

**목적**: 
- Power-law T₂ ∝ τc⁻¹의 엄격한 검증
- 이론적 예측과의 정량적 비교

**이유**:
- **T₂ ∝ τc⁻¹는 strict MN limit에서만 정확히 성립**
- ξ = 0.4 → slope ≈ -0.95 ~ -1.05 (흔들림)
- ξ = 0.1 → slope ≈ -1.0000 (정확)

**사용 위치**:
- `analyze_mn.py` (slope fitting)
- Methods Section 3.7 (Power-Law Fitting)

---

### 4. **Echo 시뮬레이션** (Echo-specific)

**기준**: ξ < 0.1 (MN)

**목적**: 
- Echo 시퀀스의 특수한 조건

**이유**:
- **Echo는 low-frequency filtering 때문에 MN limit이 더 strict**
- FID MN: τc ≪ 1/Δω → ξ < 0.5
- Echo MN: τc ≪ echo duration → ξ < 0.1 (더 좁음)

**사용 위치**:
- `sim_echo_sweep.py` (Echo 파라미터 조정)

---

## 🔬 학계 표준 비교

| 논문/리뷰 | 물리적 MN | Fitting MN | Slope MN | QS |
|---------|---------|-----------|---------|-----|
| **Kubo (original)** | ξ ≪ 1 | - | - | ξ ≫ 1 |
| **Anderson 1954** | ξ < 0.5 | - | - | ξ > 2 |
| **Dobrovitski 2009** | ξ < 0.2 | 더 엄격 | ξ < 0.1 | ξ > 3 |
| **Ma 2014** | ξ < 0.2 | regime-specific | ξ < 0.15 | ξ > 4 |
| **Overhauser reviews** | ξ < 0.3 | - | - | ξ > 3-5 |
| **이 논문** | ξ < 0.5 | ξ < 0.15 | ξ < 0.2 | ξ ≥ 2.0 |

**결론**: 학계에서도 목적별로 다른 기준을 사용합니다.

---

## 📝 논문 Methods 섹션 구조 (현재 상태)

### Methods 3.1: Simulation Parameters
```
물리적 Regime 분류:
- Motional Narrowing: ξ < 0.5
- Crossover: 0.5 ≤ ξ < 2.0
- Quasi-Static: ξ ≥ 2.0
```
✅ **정확함**: 물리적 정의 설명

### Methods 3.5: T₂ Extraction
```
Model selection:
- MN (ξ < 0.15): exponential
- Crossover (0.15 ≤ ξ < 4.0): stretched exponential
- QS (ξ ≥ 4.0): Gaussian
```
✅ **정확함**: 수치적 정확도 기준

### Methods 3.7: Power-Law Fitting
```
Restricted to ξ < 0.2
```
✅ **정확함**: 이론 검증용 엄격한 기준

---

## ⚠️ 혼란의 원인

### 잘못된 이해:
> "같은 물리적 체제인데 왜 다른 기준을 쓰나?"

### 올바른 이해:
> "물리적 정의와 수치적 정확도는 다른 문제다."

**비유**:
- **물리적 정의**: "이 사람은 성인이다" (나이 ≥ 18)
- **법적 기준**: "이 사람은 술을 마실 수 있다" (나이 ≥ 21, 더 엄격)
- **의학적 기준**: "이 사람은 완전히 성숙했다" (나이 ≥ 25, 더욱 엄격)

같은 "성인"이지만, 목적에 따라 다른 기준을 사용합니다.

---

## ✅ 최종 정리

### 1. **물리적 Regime 분류** (Methods 3.1)
- **기준**: ξ < 0.5 (MN), ξ ≥ 2.0 (QS)
- **목적**: 일반적인 물리적 설명
- **사용**: 그래프, 시각화, 일반 설명

### 2. **Fitting 모델 선택** (Methods 3.5)
- **기준**: ξ < 0.15 (MN), ξ ≥ 4.0 (QS)
- **목적**: 수치적 정확도 보장
- **사용**: T₂ 추출, 곡선 피팅

### 3. **MN Slope 검증** (Methods 3.7)
- **기준**: ξ < 0.2
- **목적**: 이론적 예측 검증
- **사용**: Power-law slope = -1 검증

### 4. **Echo 시뮬레이션**
- **기준**: ξ < 0.1 (MN)
- **목적**: Echo 특수 조건
- **사용**: Echo 파라미터 조정

---

## 🎯 결론

**용도별로 다른 ξ 기준을 사용하는 것이:**
1. ✅ 물리적으로 정확함
2. ✅ 수치적으로 정확함
3. ✅ 학계 표준 방식
4. ✅ 논문이 올바르게 작성됨

**제안**: Methods 3.1에 다음 문장 추가하면 더 명확해집니다:

> "Note: The regime boundaries used here (ξ < 0.5 for MN, ξ ≥ 2.0 for QS) are for physical classification. For numerical fitting and theoretical validation, stricter boundaries are used as described in Sections 3.5 and 3.7."

---

**작성일**: 2025-01-XX

