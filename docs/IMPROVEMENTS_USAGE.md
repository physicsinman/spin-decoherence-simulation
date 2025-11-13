# 개선된 코드 사용 가이드 (Improved Code Usage Guide)

## 개요 (Overview)

이 문서는 새로 구현된 파라미터 검증, 메모리 효율적 시뮬레이션, 그리고 실시간 모니터링 기능의 사용법을 설명합니다.

This document explains how to use the newly implemented parameter validation, memory-efficient simulation, and real-time monitoring features.

---

## 1. 파라미터 검증 및 재설정 (Parameter Validation)

### 기본 사용법

```python
from parameter_validation import SimulationParameters, validate_simulation_parameters

# Si:P 시스템, Motional-Narrowing regime용 파라미터 생성
params = SimulationParameters(system='Si_P', target_regime='motional_narrowing')

# 파라미터 검증
report = params.validate()

# 파라미터를 딕셔너리로 변환 (기존 코드와 호환)
params_dict = params.to_dict()
```

### 현재 파라미터와 비교

```python
# 현재 사용 중인 파라미터와 문헌값 비교
comparison = validate_simulation_parameters(
    system='Si_P',
    target_regime='all',
    B_rms_current=5e-6,  # 현재 사용 중인 값 (5 µT)
    T_max_current=30e-6   # 현재 사용 중인 값 (30 µs)
)

# 비교 결과에서 권장사항 확인
for rec in comparison['recommendations']:
    print(rec)
```

### 출력 예시

```
============================================================
Parameter Comparison for Si_P
============================================================

B_rms Comparison:
  Literature (required): 3.214 nT
  Current simulation:   5000.000 nT
  Ratio: 1555.6×

T_max Comparison:
  Required (≥5×T2*): 12500.0 µs
  Current simulation:  30.0 µs
  Ratio: 0.00×

💡 RECOMMENDATIONS:
   - B_rms is 1555.6× too large. Update to 3.214 nT
   - T_max is 416.7× too short. Update to ≥ 12500.0 µs
```

---

## 2. 메모리 효율적 시뮬레이션 (Memory-Efficient Simulation)

### 기본 사용법

```python
from parameter_validation import SimulationParameters
from memory_efficient_sim import MemoryEfficientSimulation

# 검증된 파라미터 생성
params = SimulationParameters(system='GaAs', target_regime='all')
params.n_ensemble = 100  # 앙상블 크기 조정

# 메모리 효율적 시뮬레이션 생성
sim = MemoryEfficientSimulation(params)

# 단일 tau_c에 대한 시뮬레이션
tau_c = 0.1e-6  # 0.1 µs
coherence, coherence_std = sim.simulate_coherence_chunked(
    tau_c, 
    sequence='FID',
    seed=42
)

print(f"Coherence: {coherence:.6f} ± {coherence_std:.6f}")
```

### 시간 시리즈 시뮬레이션

```python
import numpy as np

# 시간 포인트 정의
time_points = np.linspace(0, params.total_time, 100)

# 여러 시간 포인트에서 coherence 계산
coherence_series, coherence_std_series = sim.simulate_coherence_time_series(
    tau_c,
    sequence='FID',
    time_points=time_points,
    seed=42
)

# 결과 플롯
import matplotlib.pyplot as plt
plt.plot(time_points * 1e6, coherence_series)
plt.xlabel('Time (µs)')
plt.ylabel('Coherence')
plt.show()
```

---

## 3. 실시간 모니터링 (Real-Time Monitoring)

### 기본 사용법

```python
from parameter_validation import SimulationParameters
from simulation_monitor import SimulationMonitor

# 파라미터 생성
params = SimulationParameters(system='Si_P', target_regime='motional_narrowing')

# 모니터 생성
monitor = SimulationMonitor(params)

# 검증 체크 실행
monitor.check_noise_amplitude()
monitor.check_simulation_time()
monitor.check_time_step(tau_c=0.1e-6)
monitor.check_memory_requirement()

# 결과 리포트
report = monitor.report()
```

### 시뮬레이션 중 모니터링

```python
# 시뮬레이션 루프에서 사용
tau_c_values = np.logspace(-7, -4, 20)  # 0.1 µs to 100 µs

for tau_c in tau_c_values:
    # Time step 검증
    if not monitor.check_time_step(tau_c):
        print(f"Warning: dt may be too large for tau_c = {tau_c*1e6:.3f} µs")
        continue
    
    # 시뮬레이션 실행
    coherence, std = sim.simulate_coherence_chunked(tau_c, seed=42)
    
    # T2 추출 (간단한 예시)
    # 실제로는 fitting을 사용해야 함
    # T2_measured = extract_T2(time_points, coherence_series)
    # monitor.check_T2_vs_literature(T2_measured)
```

---

## 4. 통합 워크플로우 (Integrated Workflow)

### 완전한 예시

```python
import numpy as np
from parameter_validation import SimulationParameters
from memory_efficient_sim import MemoryEfficientSimulation
from simulation_monitor import SimulationMonitor

def run_validated_simulation(system='Si_P', target_regime='motional_narrowing'):
    """
    검증된 파라미터로 전체 시뮬레이션 실행
    Run complete simulation with validated parameters
    """
    print("="*70)
    print("Step 1: Parameter Setup and Validation")
    print("="*70)
    
    # 1. 파라미터 설정 및 검증
    params = SimulationParameters(system=system, target_regime=target_regime)
    report = params.validate()
    
    if not report['valid']:
        print("ERROR: Parameter validation failed!")
        return None
    
    # 2. 모니터 초기화
    monitor = SimulationMonitor(params)
    
    # 3. 초기 검증
    if not monitor.check_noise_amplitude():
        print("ERROR: Noise amplitude validation failed!")
        return None
    
    if not monitor.check_simulation_time():
        print("WARNING: Simulation time may be insufficient!")
    
    # 4. 메모리 효율적 시뮬레이션 실행
    print("\n" + "="*70)
    print("Step 2: Running Memory-Efficient Simulation")
    print("="*70)
    
    sim = MemoryEfficientSimulation(params)
    
    # tau_c 범위 설정
    tau_c_values = np.logspace(
        np.log10(params.min_tau_c),
        np.log10(params.max_tau_c),
        20  # 20 points
    )
    
    results = {
        'tau_c': tau_c_values,
        'coherence_FID': [],
        'coherence_echo': [],
        'coherence_FID_std': [],
        'coherence_echo_std': []
    }
    
    for tau_c in tau_c_values:
        # Time step 검증
        monitor.check_time_step(tau_c)
        
        # FID simulation
        coherence_FID, std_FID = sim.simulate_coherence_chunked(
            tau_c, sequence='FID', seed=42
        )
        
        # Echo simulation
        coherence_echo, std_echo = sim.simulate_coherence_chunked(
            tau_c, sequence='Echo', seed=42
        )
        
        results['coherence_FID'].append(coherence_FID)
        results['coherence_echo'].append(coherence_echo)
        results['coherence_FID_std'].append(std_FID)
        results['coherence_echo_std'].append(std_echo)
    
    # 5. 최종 검증
    print("\n" + "="*70)
    print("Step 3: Final Validation")
    print("="*70)
    
    final_report = monitor.report()
    
    return results, params, final_report

# 실행
if __name__ == '__main__':
    results, params, report = run_validated_simulation(
        system='GaAs',  # GaAs는 더 빠르므로 테스트에 적합
        target_regime='motional_narrowing'
    )
```

---

## 5. 기존 코드와 통합 (Integration with Existing Code)

### profiles.yaml 업데이트

새로운 파라미터를 `profiles.yaml`에 반영:

```yaml
Si_P:
  # ... existing parameters ...
  
  # Validated parameters (from SimulationParameters)
  validated:
    B_rms: 3.214e-9  # 3.214 nT (from T2* = 2.5 ms)
    T_max: 12.5e-3   # 12.5 ms (5 × T2*)
    dt: 0.2e-9       # 0.2 ns
```

### simulate_materials.py 수정 예시

```python
from parameter_validation import SimulationParameters, validate_simulation_parameters

# 기존 코드에서 파라미터 로드
materials = load_profiles('profiles.yaml')

# 각 물질에 대해 검증
for material_name, profile in materials.items():
    # 현재 파라미터 추출
    B_rms_current = profile['OU']['B_rms']
    T_max_current = profile['T_max']
    
    # 검증 및 비교
    comparison = validate_simulation_parameters(
        system=material_name,
        target_regime='all',
        B_rms_current=B_rms_current,
        T_max_current=T_max_current
    )
    
    # 권장사항 출력
    if comparison['recommendations']:
        print(f"\n{material_name} 파라미터 권장사항:")
        for rec in comparison['recommendations']:
            print(f"  - {rec}")
```

---

## 6. 주의사항 (Important Notes)

### 메모리 제한

- **Si:P의 Motional-Narrowing regime**: 메모리 요구량이 매우 큼 (200 GB)
  - 해결책: `n_ensemble` 감소, `dt` 증가, 또는 `target_regime='quasi_static'` 사용

### 파라미터 선택

- **B_rms는 T2*에서 역산됨**: 문헌 T2* 값이 정확해야 함
- **T_max는 T2*의 5배 이상 필요**: 충분한 decay capture를 위해

### 청크 크기

- **Chunked processing**: 각 청크는 독립적으로 생성되므로, 매우 작은 tau_c에 대해서는 부정확할 수 있음
- **해결책**: `chunk_size_sec`를 명시적으로 설정하여 `tau_c`보다 크게 유지

---

## 7. 다음 단계 (Next Steps)

구현된 기능:
- ✅ 파라미터 검증 및 재설정
- ✅ 메모리 효율적 시뮬레이션
- ✅ 실시간 모니터링

추가로 구현할 기능 (우선순위 순):
1. **적응형 시뮬레이션** (Adaptive Simulation)
2. **개선된 T2 추출** (Improved T2 Extraction)
3. **Regime-aware Bootstrap**

이 기능들은 `IMPROVEMENTS_PLAN.md`에 상세히 설명되어 있습니다.

