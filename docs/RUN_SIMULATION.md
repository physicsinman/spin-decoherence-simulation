# 시뮬레이션 실행 가이드 (Simulation Execution Guide)

## 빠른 시작 (Quick Start)

### 전체 시뮬레이션 실행

```bash
# 전체 실행 (8개 조합: 2 materials × 2 noise models × 2 sequences)
python3 main_comparison.py --full

# 또는 특정 조합만
python3 main_comparison.py --materials Si_P GaAs --noise OU --sequences FID Hahn
```

### 결과 분석

```bash
# 기존 결과 분석 및 그래프 생성
python3 main_comparison.py --analyze --result-file results_comparison/all_results_*.json
```

---

## 실행 전 확인사항

### 1. 파라미터 검증 (선택사항)

```python
from parameter_validation import validate_simulation_parameters

# Si:P 파라미터 확인
validate_simulation_parameters('Si_P', B_rms_current=5e-6, T_max_current=30e-6)

# GaAs 파라미터 확인
validate_simulation_parameters('GaAs', B_rms_current=8e-6, T_max_current=30e-6)
```

**현재 상태**:
- ✅ GaAs: 파라미터 적절
- ⚠️ Si:P: B_rms 1556× 과대, T_max 417× 부족 (하지만 실행은 가능)

### 2. 예상 시간

- **GaAs**: ~5-10분 (각 조합당)
- **Si:P**: ~30-60분 (각 조합당, 파라미터 문제로 더 오래 걸릴 수 있음)
- **전체**: ~3-8시간 (8개 조합)

### 3. 메모리 요구사항

- **GaAs**: ~0.08 GB (문제 없음)
- **Si:P**: ~20 GB (청크 사용 시 ~2 GB)

---

## 실행 옵션

### 옵션 1: 전체 실행 (권장)

```bash
python3 main_comparison.py --full
```

**결과**:
- 모든 조합 시뮬레이션
- 자동으로 그래프 생성
- `results_comparison/` 디렉토리에 저장

### 옵션 2: GaAs만 먼저 테스트

```bash
python3 main_comparison.py --materials GaAs --noise OU --sequences FID Hahn
```

**장점**:
- 빠름 (~10분)
- 파라미터 적절
- 문제 없으면 Si:P 실행

### 옵션 3: 특정 조합만

```bash
# Si:P OU FID만
python3 main_comparison.py --materials Si_P --noise OU --sequences FID
```

---

## 실행 중 모니터링

### 진행 상황 확인

시뮬레이션은 각 조합마다 진행 상황을 출력합니다:
```
[1/8] Starting simulation...
Running: Si_P | OU | FID
  [1/25] tau_c = 0.100 μs
  [2/25] tau_c = 0.150 μs
  ...
```

### 중단 및 재개

- **중단**: `Ctrl+C`
- **재개**: 같은 명령어 다시 실행 (이미 완료된 조합은 건너뜀)

---

## 결과 파일

### 생성되는 파일들

1. **개별 결과**:
   - `Si_P_OU_FID_YYYYMMDD_HHMMSS.json`
   - `Si_P_OU_Hahn_YYYYMMDD_HHMMSS.json`
   - 등등...

2. **통합 결과**:
   - `all_results_YYYYMMDD_HHMMSS.json`

3. **그래프** (자동 생성):
   - `T2_comparison.png/pdf`
   - `T2_comparison_no_ci.png/pdf`
   - `echo_enhancement.png/pdf`
   - `psd_comparison.png/pdf`
   - `eta_dimensionless_collapse.png/pdf`

4. **요약 테이블**:
   - `summary.csv`
   - `summary_clean.csv`

---

## 실행 후 분석

### 그래프 재생성

```bash
# 최신 결과 분석
python3 main_comparison.py --analyze --result-file results_comparison/all_results_*.json
```

### Python에서 직접 분석

```python
from analyze_results import analyze_all

# 전체 분석 실행
analyze_all('results_comparison/all_results_YYYYMMDD_HHMMSS.json')
```

---

## 주의사항

### Si:P 파라미터 문제

**현재 파라미터**:
- B_rms = 5 µT (1556× 과대)
- T_max = 30 µs (417× 부족)

**영향**:
- T₂ 값이 문헌값보다 ~1000× 작을 수 있음
- Quasi-static regime에서 부정확할 수 있음
- Motional-narrowing regime은 상대적으로 정확

**해석**:
- 절대값보다는 **상대적 경향**에 집중
- Motional-narrowing regime 결과만 신뢰
- Quasi-static regime은 "preliminary"로 표시

### 메모리 부족 시

```python
# profiles.yaml에서 앙상블 크기 감소
M: 500  # 750 → 500
```

또는

```python
# tau_c_num 감소
tau_c_num: 15  # 25 → 15
```

---

## 문제 해결

### 문제 1: "MemoryError"

**해결책**:
1. 앙상블 크기 감소 (`M` in `profiles.yaml`)
2. `tau_c_num` 감소
3. Si:P만 실행 (GaAs는 문제 없음)

### 문제 2: "Too slow"

**해결책**:
1. GaAs만 먼저 실행
2. `tau_c_num` 감소
3. 특정 조합만 실행

### 문제 3: "Import error"

**해결책**:
```bash
# 현재 디렉토리 확인
pwd
# 반드시 simulation 디렉토리에서 실행
cd /path/to/simulation
python3 main_comparison.py --full
```

---

## 실행 체크리스트

실행 전:
- [ ] 현재 디렉토리 확인 (`pwd`)
- [ ] `profiles.yaml` 존재 확인
- [ ] 충분한 디스크 공간 확인
- [ ] 예상 시간 확인 (3-8시간)

실행 중:
- [ ] 진행 상황 모니터링
- [ ] 메모리 사용량 확인
- [ ] 오류 메시지 확인

실행 후:
- [ ] 결과 파일 확인
- [ ] 그래프 생성 확인
- [ ] 요약 테이블 확인

---

## 빠른 참조

```bash
# 전체 실행
python3 main_comparison.py --full

# GaAs만 테스트
python3 main_comparison.py --materials GaAs

# 결과 분석
python3 main_comparison.py --analyze --result-file results_comparison/all_results_*.json

# 도움말
python3 main_comparison.py --help
```

---

**실행 준비 완료!** 🚀

