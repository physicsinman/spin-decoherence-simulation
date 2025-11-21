# Echo 시뮬레이션 문제 요약 및 해결 방안

## 🔴 심각한 문제점

### 1. 포인트 불일치 (매우 심각)
- **FID와 Echo가 다른 tau_c grid 사용**
- 둘 다 있는 포인트: 25개만 (73개 중)
- FID만: 48개, Echo만: 58개
- **결과**: Echo gain 계산 불가능 (매칭 안 됨)

### 2. Echo Fitting 실패 (심각)
- **R² < 0**: 14개 포인트 (fitting 완전 실패)
- **R² 없음**: 10개 포인트
- **평균 R²**: 0.6928 (FID는 0.9414)
- **R² < 0.9**: 16개 포인트

### 3. Echo Gain 문제
- **NaN**: 6개 포인트
- **gain < 1**: 4개 포인트 (물리적으로 불가능)

## 🔍 원인 분석

### 포인트 불일치 원인
1. `run_echo_sweep.py`가 `run_fid_sweep.py`와 다른 tau_c grid 생성
2. Echo sweep이 FID sweep 이후에 실행되어 다른 포인트 사용 가능

### Echo Fitting 실패 원인
1. **QS regime에서 flat curve**: Echo decay가 거의 없어서 fitting 실패
2. **T_max_echo 부족**: Echo decay를 충분히 관측하지 못함
3. **Window selection 문제**: Echo-optimized window가 너무 보수적

## ✅ 해결 방안

### 1. 포인트 동기화 (최우선)
- FID sweep의 tau_c grid를 저장하고 Echo sweep에서 재사용
- 또는 두 sweep을 동시에 실행하여 같은 grid 보장

### 2. Echo Fitting 개선
- Flat curve detection 개선
- 더 robust한 fitting 방법
- T_max_echo 추가 증가

### 3. Echo Gain 계산 개선
- FID와 Echo를 tau_c로 정확히 매칭
- Nearest neighbor matching으로 보완

## 🚀 즉시 실행 가능한 해결책

### 옵션 1: FID grid 재사용 (권장)
```python
# run_echo_sweep.py에서
df_fid = pd.read_csv('results_comparison/t2_vs_tau_c.csv')
tau_c_sweep = df_fid['tau_c'].values  # FID와 동일한 grid 사용
```

### 옵션 2: 동시 실행
- FID와 Echo를 같은 스크립트에서 실행하여 같은 grid 보장

### 옵션 3: Echo만 재실행
- FID grid를 사용하여 Echo만 재실행

