# FID Fitting Issues 발견

## 🔴 Critical Finding: FID Fitting 실패가 Echo Gain 문제의 원인

### 발견 사항

**τc = 0.300 μs에서 FID fitting이 실패했습니다:**

```
τc = 0.257 μs: R² = 0.9992 (좋음)
τc = 0.300 μs: R² = 0.7494 (실패!) ← Echo gain 급격한 변화의 원인
τc = 0.350 μs: R² = 0.7005 (실패!)
```

### 영향

1. **잘못된 T2_FID 값**: R² < 0.75이면 fitting이 신뢰할 수 없음
2. **Echo gain 계산 오류**: T2_FID가 잘못되었으므로 gain도 잘못됨
3. **비물리적 변동**: 실제 echo gain이 변한 것이 아니라, FID 측정 오류

### 해결 방안

#### Option 1: 해당 포인트 재시뮬레이션 (권장)
- τc = 0.300, 0.350 μs 재시뮬레이션
- N_traj 증가 (2000 → 5000)
- T_max 증가 (더 긴 decay 관찰)

#### Option 2: R² < 0.9인 포인트 제외
- Echo gain 계산에서 제외
- Discussion에서 언급: "Points with R² < 0.9 were excluded due to poor fitting quality"

#### Option 3: Discussion에서 명시적 언급
- "Some points in the crossover regime show lower fitting quality (R² < 0.8) due to statistical fluctuations. These points are marked in the figure."

### 권장 사항

**Before Meeting:**
1. R² < 0.8인 포인트를 Echo gain 그래프에서 제외하거나 명시적으로 표시
2. Discussion에서 언급: "FID fitting quality varies in crossover regime"

**After Meeting:**
1. R² < 0.9인 포인트 재시뮬레이션 (N_traj 증가)

