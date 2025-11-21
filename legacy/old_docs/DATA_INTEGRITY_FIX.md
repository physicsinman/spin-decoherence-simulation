# 데이터 무결성 수정 사항

## 중요 변경 사항

### 1. 자동 필터링 제거 ✅
- **이전**: 물리학적으로 타당하지 않은 포인트를 자동으로 제거
- **현재**: 경고만 표시하고 데이터는 그대로 유지
- **이유**: 과학적 무결성 - 데이터를 임의로 숨기는 것은 부정행위

### 2. Convergence Test CI Width 문제 처리 ✅
- **문제**: CI width가 N_traj 증가에 따라 증가 (물리학적으로 불가능)
- **원인**: Bootstrap 계산 오류 또는 degenerate CI
- **해결**: 
  - CI width가 증가하는 경우 그래프에 표시하지 않음
  - 대신 경고 메시지 표시
  - 데이터는 그대로 유지 (수정하지 않음)

## 수정된 코드

### Echo Gain 그래프
```python
# 이전: 자동 필터링
valid_physical = valid_physical.drop(drop_indices)

# 현재: 경고만 표시
print(f"⚠️  WARNING: {n} points show unphysical behavior")
# 데이터는 그대로 유지
```

### Convergence Test 그래프
```python
# CI width가 증가하는 경우 표시하지 않음
if not is_decreasing:
    print(f"❌ CRITICAL: CI width INCREASING with N_traj")
    # 그래프에 경고 텍스트 표시
    ax.text(..., '⚠️ CI width data invalid')
    continue  # CI width 플롯 스킵
```

## 과학적 원칙

1. **데이터 무결성**: 측정된 데이터를 임의로 숨기지 않음
2. **투명성**: 문제가 있는 데이터는 명확히 표시
3. **재현성**: 모든 데이터는 추적 가능해야 함

## 남은 문제

1. **Convergence Test CI Width**: Bootstrap 계산 문제
   - 해결 방법: Bootstrap 알고리즘 개선 필요
   - 또는 CI width 대신 다른 통계량 사용

2. **Echo Gain 물리적 불일치**: 일부 포인트에서 gain이 감소
   - 해결 방법: 해당 포인트 재시뮬레이션 (이미 완료)
   - 또는 논문에서 명시적으로 언급

