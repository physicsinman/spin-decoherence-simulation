# 📝 실행 명령어 정리

## 🎯 가장 간단한 방법 (권장)

```bash
python run_all.py
```

**이것만 실행하면 모든 시뮬레이션이 자동으로 실행됩니다!**

---

## 📋 단계별 명령어

### **1단계: FID 시뮬레이션**

```bash
# FID 전체 sweep (20 포인트)
python sim_fid_sweep.py

# FID 대표 곡선 (4개 파일)
python sim_fid_curves.py

# Motional Narrowing 분석
python analyze_mn.py
```

### **2단계: Hahn Echo 시뮬레이션**

```bash
# Echo 전체 sweep (20 포인트)
python sim_echo_sweep.py

# Echo 대표 곡선 (4개 파일)
python sim_echo_curves.py

# Echo Gain 분석
python3 analyze_echo_gain.py
```

### **3단계: 노이즈 예제**

```bash
# 노이즈 궤적 예제 생성
python generate_noise_data.py
```

### **선택적 (Optional)**

```bash
# Bootstrap 분포 분석
python3 run_bootstrap.py

# Convergence 테스트
python3 run_convergence_test.py
```

---

## 🔄 실행 순서 요약

```
1. run_fid_sweep.py          (~1-2시간)
2. run_fid_curves.py         (~10분)
3. analyze_motional_narrowing.py (즉시)
4. run_echo_sweep.py         (~1-2시간)
5. run_echo_curves.py        (~10분)
6. analyze_echo_gain.py       (즉시)
7. generate_noise_examples.py (즉시)
```

**총 예상 시간: ~3-4시간**

---

## ✅ 결과 확인

```bash
# 생성된 파일 확인
ls -lh results/

# CSV 파일 확인
ls results/*.csv

# 텍스트 파일 확인
ls results/*.txt
```

---

## 💡 팁

- **중단된 경우:** 각 스크립트를 개별적으로 실행 가능
- **빠른 테스트:** `N_traj`를 줄이거나 `tau_c_npoints`를 줄이세요
- **논문용:** `run_fid_curves.py`와 `run_echo_curves.py`에서 7개 포인트 옵션 활성화

---

**더 자세한 내용은 `QUICK_START.md` 참고**

