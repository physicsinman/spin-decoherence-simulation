#!/bin/bash
# 전체 시뮬레이션 재실행 스크립트 (개선된 파라미터)

echo "=========================================="
echo "전체 시뮬레이션 재실행 (개선된 파라미터)"
echo "=========================================="
echo ""
echo "개선 사항:"
echo "  - QS regime: T_max = 50-100×T2 (기존: 30×)"
echo "  - Crossover: 포인트 증가 (24 → 30)"
echo "  - QS regime: 포인트 증가 (20 → 25)"
echo "  - Timestep: 더 정밀 (100 → 150 steps/tau_c)"
echo "  - Crossover: T_max = 15×T2 (기존: 10×)"
echo ""
echo "예상 시간: 2-3시간"
echo ""
read -p "계속하시겠습니까? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# 1. FID 전체 sweep
echo ""
echo "=========================================="
echo "1/4: FID 전체 sweep"
echo "=========================================="
python3 run_fid_sweep.py

# 2. Echo 전체 sweep (선택적)
echo ""
echo "=========================================="
echo "2/4: Echo 전체 sweep"
echo "=========================================="
read -p "Echo sweep도 실행하시겠습니까? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python3 run_echo_sweep.py
else
    echo "Echo sweep 건너뜀"
fi

# 3. Echo gain 분석
echo ""
echo "=========================================="
echo "3/4: Echo gain 분석"
echo "=========================================="
python3 analyze_echo_gain.py

# 4. Convergence test (선택적)
echo ""
echo "=========================================="
echo "4/4: Convergence test"
echo "=========================================="
read -p "Convergence test도 실행하시겠습니까? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python3 run_convergence_test.py
else
    echo "Convergence test 건너뜀"
fi

# 5. 그림 생성
echo ""
echo "=========================================="
echo "그림 생성"
echo "=========================================="
python3 generate_dissertation_plots.py

echo ""
echo "=========================================="
echo "✅ 전체 시뮬레이션 완료!"
echo "=========================================="

