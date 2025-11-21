#!/bin/bash
# Post-Meeting Improvements - All-in-One Script
# 모든 개선 작업을 순차적으로 실행

set -e  # 에러 발생 시 중단

echo "=================================================================================="
echo "Post-Meeting Improvements - All-in-One Execution"
echo "=================================================================================="
echo ""
echo "This will run all improvement scripts sequentially:"
echo "  1. FID re-simulation (R² < 0.8 points)"
echo "  2. Echo re-simulation (problematic points)"
echo "  3. Echo gain re-analysis"
echo "  4. Convergence test improvement"
echo "  5. Final validation"
echo "  6. Graph regeneration"
echo ""
echo "Estimated total time: ~5-6 hours"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

# Start time
START_TIME=$(date +%s)

# 1. FID 재시뮬레이션
echo ""
echo "=================================================================================="
echo "Step 1/6: Re-running FID simulations for poor fitting points"
echo "=================================================================================="
python3 rerun_poor_fid_points.py
if [ $? -ne 0 ]; then
    echo "❌ FID re-simulation failed!"
    exit 1
fi
echo "✅ Step 1 completed"

# 2. Echo 재시뮬레이션
echo ""
echo "=================================================================================="
echo "Step 2/6: Re-running Echo simulations for problematic points"
echo "=================================================================================="
python3 rerun_echo_problem_points.py
if [ $? -ne 0 ]; then
    echo "❌ Echo re-simulation failed!"
    exit 1
fi
echo "✅ Step 2 completed"

# 3. Echo gain 재분석
echo ""
echo "=================================================================================="
echo "Step 3/6: Re-analyzing Echo gain"
echo "=================================================================================="
python3 analyze_echo_gain.py
if [ $? -ne 0 ]; then
    echo "⚠️  Echo gain analysis failed (non-critical)"
fi
echo "✅ Step 3 completed"

# 4. Convergence test 개선
echo ""
echo "=================================================================================="
echo "Step 4/6: Improving convergence test"
echo "=================================================================================="
python3 improve_convergence_test.py
if [ $? -ne 0 ]; then
    echo "❌ Convergence test improvement failed!"
    exit 1
fi
echo "✅ Step 4 completed"

# 5. 최종 검증
echo ""
echo "=================================================================================="
echo "Step 5/6: Final validation"
echo "=================================================================================="
python3 final_validation.py
if [ $? -ne 0 ]; then
    echo "⚠️  Validation completed with warnings"
fi
echo "✅ Step 5 completed"

# 6. 그래프 재생성
echo ""
echo "=================================================================================="
echo "Step 6/6: Regenerating all figures"
echo "=================================================================================="
python3 generate_dissertation_plots.py
if [ $? -ne 0 ]; then
    echo "❌ Graph generation failed!"
    exit 1
fi
echo "✅ Step 6 completed"

# End time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=================================================================================="
echo "✅ All improvements completed successfully!"
echo "=================================================================================="
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Next steps:"
echo "  1. Check results_comparison/figures/ for updated graphs"
echo "  2. Review final_validation.py output for quality metrics"
echo "  3. Verify all improvements meet expectations"
echo "=================================================================================="

