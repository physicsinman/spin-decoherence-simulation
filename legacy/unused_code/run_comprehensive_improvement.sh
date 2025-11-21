#!/bin/bash
# Comprehensive Improvement - All-in-One
# 시뮬레이션 정당성과 그래프 품질 전반적 개선

set -e

echo "=================================================================================="
echo "Comprehensive Improvement: 시뮬레이션 정당성 & 그래프 품질 향상"
echo "=================================================================================="
echo ""
echo "This will:"
echo "  1. Re-simulate all FID points with R² < 0.95 (N_traj=5000, improved T_max)"
echo "  2. Re-simulate problematic echo points"
echo "  3. Improve convergence test"
echo "  4. Regenerate all graphs with maximum quality"
echo ""
echo "⚠️  Estimated time: 6-10 hours"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

START_TIME=$(date +%s)

# Step 1: Comprehensive simulation improvement
echo ""
echo "=================================================================================="
echo "Step 1: Comprehensive Simulation Improvement"
echo "=================================================================================="
python3 comprehensive_improvement.py
if [ $? -ne 0 ]; then
    echo "❌ Comprehensive improvement failed!"
    exit 1
fi

# Step 2: Re-analyze echo gain
echo ""
echo "=================================================================================="
echo "Step 2: Re-analyzing Echo Gain"
echo "=================================================================================="
python3 analyze_echo_gain.py

# Step 3: Final validation
echo ""
echo "=================================================================================="
echo "Step 3: Final Validation"
echo "=================================================================================="
python3 final_validation.py

# Step 4: Regenerate all graphs with maximum quality
echo ""
echo "=================================================================================="
echo "Step 4: Regenerating All Graphs (Maximum Quality)"
echo "=================================================================================="
python3 generate_dissertation_plots.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "=================================================================================="
echo "✅ Comprehensive improvement completed!"
echo "=================================================================================="
echo "Total elapsed time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Check results_comparison/figures/ for updated graphs"
echo "=================================================================================="

