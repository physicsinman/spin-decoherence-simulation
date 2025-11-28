#!/bin/bash
# Echo sweep simulation status checker

LOG_FILE="echo_sweep_improved_v3.log"
RESULT_FILE="results/t2_echo_vs_tau_c.csv"

echo "=== Echo Sweep Simulation Status ==="
echo ""

# Check if process is running
if pgrep -f "sim_echo_sweep.py" > /dev/null; then
    echo "✅ Process is RUNNING"
    ps aux | grep sim_echo_sweep.py | grep -v grep | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "Time:", $10}'
else
    echo "❌ Process is NOT running"
fi

echo ""

# Check log file
if [ -f "$LOG_FILE" ]; then
    LOG_LINES=$(wc -l < "$LOG_FILE")
    echo "📝 Log file: $LOG_FILE ($LOG_LINES lines)"
    
    # Check for completion message
    if grep -q "Results saved to" "$LOG_FILE"; then
        echo "✅ Simulation COMPLETED!"
        echo ""
        tail -10 "$LOG_FILE" | grep -A 10 "Results saved"
    else
        echo "⏳ Simulation in progress..."
        echo ""
        echo "Last 5 lines:"
        tail -5 "$LOG_FILE"
    fi
else
    echo "⚠️  Log file not found"
fi

echo ""

# Check result file
if [ -f "$RESULT_FILE" ]; then
    RESULT_LINES=$(wc -l < "$RESULT_FILE")
    echo "📊 Result file: $RESULT_FILE ($RESULT_LINES lines)"
    
    # Check if file was recently modified (within last 5 minutes)
    if [ $(find "$RESULT_FILE" -mmin -5 | wc -l) -gt 0 ]; then
        echo "   ⚠️  File was modified recently (may still be updating)"
    else
        echo "   ✅ File appears stable"
    fi
else
    echo "⚠️  Result file not found yet"
fi

echo ""
echo "To monitor progress in real-time:"
echo "  tail -f $LOG_FILE"

