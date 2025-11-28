#!/bin/bash

LOG_FILE="echo_curves_high_accuracy.log"

echo "=== Echo Curves Simulation Status ==="

# Check if process is running
PID=$(ps aux | grep "python.*sim_echo_curves.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo -e "\n✅ Process is RUNNING"
    ps aux | grep "$PID" | grep -v grep | awk '{print "   PID: "$2" CPU: "$3"% MEM: "$4"% Time: "$10}'
    echo -e "\n📝 Log file: $LOG_FILE ($(wc -l < "$LOG_FILE" 2>/dev/null || echo 0) lines)"
    echo "⏳ Simulation in progress..."
else
    echo -e "\n❌ Process is NOT running"
    echo -e "\n📝 Log file: $LOG_FILE ($(wc -l < "$LOG_FILE" 2>/dev/null || echo 0) lines)"
    if grep -q "All curves saved" "$LOG_FILE" 2>/dev/null; then
        echo "✅ Simulation COMPLETED!"
        grep "All curves saved" "$LOG_FILE"
    else
        echo "⚠️ Simulation may have FAILED or not started correctly."
    fi
fi

echo -e "\nLast 10 lines:"
tail -n 10 "$LOG_FILE" 2>/dev/null || echo "Log file not found"

echo -e "\n📊 Generated files:"
ls -lh results/echo_tau_c_*.csv 2>/dev/null | tail -5 || echo "No echo curve files found"

echo -e "\nTo monitor progress in real-time:"
echo "  tail -f $LOG_FILE"

