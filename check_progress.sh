#!/bin/bash

# Progress Checker for Multi-Model Fine-Tuning
# Usage: bash check_progress.sh

echo "================================"
echo "Multi-Model Training Progress"
echo "================================"
echo

# Check if training is running
if pgrep -f "multi_model_runner.py" > /dev/null; then
    echo "✓ Training is RUNNING"
    echo

    # Show current Python processes
    echo "Active processes:"
    ps aux | grep -E "(multi_model_runner|rct_ft_single)" | grep -v grep | awk '{print "  ", $11, $12, $13, $14}'
    echo
else
    echo "○ No training running"
    echo
fi

# Check completed models
echo "Completed models:"
echo "----------------"
for dir in ./finetuned-*/; do
    if [ -d "$dir" ]; then
        model_name=$(basename "$dir")
        if [ -f "$dir/adapters.safetensors" ]; then
            echo "  ✓ $model_name"
        else
            echo "  ○ $model_name (incomplete)"
        fi
    fi
done
echo

# Check results
echo "Results files:"
echo "-------------"
if [ -d "./results" ]; then
    result_count=$(ls -1 ./results/*.json 2>/dev/null | wc -l | xargs)
    if [ "$result_count" -gt 0 ]; then
        echo "  $result_count result files found"
        echo
        echo "  Most recent:"
        ls -lt ./results/*.json | head -3 | awk '{print "    ", $9, "("$6, $7, $8")"}'
    else
        echo "  No results yet"
    fi
else
    echo "  Results directory not found"
fi
echo

# Memory usage
echo "System resources:"
echo "----------------"
if command -v free &> /dev/null; then
    free -h | grep Mem | awk '{print "  Memory: "$3" used / "$2" total ("$3/$2*100"%)"}'
elif command -v vm_stat &> /dev/null; then
    # macOS
    vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages active:\s+(\d+)/ and printf("  Active Memory: %.2f GB\n", $1 * $size / 1073741824); /Pages free:\s+(\d+)/ and printf("  Free Memory: %.2f GB\n", $1 * $size / 1073741824);'
fi

# Disk usage for outputs
du -sh ./finetuned-* 2>/dev/null | awk '{print "  ", $0}' || echo "  No model directories yet"
echo

# Tail of log if it exists
if [ -f "training.log" ]; then
    echo "Latest log output:"
    echo "-----------------"
    tail -20 training.log | sed 's/^/  /'
    echo
    echo "Full log: tail -f training.log"
fi

echo "================================"
echo "Check complete!"
echo "================================"
