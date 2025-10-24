# Multi-Model LLM Fine-Tuning Guide

## Overview
This setup allows you to efficiently compare 4-5 LLMs sequentially with automatic resource management and results aggregation.

## Files Created
- **`model_configs.json`** - Configuration for all models to test
- **`rct_ft_single.py`** - Refactored single-model training script
- **`multi_model_runner.py`** - Master orchestration script
- **`analyze_results.py`** - Results analysis and visualization

## Quick Start

### 1. Configure Your Models
Edit `model_configs.json` to add/modify models:

```json
{
  "models": [
    {
      "name": "Llama-3.2-3B",
      "model_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "output_dir": "./finetuned-llama-3.2-3B",
      "lora_rank": 8,
      "lora_alpha": 16,
      "learning_rate": 9e-7,
      "num_iters": 600,
      "batch_size": 4
    }
  ]
}
```

**Suggested Models to Try:**
- `mlx-community/Llama-3.2-3B-Instruct-4bit` (3B params - balanced)
- `mlx-community/Llama-3.2-1B-Instruct-4bit` (1B params - fast)
- `mlx-community/gemma-2-2b-it-4bit` (2B params - Google)
- `mlx-community/Phi-3-mini-4k-instruct-4bit` (3.8B params - Microsoft)
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (7B params - larger, use batch_size=2)

### 2. Run All Models
```bash
# Run all models in config file
python multi_model_runner.py

# Run with options
python multi_model_runner.py --skip-baseline --skip-classical-ml

# Run specific models only
python multi_model_runner.py --models "Llama-3.2-3B" "Gemma-2-2B"

# Custom sleep time between models (default 30s)
python multi_model_runner.py --sleep-between 60
```

### 3. Analyze Results
```bash
# Analyze most recent results
python analyze_results.py

# Analyze specific file
python analyze_results.py results/combined_results_20241022_143025.json
```

## Resource Conservation Features

### Automatic Memory Cleanup
- Clears Python garbage collection between models
- Clears MLX Metal cache (GPU memory)
- Configurable sleep time between runs

### Resource Monitoring
```bash
# Install optional monitoring (recommended)
pip install psutil
```

With psutil installed, you'll see:
- Process memory usage
- System memory percentage
- Available memory in GB

### Sequential Execution
Models run one at a time to prevent:
- Memory overflow
- System crashes
- GPU conflicts

## Workflow for 3-Day Comparison

### Day 1: Setup & Quick Test
```bash
# Test with smallest model and reduced iterations
# Edit model_configs.json - set num_iters: 200 for testing
python multi_model_runner.py --models "Llama-3.2-1B"

# If successful, reset num_iters to 600 and run overnight
python multi_model_runner.py
```

### Day 2: Run Remaining Models
```bash
# Check what completed
ls -la ./finetuned-*/

# Run any failed/remaining models
python multi_model_runner.py --models "Gemma-2-2B" "Phi-3-Mini"
```

### Day 3: Analysis
```bash
# Analyze results
python analyze_results.py

# Generate combined report
python analyze_results.py > final_comparison.txt
```

## Time Estimates

Approximate times per model (600 iterations, batch_size=4):
- **1B model**: ~30-45 minutes
- **2-3B model**: ~45-75 minutes
- **7B model**: ~90-150 minutes

**Total for 5 models (1B, 3B, 2B, 3B, 7B)**: ~5-7 hours

## Tips for Efficiency

### 1. Skip Redundant Baselines
Classical ML (LASSO/RF) is the same across all models:
```bash
python multi_model_runner.py --skip-classical-ml
```

### 2. Run Overnight
Start before bed:
```bash
# Run in background with output logging
nohup python multi_model_runner.py > training.log 2>&1 &

# Check progress
tail -f training.log
```

### 3. Stagger Your Schedule
- **Evening**: Start largest model (7B)
- **Overnight**: Run 3-4 medium models
- **Morning**: Analyze results, run any remaining

### 4. Test First
Always test with reduced iterations before full run:
```json
"num_iters": 200  // Test
"num_iters": 600  // Production
```

### 5. Monitor Resources
```bash
# In another terminal, monitor system
watch -n 5 'ps aux | grep python | head -5'

# Or use Activity Monitor on macOS
open -a "Activity Monitor"
```

## Output Structure

```
./results/
├── combined_results_20241022_143025.json    # All models aggregated
├── Llama-3.2-3B_20241022_140000.json       # Individual results
├── Gemma-2-2B_20241022_142030.json
└── ...

./finetuned-llama-3.2-3B/
├── adapters.safetensors                     # Fine-tuned weights
├── data/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── ...
```

## Troubleshooting

### Out of Memory
1. Reduce `batch_size` to 2 or 1
2. Reduce `lora_rank` to 4
3. Increase `--sleep-between` to 60+
4. Close other applications

### Model Fails to Load
```bash
# Clear any cached models
rm -rf ~/.cache/huggingface/

# Re-download
python -c "from mlx_lm.utils import load; load('mlx-community/Llama-3.2-3B-Instruct-4bit')"
```

### Training Hangs
- Check `training.log` for errors
- Ensure data file exists: `Trial 9/trial9.csv`
- Verify MLX installation: `pip install -U mlx-lm`

## Advanced Usage

### Run Single Model Directly
```bash
python rct_ft_single.py \
    --model "mlx-community/Llama-3.2-3B-Instruct-4bit" \
    --output-dir "./custom-output" \
    --num-iters 600 \
    --batch-size 4
```

### Custom Configuration File
```bash
python multi_model_runner.py --config my_custom_config.json
```

### Results Analysis Options
```bash
# Just see rankings
python analyze_results.py | grep -A 20 "MODEL RANKINGS"

# Export to file
python analyze_results.py > comparison_report.txt
```

## What Changed from Original Script

### Minimal Changes
The refactoring was minimal - your original logic is preserved:

1. **Added**: Command-line argument parsing
2. **Added**: Memory cleanup functions
3. **Added**: Results JSON export
4. **Added**: Skip options for efficiency
5. **Kept**: All original training logic, metrics, and evaluation

### Original Script Still Works
Your original `rct_ft_llama.py` remains unchanged and functional!

## Performance Comparison Table
After running, you'll get tables like:

```
Model                      Accuracy    Precision   Recall      Specificity F1          Balance Gap
--------------------------------------------------------------------------------------------------------
Llama-3.2-3B              85.5%       83.2%       78.9%       88.3%       81.0%       9.4%
Gemma-2-2B                83.2%       80.5%       82.1%       84.0%       81.3%       1.9%       ← Best Balance
Phi-3-Mini                84.8%       82.0%       79.5%       87.2%       80.7%       7.7%
```

## Next Steps

1. **Edit** `model_configs.json` with your 4-5 models
2. **Test** with one small model first
3. **Run** overnight: `nohup python multi_model_runner.py > training.log 2>&1 &`
4. **Analyze** in the morning: `python analyze_results.py`
5. **Choose** best model for your clinical use case

Good luck with your comparisons!
