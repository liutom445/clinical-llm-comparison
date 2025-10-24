# Quick Start: Compare 4-5 LLMs in 3 Days

## Setup (5 minutes)

### 1. Edit Model Configuration
Open [`model_configs.json`](model_configs.json) and customize:

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
    // Add 3-4 more models here
  ]
}
```

**Recommended Models:**
- ✓ Already added: Llama-3.2-3B, Llama-3.2-1B, Gemma-2-2B
- To add: Phi-3-Mini, Mistral-7B (see README_MULTI_MODEL.md)

### 2. Optional: Install Resource Monitor
```bash
pip install psutil
```

## Day 1: Test & Launch (Evening)

### Test Run (5 minutes)
```bash
# Test with smallest model
python multi_model_runner.py --models "Llama-3.2-1B"
```

### Full Run (Overnight)
```bash
# Start all models
nohup python multi_model_runner.py > training.log 2>&1 &

# Check it started
tail -20 training.log
```

## Day 2: Monitor & Adjust (Morning/Evening)

### Check Progress
```bash
# Quick status
bash check_progress.sh

# Or watch live
tail -f training.log
```

### If Some Failed
```bash
# Run only failed models
python multi_model_runner.py --models "ModelName1" "ModelName2"
```

## Day 3: Analyze Results (Morning)

### Run Analysis
```bash
python analyze_results.py
```

### Save Report
```bash
python analyze_results.py > final_comparison.txt
```

## One-Line Commands

```bash
# Full run (background)
nohup python multi_model_runner.py > training.log 2>&1 &

# Check progress
bash check_progress.sh

# Analyze results
python analyze_results.py

# Fast mode (skip baselines)
python multi_model_runner.py --skip-baseline --skip-classical-ml
```

## File Summary

| File | Purpose |
|------|---------|
| `model_configs.json` | **EDIT THIS** - Add your 4-5 models |
| `multi_model_runner.py` | Run all models automatically |
| `rct_ft_single.py` | Single model script (called by runner) |
| `analyze_results.py` | Generate comparison reports |
| `check_progress.sh` | Monitor running jobs |
| `README_MULTI_MODEL.md` | Full documentation |

## Expected Timeline

- **Setup**: 5 minutes
- **Testing**: 30-45 minutes
- **Full training**: 5-7 hours (overnight)
- **Analysis**: 5 minutes

**Total**: ~1 day of actual work, 2-3 days elapsed

## Troubleshooting

### Out of Memory?
```bash
# Edit model_configs.json, reduce batch_size to 2
# Or reduce num_iters to 400
```

### Training Stopped?
```bash
# Check log
tail -50 training.log

# Restart remaining models
python multi_model_runner.py
# (will skip already completed models)
```

### Can't Find Results?
```bash
ls -la ./results/
python analyze_results.py --results-dir ./results
```

## What You'll Get

1. **Comparison Table** showing all metrics across models
2. **Rankings** for each metric (accuracy, precision, recall, etc.)
3. **Recommendations** for clinical use cases
4. **Improvement Analysis** (baseline vs fine-tuned)

## Your Original Script

Your original [`rct_ft_llama.py`](rct_ft_llama.py) still works unchanged!

This is just an efficient wrapper for comparing multiple models.

---

**Ready?** Edit `model_configs.json` and run:
```bash
python multi_model_runner.py
```

Good luck! 🚀
