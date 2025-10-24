# Your Optimized Model Configuration (24GB RAM)

## ✅ Configuration Updated!

I've configured **4 models** optimized for your **24GB RAM** MacBook Pro, including a DeepSeek model as requested.

---

## 📊 Your 4 Models

### 1. **Gemma-3-4B-QAT** ✅ Fast & Efficient
- **Model:** `mlx-community/gemma-3-4b-it-qat-4bit`
- **Size:** 4B parameters
- **Memory:** ~3-4 GB
- **Batch Size:** 4
- **Training Time:** ~45-60 min
- **Notes:** Quantization-Aware Training = high quality despite 4-bit

### 2. **Qwen3-4B** ✅ Long Context
- **Model:** `mlx-community/Qwen3-4B-4bit`
- **Size:** 4B parameters (629M active)
- **Memory:** ~2.3 GB
- **Batch Size:** 4
- **Training Time:** ~30-45 min
- **Notes:** 40K context window, very efficient
- **⚠️ Requires:** `transformers>=4.52.4` and `mlx_lm>=0.25.2`

### 3. **DeepSeek-R1-Distill-Qwen-7B** ✅ Reasoning Model (Your Request!)
- **Model:** `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`
- **Size:** 7B parameters
- **Memory:** ~5-6 GB
- **Batch Size:** 2 (reduced for stability)
- **Training Time:** ~75-90 min
- **Notes:**
  - DeepSeek's reasoning model distilled into Qwen
  - Text-only (compatible with your script)
  - May output "thinking" tokens - can be handled
  - Good for complex medical reasoning

### 4. **Mistral-Small-24B-3bit** ⚠️ Largest Model
- **Model:** `mlx-community/Mistral-Small-3.1-24B-Instruct-TXT-2503-3bit-AWQ`
- **Size:** 24B parameters
- **Memory:** ~10-12 GB (will use most of your RAM)
- **Batch Size:** 1 (required)
- **LoRA Rank:** 4 (reduced to save memory)
- **Training Time:** ~3-4 hours
- **Iterations:** 400 (reduced from 600)
- **Notes:**
  - This will stress your 24GB RAM
  - Close all other apps when running
  - Runs LAST (after smaller models succeed)

---

## ⏱️ Total Training Time

**Sequential execution:**
- Gemma-3-4B: ~45-60 min
- Qwen3-4B: ~30-45 min
- DeepSeek-R1-7B: ~75-90 min
- Mistral-24B: ~3-4 hours

**Total: ~6-8 hours** (perfect for overnight!)

---

## 🎯 Model Diversity

Your configuration gives you excellent model diversity:

| Model | Size | Specialty | Quantization |
|-------|------|-----------|--------------|
| Gemma-3-4B | 4B | Google QAT, balanced | 4-bit QAT |
| Qwen3-4B | 4B | Long context, efficient | 4-bit |
| DeepSeek-R1-7B | 7B | Reasoning, medical logic | 4-bit |
| Mistral-Small-24B | 24B | Large, comprehensive | 3-bit AWQ |

---

## ⚙️ Before You Start

### 1. Update Dependencies (Required for Qwen3)
```bash
pip install --upgrade transformers mlx-lm

# Verify versions
pip show transformers mlx-lm
```

**Required:**
- `transformers >= 4.52.4`
- `mlx-lm >= 0.25.2`

### 2. Check Available Disk Space
```bash
df -h .
```

**Required:** ~30-40 GB free space
- Models download: ~15-20 GB
- Fine-tuned adapters: ~5-10 GB
- Training data: ~1 GB

### 3. Prepare Your System
```bash
# Close unnecessary applications
# Especially before Mistral-24B run

# Monitor memory during training
watch -n 5 'vm_stat | head -10'
```

---

## 🚀 How to Run

### Quick Test (5 minutes)
Test with the smallest model first:
```bash
python multi_model_runner.py --models "Qwen3-4B"
```

### Full Run (6-8 hours)
```bash
# Background execution with logging
nohup python multi_model_runner.py > training.log 2>&1 &

# Check progress
bash check_progress.sh

# Or watch live
tail -f training.log
```

### Run Without Baselines (Faster)
```bash
# Skip classical ML and baseline evals to save ~30min per model
python multi_model_runner.py --skip-baseline --skip-classical-ml
```

---

## 📈 Expected Results

Based on your current script showing:
- Llama-3.2-3B: ~85% accuracy, balanced recall/specificity
- LASSO/RF: ~83-84% accuracy

**Predictions for your 4 models:**

1. **Gemma-3-4B:** Similar to Llama (~84-86%)
2. **Qwen3-4B:** Fast, efficient (~83-85%)
3. **DeepSeek-R1-7B:** May excel at reasoning (~85-88%)
4. **Mistral-24B:** Potentially highest accuracy (~86-90%)

---

## ⚠️ Potential Issues & Solutions

### Issue 1: Qwen3 Fails to Load
**Error:** "requires transformers>=4.52.4"

**Solution:**
```bash
pip install --upgrade transformers mlx-lm
```

### Issue 2: DeepSeek Outputs "Thinking" Tokens
**Symptom:** Model outputs reasoning process before answer

**Solution:** This is normal for R1 models. The evaluation will still extract "Yes"/"No" from output.

### Issue 3: Mistral-24B Out of Memory
**Error:** "Out of memory" or system freeze

**Solutions:**
1. Close all other applications
2. Reduce LoRA rank to 2:
   ```json
   "lora_rank": 2  // in model_configs.json
   ```
3. Skip Mistral-24B if issues persist:
   ```bash
   python multi_model_runner.py --models "Gemma-3-4B-QAT" "Qwen3-4B" "DeepSeek-R1-Distill-Qwen-7B"
   ```

### Issue 4: Training Too Slow
**Solution:** Reduce iterations for testing:
```json
"num_iters": 300  // instead of 600
```

---

## 📝 Recommended Execution Order

The runner will execute in this order (as configured):

1. **Gemma-3-4B** - Warm up, verify setup works
2. **Qwen3-4B** - Fast execution, build confidence
3. **DeepSeek-R1-7B** - Medium size, test reasoning
4. **Mistral-24B** - Final challenge, runs last when others are done

This order is intentional:
- ✅ Start with reliable models
- ✅ Test your setup early
- ✅ Save risky large model for last
- ✅ If Mistral-24B fails, you still have 3 models!

---

## 🎯 Success Criteria

After the run completes, you should have:

1. ✅ 4 trained model adapters in `./finetuned-*` directories
2. ✅ Individual results JSON files in `./results/`
3. ✅ Combined comparison in `./results/combined_results_*.json`
4. ✅ Comparison table showing all metrics

**Minimum acceptable:** 3 out of 4 models complete successfully

---

## 📊 What Changed from Original List

| Original | Status | Replacement |
|----------|--------|-------------|
| `gemma-3-4b-it-qat-4bit` | ✅ Kept | - |
| `Qwen3-4B-4bit` | ✅ Kept | - |
| `deepseek-vl2-8bit` | ❌ Removed | **DeepSeek-R1-Distill-Qwen-7B** (text-only) |
| `Mistral-Small-24B-3bit` | ⚠️ Kept with adjustments | Reduced batch_size=1, lora_rank=4 |

**Why DeepSeek-VL2 was removed:**
- It's a **vision-language model** (requires images)
- Your task is **text-only** (clinical tabular data)
- Would require completely different script (`mlx-vlm` not `mlx-lm`)

**DeepSeek-R1-Distill-Qwen-7B is better because:**
- ✅ Text-only (compatible with your script)
- ✅ Reasoning capabilities (good for medical decisions)
- ✅ 7B size (manageable on 24GB RAM)
- ✅ Distilled from DeepSeek's powerful R1 model

---

## 🔍 Next Steps

1. **Update dependencies:**
   ```bash
   pip install --upgrade transformers mlx-lm
   ```

2. **Quick test:**
   ```bash
   python multi_model_runner.py --models "Qwen3-4B"
   ```

3. **If test succeeds, run all models:**
   ```bash
   nohup python multi_model_runner.py > training.log 2>&1 &
   ```

4. **Check progress periodically:**
   ```bash
   bash check_progress.sh
   ```

5. **When complete, analyze:**
   ```bash
   python analyze_results.py
   ```

---

## 💡 Pro Tips

1. **Start in the evening** so Mistral-24B can run overnight
2. **Monitor the first model** to ensure setup is correct
3. **Don't interrupt Mistral-24B** - it takes 3-4 hours, interruption wastes that time
4. **Save your training.log** - useful for debugging
5. **If Mistral-24B crashes your system**, skip it next time - 3 models is enough for comparison

---

## ✅ Your Configuration is Ready!

Your [model_configs.json](model_configs.json) is now optimized for:
- ✅ Your 24GB RAM
- ✅ Including DeepSeek (as requested)
- ✅ All text-only, compatible models
- ✅ Realistic training times
- ✅ Good model diversity

**Ready to start?** Run:
```bash
pip install --upgrade transformers mlx-lm
python multi_model_runner.py --models "Qwen3-4B"  # Test first
```

Good luck! 🚀
