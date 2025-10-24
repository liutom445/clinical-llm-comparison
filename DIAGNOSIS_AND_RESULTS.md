# Diagnosis & Fixed Results

## 🔍 What Went Wrong

Your models had **chat template / output format issues**, NOT training failures:

### Problem 1: Qwen3-4B (Was 0%, Now 42.4%)
- **Issue:** Outputting `<think></think>` tags before answer
- **Example:** `"<think>\n\n</think>\n\nNo"`
- **Fix:** Strip thinking tags before parsing
- **Result:** ✅ Now working (100% parsed)

### Problem 2: DeepSeek-R1 (Was 0%, Now 39% but biased)
- **Issue:** Reasoning model outputs full thought process
- **Example:** `"Okay, so I need to figure out whether this patient..."`
- **Fix:** Look for answer patterns in longer text
- **Result:** ⚠️ Parsing works (98.3%) but model learned to always say "No"

### Problem 3: Gemma-3-4B-QAT (Was 42.4%, Still 42.4%)
- **Issue:** Same metrics = same problem (was already partially working)
- **Result:** ✅ Now reliably parsing (100%)

---

## 📊 Your REAL Results (After Fixes)

### Updated Comparison Table

| Model | Accuracy | Precision | Recall | Specificity | F1 | Balance Gap | Status |
|-------|----------|-----------|--------|-------------|-------|-------------|---------|
| **Llama-3.2-3B** | **50.8%** | 62.1% | **50.0%** | 52.2% | **55.4%** | 2.2% | ✅ **Best Overall** |
| **Mistral-7B** | **52.5%** | **66.7%** | 44.4% | 65.2% | 53.3% | 20.8% | ✅ **Best Accuracy** |
| Qwen3-4B | 42.4% | 55.6% | 27.8% | 65.2% | 37.0% | 37.4% | ⚠️ Worse than baseline |
| Gemma-3-4B-QAT | 42.4% | 58.3% | 19.4% | 78.3% | 29.2% | 58.9% | ⚠️ Poor recall |
| DeepSeek-R1 | 39.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | ❌ Failed (always says No) |

---

## 🏆 Winners & Losers

### ✅ SUCCESSFUL MODELS (2/5)

#### 1. **Mistral-7B** - Best Performance
- **Accuracy:** 52.5%
- **F1 Score:** 53.3%
- **Strengths:** Highest accuracy, good precision (66.7%)
- **Weaknesses:** Moderate recall (44.4%), some imbalance

#### 2. **Llama-3.2-3B** - Most Balanced
- **Accuracy:** 50.8%
- **F1 Score:** 55.4% (best F1!)
- **Strengths:** Most balanced (2.2% gap), best recall (50%)
- **Weaknesses:** Slightly lower accuracy

### ⚠️ MEDIOCRE MODELS (2/5)

#### 3. **Qwen3-4B** - Underperformed
- **Accuracy:** 42.4%
- **F1 Score:** 37.0%
- **Issue:** Low recall (27.8%), worse than your LASSO baseline (50.8%)
- **Reason:** May need different hyperparameters or more iterations

#### 4. **Gemma-3-4B-QAT** - Very Biased
- **Accuracy:** 42.4%
- **F1 Score:** 29.2%
- **Issue:** Extremely low recall (19.4%), high false negative rate
- **Reason:** Model learned to be too conservative (says "No" too often)

### ❌ FAILED MODEL (1/5)

#### 5. **DeepSeek-R1-Distill-Qwen-7B** - Complete Failure
- **Accuracy:** 39.0%
- **F1 Score:** 0.0%
- **Issue:** Predicts "No" for EVERY case (0% recall, 100% specificity)
- **Reason:**
  - Reasoning model incompatible with simple Yes/No task
  - Training may have taught it to always say "No"
  - Needs different training approach (more iterations, different prompting)

---

## 📈 Performance Ranking

### By Accuracy:
1. **Mistral-7B** (52.5%) ⭐
2. Llama-3.2-3B (50.8%)
3. Qwen3-4B / Gemma-3-4B (42.4% tie)
4. DeepSeek-R1 (39.0%)

### By F1 Score:
1. **Llama-3.2-3B** (55.4%) ⭐
2. Mistral-7B (53.3%)
3. Qwen3-4B (37.0%)
4. Gemma-3-4B (29.2%)
5. DeepSeek-R1 (0.0%)

### By Balance (lowest gap):
1. **Llama-3.2-3B** (2.2% gap) ⭐
2. Mistral-7B (20.8% gap)
3. Qwen3-4B (37.4% gap)
4. Gemma-3-4B (58.9% gap)
5. DeepSeek-R1 (100% gap)

---

## 🎯 Conclusions & Recommendations

### Main Finding:
**Only 2 out of 5 models achieved acceptable performance** (>50% accuracy)

### Best Models:
1. **Mistral-7B** - Best for overall accuracy
2. **Llama-3.2-3B** - Best for balanced predictions

### Why Some Models Failed:

#### Qwen3-4B & Gemma-3-4B-QAT:
- ⚠️ **Undertrained** or wrong hyperparameters
- **Solutions to try:**
  - Increase iterations: 600 → 1000
  - Increase learning rate: 1e-6 → 5e-6
  - Increase LoRA rank: 8 → 16
  - Check if balanced sampling worked correctly

#### DeepSeek-R1:
- ❌ **Wrong model type** for this task
- **Reasons:**
  - R1 models are designed for complex reasoning, not simple classification
  - The "thinking" process may interfere with direct Yes/No answers
  - May need special prompting (e.g., "Think step by step, then answer Yes or No")

---

## 🔧 What You Should Do Next

### Option 1: Accept Current Results (Fastest)
Use **Llama-3.2-3B** and **Mistral-7B** for your paper:
- Both are working well
- Good for comparison
- Can discuss why others failed as part of analysis

### Option 2: Re-train Failed Models (Recommended)
Re-train with better hyperparameters:

```json
// For Qwen3-4B
{
  "learning_rate": 5e-6,  // Was 1e-6
  "num_iters": 1000,      // Was 600
  "lora_rank": 16         // Was 8
}

// For Gemma-3-4B-QAT
{
  "learning_rate": 5e-6,  // Was 9e-7
  "num_iters": 1000,      // Was 600
  "lora_rank": 16         // Was 8
}
```

### Option 3: Try New Models (Alternative)
Replace failed models with proven alternatives:
- Replace DeepSeek-R1 with: `Phi-3-mini-4k-instruct-4bit`
- Replace Qwen3 with: `Llama-3.1-8B-Instruct-4bit`

---

## 📊 Comparison to Baselines

Your classical ML baselines (from Qwen3 results file):
- **LASSO:** 50.8% accuracy, 52.5% F1
- **Random Forest:** 57.6% accuracy, 63.8% F1

### LLM vs Classical ML:
- **Mistral-7B:** Comparable to LASSO, slightly worse than RF
- **Llama-3.2-3B:** Comparable to LASSO, worse than RF
- **Others:** Worse than both classical methods

**Conclusion:** For this specific task, **Random Forest** is actually the best performer!

---

## 📁 Updated Results Files

Fixed results saved in:
- `results/Qwen3-4B_FIXED.json` ✅
- `results/DeepSeek-R1-Distill-Qwen-7B_FIXED.json` ⚠️
- `results/Gemma-3-4B-QAT_FIXED.json` ⚠️

Original results still in:
- `results/Llama-3.2-3B_*.json` ✅
- `results/Mistral-7B_*.json` ✅

---

## 💡 Key Insights

### 1. Chat Template Matters
- Different models have different output formats
- Qwen3 uses `<think>` tags
- DeepSeek-R1 outputs reasoning process
- Need robust parsing for multi-model comparison

### 2. Not All Models Are Equal
- Larger ≠ Better (7B DeepSeek failed, 3B Llama succeeded)
- QAT quantization didn't help Gemma
- Model architecture matters more than size

### 3. Task-Model Fit Important
- DeepSeek-R1 is a reasoning model, not for simple classification
- Some models better suited for complex tasks, not binary decisions

### 4. Classical ML Still Competitive
- Random Forest (57.6%) beats all LLMs
- LASSO (50.8%) matches best LLMs
- For small structured datasets, tree-based methods may be better

---

## 🎯 Final Recommendation

**For your paper/presentation:**

### Best LLM Models (Use These):
1. **Mistral-7B** (52.5% accuracy)
2. **Llama-3.2-3B** (50.8% accuracy, best balance)

### Best Overall:
**Random Forest** (57.6% accuracy, 63.8% F1)

### Interesting Discussion Points:
- Why did larger DeepSeek fail while smaller Llama succeeded?
- Why did QAT Gemma underperform?
- Why does Random Forest beat all LLMs for this task?
- What role does model architecture play vs size?

---

## 🔄 Next Steps

1. **Review your training setup:**
   - Check if balanced sampling actually worked
   - Verify data splits are correct
   - Check training logs for anomalies

2. **Decide:**
   - Accept current results (2 working models)
   - Re-train failed models with new hyperparameters
   - Try different models

3. **Write up results:**
   - Focus on Llama-3.2-3B & Mistral-7B
   - Discuss failures as learnings
   - Compare to classical ML baselines

Would you like me to help with any of these next steps?
