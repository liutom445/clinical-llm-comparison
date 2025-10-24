# Multi-Model LLM Fine-Tuning for Clinical Prediction

An ongoing project of fine-tuning and comparing Large Language Models (LLMs) for predicting manual removal of retained placenta using patient clinical data.

## 📋 Overview

This repository contains code for fine-tuning multiple LLM models on Apple Silicon (MLX framework) for a clinical prediction task. The project compares LLM performance against classical machine learning baselines (LASSO, Random Forest).

**Task:** Predict whether a patient with retained placenta will require manual removal based on clinical features.

## Foundations of LLMs and Fine-Tuning

**Large Language Models** are deep neural networks trained on vast text corpora to learn probabilistic mappings from input sequences (prompts) to output sequences (completions). At their core, LLMs perform next-token prediction using transformer architectures with billions of parameters, essentially learning high-dimensional probability distributions P(token|context) through maximum likelihood estimation on training data.  However, this flexibility comes at a cost: pre-trained LLMs are optimized for general language tasks and may perform poorly on specialized domains like clinical prediction without adaptation.

**Fine-tuning** addresses this by continuing the training process on domain-specific data, essentially performing transfer learning where the model's parameters are updated to minimize loss on the target task. In the context of tabular clinical prediction, we convert structured patient data (age, vitals, lab values) into natural language prompts and fine-tune the LLM to output classification decisions, leveraging the model's learned linguistic patterns to capture complex, non-linear relationships between features—analogous to how a generalized additive model learns smooth functions, but with far greater representational capacity. This approach is particularly relevant as LLMs have demonstrated strong performance on various structured prediction tasks, though as our results show (Llama-3.2-3B-v2: 57.6% accuracy vs Random Forest: 57.6%), they match but do not necessarily surpass well-tuned classical methods on small tabular datasets.

Huggingface: [https://huggingface.co/] is a good place to look at for open sourced LLMs. 

Here are the metrics we use for comparisons: 

Recall (Sensitivity, TPR): How many actual positives you correctly catch; High recall means few misses.

Specificity (TNR): How many actual negatives you correctly reject; High specificity means few false alarms.

F1 Score: Harmonic mean of precision and recall, rewarding balance. 

### Fine-Tuned LLM Performance (v2 - Optimized)

| Model | Accuracy | F1 Score | Recall | Specificity | Balance Gap |
|-------|----------|----------|--------|-------------|-------------|
| **Llama-3.2-3B-v2** 🏆 | **57.6%** | **70.6%** | **83.3%** | 17.4% | **65.9%** |


### Previous Results (v1 - Initial Training)

| Model | Accuracy | F1 Score | Recall | Specificity | Balance Gap |
|-------|----------|----------|--------|-------------|-------------|
| **Llama-3.2-3B** ⭐ | 54.2% | 59.7% | 55.6% | 52.2% | 3.4% |
| Mistral-7B | 52.5% | 53.3% | 44.4% | 65.2% | 20.8% |
| Phi-3-mini | 50.8% | 54.0% | 47.2% | 56.5% | 9.3% |
| Llama-3.1-8B | 47.5% | 39.2% | 27.8% | 78.3% | 50.5% |
| Qwen2.5-7B | 44.1% | 37.7% | 27.8% | 69.6% | 41.8% |

### Baseline Comparisons

| Model | Accuracy | F1 Score | Recall | Specificity |
|-------|----------|----------|--------|-------------|
| **Random Forest** | **57.6%** | 63.8% | 61.1% | 52.2% |
| **Llama-3.2-3B-v2**  | **57.6%** | **70.6%** | **83.3%** | 17.4% |
| LASSO | 50.8% | 52.5% | 44.4% | 60.9% |



## 🏗️ Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── model_configs.json                 # Model configurations
│
│
├── results/                           # Training results
│   ├── combined_results_*.json       # Aggregated results
│   ├── Llama-3.2-3B_*.json          # Individual model results
│   ├── Mistral-7B_*.json
│   └── *_FIXED.json                  # Re-evaluated results
│
├── docs/
│   ├── ALTERNATIVE_MODELS.md         # Alternative model recommendations
│   ├── DIAGNOSIS_AND_RESULTS.md      # Failure analysis and findings
│   ├── FEASIBILITY_ASSESSMENT.md     # Initial model feasibility study
│   ├── NEW_CONFIG_SUMMARY.md         # Updated configuration guide
│   └── QUICK_START.md                # Quick start guide
│
└── scripts/
    └── check_progress.sh             # Progress monitoring script
```

## 🚀 Quick Start

### Prerequisites

- Apple Silicon Mac (M1/M2/M3) with 16GB+ RAM
- Python 3.9+
- MLX framework

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clinical-llm-comparison.git
cd clinical-llm-comparison

# Install dependencies
pip install -r requirements.txt
```


## 📊 Models Tested

### Successful Models ✅

1. **Llama-3.2-3B** (Meta)
   - Size: 3B parameters
   - Accuracy: 50.8%
   - Best balanced predictions (2.2% gap)

2. **Mistral-7B** (Mistral AI)
   - Size: 7B parameters
   - Accuracy: 52.5% (best LLM)
   - High precision (66.7%)

### New Models to Test

3. **Phi-3-mini** (Microsoft) - 3.8B params
4. **Llama-3.1-8B** (Meta) - 8B params
5. **Qwen2.5-7B** (Alibaba) - 7B params

### Failed Models ❌

- **Qwen3-4B**: Output format issues (`<think>` tags)
- **Gemma-3-4B-QAT**: Poor recall (19.4%), biased
- **DeepSeek-R1**: Reasoning model incompatible with binary classification

## 🔧 Configuration

Edit `model_configs.json` to customize:

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
    // ... more models
  ]
}
```

## 📖 Documentation

Detailed documentation available in `/docs`:

- **[QUICK_START.md](docs/QUICK_START.md)** - Getting started guide
- **[ALTERNATIVE_MODELS.md](docs/ALTERNATIVE_MODELS.md)** - Model recommendations
- **[DIAGNOSIS_AND_RESULTS.md](docs/DIAGNOSIS_AND_RESULTS.md)** - Detailed analysis
- **[NEW_CONFIG_SUMMARY.md](docs/NEW_CONFIG_SUMMARY.md)** - Latest configuration

## 🔬 Methodology

### Data Processing
- **Dataset**: 577 patients with retained placenta
- **Split**: 80% train, 10% validation, 10% test
- **Balancing**: Oversampling minority class in training data
- **Features**: Age, country, treatment, vital signs, hemoglobin, gestational age, birth weight

### Fine-Tuning
- **Method**: LoRA (Low-Rank Adaptation)
- **Framework**: MLX (Apple Silicon optimized)
- **Hyperparameters**:
  - LoRA rank: 8
  - Learning rate: 8e-7 to 1e-6
  - Iterations: 600
  - Batch size: 2-4 (model-dependent)

### Evaluation Metrics
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- Balance Gap (|Recall - Specificity|)

## 🐛 Known Issues & Solutions

### Issue 1: Chat Template Incompatibility
**Problem**: Some models output thinking tags or reasoning text
**Solution**: Use `reevaluate_fixed.py` with robust parsing

### Issue 2: Reasoning Models Fail
**Problem**: DeepSeek-R1 outputs process instead of answer
**Solution**: Avoid reasoning models for simple classification

### Issue 3: Model Bias
**Problem**: Some models predict "No" for all cases
**Solution**: Check training logs, adjust learning rate or iterations

## 📈 Performance Comparison

### LLM vs Classical ML

**Advantages of LLMs:**
- Better interpretability (can explain reasoning)
- Handles missing data naturally
- Can incorporate unstructured clinical notes

**Advantages of Classical ML:**
- Better accuracy on structured data (57.6% vs 52.5%)
- Faster training (minutes vs hours)
- Lower computational requirements
- More stable predictions

## 🔄 Reproducibility

All experiments use:
- Fixed random seed: 42
- Consistent data splits
- Same hyperparameters per model size class
- Version-controlled configurations

## 📄 Citation

If you use this code, please cite:

```bibtex
@misc{clinical-llm-comparison-2025,
  title={Multi-Model LLM Fine-Tuning for Clinical Prediction},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/clinical-llm-comparison}
}
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

- **Author**: Tom Liu
- **Email**: liutom@umich.edu
- **Institution**: University of Michigan

## 🙏 Acknowledgments

- MLX framework by Apple
- Meta AI (Llama models)
- Mistral AI (Mistral models)
- Microsoft (Phi models)
- Alibaba (Qwen models)
- Google (Gemma models)

## ⚠️ Disclaimer

This is research code for academic purposes. Clinical predictions should not be used for patient care without proper validation and regulatory approval.

---

**Status**: Active Development
**Last Updated**: October 2025
**Version**: 2.0
