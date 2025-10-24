"""
Re-evaluate failed models with robust output parsing

This will:
1. Load the already-trained adapters (no re-training needed!)
2. Use improved Yes/No extraction
3. Re-generate results JSON files
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter
from mlx_lm.utils import load
from mlx_lm.generate import generate
from typing import Dict, List

# Configuration
DATA_PATH = "Trial 9/trial9.csv"
RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10

# Models to re-evaluate
MODELS_TO_FIX = [
    {
        "name": "Qwen3-4B",
        "model_id": "mlx-community/Qwen3-4B-4bit",
        "adapter_path": "./finetuned-qwen3-4b",
        "results_file": "results/Qwen3-4B_FIXED.json"
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-7B",
        "model_id": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        "adapter_path": "./finetuned-deepseek-r1-qwen-7b",
        "results_file": "results/DeepSeek-R1-Distill-Qwen-7B_FIXED.json"
    },
    {
        "name": "Gemma-3-4B-QAT",
        "model_id": "mlx-community/gemma-3-4b-it-qat-4bit",
        "adapter_path": "./finetuned-gemma-3-4b-qat",
        "results_file": "results/Gemma-3-4B-QAT_FIXED.json"
    },
]

np.random.seed(RANDOM_SEED)


def extract_yes_no_robust(output_text):
    """
    Robustly extract Yes/No from model output
    """
    if not output_text or not output_text.strip():
        return None

    text = output_text.strip().lower()

    # Remove thinking tags (Qwen3)
    text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Look for explicit answer patterns (DeepSeek-R1)
    answer_patterns = [
        r'(?:final\s+)?answer\s*(?:is)?\s*:\s*(yes|no)',
        r'(?:the\s+)?answer\s+is\s+(yes|no)',
        r'conclusion\s*:\s*(yes|no)',
        r'therefore\s*,?\s*(yes|no)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # Check cleaned text first line/word
    if text_no_think.strip():
        first_line = text_no_think.strip().split('\n')[0].strip()
        first_word = first_line.split()[0].rstrip('.,!?;:') if first_line.split() else ""
        if first_word in ['yes', 'no']:
            return first_word

        # Check last line/word
        last_line = text_no_think.strip().split('\n')[-1].strip()
        last_word = last_line.split()[-1].rstrip('.,!?;') if last_line.split() else ""
        if last_word in ['yes', 'no']:
            return last_word

    # Check each line
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if not words:
            continue
        first_word = words[0].rstrip('.,!?;:')
        if first_word in ['yes', 'no']:
            return first_word

    # Search anywhere (last resort)
    if 'yes' in text and 'no' not in text:
        return 'yes'
    if 'no' in text and 'yes' not in text:
        return 'no'

    return None


def safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def create_prompt(row: pd.Series) -> str:
    """Convert patient data to clinical prompt"""
    prompt = f"""Evaluate this patient with retained placenta.

PATIENT DATA:
- Age: {row.get('X_age_0d', 'unknown')} years
- Country: {row.get('X_country_0d', 'unknown')}
- Treatment: {row.get('Treatment', 'unknown')}
- Pulse: {row.get('X_pulse_0d', 'unknown')} bpm
- Blood Pressure: {row.get('X_bp_sys_0d', 'unknown')}/{row.get('X_bp_dia_0d', 'unknown')} mmHg
- Hemoglobin: {row.get('X_hb_before_delivery_0d', 'unknown')} g/dL
- Gestational age: {row.get('X_GestationWeeks_0d', 'unknown')} weeks
- Birth weight: {row.get('X_birth_weight_0d', 'unknown')}g

CLINICAL QUESTION:
Will this patient require manual removal of the placenta?

Answer with only 'Yes' or 'No'."""
    return prompt


def format_example(row: pd.Series) -> Dict:
    """Create training example"""
    answer = "Yes" if row['YP_manual_removal'] == "Yes" else "No"
    return {
        'messages': [
            {'role': 'user', 'content': create_prompt(row)},
            {'role': 'assistant', 'content': answer}
        ]
    }


def evaluate_model(model, tokenizer, test_data: List[Dict], model_name: str) -> Dict:
    """Evaluate model with robust parsing"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}\n")

    predictions, ground_truths = [], []
    parsed_count = 0
    unparsed_outputs = []

    for idx, example in enumerate(test_data, 1):
        gt = example['messages'][1]['content'].strip().lower()
        ground_truths.append(gt)

        prompt = tokenizer.apply_chat_template(
            [example['messages'][0]],
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            # Increase max_tokens for reasoning models
            raw_output = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)

            # Use robust extraction
            extracted = extract_yes_no_robust(raw_output)

            if extracted:
                predictions.append(extracted)
                parsed_count += 1
            else:
                predictions.append("error")
                unparsed_outputs.append((idx, raw_output[:100]))

        except Exception as e:
            predictions.append("error")
            unparsed_outputs.append((idx, f"EXCEPTION: {str(e)}"))

        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(test_data)} ({parsed_count} parsed successfully)")

    # Calculate metrics
    correct = sum(1 for gt, pred in zip(ground_truths, predictions)
                  if gt == pred)
    accuracy = safe_div(correct, len(test_data))

    gt_classes = [1 if gt == 'yes' else 0 for gt in ground_truths]
    pred_classes = []
    for pred in predictions:
        if pred == 'yes':
            pred_classes.append(1)
        elif pred == 'no':
            pred_classes.append(0)
        else:
            pred_classes.append(-1)  # Unparsed

    # Confusion matrix
    tp = tn = fp = fn = 0
    for gt, pred in zip(gt_classes, pred_classes):
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 0 and gt == 0:
            tn += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 1:
            fn += 1
        # pred == -1 is ignored

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Parsed: {parsed_count}/{len(test_data)} ({parsed_count/len(test_data):.1%})")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  Specificity: {specificity:.1%}")
    print(f"  F1 Score: {f1:.1%}")
    print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

    if unparsed_outputs:
        print(f"\n  Warning: {len(unparsed_outputs)} unparsed outputs:")
        for idx, output in unparsed_outputs[:5]:
            print(f"    #{idx}: {output}...")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'parsed_rate': parsed_count / len(test_data)
    }


# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} patients\n")

# Create test set (same seed as original)
shuffled_df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
examples = [format_example(row) for _, row in shuffled_df.iterrows()]

n = len(examples)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)
test_examples = examples[val_end:]

print(f"Test set: {len(test_examples)} examples\n")

# Re-evaluate each model
all_results = {}

for model_config in MODELS_TO_FIX:
    model_name = model_config['name']
    model_id = model_config['model_id']
    adapter_path = model_config['adapter_path']
    results_file = model_config['results_file']

    try:
        print(f"\n{'#'*80}")
        print(f"# {model_name}")
        print(f"{'#'*80}")

        # Load model
        print(f"\nLoading {model_id} with adapters from {adapter_path}...")
        model, tokenizer = load(model_id, adapter_path=adapter_path)
        print("✓ Model loaded")

        # Evaluate
        results = evaluate_model(model, tokenizer, test_examples, model_name)
        all_results[model_name] = results

        # Save
        output_data = {
            'model_name': model_name,
            'model_id': model_id,
            'adapter_path': adapter_path,
            'results': {'finetuned': results},
            'note': 'Re-evaluated with robust Yes/No extraction'
        }

        Path("results").mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Results saved to {results_file}")

        # Cleanup
        del model, tokenizer

    except Exception as e:
        print(f"✗ Failed to evaluate {model_name}: {e}")
        import traceback
        traceback.print_exc()

# Print summary
print(f"\n\n{'='*80}")
print("RE-EVALUATION SUMMARY")
print(f"{'='*80}\n")

print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Parsed':<10}")
print("-" * 95)

for model_name, results in all_results.items():
    print(f"{model_name:<30} ", end="")
    print(f"{results['accuracy']:<12.1%} ", end="")
    print(f"{results['precision']:<12.1%} ", end="")
    print(f"{results['recall']:<12.1%} ", end="")
    print(f"{results['f1']:<12.1%} ", end="")
    print(f"{results['parsed_rate']:<10.1%}")

print("\n" + "="*80)
print("COMPLETE! Check results/ directory for *_FIXED.json files")
print("="*80)
