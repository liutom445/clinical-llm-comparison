"""
Single Model Fine-Tuning Script (Refactored for Multi-Model Comparison)
Can be run standalone or called by multi_model_runner.py

Usage:
  python rct_ft_single.py  # Uses default config
  python rct_ft_single.py --model "mlx-community/Llama-3.2-3B-Instruct-4bit" --output-dir "./output"
"""

import pandas as pd
import numpy as np
import json
import argparse
import gc
from pathlib import Path
from collections import Counter
from mlx_lm.utils import load
from mlx_lm.generate import generate
from typing import Dict, List
import subprocess
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import MLX for memory cleanup
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune a single LLM model')

    # Model configuration
    parser.add_argument('--model', type=str, default='mlx-community/Llama-3.2-3B-Instruct-4bit',
                       help='Model ID to use')
    parser.add_argument('--output-dir', type=str, default='./release-finetuned-llama',
                       help='Output directory for adapters')
    parser.add_argument('--model-name', type=str, default='Model',
                       help='Friendly name for the model')

    # Hyperparameters
    parser.add_argument('--lora-rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--learning-rate', type=float, default=9e-7,
                       help='Learning rate')
    parser.add_argument('--num-iters', type=int, default=600,
                       help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--steps-per-eval', type=int, default=50,
                       help='Steps per evaluation')
    parser.add_argument('--steps-per-report', type=int, default=50,
                       help='Steps per report')

    # Data configuration
    parser.add_argument('--data-path', type=str, default='Trial 9/trial9.csv',
                       help='Path to data CSV')
    parser.add_argument('--train-ratio', type=float, default=0.80,
                       help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.10,
                       help='Validation data ratio')
    parser.add_argument('--test-ratio', type=float, default=0.10,
                       help='Test data ratio')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use-balanced-sampling', action='store_true', default=True,
                       help='Use balanced sampling')

    # Execution options
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline evaluation')
    parser.add_argument('--skip-classical-ml', action='store_true',
                       help='Skip classical ML baselines')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip fine-tuning (only evaluate)')
    parser.add_argument('--results-file', type=str, default=None,
                       help='Save results to JSON file')

    return parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

args = parse_args()

# Set configuration from args
DATA_PATH = args.data_path
MODEL_NAME = args.model
OUTPUT_DIR = args.output_dir
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = args.train_ratio, args.val_ratio, args.test_ratio
RANDOM_SEED = args.random_seed

LORA_RANK = args.lora_rank
LORA_ALPHA = args.lora_alpha
LEARNING_RATE = args.learning_rate
NUM_ITERS = args.num_iters
BATCH_SIZE = args.batch_size
STEPS_PER_EVAL = args.steps_per_eval
STEPS_PER_REPORT = args.steps_per_report

USE_BALANCED_SAMPLING = args.use_balanced_sampling

np.random.seed(RANDOM_SEED)

LINE_LEN = 80
METRIC_KEYS = ('accuracy', 'precision', 'recall', 'specificity', 'f1')


def section(title: str, char: str = "=") -> None:
    """Print a formatted banner headline."""
    line = char * LINE_LEN
    print(f"\n{line}\n{title}\n{line}")


def safe_div(num: float, denom: float) -> float:
    """Guarded division for metrics."""
    return num / denom if denom else 0.0


def message_counts(examples: List[Dict]) -> Counter:
    """Count Yes/No labels within chat-formatted examples."""
    return Counter(ex["messages"][1]["content"] for ex in examples)


def log_counts(counts, indent: str = "  ") -> None:
    """Pretty-print label counts with percentages."""
    items = list(counts.items()) if hasattr(counts, "items") else list(counts)
    total = sum(count for _, count in items) or 1
    for label, count in items:
        print(f"{indent}{label}: {count} ({count / total:.1%})")


def balance_examples(examples: List[Dict]) -> List[Dict]:
    """Oversample minority class to match majority."""
    counts = message_counts(examples)
    if len(counts) < 2 or counts['Yes'] == counts['No']:
        return examples

    majority_label = max(counts, key=counts.get)
    minority_label = min(counts, key=counts.get)
    majority = [ex for ex in examples if ex['messages'][1]['content'] == majority_label]
    minority = [ex for ex in examples if ex['messages'][1]['content'] == minority_label]

    if not minority:
        return examples

    repeats, remainder = divmod(len(majority), len(minority))
    oversampled = minority * repeats + minority[:remainder]
    balanced = majority + oversampled
    np.random.shuffle(balanced)
    return balanced


def confusion_dict(gt_labels: List[int], pred_labels: List[int]) -> Dict[str, int]:
    """Replicate manual confusion tally while ignoring unparseable predictions."""
    tp = tn = fp = fn = 0
    for gt, pred in zip(gt_labels, pred_labels):
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 0 and gt == 0:
            tn += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 1:
            fn += 1
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def print_binary_report(name: str, result: Dict, pred_counts: Dict[str, int]) -> None:
    """Standard formatter for binary classification metrics."""
    print(f"\n{name} Results:")
    print(f"  Accuracy:    {result['accuracy']:.1%}")
    print(f"  Precision:   {result['precision']:.1%}")
    print(f"  Recall:      {result['recall']:.1%}")
    print(f"  Specificity: {result['specificity']:.1%}")
    print(f"  F1 Score:    {result['f1']:.1%}")
    conf = result['confusion']
    print(f"  Confusion: TP={conf['tp']} TN={conf['tn']} FP={conf['fp']} FN={conf['fn']}")
    if pred_counts:
        print(f"  Predictions: Yes={pred_counts.get('Yes', 0)}, No={pred_counts.get('No', 0)}")


def cleanup_memory():
    """Clean up memory between model runs"""
    print("\nCleaning up memory...")
    gc.collect()
    if MLX_AVAILABLE:
        try:
            mx.metal.clear_cache()
            print("  MLX cache cleared")
        except Exception as e:
            print(f"  Warning: Could not clear MLX cache: {e}")


# ============================================================================
# DATA LOADING
# ============================================================================

section(f"FINE-TUNING: {args.model_name}")
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} patients")

manual_removal_counts = df['YP_manual_removal'].value_counts()
print("\nClass distribution:")
log_counts(manual_removal_counts)

# ============================================================================
# PROMPT FORMATTING
# ============================================================================

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

# ============================================================================
# DATA SPLITTING & BALANCING
# ============================================================================

print("\nProcessing data...")
shuffled_df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
examples = [format_example(row) for _, row in shuffled_df.iterrows()]

n = len(examples)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_examples = examples[:train_end]
val_examples = examples[train_end:val_end]
test_examples = examples[val_end:]

# Balance training set
if USE_BALANCED_SAMPLING:
    print("\nBalancing training set...")
    print("  Original:")
    log_counts(message_counts(train_examples), indent="    ")
    train_examples = balance_examples(train_examples)
    print("  Balanced:")
    log_counts(message_counts(train_examples), indent="    ")

# Save datasets
splits = {
    'train': train_examples,
    'valid': val_examples,
    'test': test_examples
}

data_dir = Path(OUTPUT_DIR) / "data"
data_dir.mkdir(parents=True, exist_ok=True)

print(f"\nSaving to {data_dir}:")
for split_name, data in splits.items():
    filepath = data_dir / f"{split_name}.jsonl"
    with open(filepath, 'w') as f:
        for ex in data:
            f.write(json.dumps(ex) + '\n')

    counts = message_counts(data)
    yes = counts.get('Yes', 0)
    no = counts.get('No', 0)
    print(f"  {split_name}: {len(data)} (Yes={yes}, No={no})")

# ============================================================================
# BASELINE EVALUATION
# ============================================================================

baseline_results = None
if not args.skip_baseline:
    section("BASELINE EVALUATION")

    print("\nLoading baseline model...")
    model, tokenizer = load(MODEL_NAME)

    def evaluate(model, tokenizer, test_data: List[Dict], name: str) -> Dict:
        """Evaluate model on test data."""
        print(f"\nEvaluating {name} on {len(test_data)} examples...")
        predictions, ground_truths = [], []

        for idx, example in enumerate(test_data, 1):
            gt = example['messages'][1]['content'].strip().lower()
            ground_truths.append(gt)

            prompt = tokenizer.apply_chat_template(
                [example['messages'][0]],
                tokenize=False,
                add_generation_prompt=True
            )

            try:
                pred = generate(model, tokenizer, prompt=prompt, max_tokens=10).strip().lower()
            except Exception:
                pred = "error"
            predictions.append(pred)

            if idx % 20 == 0:
                print(f"  {idx}/{len(test_data)}")

        correct = sum(1 for gt, pred in zip(ground_truths, predictions)
                      if gt in pred or pred in gt)
        accuracy = safe_div(correct, len(test_data))

        gt_classes = [1 if gt == 'yes' else 0 for gt in ground_truths]
        pred_classes = []
        for pred in predictions:
            first = (pred.split() or [""])[0]
            if 'yes' in first:
                pred_classes.append(1)
            elif 'no' in first:
                pred_classes.append(0)
            else:
                pred_classes.append(-1)

        conf = confusion_dict(gt_classes, pred_classes)
        precision = safe_div(conf['tp'], conf['tp'] + conf['fp'])
        recall = safe_div(conf['tp'], conf['tp'] + conf['fn'])
        specificity = safe_div(conf['tn'], conf['tn'] + conf['fp'])
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'confusion': conf,
            'predictions': predictions,
            'ground_truths': ground_truths
        }

        pred_summary = Counter(pred_classes)
        print_binary_report(
            name,
            result,
            {'Yes': pred_summary.get(1, 0), 'No': pred_summary.get(0, 0)}
        )
        return result

    test_subset = splits['test']
    baseline_results = evaluate(model, tokenizer, test_subset, "Baseline")

    # Clean up baseline model
    del model, tokenizer
    cleanup_memory()

# ============================================================================
# CLASSICAL ML BASELINES (LASSO & Random Forest)
# ============================================================================

lasso_results = None
rf_results = None

if not args.skip_classical_ml:
    section("CLASSICAL ML BASELINES")

    print("\nPreparing classical ML features...")

    feature_cols = [
        'X_age_0d', 'X_pulse_0d', 'X_bp_sys_0d', 'X_bp_dia_0d',
        'X_birth_weight_0d', 'X_GestationWeeks_0d', 'X_hb_before_delivery_0d'
    ]

    df_ml = shuffled_df.copy()
    df_ml['Treatment_Oxytocin'] = (df_ml['Treatment'] == 'Oxytocin').astype(int)
    feature_cols.append('Treatment_Oxytocin')

    train_df = df_ml.iloc[:train_end]
    val_df = df_ml.iloc[train_end:val_end]
    test_df = df_ml.iloc[val_end:]

    X_train = train_df[feature_cols].values
    y_train = (train_df['YP_manual_removal'] == 'Yes').astype(int).values

    X_val = val_df[feature_cols].values
    y_val = (val_df['YP_manual_removal'] == 'Yes').astype(int).values

    X_test = test_df[feature_cols].values
    y_test = (test_df['YP_manual_removal'] == 'Yes').astype(int).values

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    print(f"Training set: {len(X_train)} samples")
    print(f"Features: {len(feature_cols)}")

    # LASSO
    section("Training LASSO Logistic Regression", char='-')

    lasso_model = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED
    )

    lasso_model.fit(X_train_scaled, y_train)

    y_pred_lasso = lasso_model.predict(X_test_scaled)
    cm_lasso = confusion_matrix(y_test, y_pred_lasso)
    tn_lasso, fp_lasso, fn_lasso, tp_lasso = cm_lasso.ravel()

    lasso_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_lasso),
        'precision': precision_score(y_test, y_pred_lasso, zero_division=0),
        'recall': recall_score(y_test, y_pred_lasso, zero_division=0),
        'specificity': safe_div(tn_lasso, tn_lasso + fp_lasso),
        'f1': f1_score(y_test, y_pred_lasso, zero_division=0)
    }
    lasso_results = {**lasso_metrics,
                     'confusion': {'tp': int(tp_lasso), 'tn': int(tn_lasso),
                                   'fp': int(fp_lasso), 'fn': int(fn_lasso)}}

    print_binary_report(
        "LASSO Logistic Regression",
        lasso_results,
        {'Yes': int(y_pred_lasso.sum()), 'No': int(len(y_pred_lasso) - y_pred_lasso.sum())}
    )

    # Random Forest
    section("Training Random Forest", char='-')

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_test_scaled)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'specificity': safe_div(tn_rf, tn_rf + fp_rf),
        'f1': f1_score(y_test, y_pred_rf, zero_division=0)
    }
    rf_results = {**rf_metrics,
                  'confusion': {'tp': int(tp_rf), 'tn': int(tn_rf),
                                'fp': int(fp_rf), 'fn': int(fn_rf)}}

    print_binary_report(
        "Random Forest",
        rf_results,
        {'Yes': int(y_pred_rf.sum()), 'No': int(len(y_pred_rf) - y_pred_rf.sum())}
    )

# ============================================================================
# FINE-TUNING
# ============================================================================

if not args.skip_training:
    section("FINE-TUNING")

    adapter_path = Path(OUTPUT_DIR)
    adapters_file = adapter_path / "adapters.safetensors"

    if adapters_file.exists():
        print(f"\n✓ Model exists: {OUTPUT_DIR}")
        print("To retrain: rm -rf " + OUTPUT_DIR)
    else:
        print(f"\nConfiguration:")
        print(f"  LoRA Rank: {LORA_RANK}")
        print(f"  Learning Rate: {LEARNING_RATE}")
        print(f"  Iterations: {NUM_ITERS}")

        train_cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", MODEL_NAME,
            "--train",
            "--data", str(data_dir),
            "--iters", str(NUM_ITERS),
            "--batch-size", str(BATCH_SIZE),
            "--steps-per-eval", str(STEPS_PER_EVAL),
            "--steps-per-report", str(STEPS_PER_REPORT),
            "--adapter-path", OUTPUT_DIR,
            "--learning-rate", str(LEARNING_RATE)
        ]

        print(f"\nCommand: {' '.join(train_cmd)}\n")

        try:
            subprocess.run(train_cmd, check=True, text=True)
            print("\n✓ Training complete!")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed: {e.returncode}")
            sys.exit(1)

# ============================================================================
# EVALUATION
# ============================================================================

section("FINE-TUNED EVALUATION")

print("\nLoading fine-tuned model...")
try:
    finetuned_model, finetuned_tokenizer = load(MODEL_NAME, adapter_path=OUTPUT_DIR)
    print("✓ Loaded")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

finetuned_results = evaluate(finetuned_model, finetuned_tokenizer, test_subset, "Fine-tuned")

# Clean up fine-tuned model
del finetuned_model, finetuned_tokenizer
cleanup_memory()

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

section("MODEL COMPARISON")

all_models = {}
if baseline_results:
    all_models['Baseline LLM'] = baseline_results
if lasso_results:
    all_models['LASSO'] = lasso_results
if rf_results:
    all_models['Random Forest'] = rf_results
all_models['Fine-tuned LLM'] = finetuned_results

print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1':<12} {'Balance Gap':<12}")
print("-" * 110)

for model_name, results in all_models.items():
    balance_gap = abs(results['recall'] - results['specificity'])
    print(f"{model_name:<20} ", end="")
    for metric in METRIC_KEYS:
        print(f"{results[metric]:<12.1%} ", end="")
    print(f"{balance_gap:<12.1%}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

if args.results_file:
    results_to_save = {
        'model_name': args.model_name,
        'model_id': MODEL_NAME,
        'output_dir': OUTPUT_DIR,
        'hyperparameters': {
            'lora_rank': LORA_RANK,
            'lora_alpha': LORA_ALPHA,
            'learning_rate': LEARNING_RATE,
            'num_iters': NUM_ITERS,
            'batch_size': BATCH_SIZE,
        },
        'results': {
            'baseline': baseline_results if baseline_results else None,
            'lasso': lasso_results if lasso_results else None,
            'random_forest': rf_results if rf_results else None,
            'finetuned': finetuned_results
        }
    }

    # Remove predictions/ground_truths to keep file size reasonable
    for model_key in ['baseline', 'finetuned']:
        if results_to_save['results'][model_key]:
            results_to_save['results'][model_key].pop('predictions', None)
            results_to_save['results'][model_key].pop('ground_truths', None)

    with open(args.results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n✓ Results saved to: {args.results_file}")

section("COMPLETE")
print(f"Model: {args.model_name}")
print(f"Output: {OUTPUT_DIR}")
