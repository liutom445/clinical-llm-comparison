"""
Release Study Fine-Tuning with Llama 3.2
Uses balanced sampling + Llama model (known for stability)

Llama 3.2 advantages:
- More stable training than Gemma
- Better instruction following
- Good balance at smaller sizes

Usage: python rct_ft_llama.py
"""

import pandas as pd
import numpy as np
import json
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

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "Trial 9/trial9.csv"

# Llama model options (try in this order):
# MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # Recommended: stable, good size
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # Alternative: larger, more capable
# MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"  # Smallest: fastest

#MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
OUTPUT_DIR = "./release-finetuned-llama"
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10 
RANDOM_SEED = 42 

# Conservative hyperparameters for Llama
LORA_RANK = 8  # Lower than Gemma - Llama needs less
LORA_ALPHA = 16
LEARNING_RATE = 9e-7  # Conservative for stability
NUM_ITERS = 600  # Moderate length
BATCH_SIZE = 4
STEPS_PER_EVAL = 25
STEPS_PER_REPORT = 25

# Balanced sampling
USE_BALANCED_SAMPLING = True

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
# ============================================================================
# DATA LOADING
# ============================================================================

section("LLAMA 3.2 FINE-TUNING WITH BALANCED SAMPLING")
print(f"Model: {MODEL_NAME}")
print("Strategy: Balanced training data + Conservative hyperparameters")
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

data_dir = Path("./data_llama")
data_dir.mkdir(exist_ok=True)

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

# ============================================================================
# CLASSICAL ML BASELINES (LASSO & Random Forest)
# ============================================================================

section("CLASSICAL ML BASELINES")

# Prepare feature matrix from shuffled dataframe to align with splits
print("\nPreparing classical ML features...")

# Define feature columns (clinical measurements only)
feature_cols = [
    'X_age_0d', 'X_pulse_0d', 'X_bp_sys_0d', 'X_bp_dia_0d',
    'X_birth_weight_0d', 'X_GestationWeeks_0d', 'X_hb_before_delivery_0d'
]

# Add treatment as binary feature
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

# Handle missing values with median imputation
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Features: {len(feature_cols)}")

# ============================================================================
# LASSO Logistic Regression
# ============================================================================

section("Training LASSO Logistic Regression", char='-')

# Use class_weight='balanced' to handle imbalance
lasso_model = LogisticRegression(
    penalty='l1',
    solver='saga',
    C=1.0,  # Inverse of regularization strength
    class_weight='balanced',
    max_iter=1000,
    random_state=RANDOM_SEED
)

lasso_model.fit(X_train_scaled, y_train)

# Evaluate on test set
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

important_features = [(feat, coef) for feat, coef in zip(feature_cols, lasso_model.coef_[0])
                      if abs(coef) > 0.01]
if important_features:
    print("\nFeature Importance (LASSO coefficients):")
    for feat, coef in important_features:
        print(f"  {feat}: {coef:.3f}")

# ============================================================================
# Random Forest
# ============================================================================

section("Training Random Forest", char='-')

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Prevent overfitting on small dataset
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Evaluate on test set
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

feature_importance = sorted(zip(feature_cols, rf_model.feature_importances_),
                            key=lambda x: x[1], reverse=True)
significant_importance = [(feat, imp) for feat, imp in feature_importance if imp > 0.05]
if significant_importance:
    print("\nFeature Importance (Random Forest):")
    for feat, imp in significant_importance:
        print(f"  {feat}: {imp:.3f}")

# ============================================================================
# FINE-TUNING
# ============================================================================

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

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

section("COMPREHENSIVE MODEL COMPARISON")

# Collect all model results
all_models = {
    'Baseline LLM': baseline_results,
    'LASSO': lasso_results,
    'Random Forest': rf_results,
    'Fine-tuned LLM': finetuned_results
}

# Print comparison table
print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1':<12} {'Balance Gap':<12}")
print("-" * 110)

for model_name, results in all_models.items():
    balance_gap = abs(results['recall'] - results['specificity'])
    print(f"{model_name:<20} ", end="")
    for metric in METRIC_KEYS:
        print(f"{results[metric]:<12.1%} ", end="")
    print(f"{balance_gap:<12.1%}")

# Find best model for each metric
section("BEST PERFORMERS")

for metric in METRIC_KEYS + ('balance',):
    if metric == 'balance':
        best_model = min(all_models.items(),
                        key=lambda x: abs(x[1]['recall'] - x[1]['specificity']))
        best_val = abs(best_model[1]['recall'] - best_model[1]['specificity'])
        print(f"\nBest Balance (lowest gap): {best_model[0]}")
        print(f"  Gap: {best_val:.1%}")
        print(f"  Recall: {best_model[1]['recall']:.1%}, Specificity: {best_model[1]['specificity']:.1%}")
    else:
        best_model = max(all_models.items(), key=lambda x: x[1][metric])
        print(f"\nBest {metric.capitalize()}: {best_model[0]} ({best_model[1][metric]:.1%})")

# Detailed confusion matrices
section("CONFUSION MATRICES")

for model_name, results in all_models.items():
    conf = results['confusion']
    print(f"\n{model_name}:")
    print(f"  TP={conf['tp']:<3} FP={conf['fp']:<3}")
    print(f"  FN={conf['fn']:<3} TN={conf['tn']:<3}")

    # Calculate total predictions
    total_yes = conf['tp'] + conf['fp']
    total_no = conf['tn'] + conf['fn']
    actual_yes = conf['tp'] + conf['fn']
    actual_no = conf['tn'] + conf['fp']
    print(f"  Predicted: Yes={total_yes}, No={total_no}")
    print(f"  Actual: Yes={actual_yes}, No={actual_no}")

ft_spec = finetuned_results['specificity']
ft_recall = finetuned_results['recall']
balance_diff = abs(ft_recall - ft_spec)

section("ANALYSIS & RECOMMENDATIONS")

ft_avg = (ft_recall + ft_spec) / 2
print("\nFine-tuned LLM Balance Check:")
print(f"  Recall:      {ft_recall:.1%}")
print(f"  Specificity: {ft_spec:.1%}")
print(f"  Difference:  {balance_diff:.1%}")
print(f"  Average:     {ft_avg:.1%}")

if balance_diff < 0.15 and ft_avg > 0.65:
    print("\n✓✓✓ SUCCESS: Balanced performance achieved with Llama!")
elif balance_diff < 0.25:
    print("\n✓ GOOD: Decent balance")
elif ft_spec > 0.9 or ft_recall > 0.9:
    print("\n⚠ Still biased")
    print("  Try: Adjust NUM_ITERS (reduce if too much bias)")
else:
    print("\n→ Partial improvement")

section("RECOMMENDATION", char='-')

best_accuracy = max(all_models.items(), key=lambda x: x[1]['accuracy'])
best_f1 = max(all_models.items(), key=lambda x: x[1]['f1'])
best_balance = min(all_models.items(),
                   key=lambda x: abs(x[1]['recall'] - x[1]['specificity']))

print(f"\nFor Clinical Use:")
print(f"  Best Overall Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.1%})")
print(f"  Best F1 Score: {best_f1[0]} ({best_f1[1]['f1']:.1%})")
print(f"  Best Balance: {best_balance[0]} (gap: {abs(best_balance[1]['recall'] - best_balance[1]['specificity']):.1%})")

print(f"\nTrade-off Considerations:")
if finetuned_results['recall'] > 0.75:
    print(f"  - Fine-tuned LLM: High recall ({finetuned_results['recall']:.1%}) - fewer missed cases")
if lasso_results['specificity'] > 0.6 or rf_results['specificity'] > 0.6:
    print(f"  - Classical ML: Better specificity - fewer false alarms")
if best_balance[0] != 'Fine-tuned LLM':
    print(f"  - {best_balance[0]}: Most balanced predictions")

section("ARTIFACTS")
print(f"\nModels saved to: {OUTPUT_DIR}")

## Presentation on LLM/API/Fine-Tuning 
## Write a tutorial on how to fine-tune Llama 3.2 using LoRA for clinical tasks 
