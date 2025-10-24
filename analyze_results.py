"""
Results Analysis Script
Analyze and visualize results from multi-model comparison

Usage:
  python analyze_results.py results/combined_results_20241022_*.json
  python analyze_results.py --results-dir ./results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys


def load_results(file_path: str) -> Dict:
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_detailed_comparison(results: Dict):
    """Print detailed comparison of all models"""
    models = results['models']

    print("\n" + "="*140)
    print("DETAILED MODEL COMPARISON")
    print("="*140 + "\n")

    # Fine-tuned LLM comparison
    print("FINE-TUNED LLM PERFORMANCE:")
    print("-" * 140)
    print(f"{'Model':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'Spec':<8} {'F1':<8} {'Gap':<8} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
    print("-" * 140)

    for model in models:
        name = model['model_name']
        ft = model['results']['finetuned']
        conf = ft['confusion']
        gap = abs(ft['recall'] - ft['specificity'])

        print(f"{name:<30} ", end="")
        print(f"{ft['accuracy']:<8.1%} ", end="")
        print(f"{ft['precision']:<8.1%} ", end="")
        print(f"{ft['recall']:<8.1%} ", end="")
        print(f"{ft['specificity']:<8.1%} ", end="")
        print(f"{ft['f1']:<8.1%} ", end="")
        print(f"{gap:<8.1%} ", end="")
        print(f"{conf['tp']:<6} ", end="")
        print(f"{conf['tn']:<6} ", end="")
        print(f"{conf['fp']:<6} ", end="")
        print(f"{conf['fn']:<6}")

    # Baseline comparison (if available)
    baseline_available = any(m['results'].get('baseline') for m in models)
    if baseline_available:
        print("\n\nBASELINE (PRE-FINETUNING) PERFORMANCE:")
        print("-" * 140)
        print(f"{'Model':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'Spec':<8} {'F1':<8} {'Gap':<8} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
        print("-" * 140)

        for model in models:
            if not model['results'].get('baseline'):
                continue

            name = model['model_name']
            bl = model['results']['baseline']
            conf = bl['confusion']
            gap = abs(bl['recall'] - bl['specificity'])

            print(f"{name:<30} ", end="")
            print(f"{bl['accuracy']:<8.1%} ", end="")
            print(f"{bl['precision']:<8.1%} ", end="")
            print(f"{bl['recall']:<8.1%} ", end="")
            print(f"{bl['specificity']:<8.1%} ", end="")
            print(f"{bl['f1']:<8.1%} ", end="")
            print(f"{gap:<8.1%} ", end="")
            print(f"{conf['tp']:<6} ", end="")
            print(f"{conf['tn']:<6} ", end="")
            print(f"{conf['fp']:<6} ", end="")
            print(f"{conf['fn']:<6}")

    # Improvement analysis
    if baseline_available:
        print("\n\nFINE-TUNING IMPROVEMENT:")
        print("-" * 100)
        print(f"{'Model':<30} {'Acc Δ':<12} {'Prec Δ':<12} {'Rec Δ':<12} {'Spec Δ':<12} {'F1 Δ':<12}")
        print("-" * 100)

        for model in models:
            if not model['results'].get('baseline'):
                continue

            name = model['model_name']
            bl = model['results']['baseline']
            ft = model['results']['finetuned']

            acc_delta = ft['accuracy'] - bl['accuracy']
            prec_delta = ft['precision'] - bl['precision']
            rec_delta = ft['recall'] - bl['recall']
            spec_delta = ft['specificity'] - bl['specificity']
            f1_delta = ft['f1'] - bl['f1']

            print(f"{name:<30} ", end="")
            print(f"{acc_delta:+11.1%} ", end="")
            print(f"{prec_delta:+11.1%} ", end="")
            print(f"{rec_delta:+11.1%} ", end="")
            print(f"{spec_delta:+11.1%} ", end="")
            print(f"{f1_delta:+11.1%}")


def print_rankings(results: Dict):
    """Print model rankings for each metric"""
    models = results['models']

    print("\n" + "="*100)
    print("MODEL RANKINGS")
    print("="*100 + "\n")

    metrics = [
        ('accuracy', 'Accuracy', False),
        ('precision', 'Precision', False),
        ('recall', 'Recall', False),
        ('specificity', 'Specificity', False),
        ('f1', 'F1 Score', False),
        ('balance', 'Balance (lowest gap)', True)
    ]

    for metric_key, metric_name, reverse in metrics:
        print(f"\n{metric_name}:")
        print("-" * 60)

        if metric_key == 'balance':
            sorted_models = sorted(
                models,
                key=lambda m: abs(m['results']['finetuned']['recall'] -
                                m['results']['finetuned']['specificity']),
                reverse=reverse
            )
        else:
            sorted_models = sorted(
                models,
                key=lambda m: m['results']['finetuned'][metric_key],
                reverse=not reverse
            )

        for i, model in enumerate(sorted_models, 1):
            ft = model['results']['finetuned']
            if metric_key == 'balance':
                value = abs(ft['recall'] - ft['specificity'])
                print(f"  {i}. {model['model_name']:<30} Gap: {value:.1%} "
                      f"(Rec: {ft['recall']:.1%}, Spec: {ft['specificity']:.1%})")
            else:
                value = ft[metric_key]
                print(f"  {i}. {model['model_name']:<30} {value:.1%}")


def print_hyperparameter_comparison(results: Dict):
    """Print hyperparameters used for each model"""
    models = results['models']

    print("\n" + "="*100)
    print("HYPERPARAMETER COMPARISON")
    print("="*100 + "\n")

    print(f"{'Model':<30} {'LoRA Rank':<12} {'LoRA Alpha':<12} {'Learn Rate':<15} {'Iterations':<12} {'Batch Size':<12}")
    print("-" * 100)

    for model in models:
        name = model['model_name']
        hp = model['hyperparameters']

        print(f"{name:<30} ", end="")
        print(f"{hp['lora_rank']:<12} ", end="")
        print(f"{hp['lora_alpha']:<12} ", end="")
        print(f"{hp['learning_rate']:<15.2e} ", end="")
        print(f"{hp['num_iters']:<12} ", end="")
        print(f"{hp['batch_size']:<12}")


def print_recommendations(results: Dict):
    """Print recommendations based on results"""
    models = results['models']

    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100 + "\n")

    # Best overall
    best_f1 = max(models, key=lambda m: m['results']['finetuned']['f1'])
    print(f"Best Overall (F1 Score): {best_f1['model_name']}")
    print(f"  F1: {best_f1['results']['finetuned']['f1']:.1%}")
    print(f"  Accuracy: {best_f1['results']['finetuned']['accuracy']:.1%}")

    # Best balance
    best_balance = min(models,
                       key=lambda m: abs(m['results']['finetuned']['recall'] -
                                        m['results']['finetuned']['specificity']))
    gap = abs(best_balance['results']['finetuned']['recall'] -
              best_balance['results']['finetuned']['specificity'])
    print(f"\nBest Balanced: {best_balance['model_name']}")
    print(f"  Balance Gap: {gap:.1%}")
    print(f"  Recall: {best_balance['results']['finetuned']['recall']:.1%}")
    print(f"  Specificity: {best_balance['results']['finetuned']['specificity']:.1%}")

    # Best sensitivity (recall)
    best_recall = max(models, key=lambda m: m['results']['finetuned']['recall'])
    print(f"\nBest Sensitivity (fewer missed cases): {best_recall['model_name']}")
    print(f"  Recall: {best_recall['results']['finetuned']['recall']:.1%}")
    print(f"  Specificity: {best_recall['results']['finetuned']['specificity']:.1%}")

    # Clinical use case recommendations
    print("\n\nClinical Use Case Recommendations:")
    print("-" * 80)
    print("\n1. Screening (minimize missed cases):")
    print(f"   → Use: {best_recall['model_name']}")
    print(f"   Rationale: High recall ({best_recall['results']['finetuned']['recall']:.1%}) catches most cases")

    print("\n2. Balanced Clinical Decision Support:")
    print(f"   → Use: {best_balance['model_name']}")
    print(f"   Rationale: Good balance between sensitivity and specificity")

    print("\n3. Overall Performance:")
    print(f"   → Use: {best_f1['model_name']}")
    print(f"   Rationale: Best F1 score indicates optimal precision-recall balance")


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-model results')
    parser.add_argument('file', nargs='?', help='Path to combined results JSON file')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing results (uses most recent if file not specified)')

    args = parser.parse_args()

    # Find results file
    if args.file:
        results_file = Path(args.file)
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)

        # Find most recent combined results file
        combined_files = list(results_dir.glob("combined_results_*.json"))
        if not combined_files:
            print(f"Error: No combined results files found in {results_dir}")
            sys.exit(1)

        results_file = max(combined_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent results: {results_file.name}")

    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    # Load and analyze
    results = load_results(results_file)

    print(f"\nAnalyzing results from: {results_file}")
    print(f"Timestamp: {results.get('timestamp', 'unknown')}")
    print(f"Total time: {results.get('total_time_seconds', 0)/60:.1f} minutes")
    print(f"Models: {len(results['models'])}")

    # Print all analyses
    print_detailed_comparison(results)
    print_rankings(results)
    print_hyperparameter_comparison(results)
    print_recommendations(results)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
