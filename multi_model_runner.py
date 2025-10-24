"""
Multi-Model Fine-Tuning Runner
Runs multiple LLM models sequentially with resource management and comparison

Usage:
  python multi_model_runner.py
  python multi_model_runner.py --config custom_config.json
  python multi_model_runner.py --skip-baseline --skip-classical-ml  # Only fine-tune LLMs
"""

import json
import subprocess
import sys
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import argparse

# Try to import MLX and psutil for resource monitoring
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available for memory cleanup")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available for resource monitoring (pip install psutil)")


class ResourceMonitor:
    """Monitor system resources during training"""

    def __init__(self):
        self.available = PSUTIL_AVAILABLE

    def get_memory_info(self):
        """Get current memory usage"""
        if not self.available:
            return None

        process = psutil.Process()
        mem_info = process.memory_info()
        virtual_mem = psutil.virtual_memory()

        return {
            'process_mb': mem_info.rss / 1024 / 1024,
            'system_percent': virtual_mem.percent,
            'system_available_gb': virtual_mem.available / 1024 / 1024 / 1024
        }

    def print_memory_status(self, prefix=""):
        """Print current memory status"""
        mem = self.get_memory_info()
        if mem:
            print(f"{prefix}Memory: Process={mem['process_mb']:.1f}MB, "
                  f"System={mem['system_percent']:.1f}% used, "
                  f"Available={mem['system_available_gb']:.1f}GB")


def cleanup_memory():
    """Aggressive memory cleanup between models"""
    print("\nCleaning up memory...")
    gc.collect()
    if MLX_AVAILABLE:
        try:
            mx.metal.clear_cache()
            print("  ✓ MLX cache cleared")
        except Exception as e:
            print(f"  Warning: Could not clear MLX cache: {e}")
    gc.collect()  # Second pass
    print("  ✓ Python GC completed")


def load_config(config_path: str) -> Dict:
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_single_model(model_config: Dict, common_config: Dict,
                     skip_baseline: bool = False,
                     skip_classical_ml: bool = False) -> Dict:
    """Run fine-tuning for a single model"""

    model_name = model_config['name']
    print(f"\n{'='*80}")
    print(f"STARTING: {model_name}")
    print(f"{'='*80}\n")

    # Prepare results file path
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{model_name.replace('/', '_')}_{timestamp}.json"

    # Build command
    cmd = [
        sys.executable,
        "rct_ft_single.py",
        "--model", model_config['model_id'],
        "--output-dir", model_config['output_dir'],
        "--model-name", model_name,
        "--lora-rank", str(model_config['lora_rank']),
        "--lora-alpha", str(model_config['lora_alpha']),
        "--learning-rate", str(model_config['learning_rate']),
        "--num-iters", str(model_config['num_iters']),
        "--batch-size", str(model_config['batch_size']),
        "--data-path", common_config['data_path'],
        "--train-ratio", str(common_config['train_ratio']),
        "--val-ratio", str(common_config['val_ratio']),
        "--test-ratio", str(common_config['test_ratio']),
        "--random-seed", str(common_config['random_seed']),
        "--steps-per-eval", str(common_config['steps_per_eval']),
        "--steps-per-report", str(common_config['steps_per_report']),
        "--results-file", str(results_file)
    ]

    if common_config.get('use_balanced_sampling', True):
        cmd.append("--use-balanced-sampling")

    if skip_baseline:
        cmd.append("--skip-baseline")

    if skip_classical_ml:
        cmd.append("--skip-classical-ml")

    print(f"Command: {' '.join(cmd)}\n")

    # Run the model
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        elapsed = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"✓ COMPLETED: {model_name} (took {elapsed/60:.1f} minutes)")
        print(f"{'='*80}\n")

        # Load and return results
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Results file not found: {results_file}")
            return None

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✗ FAILED: {model_name} (after {elapsed/60:.1f} minutes)")
        print(f"Error code: {e.returncode}")
        print(f"{'='*80}\n")
        return None


def print_summary_table(all_results: List[Dict]):
    """Print comparison table of all models"""
    print("\n" + "="*120)
    print("FINAL COMPARISON: ALL MODELS")
    print("="*120 + "\n")

    # Header
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1':<12} {'Balance Gap':<12}")
    print("-" * 120)

    # Each model
    for result in all_results:
        if not result or 'results' not in result:
            continue

        model_name = result['model_name']
        ft_results = result['results']['finetuned']

        balance_gap = abs(ft_results['recall'] - ft_results['specificity'])

        print(f"{model_name:<25} ", end="")
        print(f"{ft_results['accuracy']:<12.1%} ", end="")
        print(f"{ft_results['precision']:<12.1%} ", end="")
        print(f"{ft_results['recall']:<12.1%} ", end="")
        print(f"{ft_results['specificity']:<12.1%} ", end="")
        print(f"{ft_results['f1']:<12.1%} ", end="")
        print(f"{balance_gap:<12.1%}")

    print("\n" + "="*120)


def print_best_models(all_results: List[Dict]):
    """Print best performing models for each metric"""
    print("\nBEST PERFORMERS:")
    print("-" * 80)

    valid_results = [r for r in all_results if r and 'results' in r]
    if not valid_results:
        print("No valid results to compare")
        return

    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']

    for metric in metrics:
        best = max(valid_results,
                   key=lambda r: r['results']['finetuned'][metric])
        value = best['results']['finetuned'][metric]
        print(f"\nBest {metric.capitalize()}: {best['model_name']}")
        print(f"  Value: {value:.1%}")

    # Best balance
    best_balance = min(valid_results,
                       key=lambda r: abs(r['results']['finetuned']['recall'] -
                                        r['results']['finetuned']['specificity']))
    gap = abs(best_balance['results']['finetuned']['recall'] -
              best_balance['results']['finetuned']['specificity'])
    print(f"\nBest Balance: {best_balance['model_name']}")
    print(f"  Gap: {gap:.1%}")
    print(f"  Recall: {best_balance['results']['finetuned']['recall']:.1%}, "
          f"Specificity: {best_balance['results']['finetuned']['specificity']:.1%}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run multiple model fine-tuning jobs')
    parser.add_argument('--config', type=str, default='model_configs.json',
                       help='Path to model configuration file')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline evaluation for all models')
    parser.add_argument('--skip-classical-ml', action='store_true',
                       help='Skip classical ML for all models')
    parser.add_argument('--sleep-between', type=int, default=30,
                       help='Seconds to sleep between models')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Only run specific models by name')

    args = parser.parse_args()

    print("="*80)
    print("MULTI-MODEL FINE-TUNING RUNNER")
    print("="*80)

    # Load configuration
    config = load_config(args.config)
    models = config['models']
    common_config = config['common_config']

    # Filter models if specified
    if args.models:
        models = [m for m in models if m['name'] in args.models]
        print(f"\nRunning selected models: {[m['name'] for m in models]}")

    print(f"\nConfiguration loaded: {len(models)} models to process")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']}")

    print(f"\nSettings:")
    print(f"  Data: {common_config['data_path']}")
    print(f"  Skip baseline: {args.skip_baseline}")
    print(f"  Skip classical ML: {args.skip_classical_ml}")
    print(f"  Sleep between models: {args.sleep_between}s")

    # Initialize resource monitor
    monitor = ResourceMonitor()
    monitor.print_memory_status("\nInitial ")

    # Run each model
    all_results = []
    start_time = time.time()

    for i, model_config in enumerate(models, 1):
        print(f"\n\n{'#'*80}")
        print(f"# MODEL {i}/{len(models)}")
        print(f"{'#'*80}\n")

        monitor.print_memory_status("Before ")

        # Run the model
        result = run_single_model(
            model_config,
            common_config,
            skip_baseline=args.skip_baseline,
            skip_classical_ml=args.skip_classical_ml
        )

        if result:
            all_results.append(result)

        # Cleanup and wait
        cleanup_memory()
        monitor.print_memory_status("After cleanup ")

        if i < len(models):
            print(f"\nWaiting {args.sleep_between}s before next model...")
            time.sleep(args.sleep_between)

    # Print final summary
    total_time = time.time() - start_time

    print("\n\n" + "="*120)
    print("ALL MODELS COMPLETED")
    print("="*120)
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Successful: {len(all_results)}/{len(models)} models")

    if all_results:
        print_summary_table(all_results)
        print_best_models(all_results)

        # Save combined results
        combined_file = Path("./results") / f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(combined_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_time_seconds': total_time,
                'models': all_results
            }, f, indent=2)
        print(f"\n✓ Combined results saved to: {combined_file}")

    print("\n" + "="*120)


if __name__ == "__main__":
    main()
