"""
Compare LoRA vs QLoRA training results.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_metrics(run_dir: str) -> Dict:
    """Load training metrics from a run directory."""
    metrics_file = Path(run_dir) / "trainer_state.json"
    
    if not metrics_file.exists():
        return {}
    
    with open(metrics_file, 'r') as f:
        state = json.load(f)
    
    # Extract best metrics
    log_history = state.get('log_history', [])
    
    best_metrics = {}
    for entry in log_history:
        if 'eval_avg_auprc' in entry:
            for key, value in entry.items():
                if key.startswith('eval_'):
                    metric_name = key.replace('eval_', '')
                    if metric_name not in best_metrics or value > best_metrics.get(metric_name, 0):
                        best_metrics[metric_name] = value
    
    return best_metrics


def get_model_size(run_dir: str) -> float:
    """Get model checkpoint size in MB."""
    checkpoint_dir = Path(run_dir) / "best_model"
    
    if not checkpoint_dir.exists():
        return 0.0
    
    total_size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
    return total_size / (1024 * 1024)  # Convert to MB


def get_training_time(run_dir: str) -> float:
    """Get total training time in hours."""
    metrics_file = Path(run_dir) / "trainer_state.json"
    
    if not metrics_file.exists():
        return 0.0
    
    with open(metrics_file, 'r') as f:
        state = json.load(f)
    
    log_history = state.get('log_history', [])
    
    if len(log_history) < 2:
        return 0.0
    
    # Calculate time from first to last log entry
    start_time = log_history[0].get('epoch', 0)
    end_time = log_history[-1].get('epoch', 0)
    
    # Rough estimate (actual training time would need timestamps)
    return end_time - start_time


def compare_runs():
    """Compare LoRA and QLoRA runs."""
    
    print("\n" + "="*80)
    print("  LoRA vs QLoRA Comparison Report")
    print("="*80 + "\n")
    
    # Directories
    qlora_dir = "../outputs/qlora_run"
    lora_dir = "../outputs/lora_run"
    
    # Load metrics
    qlora_metrics = load_metrics(qlora_dir)
    lora_metrics = load_metrics(lora_dir)
    
    if not qlora_metrics or not lora_metrics:
        print("❌ Could not find metrics for both runs.")
        print(f"QLoRA metrics found: {bool(qlora_metrics)}")
        print(f"LoRA metrics found: {bool(lora_metrics)}")
        return
    
    # Get model sizes
    qlora_size = get_model_size(qlora_dir)
    lora_size = get_model_size(lora_dir)
    
    # Create comparison table
    comparison = {
        'Metric': [],
        'QLoRA (4-bit)': [],
        'LoRA (16-bit)': [],
        'Difference': []
    }
    
    # Performance metrics
    metric_names = {
        'avg_auprc': 'Average AUPRC',
        'avg_auroc': 'Average AUROC',
        'avg_f1': 'Average F1',
        'h6_auprc': 'H6 AUPRC',
        'h12_auprc': 'H12 AUPRC',
        'h24_auprc': 'H24 AUPRC',
        'h48_auprc': 'H48 AUPRC'
    }
    
    for metric_key, metric_name in metric_names.items():
        qlora_val = qlora_metrics.get(metric_key, 0.0)
        lora_val = lora_metrics.get(metric_key, 0.0)
        diff = lora_val - qlora_val
        
        comparison['Metric'].append(metric_name)
        comparison['QLoRA (4-bit)'].append(f"{qlora_val:.4f}")
        comparison['LoRA (16-bit)'].append(f"{lora_val:.4f}")
        comparison['Difference'].append(f"{diff:+.4f}")
    
    # Model size comparison
    comparison['Metric'].append('Model Size (MB)')
    comparison['QLoRA (4-bit)'].append(f"{qlora_size:.1f}")
    comparison['LoRA (16-bit)'].append(f"{lora_size:.1f}")
    comparison['Difference'].append(f"{lora_size - qlora_size:+.1f}")
    
    # Create DataFrame
    df = pd.DataFrame(comparison)
    
    # Print table
    print(df.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    # Summary
    print("📊 Summary:")
    print(f"  • QLoRA Average AUPRC: {qlora_metrics.get('avg_auprc', 0):.4f}")
    print(f"  • LoRA Average AUPRC:  {lora_metrics.get('avg_auprc', 0):.4f}")
    print(f"  • Performance difference: {lora_metrics.get('avg_auprc', 0) - qlora_metrics.get('avg_auprc', 0):+.4f}")
    print(f"\n  • QLoRA model size: {qlora_size:.1f} MB")
    print(f"  • LoRA model size:  {lora_size:.1f} MB")
    print(f"  • Size ratio: {lora_size / qlora_size:.2f}x\n")
    
    # Recommendations
    print("💡 Recommendation:")
    if abs(lora_metrics.get('avg_auprc', 0) - qlora_metrics.get('avg_auprc', 0)) < 0.01:
        print("  ✅ QLoRA recommended - similar performance, much smaller size")
    elif lora_metrics.get('avg_auprc', 0) > qlora_metrics.get('avg_auprc', 0) + 0.02:
        print("  ⚠️  LoRA shows better performance - consider if worth the size")
    else:
        print("  ✅ QLoRA recommended - good performance/size trade-off")
    
    print("\n" + "="*80 + "\n")
    
    # Save report
    report_path = "../outputs/comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("  LoRA vs QLoRA Comparison Report\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Summary:\n")
        f.write(f"  QLoRA AUPRC: {qlora_metrics.get('avg_auprc', 0):.4f}\n")
        f.write(f"  LoRA AUPRC:  {lora_metrics.get('avg_auprc', 0):.4f}\n")
        f.write(f"  Difference:  {lora_metrics.get('avg_auprc', 0) - qlora_metrics.get('avg_auprc', 0):+.4f}\n")
    
    print(f"📄 Report saved to: {report_path}")


if __name__ == "__main__":
    compare_runs()
