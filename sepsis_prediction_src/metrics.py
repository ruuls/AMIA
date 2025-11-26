"""
Evaluation Metrics for Multi-Label Classification

Comprehensive metrics for evaluating sepsis prediction model:
- Per-label metrics: Precision, Recall, F1, AUROC, AUPRC
- Confusion matrices
- Aggregate metrics
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MultiLabelMetrics:
    """Compute comprehensive metrics for multi-label classification"""
    
    def __init__(self, label_names: List[str], threshold: float = 0.5):
        """
        Args:
            label_names: Names of labels (e.g., ['h6', 'h12', 'h24', 'h48'])
            threshold: Decision threshold for binary classification
        """
        self.label_names = label_names
        self.threshold = threshold
        self.num_labels = len(label_names)
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        return_per_label: bool = True
    ) -> Dict:
        """
        Compute all metrics
        
        Args:
            predictions: Predicted probabilities [num_samples, num_labels]
            labels: Ground truth labels [num_samples, num_labels]
            return_per_label: Whether to return per-label metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert probabilities to binary predictions
        pred_binary = (predictions >= self.threshold).astype(int)
        
        # Overall metrics (macro-averaged)
        metrics['accuracy'] = accuracy_score(labels, pred_binary)
        metrics['precision_macro'] = precision_score(labels, pred_binary, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(labels, pred_binary, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(labels, pred_binary, average='macro', zero_division=0)
        
        # Micro-averaged metrics
        metrics['precision_micro'] = precision_score(labels, pred_binary, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(labels, pred_binary, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(labels, pred_binary, average='micro', zero_division=0)
        
        # Weighted metrics (by support)
        metrics['precision_weighted'] = precision_score(labels, pred_binary, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(labels, pred_binary, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels, pred_binary, average='weighted', zero_division=0)
        
        # AUROC and AUPRC (macro-averaged)
        try:
            metrics['auroc_macro'] = roc_auc_score(labels, predictions, average='macro')
            metrics['auprc_macro'] = average_precision_score(labels, predictions, average='macro')
        except ValueError as e:
            # Handle case where some classes have no positive samples
            metrics['auroc_macro'] = 0.0
            metrics['auprc_macro'] = 0.0
        
        # Per-label metrics
        if return_per_label:
            for i, label_name in enumerate(self.label_names):
                label_pred = predictions[:, i]
                label_true = labels[:, i]
                label_pred_binary = pred_binary[:, i]
                
                # Basic metrics
                metrics[f'{label_name}_precision'] = precision_score(
                    label_true, label_pred_binary, zero_division=0
                )
                metrics[f'{label_name}_recall'] = recall_score(
                    label_true, label_pred_binary, zero_division=0
                )
                metrics[f'{label_name}_f1'] = f1_score(
                    label_true, label_pred_binary, zero_division=0
                )
                
                # AUROC and AUPRC
                try:
                    if len(np.unique(label_true)) > 1:  # Need both classes
                        metrics[f'{label_name}_auroc'] = roc_auc_score(label_true, label_pred)
                        metrics[f'{label_name}_auprc'] = average_precision_score(label_true, label_pred)
                    else:
                        metrics[f'{label_name}_auroc'] = 0.0
                        metrics[f'{label_name}_auprc'] = 0.0
                except ValueError:
                    metrics[f'{label_name}_auroc'] = 0.0
                    metrics[f'{label_name}_auprc'] = 0.0
                
                # Support (number of positive samples)
                metrics[f'{label_name}_support'] = int(label_true.sum())
        
        # Average AUPRC across labels (important for imbalanced data)
        auprc_values = [metrics[f'{label}_auprc'] for label in self.label_names]
        metrics['avg_auprc'] = np.mean(auprc_values)
        
        # Average AUROC across labels
        auroc_values = [metrics[f'{label}_auroc'] for label in self.label_names]
        metrics['avg_auroc'] = np.mean(auroc_values)
        
        return metrics
    
    def compute_confusion_matrices(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrix for each label
        
        Args:
            predictions: Predicted probabilities [num_samples, num_labels]
            labels: Ground truth labels [num_samples, num_labels]
            
        Returns:
            Dictionary mapping label names to confusion matrices
        """
        pred_binary = (predictions >= self.threshold).astype(int)
        
        confusion_matrices = {}
        for i, label_name in enumerate(self.label_names):
            cm = confusion_matrix(labels[:, i], pred_binary[:, i])
            confusion_matrices[label_name] = cm
        
        return confusion_matrices
    
    def print_metrics(self, metrics: Dict, title: str = "Evaluation Metrics"):
        """Pretty print metrics"""
        print("\n" + "="*80)
        print(f"{title:^80}")
        print("="*80)
        
        # Overall metrics
        print("\nOverall Metrics (Macro-Averaged):")
        print("-"*80)
        print(f"{'Accuracy:':<25} {metrics['accuracy']:.4f}")
        print(f"{'Precision (Macro):':<25} {metrics['precision_macro']:.4f}")
        print(f"{'Recall (Macro):':<25} {metrics['recall_macro']:.4f}")
        print(f"{'F1 Score (Macro):':<25} {metrics['f1_macro']:.4f}")
        print(f"{'AUROC (Macro):':<25} {metrics['auroc_macro']:.4f}")
        print(f"{'AUPRC (Macro):':<25} {metrics['auprc_macro']:.4f}")
        
        print("\nOverall Metrics (Micro-Averaged):")
        print("-"*80)
        print(f"{'Precision (Micro):':<25} {metrics['precision_micro']:.4f}")
        print(f"{'Recall (Micro):':<25} {metrics['recall_micro']:.4f}")
        print(f"{'F1 Score (Micro):':<25} {metrics['f1_micro']:.4f}")
        
        # Per-label metrics
        print("\nPer-Label Metrics:")
        print("-"*80)
        print(f"{'Label':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'AUROC':<10} | {'AUPRC':<10} | {'Support':<10}")
        print("-"*80)
        
        for label in self.label_names:
            precision = metrics.get(f'{label}_precision', 0.0)
            recall = metrics.get(f'{label}_recall', 0.0)
            f1 = metrics.get(f'{label}_f1', 0.0)
            auroc = metrics.get(f'{label}_auroc', 0.0)
            auprc = metrics.get(f'{label}_auprc', 0.0)
            support = metrics.get(f'{label}_support', 0)
            
            print(f"{label:<10} | {precision:>10.4f} | {recall:>10.4f} | {f1:>10.4f} | {auroc:>10.4f} | {auprc:>10.4f} | {support:>10}")
        
        print("="*80)
    
    def print_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray]):
        """Pretty print confusion matrices"""
        print("\n" + "="*60)
        print("CONFUSION MATRICES")
        print("="*60)
        
        for label, cm in confusion_matrices.items():
            print(f"\n{label}:")
            print("-"*40)
            print(f"{'':>15} | {'Predicted Neg':>15} | {'Predicted Pos':>15}")
            print("-"*40)
            print(f"{'Actual Neg':>15} | {cm[0,0]:>15} | {cm[0,1]:>15}")
            print(f"{'Actual Pos':>15} | {cm[1,0]:>15} | {cm[1,1]:>15}")
            
            # Calculate metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            print(f"\nSensitivity (Recall): {sensitivity:.4f}")
            print(f"Specificity:          {specificity:.4f}")
            print(f"PPV (Precision):      {ppv:.4f}")
            print(f"NPV:                  {npv:.4f}")
    
    def save_confusion_matrix_plots(
        self,
        confusion_matrices: Dict[str, np.ndarray],
        save_dir: str
    ):
        """Save confusion matrix visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for label, cm in confusion_matrices.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=ax
            )
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {label}')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/confusion_matrix_{label}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nConfusion matrix plots saved to {save_dir}")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    label_names: List[str] = None
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to run on
        label_names: Names of labels
        
    Returns:
        (metrics_dict, all_predictions, all_labels)
    """
    if label_names is None:
        label_names = ['h6', 'h12', 'h24', 'h48']
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print("Running evaluation...")
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Get probabilities and move to CPU immediately
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = labels.numpy()
            
            all_predictions.append(probs)
            all_labels.append(labels)
            
            # Clear GPU cache periodically
            if i > 0 and i % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Compute metrics
    metrics_calculator = MultiLabelMetrics(label_names)
    metrics = metrics_calculator.compute_metrics(all_predictions, all_labels)
    
    return metrics, all_predictions, all_labels


if __name__ == "__main__":
    # Test metrics computation
    print("Testing metrics computation...\n")
    
    # Create dummy data
    np.random.seed(42)
    num_samples = 100
    num_labels = 4
    label_names = ['h6', 'h12', 'h24', 'h48']
    
    # Simulate imbalanced predictions (mostly negative)
    predictions = np.random.rand(num_samples, num_labels) * 0.3  # Biased toward low probabilities
    labels = np.random.binomial(1, 0.05, (num_samples, num_labels))  # 5% positive rate
    
    # Add some true positives
    labels[0:5, :] = 1
    predictions[0:5, :] = np.random.rand(5, num_labels) * 0.5 + 0.5  # High probabilities
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Positive rate: {labels.mean():.2%}\n")
    
    # Compute metrics
    metrics_calculator = MultiLabelMetrics(label_names)
    metrics = metrics_calculator.compute_metrics(predictions, labels)
    
    # Print metrics
    metrics_calculator.print_metrics(metrics)
    
    # Compute and print confusion matrices
    confusion_matrices = metrics_calculator.compute_confusion_matrices(predictions, labels)
    metrics_calculator.print_confusion_matrices(confusion_matrices)
    
    print("\n✅ Metrics computation test PASSED!")
