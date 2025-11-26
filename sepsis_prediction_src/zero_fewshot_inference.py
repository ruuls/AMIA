"""
Zero-Shot and Few-Shot In-Context Learning Inference for Sepsis Prediction

This script performs inference on the test set using:
1. Zero-shot (no examples)
2. Few-shot (5, 10, 15, 20 shots)

Evaluates performance stratified by time horizons (h6, h12, h24, h48)
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
from tqdm import tqdm
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SepsisExample:
    """Data class for sepsis prediction examples"""
    text: str
    h6: int
    h12: int
    h24: int
    h48: int
    
    def to_dict(self):
        return {
            'text': self.text,
            'h6': self.h6,
            'h12': self.h12,
            'h24': self.h24,
            'h48': self.h48
        }


class FewShotPromptBuilder:
    """Builds prompts for zero-shot and few-shot inference"""
    
    def __init__(self, time_horizons: List[str] = ['h6', 'h12', 'h24', 'h48']):
        self.time_horizons = time_horizons
        
    def build_zero_shot_prompt(self, clinical_note: str) -> str:
        """Build zero-shot prompt with task instruction"""
        prompt = f"""You are a medical AI assistant trained to predict sepsis risk from clinical notes.

Task: Analyze the following clinical note and predict the risk of sepsis at four time horizons:
- h6: Within 6 hours
- h12: Within 12 hours  
- h24: Within 24 hours
- h48: Within 48 hours

For each time horizon, respond with either "Yes" (high risk) or "No" (low risk).

Clinical Note:
{clinical_note}

Predictions:
h6:"""
        return prompt
    
    def build_few_shot_prompt(self, 
                            clinical_note: str, 
                            examples: List[SepsisExample],
                            n_shots: int) -> str:
        """Build few-shot prompt with example demonstrations"""
        # Sample examples if needed
        if len(examples) > n_shots:
            examples = random.sample(examples, n_shots)
        
        prompt = f"""You are a medical AI assistant trained to predict sepsis risk from clinical notes.

Task: Analyze clinical notes and predict the risk of sepsis at four time horizons:
- h6: Within 6 hours
- h12: Within 12 hours  
- h24: Within 24 hours
- h48: Within 48 hours

For each time horizon, respond with either "Yes" (high risk) or "No" (low risk).

Here are {n_shots} examples:

"""
        # Add examples
        for i, example in enumerate(examples, 1):
            prompt += f"""Example {i}:
Clinical Note:
{example.text[:500]}...

Predictions:
h6: {"Yes" if example.h6 == 1 else "No"}
h12: {"Yes" if example.h12 == 1 else "No"}
h24: {"Yes" if example.h24 == 1 else "No"}
h48: {"Yes" if example.h48 == 1 else "No"}

"""
        
        # Add test case
        prompt += f"""Now analyze this new clinical note:

Clinical Note:
{clinical_note}

Predictions:
h6:"""
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, int]:
        """Parse model response to extract predictions"""
        predictions = {}
        response_lower = response.lower()
        
        for horizon in self.time_horizons:
            # Try to find the prediction for each horizon
            # Look for patterns like "h6: yes" or "h6: no"
            if f"{horizon}:" in response_lower:
                idx = response_lower.index(f"{horizon}:")
                snippet = response_lower[idx:idx+50]
                
                if "yes" in snippet:
                    predictions[horizon] = 1
                elif "no" in snippet:
                    predictions[horizon] = 0
                else:
                    # Default to negative if unclear
                    predictions[horizon] = 0
            else:
                # If horizon not found, default to negative
                predictions[horizon] = 0
        
        return predictions


class SepsisMetricsCalculator:
    """Calculate comprehensive metrics stratified by time horizon"""
    
    def __init__(self, time_horizons: List[str] = ['h6', 'h12', 'h24', 'h48']):
        self.time_horizons = time_horizons
    
    def compute_metrics(self, 
                       predictions: Dict[str, List[int]], 
                       labels: Dict[str, List[int]]) -> Dict:
        """Compute metrics for all time horizons"""
        metrics = {}
        
        for horizon in self.time_horizons:
            y_true = np.array(labels[horizon])
            y_pred = np.array(predictions[horizon])
            
            # Basic metrics
            metrics[f'{horizon}_accuracy'] = accuracy_score(y_true, y_pred)
            metrics[f'{horizon}_precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics[f'{horizon}_recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics[f'{horizon}_f1'] = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix with error handling
            try:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    # Handle edge case where only one class is present
                    if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
                        if np.unique(y_true)[0] == 0 and np.unique(y_pred)[0] == 0:
                            tn, fp, fn, tp = len(y_true), 0, 0, 0
                        elif np.unique(y_true)[0] == 1 and np.unique(y_pred)[0] == 1:
                            tn, fp, fn, tp = 0, 0, 0, len(y_true)
                        else:
                            tn, fp, fn, tp = 0, len(y_pred), len(y_true), 0
                    else:
                        tn, fp, fn, tp = 0, 0, 0, 0
            except Exception as e:
                print(f"Warning: Error computing confusion matrix for {horizon}: {e}")
                tn, fp, fn, tp = 0, 0, 0, 0
            
            metrics[f'{horizon}_true_positives'] = int(tp)
            metrics[f'{horizon}_false_positives'] = int(fp)
            metrics[f'{horizon}_true_negatives'] = int(tn)
            metrics[f'{horizon}_false_negatives'] = int(fn)
            
            # Specificity and sensitivity
            metrics[f'{horizon}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'{horizon}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Support (number of positive samples)
            metrics[f'{horizon}_support'] = int(np.sum(y_true))
        
        # Average metrics across all horizons
        metrics['avg_accuracy'] = np.mean([metrics[f'{h}_accuracy'] for h in self.time_horizons])
        metrics['avg_precision'] = np.mean([metrics[f'{h}_precision'] for h in self.time_horizons])
        metrics['avg_recall'] = np.mean([metrics[f'{h}_recall'] for h in self.time_horizons])
        metrics['avg_f1'] = np.mean([metrics[f'{h}_f1'] for h in self.time_horizons])
        
        return metrics
    
    def print_metrics(self, metrics: Dict, title: str = "Metrics"):
        """Pretty print metrics"""
        print("\n" + "="*80)
        print(f"{title}")
        print("="*80)
        
        # Print per-horizon metrics
        for horizon in self.time_horizons:
            print(f"\n{horizon.upper()} Metrics:")
            print(f"  Accuracy:    {metrics[f'{horizon}_accuracy']:.4f}")
            print(f"  Precision:   {metrics[f'{horizon}_precision']:.4f}")
            print(f"  Recall:      {metrics[f'{horizon}_recall']:.4f}")
            print(f"  F1 Score:    {metrics[f'{horizon}_f1']:.4f}")
            print(f"  Sensitivity: {metrics[f'{horizon}_sensitivity']:.4f}")
            print(f"  Specificity: {metrics[f'{horizon}_specificity']:.4f}")
            print(f"  Support:     {metrics[f'{horizon}_support']}")
            print(f"  TP/FP/TN/FN: {metrics[f'{horizon}_true_positives']}/{metrics[f'{horizon}_false_positives']}/{metrics[f'{horizon}_true_negatives']}/{metrics[f'{horizon}_false_negatives']}")
        
        # Print average metrics
        print(f"\nAverage Across All Horizons:")
        print(f"  Accuracy:  {metrics['avg_accuracy']:.4f}")
        print(f"  Precision: {metrics['avg_precision']:.4f}")
        print(f"  Recall:    {metrics['avg_recall']:.4f}")
        print(f"  F1 Score:  {metrics['avg_f1']:.4f}")
        print("="*80 + "\n")


class ZeroFewShotInference:
    """Main inference class for zero-shot and few-shot prediction"""
    
    def __init__(self, 
                 model_path: str,
                 use_quantization: bool = True,
                 device: str = 'cuda'):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to pre-trained model
            use_quantization: Whether to use 4-bit quantization
            device: Device to use for inference
        """
        self.device = device
        self.model_path = model_path
        self.time_horizons = ['h6', 'h12', 'h24', 'h48']
        
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        self.model.eval()
        print(f"Model loaded successfully on {device}")
        
        self.prompt_builder = FewShotPromptBuilder(self.time_horizons)
        self.metrics_calculator = SepsisMetricsCalculator(self.time_horizons)
    
    def load_data(self, data_path: str) -> List[SepsisExample]:
        """Load data from JSONL file"""
        examples = []
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Handle both formats: direct keys or nested in 'output'
                if 'output' in data:
                    # Format: {"input": "...", "output": {"h6": 0, ...}}
                    example = SepsisExample(
                        text=data['input'],
                        h6=data['output']['h6'],
                        h12=data['output']['h12'],
                        h24=data['output']['h24'],
                        h48=data['output']['h48']
                    )
                else:
                    # Format: {"text": "...", "h6": 0, ...}
                    example = SepsisExample(
                        text=data.get('text', data.get('input', '')),
                        h6=data['h6'],
                        h12=data['h12'],
                        h24=data['h24'],
                        h48=data['h48']
                    )
                examples.append(example)
        return examples
    
    def generate_prediction(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate prediction from model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for more deterministic outputs
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part (not the input)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def run_zero_shot(self, test_examples: List[SepsisExample]) -> Tuple[Dict, Dict]:
        """Run zero-shot inference on test set"""
        print("\n" + "="*80)
        print("RUNNING ZERO-SHOT INFERENCE")
        print("="*80)
        
        predictions = {h: [] for h in self.time_horizons}
        labels = {h: [] for h in self.time_horizons}
        
        for example in tqdm(test_examples, desc="Zero-shot inference"):
            # Build prompt
            prompt = self.prompt_builder.build_zero_shot_prompt(example.text)
            
            # Generate prediction
            response = self.generate_prediction(prompt)
            
            # Parse response
            pred = self.prompt_builder.parse_response(response)
            
            # Store predictions and labels
            for horizon in self.time_horizons:
                predictions[horizon].append(pred.get(horizon, 0))
                labels[horizon].append(getattr(example, horizon))
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_metrics(predictions, labels)
        
        return predictions, labels, metrics
    
    def run_few_shot(self, 
                     test_examples: List[SepsisExample],
                     train_examples: List[SepsisExample],
                     n_shots: int) -> Tuple[Dict, Dict, Dict]:
        """Run few-shot inference on test set"""
        print("\n" + "="*80)
        print(f"RUNNING {n_shots}-SHOT INFERENCE")
        print("="*80)
        
        predictions = {h: [] for h in self.time_horizons}
        labels = {h: [] for h in self.time_horizons}
        
        # Select diverse examples for few-shot learning
        # Balance positive and negative examples
        few_shot_examples = self._select_balanced_examples(train_examples, n_shots)
        
        for example in tqdm(test_examples, desc=f"{n_shots}-shot inference"):
            # Build prompt with examples
            prompt = self.prompt_builder.build_few_shot_prompt(
                example.text, 
                few_shot_examples, 
                n_shots
            )
            
            # Generate prediction
            response = self.generate_prediction(prompt)
            
            # Parse response
            pred = self.prompt_builder.parse_response(response)
            
            # Store predictions and labels
            for horizon in self.time_horizons:
                predictions[horizon].append(pred.get(horizon, 0))
                labels[horizon].append(getattr(example, horizon))
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_metrics(predictions, labels)
        
        return predictions, labels, metrics
    
    def _select_balanced_examples(self, 
                                  examples: List[SepsisExample], 
                                  n_shots: int) -> List[SepsisExample]:
        """Select balanced examples (mix of positive and negative cases)"""
        # Separate positive and negative examples based on h6 (earliest horizon)
        positive = [ex for ex in examples if ex.h6 == 1]
        negative = [ex for ex in examples if ex.h6 == 0]
        
        # Sample balanced set
        n_positive = min(len(positive), n_shots // 2)
        n_negative = n_shots - n_positive
        
        selected = (
            random.sample(positive, n_positive) + 
            random.sample(negative, min(len(negative), n_negative))
        )
        
        random.shuffle(selected)
        return selected


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot and Few-shot Inference for Sepsis Prediction"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/ubuntu/nahiyan_workspace/s3_bucket/models/meditron-7b',
        help='Path to pre-trained model'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/ubuntu/nahiyan_workspace/s3_bucket/data',
        help='Directory containing train/test JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/ubuntu/nahiyan_workspace/sepsis_prediction/outputs/zero_fewshot',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n_test_samples',
        type=int,
        default=None,
        help='Number of test samples to use (None for all)'
    )
    parser.add_argument(
        '--shot_configs',
        type=int,
        nargs='+',
        default=[0, 5, 10],
        help='List of shot configurations to test (e.g., 0 5 10 15 20)'
    )
    parser.add_argument(
        '--use_quantization',
        action='store_true',
        default=True,
        help='Use 4-bit quantization'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference pipeline
    inference = ZeroFewShotInference(
        model_path=args.model_path,
        use_quantization=args.use_quantization,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load data
    print("\nLoading data...")
    test_path = Path(args.data_dir) / "test.jsonl"
    train_path = Path(args.data_dir) / "train.jsonl"
    
    test_examples = inference.load_data(test_path)
    print(f"Loaded {len(test_examples)} test examples")
    
    # Limit test samples if specified
    if args.n_test_samples:
        test_examples = random.sample(test_examples, min(args.n_test_samples, len(test_examples)))
        print(f"Using {len(test_examples)} test samples")
    
    # Load training examples for few-shot learning
    if any(shot > 0 for shot in args.shot_configs):
        train_examples = inference.load_data(train_path)
        print(f"Loaded {len(train_examples)} training examples for few-shot")
    else:
        train_examples = []
    
    # Store all results
    all_results = {}
    
    # Run inference for each shot configuration
    for n_shots in args.shot_configs:
        if n_shots == 0:
            # Zero-shot
            predictions, labels, metrics = inference.run_zero_shot(test_examples)
            config_name = "zero_shot"
        else:
            # Few-shot
            predictions, labels, metrics = inference.run_few_shot(
                test_examples, 
                train_examples, 
                n_shots
            )
            config_name = f"{n_shots}_shot"
        
        # Print metrics
        inference.metrics_calculator.print_metrics(
            metrics, 
            title=f"{config_name.replace('_', '-').upper()} RESULTS"
        )
        
        # Save results
        result = {
            'config': config_name,
            'n_shots': n_shots,
            'n_test_samples': len(test_examples),
            'metrics': metrics,
            'predictions': predictions,
            'labels': labels
        }
        all_results[config_name] = result
        
        # Save individual result
        result_path = output_dir / f"{config_name}_results.json"
        with open(result_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = {
                'config': result['config'],
                'n_shots': result['n_shots'],
                'n_test_samples': result['n_test_samples'],
                'metrics': result['metrics'],
                'predictions': {k: [int(x) for x in v] for k, v in result['predictions'].items()},
                'labels': {k: [int(x) for x in v] for k, v in result['labels'].items()}
            }
            json.dump(serializable_result, f, indent=2)
        print(f"✅ Saved results to {result_path}")
    
    # Create comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    comparison = []
    for config_name, result in all_results.items():
        metrics = result['metrics']
        comparison.append({
            'Configuration': config_name,
            'Avg F1': f"{metrics['avg_f1']:.4f}",
            'Avg Precision': f"{metrics['avg_precision']:.4f}",
            'Avg Recall': f"{metrics['avg_recall']:.4f}",
            'Avg Accuracy': f"{metrics['avg_accuracy']:.4f}",
            'h6 F1': f"{metrics['h6_f1']:.4f}",
            'h12 F1': f"{metrics['h12_f1']:.4f}",
            'h24 F1': f"{metrics['h24_f1']:.4f}",
            'h48 F1': f"{metrics['h48_f1']:.4f}",
        })
    
    # Print comparison table
    import pandas as pd
    df_comparison = pd.DataFrame(comparison)
    print("\n", df_comparison.to_string(index=False))
    
    # Save comparison
    comparison_path = output_dir / "comparison_summary.csv"
    df_comparison.to_csv(comparison_path, index=False)
    print(f"\n✅ Saved comparison summary to {comparison_path}")
    
    # Save all results together
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, 'w') as f:
        serializable_all = {}
        for config_name, result in all_results.items():
            serializable_all[config_name] = {
                'config': result['config'],
                'n_shots': result['n_shots'],
                'n_test_samples': result['n_test_samples'],
                'metrics': result['metrics'],
                'predictions': {k: [int(x) for x in v] for k, v in result['predictions'].items()},
                'labels': {k: [int(x) for x in v] for k, v in result['labels'].items()}
            }
        json.dump(serializable_all, f, indent=2)
    print(f"✅ Saved all results to {all_results_path}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
