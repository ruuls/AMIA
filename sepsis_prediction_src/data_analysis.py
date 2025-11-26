"""
Data Analysis and Preprocessing Script for Sepsis Prediction

This script analyzes the JSONL dataset and provides insights on:
- Class distribution and imbalance
- Label patterns
- Input length statistics
- Data quality checks
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class SepsisDataAnalyzer:
    """Analyze sepsis prediction dataset"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.label_names = ['h6', 'h12', 'h24', 'h48']
        
    def load_jsonl(self, filename: str) -> List[Dict]:
        """Load JSONL file"""
        filepath = self.data_dir / filename
        data = []
        with open(filepath, 'r') as f:
            for line in tqdm(f, desc=f"Loading {filename}"):
                data.append(json.loads(line.strip()))
        return data
    
    def analyze_class_distribution(self, data: List[Dict]) -> Dict:
        """Analyze class distribution and imbalance"""
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Count individual label occurrences
        label_counts = {label: 0 for label in self.label_names}
        label_patterns = Counter()
        
        for record in data:
            output = record['output']
            pattern = ''.join([str(output[label]) for label in self.label_names])
            label_patterns[pattern] += 1
            
            for label in self.label_names:
                label_counts[label] += output[label]
        
        total = len(data)
        
        # Individual label statistics
        print("\nIndividual Label Statistics:")
        print("-" * 60)
        for label in self.label_names:
            count = label_counts[label]
            pct = (count / total) * 100
            imbalance_ratio = total / count if count > 0 else float('inf')
            print(f"{label:5s}: {count:6d} ({pct:5.2f}%) | Imbalance Ratio: 1:{imbalance_ratio:.1f}")
        
        # Label pattern analysis
        print("\nLabel Pattern Distribution (Top 10):")
        print("-" * 60)
        print(f"{'Pattern':^10s} | {'Count':^8s} | {'Percentage':^12s}")
        print("-" * 60)
        
        for pattern, count in label_patterns.most_common(10):
            pct = (count / total) * 100
            print(f"{pattern:^10s} | {count:^8d} | {pct:^11.2f}%")
        
        # Calculate overall statistics
        all_negative = label_patterns.get('0000', 0)
        at_least_one_positive = total - all_negative
        
        print("\nOverall Imbalance:")
        print("-" * 60)
        print(f"All Negative (0000): {all_negative:6d} ({(all_negative/total)*100:.2f}%)")
        print(f"At Least 1 Positive: {at_least_one_positive:6d} ({(at_least_one_positive/total)*100:.2f}%)")
        print(f"Overall Imbalance Ratio: 1:{all_negative/at_least_one_positive:.1f}")
        
        return {
            'label_counts': label_counts,
            'label_patterns': dict(label_patterns),
            'total': total,
            'imbalance_ratios': {
                label: total / label_counts[label] if label_counts[label] > 0 else float('inf')
                for label in self.label_names
            }
        }
    
    def analyze_input_lengths(self, data: List[Dict]) -> Dict:
        """Analyze input text lengths"""
        print("\n" + "="*60)
        print("INPUT LENGTH ANALYSIS")
        print("="*60)
        
        lengths = [len(record['input'].split()) for record in data]
        char_lengths = [len(record['input']) for record in data]
        
        stats = {
            'word_count': {
                'min': np.min(lengths),
                'max': np.max(lengths),
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'p95': np.percentile(lengths, 95),
                'p99': np.percentile(lengths, 99)
            },
            'char_count': {
                'min': np.min(char_lengths),
                'max': np.max(char_lengths),
                'mean': np.mean(char_lengths),
                'median': np.median(char_lengths),
                'std': np.std(char_lengths),
                'p95': np.percentile(char_lengths, 95),
                'p99': np.percentile(char_lengths, 99)
            }
        }
        
        print("\nWord Count Statistics:")
        print("-" * 60)
        for key, value in stats['word_count'].items():
            print(f"{key:10s}: {value:8.1f}")
        
        print("\nCharacter Count Statistics:")
        print("-" * 60)
        for key, value in stats['char_count'].items():
            print(f"{key:10s}: {value:8.1f}")
        
        # Estimate token count (rough approximation: 1 token ≈ 0.75 words)
        token_estimate = {
            'mean': stats['word_count']['mean'] * 1.3,
            'p95': stats['word_count']['p95'] * 1.3,
            'p99': stats['word_count']['p99'] * 1.3
        }
        
        print("\nEstimated Token Count (approximate):")
        print("-" * 60)
        for key, value in token_estimate.items():
            print(f"{key:10s}: {value:8.1f}")
        
        return stats
    
    def analyze_label_correlations(self, data: List[Dict]) -> np.ndarray:
        """Analyze correlations between labels"""
        print("\n" + "="*60)
        print("LABEL CORRELATION ANALYSIS")
        print("="*60)
        
        # Create label matrix
        label_matrix = np.zeros((len(data), len(self.label_names)))
        for i, record in enumerate(data):
            for j, label in enumerate(self.label_names):
                label_matrix[i, j] = record['output'][label]
        
        # Calculate correlation
        correlation = np.corrcoef(label_matrix.T)
        
        print("\nCorrelation Matrix:")
        print("-" * 60)
        
        # Print header
        print(f"{'':5s}", end='')
        for label in self.label_names:
            print(f"{label:>8s}", end='')
        print()
        
        # Print matrix
        for i, label in enumerate(self.label_names):
            print(f"{label:5s}", end='')
            for j in range(len(self.label_names)):
                print(f"{correlation[i, j]:8.3f}", end='')
            print()
        
        return correlation
    
    def calculate_class_weights(self, data: List[Dict]) -> Dict[str, float]:
        """Calculate class weights for loss function"""
        print("\n" + "="*60)
        print("RECOMMENDED CLASS WEIGHTS")
        print("="*60)
        
        total = len(data)
        label_counts = {label: 0 for label in self.label_names}
        
        for record in data:
            for label in self.label_names:
                label_counts[label] += record['output'][label]
        
        # Calculate weights using inverse frequency
        weights = {}
        for label in self.label_names:
            pos_count = label_counts[label]
            neg_count = total - pos_count
            
            # Inverse frequency weight
            weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            # Apply smoothing to prevent extremely large weights
            weight = min(weight, 50.0)  # Cap at 50
            
            weights[label] = weight
        
        print("\nClass Weights (for positive class):")
        print("-" * 60)
        for label, weight in weights.items():
            print(f"{label}: {weight:.2f}")
        
        # Also calculate for focal loss
        print("\nRecommended Focal Loss Parameters:")
        print("-" * 60)
        print("alpha: 0.25 (balance between positive/negative)")
        print("gamma: 2.0 (focus on hard examples)")
        
        return weights
    
    def check_data_quality(self, data: List[Dict]) -> Dict:
        """Check for data quality issues"""
        print("\n" + "="*60)
        print("DATA QUALITY CHECKS")
        print("="*60)
        
        issues = {
            'empty_inputs': 0,
            'missing_labels': 0,
            'invalid_labels': 0,
            'missing_meta': 0
        }
        
        for i, record in enumerate(data):
            # Check empty inputs
            if not record.get('input', '').strip():
                issues['empty_inputs'] += 1
            
            # Check missing output
            if 'output' not in record:
                issues['missing_labels'] += 1
                continue
            
            # Check invalid label values
            output = record['output']
            for label in self.label_names:
                if label not in output:
                    issues['invalid_labels'] += 1
                    break
                if output[label] not in [0, 1]:
                    issues['invalid_labels'] += 1
                    break
            
            # Check meta information
            if 'meta' not in record:
                issues['missing_meta'] += 1
        
        print("\nData Quality Summary:")
        print("-" * 60)
        for issue_type, count in issues.items():
            status = "✅ PASS" if count == 0 else f"⚠️  ISSUES FOUND"
            print(f"{issue_type:20s}: {count:6d} | {status}")
        
        return issues
    
    def generate_report(self, splits: List[str] = ['train', 'val', 'test']):
        """Generate comprehensive analysis report"""
        print("\n" + "="*70)
        print(" " * 20 + "SEPSIS DATASET ANALYSIS REPORT")
        print("="*70)
        
        all_stats = {}
        
        for split in splits:
            filename = f"{split}.jsonl"
            print(f"\n{'='*70}")
            print(f"Analyzing {split.upper()} split: {filename}")
            print(f"{'='*70}")
            
            # Load data
            data = self.load_jsonl(filename)
            print(f"\nTotal records: {len(data)}")
            
            # Run analyses
            stats = {
                'size': len(data),
                'class_dist': self.analyze_class_distribution(data),
                'input_lengths': self.analyze_input_lengths(data),
                'label_corr': self.analyze_label_correlations(data),
                'data_quality': self.check_data_quality(data)
            }
            
            if split == 'train':
                stats['class_weights'] = self.calculate_class_weights(data)
            
            all_stats[split] = stats
        
        # Summary across splits
        print("\n" + "="*70)
        print("SUMMARY ACROSS SPLITS")
        print("="*70)
        print(f"\n{'Split':10s} | {'Size':>8s} | {'Positive Rate':>15s}")
        print("-" * 70)
        
        for split in splits:
            size = all_stats[split]['size']
            pos_rate = (1 - all_stats[split]['class_dist']['label_patterns'].get('0000', 0) / size) * 100
            print(f"{split:10s} | {size:>8d} | {pos_rate:>14.2f}%")
        
        return all_stats


def main():
    """Run complete data analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze sepsis prediction dataset")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/ubuntu/nahiyan_workspace/s3_bucket/data',
        help='Directory containing JSONL files'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to analyze'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = SepsisDataAnalyzer(args.data_dir)
    stats = analyzer.generate_report(args.splits)
    
    print("\n" + "="*70)
    print("Analysis complete! Use these insights to configure your training.")
    print("="*70)


if __name__ == "__main__":
    main()
