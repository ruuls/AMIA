"""
Custom Dataset for Sepsis Prediction Multi-Label Classification

Handles loading JSONL data, tokenization, and formatting for the model.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from transformers import PreTrainedTokenizer


class SepsisDataset(Dataset):
    """
    Dataset for sepsis prediction multi-label classification
    
    Each sample contains:
    - input: Clinical notes (text)
    - labels: 4 binary labels (h6, h12, h24, h48)
    - meta: Metadata (icustay_id, subject_id, etc.)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        label_names: List[str] = None,
        use_oversampling: bool = False,
        oversample_factor: float = 2.0,
        use_sampling: bool = False,
        sample_ratio: float = 1.0
    ):
        """
        Args:
            data_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            label_names: List of label names (default: ['h6', 'h12', 'h24', 'h48'])
            use_oversampling: Whether to oversample positive cases
            oversample_factor: Factor for oversampling minority class
            use_sampling: Whether to use only a subset of data
            sample_ratio: Ratio of data to use (0.0-1.0)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = label_names or ['h6', 'h12', 'h24', 'h48']
        
        # Load data
        self.data = self._load_data()
        
        # Apply sampling if requested (before oversampling)
        if use_sampling and sample_ratio < 1.0:
            original_len = len(self.data)
            sample_size = int(len(self.data) * sample_ratio)
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(self.data), sample_size, replace=False)
            self.data = [self.data[i] for i in indices]
            print(f"Sampling: Using {len(self.data)} samples ({sample_ratio*100:.0f}%) out of {original_len}")
        
        # Apply oversampling if requested
        if use_oversampling:
            self.data = self._oversample_positive_cases(oversample_factor)
        
        print(f"Loaded {len(self.data)} samples from {self.data_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load JSONL data"""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _oversample_positive_cases(self, factor: float) -> List[Dict]:
        """Oversample cases with at least one positive label"""
        negative_cases = []
        positive_cases = []
        
        for record in self.data:
            labels = [record['output'][label] for label in self.label_names]
            if sum(labels) > 0:
                positive_cases.append(record)
            else:
                negative_cases.append(record)
        
        # Calculate how many times to duplicate positive cases
        num_duplicates = int(factor)
        
        oversampled_data = negative_cases + positive_cases * num_duplicates
        
        print(f"Oversampling: {len(negative_cases)} negative + {len(positive_cases)}x{num_duplicates} positive = {len(oversampled_data)} total")
        
        return oversampled_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary containing:
            - input_ids: Tokenized input
            - attention_mask: Attention mask
            - labels: Binary labels for 4 time horizons
        """
        record = self.data[idx]
        
        # Get input text
        input_text = record['input']
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract labels
        labels = torch.tensor(
            [record['output'][label] for label in self.label_names],
            dtype=torch.float32
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset"""
        label_counts = {label: 0 for label in self.label_names}
        
        for record in self.data:
            for label in self.label_names:
                label_counts[label] += record['output'][label]
        
        return label_counts
    
    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for weighted loss
        Uses inverse frequency with smoothing
        """
        label_counts = self.get_class_distribution()
        total = len(self.data)
        
        weights = []
        for label in self.label_names:
            pos_count = label_counts[label]
            neg_count = total - pos_count
            
            if pos_count > 0:
                weight = neg_count / pos_count
                # Cap weight at 50 to prevent extreme values
                weight = min(weight, 50.0)
            else:
                weight = 1.0
            
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class SepsisDataCollator:
    """
    Custom data collator for batching
    Handles padding and batching for variable-length sequences
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples
        
        Args:
            features: List of samples from __getitem__
            
        Returns:
            Batched dictionary
        """
        # Stack all tensors
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }
        
        return batch


def create_dataloaders(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    num_workers: int = 4,
    use_oversampling: bool = False,
    oversample_factor: float = 2.0,
    use_sampling: bool = False,
    sample_ratio: float = 1.0
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of DataLoader workers
        use_oversampling: Whether to oversample training data
        oversample_factor: Oversampling factor
        use_sampling: Whether to sample a subset of training data
        sample_ratio: Ratio of training data to use (0.0-1.0)
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    from torch.utils.data import DataLoader
    
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = SepsisDataset(
        data_dir / 'train.jsonl',
        tokenizer,
        max_length,
        use_oversampling=use_oversampling,
        oversample_factor=oversample_factor,
        use_sampling=use_sampling,
        sample_ratio=sample_ratio
    )
    
    val_dataset = SepsisDataset(
        data_dir / 'val.jsonl',
        tokenizer,
        max_length,
        use_oversampling=False,  # Never oversample validation
        use_sampling=False  # Never sample validation
    )
    
    test_dataset = SepsisDataset(
        data_dir / 'test.jsonl',
        tokenizer,
        max_length,
        use_oversampling=False,  # Never oversample test
        use_sampling=False  # Never sample test
    )
    
    # Create collator
    collator = SepsisDataCollator(tokenizer)
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Can use larger batch for eval
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    }
    
    # Print class distribution for training set
    print("\n" + "="*60)
    print("Training Set Class Distribution:")
    print("="*60)
    class_dist = train_dataset.get_class_distribution()
    for label, count in class_dist.items():
        pct = (count / len(train_dataset)) * 100
        print(f"{label}: {count} ({pct:.2f}%)")
    
    # Print class weights
    print("\n" + "="*60)
    print("Computed Class Weights:")
    print("="*60)
    class_weights = train_dataset.compute_class_weights()
    for label, weight in zip(train_dataset.label_names, class_weights):
        print(f"{label}: {weight:.2f}")
    
    return dataloaders, class_weights


if __name__ == "__main__":
    # Test dataset loading
    from transformers import AutoTokenizer
    
    model_path = "/home/ubuntu/nahiyan_workspace/s3_bucket/models/meditron-7b"
    data_dir = "/home/ubuntu/nahiyan_workspace/s3_bucket/data"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    print("\nCreating dataloaders...")
    dataloaders, class_weights = create_dataloaders(
        data_dir,
        tokenizer,
        batch_size=2,
        max_length=512,  # Smaller for testing
        num_workers=0
    )
    
    print("\nTesting batch loading...")
    batch = next(iter(dataloaders['train']))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Sample labels: {batch['labels'][0]}")
    
    print("\nDataset loading test complete!")
