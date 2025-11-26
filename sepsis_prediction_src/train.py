"""
Main Training Script for Sepsis Prediction Multi-Label Classification

This script orchestrates the complete training pipeline:
1. Load configuration
2. Setup model and data
3. Train with proper evaluation and checkpointing
4. Save best model
"""

import torch
import torch.nn as nn
import yaml
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Import custom modules
from model import load_model_and_tokenizer
from dataset import create_dataloaders
from loss import get_loss_function
from metrics import MultiLabelMetrics, evaluate_model


class Trainer:
    """Trainer class for sepsis prediction model"""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed
        self.set_seed(self.config['seed'])
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['hardware']['use_cuda'] else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.setup_directories()
        
        # Initialize tensorboard
        if self.config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(self.config['logging']['tensorboard_dir'])
        else:
            self.writer = None
        
        # Load model and tokenizer
        print("\n" + "="*70)
        print("INITIALIZING MODEL")
        print("="*70)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path=self.config['paths']['model_path'],
            num_labels=self.config['model']['num_labels'],
            lora_config=self.config['lora'],
            quantization_config=self.config['quantization']
        )
        
        # Create dataloaders
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        self.dataloaders, self.class_weights = create_dataloaders(
            data_dir=self.config['paths']['data_dir'],
            tokenizer=self.tokenizer,
            batch_size=self.config['training']['per_device_train_batch_size'],
            max_length=self.config['model']['max_length'],
            num_workers=self.config['hardware']['num_workers'],
            use_oversampling=self.config['data']['use_oversampling'],
            oversample_factor=self.config['data'].get('oversample_factor', 2.0)
        )
        
        # Setup loss function
        self.criterion = self.setup_loss_function()
        
        # Setup optimizer
        self.optimizer = self.setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self.setup_scheduler()
        
        # Metrics calculator
        self.metrics_calculator = MultiLabelMetrics(
            label_names=self.config['model']['label_names'],
            threshold=0.5
        )
        
        # Training state
        self.global_step = 0
        self.best_metric = 0.0
        self.early_stopping_counter = 0
        self.train_losses = []
        self.eval_metrics_history = []
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def setup_directories(self):
        """Create output directories"""
        for dir_path in [
            self.config['paths']['output_dir'],
            self.config['paths']['logs_dir'],
            self.config['paths']['checkpoint_dir'],
            self.config['logging']['tensorboard_dir']
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_loss_function(self):
        """Setup loss function with class weights"""
        print("\n" + "="*70)
        print("SETTING UP LOSS FUNCTION")
        print("="*70)
        
        loss_config = self.config['loss']
        loss_type = loss_config['type']
        
        # Use computed class weights or config weights
        if self.config['data']['use_class_weights']:
            class_weights = self.class_weights
        else:
            class_weights = torch.tensor(loss_config.get('class_weights', [1.0]*4), dtype=torch.float32)
        
        print(f"Loss type: {loss_type}")
        print(f"Class weights: {class_weights}")
        
        criterion = get_loss_function(
            loss_type=loss_type,
            class_weights=class_weights,
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0)
        )
        
        return criterion
    
    def setup_optimizer(self):
        """Setup optimizer"""
        # Get trainable parameters
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        print(f"\nOptimizer: AdamW")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print(f"Weight decay: {self.config['training']['weight_decay']}")
        
        return optimizer
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        num_training_steps = len(self.dataloaders['train']) * self.config['training']['num_train_epochs']
        num_warmup_steps = int(num_training_steps * self.config['training']['warmup_ratio'])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=1e-7
        )
        
        print(f"\nScheduler: CosineAnnealingLR")
        print(f"Total training steps: {num_training_steps}")
        print(f"Warmup steps: {num_warmup_steps}")
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to tensorboard
            if self.writer and self.global_step % self.config['training']['logging_steps'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)
            
            # Evaluate periodically (only if using step-based evaluation)
            eval_steps = self.config['training'].get('eval_steps')
            if eval_steps and self.global_step % eval_steps == 0:
                eval_metrics = self.evaluate()
                self.model.train()  # Back to training mode
                
                # Check if best model
                # Remove 'eval_' prefix if present in config
                metric_name = self.config['training']['metric_for_best_model'].replace('eval_', '')
                current_metric = eval_metrics[metric_name]
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint(is_best=True)
                    self.early_stopping_counter = 0
                    print(f"\n✅ New best model! {self.config['training']['metric_for_best_model']}: {current_metric:.4f}")
                else:
                    self.early_stopping_counter += 1
                
                # Early stopping
                if self.early_stopping_counter >= self.config['training']['early_stopping_patience']:
                    print(f"\n⚠️  Early stopping triggered after {self.early_stopping_counter} evaluations without improvement")
                    return total_loss / num_batches
        
        return total_loss / num_batches
    
    def evaluate(self) -> dict:
        """Evaluate on validation set"""
        print("\n" + "="*70)
        print("EVALUATING ON VALIDATION SET")
        print("="*70)
        
        # Put model in eval mode FIRST (disables dropout, etc.)
        self.model.eval()
        
        # AGGRESSIVE memory cleanup before evaluation
        if torch.cuda.is_available():
            # Clear cache multiple times
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"GPU memory before eval: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        # Use VERY SMALL batch size for evaluation (batch_size=2)
        # This is essential for QLoRA to avoid OOM during eval
        eval_batch_size = 2  # Changed from 4 to 2
        
        val_dataloader = DataLoader(
            self.dataloaders['val'].dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,  # Must be 0 to save memory
            pin_memory=False,  # Disable pin_memory
            persistent_workers=False
        )
        
        print(f"Evaluating on {len(val_dataloader.dataset)} samples (batch_size={eval_batch_size})...")
        print(f"Total eval batches: {len(val_dataloader)}")
        
        # Evaluate with no_grad to save memory
        with torch.no_grad():
            metrics, predictions, labels = evaluate_model(
                model=self.model,
                dataloader=val_dataloader,
                device=self.device,
                label_names=self.config['model']['label_names']
            )
        
        # Clear cache after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory after eval: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Print metrics
        self.metrics_calculator.print_metrics(metrics, title="Validation Metrics")
        
        # Log to tensorboard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'eval/{key}', value, self.global_step)
        
        self.eval_metrics_history.append(metrics)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        
        if is_best:
            save_path = checkpoint_dir / "best_model"
        else:
            save_path = checkpoint_dir / f"checkpoint_step_{self.global_step}"
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(save_path))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(save_path))
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config
        }
        torch.save(state, save_path / "training_state.pt")
        
        print(f"✅ Checkpoint saved to {save_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Total epochs: {self.config['training']['num_train_epochs']}")
        print(f"Train batches per epoch: {len(self.dataloaders['train'])}")
        print(f"Eval batches: {len(self.dataloaders['val'])}")
        print("="*70 + "\n")
        
        for epoch in range(self.config['training']['num_train_epochs']):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{self.config['training']['num_train_epochs']}")
            print(f"{'='*70}")
            
            # Train epoch
            avg_loss = self.train_epoch(epoch)
            self.train_losses.append(avg_loss)
            
            print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")
            
            # Evaluate at end of epoch
            eval_metrics = self.evaluate()
            
            # Check if best model
            # Remove 'eval_' prefix if present in config
            metric_name = self.config['training']['metric_for_best_model'].replace('eval_', '')
            current_metric = eval_metrics[metric_name]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(is_best=True)
                self.early_stopping_counter = 0
                print(f"\n✅ New best model! {self.config['training']['metric_for_best_model']}: {current_metric:.4f}")
            else:
                self.early_stopping_counter += 1
            
            # Regular checkpoint
            if (epoch + 1) % 1 == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.early_stopping_counter >= self.config['training']['early_stopping_patience']:
                print(f"\n⚠️  Early stopping triggered!")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Best {self.config['training']['metric_for_best_model']}: {self.best_metric:.4f}")
        
        # Final evaluation on test set
        print("\n" + "="*70)
        print("FINAL EVALUATION ON TEST SET")
        print("="*70)
        
        test_metrics, test_predictions, test_labels = evaluate_model(
            model=self.model,
            dataloader=self.dataloaders['test'],
            device=self.device,
            label_names=self.config['model']['label_names']
        )
        
        self.metrics_calculator.print_metrics(test_metrics, title="Test Set Metrics")
        
        # Save test results
        results_path = Path(self.config['paths']['output_dir']) / "test_results.pt"
        torch.save({
            'metrics': test_metrics,
            'predictions': test_predictions,
            'labels': test_labels
        }, results_path)
        
        print(f"\n✅ Test results saved to {results_path}")
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Meditron-7B for sepsis prediction")
    parser.add_argument(
        '--config',
        type=str,
        default='/home/ubuntu/nahiyan_workspace/sepsis_prediction/configs/training_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
