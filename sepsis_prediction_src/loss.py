"""
Custom Loss Functions for Imbalanced Multi-Label Classification

Implements several loss functions to handle severe class imbalance:
1. Weighted Binary Cross-Entropy Loss
2. Focal Loss
3. Asymmetric Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for imbalanced multi-label classification
    
    Applies higher weights to positive (minority) class
    """
    
    def __init__(self, pos_weights: torch.Tensor, reduction: str = 'mean'):
        """
        Args:
            pos_weights: Weight for positive class for each label [num_labels]
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.pos_weights = pos_weights
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]
            
        Returns:
            Loss value
        """
        # Move pos_weights to same device as logits
        pos_weights = self.pos_weights.to(logits.device)
        
        # Binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weights,
            reduction=self.reduction
        )
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced multi-label classification
    
    Focuses training on hard examples and down-weights easy examples
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Balance between positive/negative classes (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            pos_weights: Additional weights for positive class [num_labels]
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weights = pos_weights
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]
            
        Returns:
            Loss value
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        
        # Compute focal term: (1 - p_t)^gamma
        # p_t = p if target = 1 else (1 - p)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply focal term
        focal_loss = focal_term * bce_loss
        
        # Apply alpha balancing
        # alpha_t = alpha if target = 1 else (1 - alpha)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal_loss = alpha_t * focal_loss
        
        # Apply additional pos_weights if provided
        if self.pos_weights is not None:
            pos_weights = self.pos_weights.to(logits.device)
            # Apply weights only to positive class
            weight_term = pos_weights * labels + (1 - labels)
            focal_loss = focal_loss * weight_term
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    
    Uses different focusing parameters for positive and negative samples
    
    Reference: Ridnik et al. "Asymmetric Loss For Multi-Label Classification" (2021)
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Probability clipping value
            eps: Small value for numerical stability
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]
            
        Returns:
            Loss value
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Probability clipping
        if self.clip is not None and self.clip > 0:
            probs = probs.clamp(min=self.clip)
        
        # Positive loss
        pos_loss = labels * torch.log(probs.clamp(min=self.eps))
        pos_loss = pos_loss * (1 - probs) ** self.gamma_pos
        
        # Negative loss
        neg_loss = (1 - labels) * torch.log((1 - probs).clamp(min=self.eps))
        neg_loss = neg_loss * probs ** self.gamma_neg
        
        # Total loss
        loss = -(pos_loss + neg_loss)
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combination of multiple losses
    Useful for leveraging benefits of different loss functions
    """
    
    def __init__(
        self,
        focal_weight: float = 0.7,
        bce_weight: float = 0.3,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            focal_weight: Weight for focal loss
            bce_weight: Weight for BCE loss
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            pos_weights: Positive class weights
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        self.focal_loss = FocalLoss(alpha, gamma, pos_weights)
        self.bce_loss = WeightedBCELoss(pos_weights) if pos_weights is not None else nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]
            
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(logits, labels)
        bce = self.bce_loss(logits, labels)
        
        return self.focal_weight * focal + self.bce_weight * bce


def get_loss_function(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function by name
    
    Args:
        loss_type: Type of loss ('weighted_bce', 'focal_loss', 'asymmetric_loss', 'combined')
        class_weights: Class weights for positive class [num_labels]
        **kwargs: Additional loss-specific parameters
        
    Returns:
        Loss function module
    """
    if loss_type == 'weighted_bce':
        if class_weights is None:
            raise ValueError("class_weights required for weighted_bce loss")
        return WeightedBCELoss(class_weights)
    
    elif loss_type == 'focal_loss':
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha, gamma, class_weights)
    
    elif loss_type == 'asymmetric_loss':
        gamma_neg = kwargs.get('gamma_neg', 4.0)
        gamma_pos = kwargs.get('gamma_pos', 1.0)
        clip = kwargs.get('clip', 0.05)
        return AsymmetricLoss(gamma_neg, gamma_pos, clip)
    
    elif loss_type == 'combined':
        focal_weight = kwargs.get('focal_weight', 0.7)
        bce_weight = kwargs.get('bce_weight', 0.3)
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        return CombinedLoss(focal_weight, bce_weight, alpha, gamma, class_weights)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...\n")
    
    # Create dummy data
    batch_size = 4
    num_labels = 4
    
    logits = torch.randn(batch_size, num_labels)
    labels = torch.tensor([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0]
    ], dtype=torch.float32)
    
    class_weights = torch.tensor([10.0, 8.0, 5.0, 3.0])
    
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Class weights: {class_weights}\n")
    
    # Test each loss
    print("="*60)
    print("Testing Weighted BCE Loss")
    print("="*60)
    bce_loss = WeightedBCELoss(class_weights)
    loss_val = bce_loss(logits, labels)
    print(f"Loss: {loss_val.item():.4f}\n")
    
    print("="*60)
    print("Testing Focal Loss")
    print("="*60)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, pos_weights=class_weights)
    loss_val = focal_loss(logits, labels)
    print(f"Loss: {loss_val.item():.4f}\n")
    
    print("="*60)
    print("Testing Asymmetric Loss")
    print("="*60)
    asym_loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    loss_val = asym_loss(logits, labels)
    print(f"Loss: {loss_val.item():.4f}\n")
    
    print("="*60)
    print("Testing Combined Loss")
    print("="*60)
    combined_loss = CombinedLoss(pos_weights=class_weights)
    loss_val = combined_loss(logits, labels)
    print(f"Loss: {loss_val.item():.4f}\n")
    
    print("✅ All loss functions tested successfully!")
