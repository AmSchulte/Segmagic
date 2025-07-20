"""
Loss functions for segmentation models.
Provides a unified interface for different loss functions commonly used in segmentation tasks.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Any, Optional, Union


class ComboLoss(nn.Module):
    """Combination of two loss functions with optional weighting."""
    
    def __init__(self, loss_a: nn.Module, loss_b: nn.Module, weight_a: float = 1.0, weight_b: float = 1.0):
        super().__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_a = weight_a
        self.weight_b = weight_b
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.weight_a * self.loss_a(x, y) + self.weight_b * self.loss_b(x, y)


class WeightedFocalLoss(nn.Module):
    """Weighted focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, reduced_threshold: float = 0.2):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(
            mode="multilabel",
            gamma=gamma,
            alpha=alpha,
            reduction=None,
            reduced_threshold=reduced_threshold
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        focal_loss = self.focal_loss(x, y)
        
        if weights is not None:
            # Apply weights
            focal_loss = focal_loss * weights.flatten()
        
        return focal_loss.sum()


class DiceBCELoss(nn.Module):
    """Combination of Dice loss and BCE loss."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = smp.losses.DiceLoss(mode="multilabel", smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(x, y)
        bce = self.bce_loss(x, y)
        return self.dice_weight * dice + self.bce_weight * bce


class TverskyLoss(nn.Module):
    """Tversky loss for handling false positives and false negatives differently."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.tversky_loss = smp.losses.TverskyLoss(
            mode="multilabel",
            alpha=alpha,
            beta=beta,
            smooth=smooth
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.tversky_loss(x, y)


class LovaszLoss(nn.Module):
    """Lovasz loss for segmentation."""
    
    def __init__(self):
        super().__init__()
        self.lovasz_loss = smp.losses.LovaszLoss(mode="multilabel")
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.lovasz_loss(x, y)


class IoULoss(nn.Module):
    """IoU (Jaccard) loss."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.iou_loss = smp.losses.JaccardLoss(mode="multilabel", smooth=smooth)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.iou_loss(x, y)


def get_loss_function(loss_name: str, loss_params: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Factory function to create loss functions by name.
    
    Args:
        loss_name: Name of the loss function
        loss_params: Parameters to pass to the loss function
    
    Returns:
        Loss function instance
    
    Available losses:
        - "bce": Binary Cross Entropy with Logits
        - "dice": Dice Loss
        - "focal": Focal Loss
        - "weighted_focal": Weighted Focal Loss
        - "dice_bce": Combination of Dice and BCE
        - "tversky": Tversky Loss
        - "lovasz": Lovasz Loss
        - "iou": IoU/Jaccard Loss
        - "combo": Combination of two losses (requires loss_a and loss_b in params)
    """
    if loss_params is None:
        loss_params = {}
    
    loss_registry = {
        "bce": lambda params: nn.BCEWithLogitsLoss(**params),
        
        "dice": lambda params: smp.losses.DiceLoss(
            mode=params.get("mode", "multilabel"),
            smooth=params.get("smooth", 1.0)
        ),
        
        "focal": lambda params: smp.losses.FocalLoss(
            mode=params.get("mode", "multilabel"),
            gamma=params.get("gamma", 2.0),
            alpha=params.get("alpha", 0.75),
            reduction=params.get("reduction", "sum"),
            reduced_threshold=params.get("reduced_threshold", 0.2)
        ),
        
        "weighted_focal": lambda params: WeightedFocalLoss(
            gamma=params.get("gamma", 2.0),
            alpha=params.get("alpha", 0.75),
            reduced_threshold=params.get("reduced_threshold", 0.2)
        ),
        
        "dice_bce": lambda params: DiceBCELoss(
            dice_weight=params.get("dice_weight", 0.5),
            bce_weight=params.get("bce_weight", 0.5),
            smooth=params.get("smooth", 1.0)
        ),
        
        "tversky": lambda params: TverskyLoss(
            alpha=params.get("alpha", 0.3),
            beta=params.get("beta", 0.7),
            smooth=params.get("smooth", 1.0)
        ),
        
        "lovasz": lambda params: LovaszLoss(),
        
        "iou": lambda params: IoULoss(
            smooth=params.get("smooth", 1.0)
        ),
        
        "combo": lambda params: ComboLoss(
            loss_a=get_loss_function(params["loss_a"], params.get("loss_a_params", {})),
            loss_b=get_loss_function(params["loss_b"], params.get("loss_b_params", {})),
            weight_a=params.get("weight_a", 1.0),
            weight_b=params.get("weight_b", 1.0)
        )
    }
    
    if loss_name not in loss_registry:
        available_losses = ", ".join(loss_registry.keys())
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {available_losses}")
    
    return loss_registry[loss_name](loss_params)


# Convenience function for common loss combinations
def get_dice_bce_loss(dice_weight: float = 0.7, bce_weight: float = 0.3, **kwargs) -> DiceBCELoss:
    """Get a Dice + BCE combination loss with specified weights."""
    return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight, **kwargs)


def get_focal_dice_loss(focal_weight: float = 0.6, dice_weight: float = 0.4, **kwargs) -> ComboLoss:
    """Get a Focal + Dice combination loss."""
    focal_params = {k: v for k, v in kwargs.items() if k.startswith('focal_')}
    dice_params = {k[5:]: v for k, v in kwargs.items() if k.startswith('dice_')}
    
    focal_loss = get_loss_function("focal", focal_params)
    dice_loss = get_loss_function("dice", dice_params)
    
    return ComboLoss(focal_loss, dice_loss, focal_weight, dice_weight)


# Export commonly used losses for easy import
__all__ = [
    'ComboLoss',
    'WeightedFocalLoss', 
    'DiceBCELoss',
    'TverskyLoss',
    'LovaszLoss',
    'IoULoss',
    'get_loss_function',
    'get_dice_bce_loss',
    'get_focal_dice_loss'
]
