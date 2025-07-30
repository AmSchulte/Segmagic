"""
Optimizer factory for common optimizers used in deep learning.

This module provides a factory method to create optimizer instances
for the most commonly used optimizers in PyTorch.
"""

import torch
from typing import Dict, Any, Optional


def get_optimizer(optimizer_name: str, parameters, lr: float = 1e-3, optimizer_params: Optional[Dict[str, Any]] = None):
    """
    Factory method to get optimizer instances.
    
    Args:
        optimizer_name (str): Name of the optimizer. Supported: 'adam', 'adamw', 'ranger', 'sophia'
        parameters: Model parameters to optimize
        lr (float): Learning rate (default: 1e-3)
        optimizer_params (Dict[str, Any], optional): Additional optimizer parameters
        
    Returns:
        torch.optim.Optimizer: Configured optimizer instance
        
    Raises:
        ValueError: If optimizer_name is not supported
        ImportError: If required packages for specific optimizers are not installed
    """
    if optimizer_params is None:
        optimizer_params = {}
    
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            betas=optimizer_params.get('betas', (0.9, 0.999)),
            eps=optimizer_params.get('eps', 1e-8),
            weight_decay=optimizer_params.get('weight_decay', 0),
            amsgrad=optimizer_params.get('amsgrad', False)
        )
    
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=optimizer_params.get('betas', (0.9, 0.999)),
            eps=optimizer_params.get('eps', 1e-8),
            weight_decay=optimizer_params.get('weight_decay', 1e-2),
            amsgrad=optimizer_params.get('amsgrad', False)
        )
    
    elif optimizer_name == "ranger":
        try:
            from ranger import Ranger
        except ImportError:
            raise ImportError(
                "Ranger optimizer not found. Please install it with: pip install ranger-fm"
            )
        
        return Ranger(
            parameters,
            lr=lr,
            alpha=optimizer_params.get('alpha', 0.5),
            k=optimizer_params.get('k', 6),
            N_sma_threshhold=optimizer_params.get('N_sma_threshhold', 5),
            betas=optimizer_params.get('betas', (0.95, 0.999)),
            eps=optimizer_params.get('eps', 1e-5),
            weight_decay=optimizer_params.get('weight_decay', 1e-2),
            use_gc=optimizer_params.get('use_gc', True),
            gc_conv_only=optimizer_params.get('gc_conv_only', False)
        )
    
    elif optimizer_name == "sophia":
        try:
            from model.sophia import SophiaG
        except ImportError:
            raise ImportError(
                "Sophia optimizer not found. Please install it with: pip install sophia-opt"
            )
        
        return SophiaG(
            parameters,
            lr=lr,
            betas=optimizer_params.get('betas', (0.965, 0.99)),
            rho=optimizer_params.get('rho', 0.04),
            weight_decay=optimizer_params.get('weight_decay', 1e-1),
            maximize=optimizer_params.get('maximize', False),
            capturable=optimizer_params.get('capturable', False)
        )
    
    else:
        supported_optimizers = ['adam', 'adamw', 'ranger', 'sophia']
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Supported optimizers are: {', '.join(supported_optimizers)}"
        )


def get_available_optimizers():
    """
    Get a list of available optimizers.
    
    Returns:
        List[str]: List of available optimizer names
    """
    return ['adam', 'adamw', 'ranger', 'sophia']


def get_optimizer_defaults(optimizer_name: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific optimizer.
    
    Args:
        optimizer_name (str): Name of the optimizer
        
    Returns:
        Dict[str, Any]: Default parameters for the optimizer
        
    Raises:
        ValueError: If optimizer_name is not supported
    """
    optimizer_name = optimizer_name.lower()
    
    defaults = {
        'adam': {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
            'amsgrad': False
        },
        'adamw': {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 1e-2,
            'amsgrad': False
        },
        'ranger': {
            'lr': 1e-3,
            'alpha': 0.5,
            'k': 6,
            'N_sma_threshhold': 5,
            'betas': (0.95, 0.999),
            'eps': 1e-5,
            'weight_decay': 1e-2,
            'use_gc': True,
            'gc_conv_only': False
        },
        'sophia': {
            'lr': 1e-3,
            'betas': (0.965, 0.99),
            'rho': 0.04,
            'weight_decay': 1e-1,
            'maximize': False,
            'capturable': False
        }
    }
    
    if optimizer_name not in defaults:
        supported_optimizers = list(defaults.keys())
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Supported optimizers are: {', '.join(supported_optimizers)}"
        )
    
    return defaults[optimizer_name].copy()
