# Loss Documentation

## Overview

All loss functions are centralized in `model/losses.py` and can be easily switched using the `loss_name` and `loss_params` parameters.

## Available Loss Functions

### Basic Losses
- `"bce"`: Binary Cross Entropy with Logits
- `"dice"`: Dice Loss
- `"focal"`: Focal Loss (default)
- `"iou"`: IoU/Jaccard Loss
- `"lovasz"`: Lovasz Loss
- `"tversky"`: Tversky Loss

### Advanced Losses
- `"weighted_focal"`: Weighted Focal Loss for handling sample weights
- `"dice_bce"`: Combination of Dice and BCE losses
- `"combo"`: Generic combination of any two losses

## Usage Examples

### Basic Usage
```python
# Default focal loss
model = Model(model_params=params)

# Dice + BCE combination
model = Model(
    model_params=params,
    loss_name="dice_bce",
    loss_params={"dice_weight": 0.7, "bce_weight": 0.3}
)
```

### Advanced Usage
```python
# Custom focal loss parameters
model = Model(
    model_params=params,
    loss_name="focal",
    loss_params={
        "gamma": 2.0,
        "alpha": 0.75,
        "reduction": "sum"
    }
)

# Tversky loss for handling FP/FN differently
model = Model(
    model_params=params,
    loss_name="tversky",
    loss_params={
        "alpha": 0.3,  # Weight for false positives
        "beta": 0.7    # Weight for false negatives
    }
)

# Combination of focal and dice
model = Model(
    model_params=params,
    loss_name="combo",
    loss_params={
        "loss_a": "focal",
        "loss_a_params": {"gamma": 2.0},
        "loss_b": "dice", 
        "loss_b_params": {"smooth": 1.0},
        "weight_a": 0.6,
        "weight_b": 0.4
    }
)
```