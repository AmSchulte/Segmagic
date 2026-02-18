from torch import nn
import torch.nn.functional as F

class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d = nn.Sequential(
            nn.GroupNorm(8, in_channels) if in_channels > 32 else nn.Identity(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            ),
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        activation = nn.Identity() if activation is None else activation
        super().__init__(conv2d, upsampling, activation)

class DeepSupervisionHead(nn.Module):
    """Auxiliary segmentation head for deep supervision.
    
    This head takes intermediate decoder features and produces segmentation outputs
    at different scales, which are then upsampled to the target resolution.
    """
    
    def __init__(self, in_channels, out_channels, target_size=None, kernel_size=1):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
    def forward(self, x):
        x = self.conv(x)
        if self.target_size is not None:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x