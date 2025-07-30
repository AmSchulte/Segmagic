from torch import nn

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