import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from segmentation_models_pytorch.base import modules as md


class MLP(nn.Module):
    def __init__(self, skip_channels: int, segmentation_channels: int):
        super().__init__()

        self.linear = nn.Linear(skip_channels, segmentation_channels)

    def forward(self, x: torch.Tensor):
        batch, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.linear(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        return x


class SegformerDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        encoder_depth: int = 5,
        segmentation_channels: int = 256,
    ):
        super().__init__()

        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for Segformer decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        if encoder_channels[1] == 0:
            encoder_channels = [
                channel for index, channel in enumerate(encoder_channels) if index != 1
            ]
        encoder_channels = encoder_channels[::-1]

        self.mlp_stage = nn.ModuleList(
            [MLP(channel, segmentation_channels) for channel in encoder_channels[:-1]]
        )

        self.fuse_stage = md.Conv2dReLU(
            in_channels=(len(encoder_channels) - 1) * segmentation_channels,
            out_channels=segmentation_channels,
            kernel_size=1,
            use_norm="batchnorm",
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Resize all features to the size of the largest feature
        target_size = [dim // 4 for dim in features[0].shape[2:]]

        features = features[2:] if features[1].size(1) == 0 else features[1:]
        features = features[::-1]  # reverse channels to start from head of encoder

        resized_features = []
        for i, mlp_layer in enumerate(self.mlp_stage):
            feature = mlp_layer(features[i])
            resized_feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_features.append(resized_feature)

        output = self.fuse_stage(torch.cat(resized_features, dim=1))

        return output