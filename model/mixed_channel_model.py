from typing import Optional, Union, List
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationModel
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import nn, einsum
from segmentation_models_pytorch.base import initialization as init
from model.blocks.channel_selection import DynamicChannelSelector, ContextSelfAttention, TokenCollapser, SegmentationHead


class DynamicUnet(SegmentationModel):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def __init__(
                self,
                encoder_name: str = "resnet34",
                encoder_depth: int = 5,
                encoder_weights: Optional[str] = "imagenet",
                decoder_use_batchnorm: bool = True,
                decoder_channels: List[int] = (256, 128, 64, 32, 16),
                decoder_attention_type: Optional[str] = None,
                in_channels: int = 3,
                classes: int = 1,
                activation: Optional[Union[str, callable]] = None,
                aux_params: Optional[dict] = None,
            ):

        super().__init__()
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.mappers = nn.ModuleList([
            nn.Identity(),
            DynamicChannelSelector(in_channels=48, emb_channels=300, channels_per_selector=12),
            DynamicChannelSelector(in_channels=40, emb_channels=300, channels_per_selector=10),
            DynamicChannelSelector(in_channels=64, emb_channels=300, channels_per_selector=16),
            DynamicChannelSelector(in_channels=176, emb_channels=300, channels_per_selector=44),
            DynamicChannelSelector(in_channels=512, emb_channels=300, channels_per_selector=128),
        ])
        
        self.attention = nn.Sequential(
            ContextSelfAttention(300, 8),
            # ContextSelfAttention(300, 8),
            # ContextSelfAttention(300, 8),
            TokenCollapser(),
            ContextSelfAttention(300, 4),
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        # self.initialize()

        self.embedding = nn.Embedding(8, 300)

        self.positional_embedding = nn.Parameter(torch.randn(8,300), requires_grad=True)

        self.get_parameter_count()

    def forward(self, x, context):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        
        emb = self.embedding(context) + self.positional_embedding
        emb = self.attention(emb)
        
        for i in range(len(features)):
            if i != 0:
                features[i] = self.mappers[i](emb, features[i])
                # features[i] = self.attention[i](features[i])

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
    
    def get_parameter_count(self):
        # print number of parameters of all components [encoder, mappers, attention, decoder, segmentation_head and classification_head]

        # get encoder parameter count
        encoder_parameters = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"Encoder parameters: {encoder_parameters}")

        # get mappers parameter count
        mappers_parameters = sum(p.numel() for p in self.mappers.parameters() if p.requires_grad)
        print(f"Mappers parameters: {mappers_parameters}")

        # get attention parameter count
        attention_parameters = sum(p.numel() for p in self.attention.parameters() if p.requires_grad)
        print(f"Attention parameters: {attention_parameters}")

        # get decoder parameter count
        decoder_parameters = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f"Decoder parameters: {decoder_parameters}")

        # get segmentation_head parameter count
        segmentation_head_parameters = sum(p.numel() for p in self.segmentation_head.parameters() if p.requires_grad)
        print(f"Segmentation head parameters: {segmentation_head_parameters}")