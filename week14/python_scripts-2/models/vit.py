"""
Script constructs a Vit model for segmentation
the code is from "https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/vision-transformer-for-oxford-iiit-pet-segmentatio.ipynb"
"""
import torch
import torch.nn as nn
from .vit_utils import SelfAttentionEncoderBlock, VisionTransformerInput, OutputProjection


class SegmentViT(nn.Module):
    """
    The class constructs a ViT model for segmentation
    """
    def __init__(self, image_size, patch_size, in_channels, out_channels, embed_size,num_blocks, num_heads,dropout ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        heads = [ SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for i in range(num_blocks) ]
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            VisionTransformerInput(image_size, patch_size, in_channels, embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, patch_size, embed_size, out_channels),
        )

    def forward(self, x):
        """
        Image is passed through the model of size (Batch_size, in_channels, image_size, image_size)
        """
        x = self.layers(x)
        return x
