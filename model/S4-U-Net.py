"""
The SaShiMi backbone and the SaShiMi-based Speech Enhancement Model.

The code is conducted by Karan Goel, Albert Gu, Chris Donahue, Christopher Re.
for the work for "It's Raw! Audio Generation with State-Space Models".
It is modified by Pin-Jui Ku to apply the model to the Speech Enhancement task.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSSM_modules.s4 import (
    S4,
    Conv,
    ZeroConv1d,
    DownPool,
    UpPool,
    FFBlock,
    ResidualBlock,
)
from model.DSSM_modules.components import TransposedLN

class S4_backbone(nn.Module):
    """
    Backbone of the SaShiMi model using S4 blocks in a U-Net-like architecture.

    Args:
        d_model (int): Dimension of the model, typically 64 for experiments.
        d_state (int): Dimension of the memory state of S4 block.
        n_layers_per_block (int): Number of layers per block.
        pool (list[int]): Pooling factors at each level.
        time_resample_factor (int): Factor for time resampling.
        ff_bottleneck_expand_factor (int): Expansion factor for FF bottleneck.
        bidirectional (bool): If True, use bidirectional S4 layers.
        unet (bool): If True, use a U-Net-like architecture.
        dropout (float): Dropout rate.
        **s4_args: Additional arguments for S4 blocks.
    """

    def __init__(self, 
        d_model=64, 
        d_state=64, 
        n_layers_per_block=8, 
        pool=[4, 4], 
        time_resample_factor=2, 
        ff_bottleneck_expand_factor=2, 
        bidirectional=False,
        unet=True, 
        dropout=0.0, 
        **s4_args
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers_per_block = n_layers_per_block
        self.time_resample_factor = time_resample_factor
        self.ff_bottleneck_expand_factor = ff_bottleneck_expand_factor
        self.bidirectional = bidirectional
        self.unet = unet
        self._init_blocks(pool, dropout, **s4_args)
        self.norm = nn.LayerNorm(self.d_model)

    def _init_blocks(self, pool, dropout, **s4_args):
        """Initializes the down, center, and up blocks of the network."""
        self.current_d_model = self.d_model
        self.down_blocks = self._create_down_blocks(pool, dropout, **s4_args)
        self.center_block = self._create_center_block(dropout, **s4_args)
        self.up_blocks = self._create_up_blocks(pool, dropout, **s4_args)

    def _create_down_blocks(self, pool, dropout, **s4_args):
        """Creates down sampling blocks."""
        down_blocks = []
        for p in pool:
            block = self._create_block_layers(dropout, **s4_args)
            block.append(DownPool(self.current_d_model, self.time_resample_factor, p))
            down_blocks.append(nn.ModuleList(block))
            self.current_d_model *= self.time_resample_factor
        return nn.ModuleList(down_blocks)

    def _create_center_block(self, dropout, **s4_args):
        """Creates center block layers."""
        return nn.ModuleList(self._create_block_layers(dropout, **s4_args))

    def _create_up_blocks(self, pool, dropout, **s4_args):
        """Creates up sampling blocks."""
        up_blocks = []
        for p in reversed(pool):
            block = [UpPool(self.current_d_model, self.time_resample_factor, p)]
            self.current_d_model //= self.time_resample_factor
            block.extend(self._create_block_layers(dropout, **s4_args))
            up_blocks.append(nn.ModuleList(block))
        return nn.ModuleList(up_blocks)

    def _create_block_layers(self, dropout, **s4_args):
        """Creates layers for a block."""
        block_layers = []
        for _ in range(self.n_layers_per_block):
            block_layers.append(self._s4_block(self.current_d_model, self.d_state, self.bidirectional, dropout, **s4_args))
            if self.ff_bottleneck_expand_factor > 0:
                block_layers.append(self._ff_block(self.current_d_model, self.ff_bottleneck_expand_factor, dropout))
        return block_layers

    def _s4_block(self, dim, d_state, bidirectional, dropout, **s4_args):
        """Creates an S4 block."""
        layer = S4(d_model=dim, d_state=d_state, bidirectional=bidirectional, dropout=dropout, transposed=True, **s4_args)
        return ResidualBlock(d_model=dim, layer=layer, dropout=dropout)

    def _ff_block(self, dim, ff_bottleneck_expand_factor, dropout):
        """Creates a feed-forward block."""
        layer = FFBlock(d_model=dim, ff_bottleneck_expand_factor=ff_bottleneck_expand_factor, dropout=dropout)
        return ResidualBlock(d_model=dim, layer=layer, dropout=dropout)

    def forward(self, x: torch.Tensor, weighted_hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the S4_backbone."""
        outputs = [x]

        # Down sampling
        for block in self.down_blocks:
            for layer in block:
                #import ipdb; ipdb.set_trace()
                x = layer(x)
                outputs.append(x)

        # Center block
        for layer in self.center_block:
            x = layer(x)
        x = x + outputs.pop()
        
        if weighted_hidden_state is not None:
            batch, channel, time_length = x.shape
            weighted_hidden_state = F.interpolate(weighted_hidden_state.unsqueeze(1), size=(channel, time_length))
            x = x + weighted_hidden_state.squeeze(1)

        # Up sampling
        for block in self.up_blocks:
            for layer in block:
                x = layer(x)
                if self.unet or isinstance(layer, UpPool):
                    x = x + outputs.pop()

        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class S4UNet(nn.Module):
    def __init__(
        self, 
        in_channels=1,
        out_channels=1,
        d_model=64, 
        d_state=16,
        n_layers_per_block=4, 
        dropout=0.0, 
    ):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(in_channels, d_model, kernel_size=1), nn.ReLU()
        )

        self.backbone = S4_backbone(
            d_model=d_model, d_state=d_state, n_layers_per_block=n_layers_per_block, dropout=dropout
        )
        self.norm = TransposedLN(d_model)
        self.final_conv = nn.Sequential(
            Conv(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(d_model, out_channels),
        )
        

    def forward(self, inputs):
        org_length = inputs.size(1)
        inputs = inputs.unsqueeze(1)
        inputs = self.add_padding(inputs)

        x = self.init_conv(inputs)
        x = self.backbone(x)
        x = self.norm(x)
        x = self.final_conv(x)
        x = x.squeeze(1)
        x = self.remove_padding(x, org_length)
        return x


    def add_padding(self, inputs: torch.Tensor):
        assert inputs.dim() == 3
        if (inputs.size(2) % 16) != 0:
            padding_length = 16 - (inputs.size(2) % 16)
            padding = torch.zeros([inputs.size(0), inputs.size(1), padding_length]).cuda()
            wav = torch.concat([inputs, padding], dim=2)

        return wav

    def remove_padding(self, outputs, length):
        assert outputs.dim() == 2
        return outputs[:, :length]


class S4Compact(nn.Module):
    def __init__(
        self, 
        in_channels=1,
        out_channels=1,
        d_model=64, 
        d_state=16,
        n_layers=4,
        dropout=0.0, 
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_state = d_state
        self.ff_bottleneck_expand_factor = 2
        self.bidirectional = False
        self.init_conv = nn.Sequential(
            Conv(in_channels, d_model, kernel_size=1), nn.ReLU()
        )

        self.S4_block = nn.ModuleList(self._create_block_layers(dropout=dropout))

        self.norm = TransposedLN(d_model)
        self.final_conv = nn.Sequential(
            Conv(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(d_model, out_channels),
        )

    def _create_block_layers(self, dropout, **s4_args):
        """Creates layers for a block."""
        block_layers = []
        for _ in range(self.n_layers):
            block_layers.append(self._s4_block(self.d_model, self.d_state, self.bidirectional, dropout, **s4_args))
            if self.ff_bottleneck_expand_factor > 0:
                block_layers.append(self._ff_block(self.d_model, self.ff_bottleneck_expand_factor, dropout))
        return block_layers

    def _s4_block(self, dim, d_state, bidirectional, dropout, **s4_args):
        """Creates an S4 block."""
        layer = S4(d_model=dim, d_state=d_state, bidirectional=bidirectional, dropout=dropout, transposed=True, **s4_args)
        return ResidualBlock(d_model=dim, layer=layer, dropout=dropout)

    def _ff_block(self, dim, ff_bottleneck_expand_factor, dropout):
        """Creates a feed-forward block."""
        layer = FFBlock(d_model=dim, ff_bottleneck_expand_factor=ff_bottleneck_expand_factor, dropout=dropout)
        return ResidualBlock(d_model=dim, layer=layer, dropout=dropout)

    def forward(self, inputs):

        inputs = inputs.unsqueeze(1)
        x = self.init_conv(inputs)
        
        for layer in self.S4_block:
            #import ipdb; ipdb.set_trace()
            x = layer(x)

        x = self.norm(x)
        x = self.final_conv(x)
        x = x.squeeze(1)
        return x
