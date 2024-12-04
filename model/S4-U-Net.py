"""
The SaShiMi backbone and the SaShiMi-based Speech Enhancement Model.

The code is conducted by Karan Goel, Albert Gu, Chris Donahue, Christopher Re.
for the work for "It's Raw! Audio Generation with State-Space Models".
It is modified by Pin-Jui Ku to apply the model to the 5G channel denoising task
"""
import torch.nn as nn

from model.DSSM_modules.s4 import (
    S4,
    Conv,
    ZeroConv1d,
    FFBlock,
    ResidualBlock,
)
from model.DSSM_modules.components import TransposedLN

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
