# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2023 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# It is modified by Pin-Jui Ku to apply the model to the 5G channel denoising task


from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

class ConvPositionEmbed(Module):
    """The Convolutional Embedding to encode time information of each frame"""

    def __init__(self, dim: int, kernel_size: int, groups: Optional[int] = None):
        super().__init__()
        if (kernel_size % 2) == 0:
            raise ValueError(f"Kernel size {kernel_size} is divisible by 2!")

        if groups is None:
            groups = dim

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2), nn.GELU()
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor, shape (B, T, D)

        Return:
            out: output tensor with the same shape (B, T, D)
        """

        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(mask, 0.0)

        x = einops.rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = einops.rearrange(x, 'b c n -> b n c')

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class RMSNorm(Module):
    """The Root Mean Square Layer Normalization

    References:
      - Zhang et al., Root Mean Square Layer Normalization, 2019
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class GEGLU(Module):
    """The GeGLU activation implementation"""

    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def get_feedforward_layer(dim: int, mult: int = 4, dropout: float = 0.0):
    """
    Return a Feed-Forward layer for the Transformer Layer.
    GeGLU activation is used in this FF layer
    """
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim_inner, dim))


class BackboneTransformer(nn.Module):
    """
    Implementation of the transformer Encoder Model with U-Net structure used in
    VoiceBox and AudioBox

    References:
        Le et al., Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale, 2023
        Vyas et al., Audiobox: Unified Audio Generation with Natural Language Prompts, 2023
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 8,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        """
        Args:
            dim: Embedding dimension
            depth: Number of Transformer Encoder Layers
            heads: Number of heads in MHA
            ff_mult: The multiplier for the feedforward dimension (ff_dim = ff_mult * dim)
            attn_dropout: dropout rate for the MHA layer
            ff_dropout: droupout rate for the feedforward layer
            use_unet_skip_connection: Whether to use U-Net or not
        """
        super().__init__()
        if (depth % 2) != 0:
            raise ValueError(f"Number of layers {depth} is not divisible by 2!")
        self.layers = nn.ModuleList([])

        self.skip_connect_scale = 2**-0.5

        for ind in range(depth):
            layer = ind + 1
            has_skip = layer > (depth // 2)
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(dim * 2, dim) if has_skip else None,
                        RMSNorm(dim=dim),
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            batch_first=True,
                        ),
                        RMSNorm(dim=dim),
                        get_feedforward_layer(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.final_norm = RMSNorm(dim)


    def forward(self, x):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, T, C)
        """
        skip_connects = []

        for skip_combiner, attn_prenorm, attn, ff_prenorm, ff in self.layers:

            if skip_combiner is None:
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                x = torch.cat((x, skip_connect), dim=-1)
                x = skip_combiner(x)

            attn_input = attn_prenorm(x)

            attn_output, _ = attn(
                query=attn_input,
                key=attn_input,
                value=attn_input,
                need_weights=False,
            )
            x = x + attn_output

            ff_input = ff_prenorm(x)
            x = ff(ff_input) + x

        return self.final_norm(x)

class TransformerUNet(nn.Module):
    def __init__(
        self,
        symbol_length: int = 32,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        ff_mult: int = 2,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        conv_pos_embed_kernel_size: int = 31,
        conv_pos_embed_groups: Optional[int] = None,
    ):
        super().__init__()
        self.symbol_length = symbol_length
        self.proj_in = nn.Linear(symbol_length , dim)

        self.conv_embed = ConvPositionEmbed(
            dim=dim, kernel_size=conv_pos_embed_kernel_size, groups=conv_pos_embed_groups
        )

        self.transformerunet = BackboneTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
        )

        self.proj_out = nn.Linear(dim, symbol_length)



    def forward(self, input):

        input = einops.rearrange(input, 'B (D T) -> B T D', D=self.symbol_length)

        x = self.proj_in(input)
        x = self.conv_embed(x) + x

        x = self.transformerunet(x=x)

        output = self.proj_out(x)
        output = einops.rearrange(output, "B T D -> B (D T)")

        return output