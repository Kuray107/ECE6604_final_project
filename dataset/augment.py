# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import random
import torch
import torchaudio
from torch import nn

from util import dsp


class Remix(nn.Module):
    """Remix.
    Mixes different noises with clean speech within a given batch
    """

    def forward(self, sources):
        noise, clean = sources
        bs, *other = noise.shape
        device = noise.device
        perm = torch.argsort(torch.rand(bs, device=device), dim=0)
        return torch.stack([noise[perm], clean])


class BandMask(nn.Module):
    """BandMask.
    Maskes bands of frequencies. Similar to Park, Daniel S., et al.
    "Specaugment: A simple data augmentation method for automatic speech recognition."
    (https://arxiv.org/pdf/1904.08779.pdf) but over the waveform.
    """

    def __init__(self, maxwidth=0.2, bands=120, sample_rate=16000):
        """__init__.
        :param maxwidth: the maximum width to remove
        :param bands: number of bands
        :param sample_rate: signal sample rate
        """
        super().__init__()
        self.maxwidth = maxwidth
        self.bands = bands
        self.sample_rate = sample_rate

    def forward(self, wav):
        bands = self.bands
        bandwidth = int(abs(self.maxwidth) * bands)
        mels = dsp.mel_frequencies(bands, 40, self.sample_rate / 2) / self.sample_rate
        low = random.randrange(bands)
        high = random.randrange(low, min(bands, low + bandwidth))
        filters = dsp.LowPassFilters([mels[low], mels[high]]).to(wav.device)
        low, midlow = filters(wav)
        # band pass filtering
        out = wav - midlow + low
        return out


class SpeedChange(nn.Module):
    """SpeedChange
    Change the speed of the input wavs. Note that in forward we first put the
    wav to CPU, do the speed change and then put it back to GPU.
    It is a stupid way to do so, but I don't see any other easy way for now.
    """

    def __init__(self, min_speed=0.9, max_speed=1.1, sample_rate=16000):
        super().__init__()
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.sample_rate = sample_rate

    def forward(self, wav):
        # The torchaudio.sox_effect doesn't support tensors in GPU. This is a stupid way to do so
        noise, clean = wav.cpu()
        speed = random.uniform(self.min_speed, self.max_speed)
        effects = [
            ["speed", f"{speed}"],  # change the speed
            ["rate", f"{self.sample_rate}"],  # This step is necessary
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_tensor(noise, self.sample_rate, effects)
        clean, _ = torchaudio.sox_effects.apply_effects_tensor(clean, self.sample_rate, effects)
        wav = torch.stack([noise, clean]).cuda()
        return wav
