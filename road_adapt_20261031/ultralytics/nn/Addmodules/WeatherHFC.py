"""Opt-in HWD/FreqSelect adaptations for the Chapter 4 weather detector.

H: Xu et al., Haar Wavelet Downsampling, Pattern Recognition 143 (2023), 109819.
   https://github.com/apple1986/HWD/blob/main/HWD.py
   This is a shared-transform residual adaptation, not the standalone Down_wt module.
F: Chen et al., Frequency-Adaptive Dilated Convolution, CVPR 2024, pp. 3414-3425.
   FrequencySelection FFT branch, source commit 88e439d9419c9b7f9d2aca077d02a27d405891e9:
   https://github.com/Linwei-Chen/FADC/blob/88e439d9419c9b7f9d2aca077d02a27d405891e9/FADC_only/conv_custom.py
   The source header attributes copyright to Meta Platforms, Inc. and affiliates;
   consult that repository's LICENSE for the original implementation's terms.

Only FreqSelect's selected FFT configuration is adapted; AdaDR, AdaKern, MMCV and
global complex gates are not included. Differences from the author FFT code:
* FFTs and gate arithmetic run in float32 with autocast disabled, then restore
  the input dtype. This supports non-power-of-two feature maps under CUDA AMP.
* Unshifted, conjugate-symmetric rectangular cutoffs replace rounded half-open
  shifted slices. Each axis retains |f| <= 1/(2*k), including both boundary
  frequencies. This avoids unpaired boundary bins and keeps DC on tiny/odd maps.
  The source's fftshift call also shifts B/C axes; only spatial FFT axes are
  used here. Gates, band differences and the untouched final low band retain
  the selected source configuration and identity initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv
from .ResNet import BasicBlock, ConvNormLayer

__all__ = ('HaarDownsample', 'HaarResidualBasicBlock', 'HaarResNetStage',
           'WeatherFrequencySelection', 'WeatherFreqSelectConv')


class HaarDownsample(nn.Module):
    """One orthonormal Haar level, concatenated LL/LH/HL/HH along channels.

    A cell is anchored at input (2*r, 2*c). Zero-pad only the right/bottom
    boundary when needed, giving ceil(H/2), ceil(W/2) output coordinates.
    For cell [[a,b],[c,d]], LH=(a+b-c-d)/2 and HL=(a-b+c-d)/2.
    Both residual projections consume the same transformed tensor.
    """

    def forward(self, x):
        height, width = x.shape[-2:]
        x = F.pad(x, (0, width % 2, 0, height % 2))
        a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
        c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        return torch.cat(((a + b + c + d) * .5,
                          (a + b - c - d) * .5,
                          (a - b + c - d) * .5,
                          (a - b - c + d) * .5), dim=1)


class HaarResidualBasicBlock(nn.Module):
    """Replace only the first P2->P3 block's two stride-2 projections."""

    expansion = 1

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.haar = HaarDownsample()
        # Retain baseline key paths for unchanged branch2b and compatible BN.
        self.branch2a = ConvNormLayer(4 * ch_in, ch_out, 1, 1, act='relu')
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.short = ConvNormLayer(4 * ch_in, ch_out, 1, 1, act=None)
        self.shortcut = False
        self.att = False

    def forward(self, x):
        transformed = self.haar(x)
        main = self.branch2b(self.branch2a(transformed))
        return F.relu(main + self.short(transformed))


class HaarResNetStage(nn.Module):
    """Parser-compatible Blocks replacement limited to BasicBlock stage 3.

    The layer index and blocks.0.branch2b / blocks.1.* checkpoint paths are
    unchanged. Only the first block's new 4C->C projections change shapes.
    Native shape-intersection loading therefore preserves baseline semantics.
    """

    def __init__(self, ch_in, ch_out, count, block='BasicBlock', stage_num=3, att=False, variant='b'):
        super().__init__()
        if block != 'BasicBlock' or stage_num != 3 or att or variant != 'b' or count < 1:
            raise ValueError('HaarResNetStage supports only variant-b BasicBlock stage 3 without attention')
        self.blocks = nn.ModuleList([HaarResidualBasicBlock(ch_in, ch_out)])
        self.blocks.extend(BasicBlock(ch_out, ch_out, stride=1, shortcut=True, variant=variant, att=False)
                           for _ in range(count - 1))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class WeatherFrequencySelection(nn.Module):
    """FADC FreqSelect FFT-only configuration for projected 256-channel P4.

    Three 3x3 input-conditioned, channel-shared spatial gates use 2*sigmoid.
    k_list=(2,4,8), spatial_group=1, lowfreq_att=False, global_selection=False.
    Four telescoping bands sum to X; zero gate initialization starts at Y=X.
    No extra X residual is added. Channel count remains configurable for tests.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.k_list = (2, 4, 8)
        self.lowfreq_att = False
        self.lp_type = 'freq'
        self.spatial_group = 1
        self.spatial_kernel = 3
        self.act = 'sigmoid'
        self.global_selection = False
        self.freq_weight_conv_list = nn.ModuleList(nn.Conv2d(in_channels, 1, 3, padding=1, bias=True)
                                                   for _ in self.k_list)
        for gate in self.freq_weight_conv_list:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def lowpass_masks(self, height, width, device):
        """Rectangles in unshifted FFT ordering; +/- bins share inclusion."""
        row = torch.arange(height, device=device)
        col = torch.arange(width, device=device)
        # Compare integer bin distances to the exact rational cutoff. Avoid
        # float rounding at boundaries and handle odd dimensions and size one.
        row_distance = torch.minimum(row, height - row)
        col_distance = torch.minimum(col, width - col)
        return tuple(((row_distance * (2 * k) <= height)[:, None]
                      & (col_distance * (2 * k) <= width)[None, :]) for k in self.k_list)

    def decompose(self, x):
        """Return float32 [X-L2, L2-L4, L4-L8, L8] with spatial dimensions intact."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            source = x.float()
            spectrum = torch.fft.fft2(source, dim=(-2, -1), norm='ortho')
            previous = source
            bands = []
            for mask in self.lowpass_masks(*source.shape[-2:], source.device):
                low = torch.fft.ifft2(spectrum * mask, dim=(-2, -1), norm='ortho').real
                bands.append(previous - low)
                previous = low
            bands.append(previous)
            return bands

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            source = x.float()
            bands = self.decompose(source)
            output = bands[-1]
            for gate, band in zip(self.freq_weight_conv_list, bands[:-1]):
                # Explicit casts also support model.half() on odd/non-power-of-
                # two inputs while retaining gradients to the original weights.
                logits = F.conv2d(source, gate.weight.float(), gate.bias.float(), padding=1)
                output = output + (2 * logits.sigmoid()) * band
            return output.to(dtype=dtype)


class WeatherFreqSelectConv(Conv):
    """P4 projection followed by FreqSelect, keeping model.12.conv/bn keys.

    Replacing Conv in place avoids shifting any baseline graph/checkpoint
    index. Overriding forward_fuse retains FreqSelect after normal BN fusion.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.freq_select = WeatherFrequencySelection(c2)

    def forward(self, x):
        return self.freq_select(super().forward(x))

    def forward_fuse(self, x):
        return self.freq_select(self.act(self.conv(x)))
