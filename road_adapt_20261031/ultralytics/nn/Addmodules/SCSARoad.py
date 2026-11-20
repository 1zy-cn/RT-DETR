"""Opt-in SCSA compression controls for the road-scene experiments.

The original SCSA/SCSA residual blocks remain unchanged.  This module reuses
their spatial attention and channel-attention forward path and replaces only
the compression operator.  These implementations do not imply an accuracy
improvement; that requires independently recorded detection experiments.
"""

from numbers import Integral
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .SCSA import BasicBlock_SCSA, SCSA


__all__ = ['RoadCompression', 'SCSARoad', 'BasicBlock_SCSARoad', 'BlocksSCSARoad']


class RoadCompression(nn.Module):
    """Pool a feature map with one of five explicit experimental controls.

    Adaptive modes keep the old floor(H/window) by floor(W/window) output
    grid when the input is at least one window, while covering every input
    position.  Unlike fixed-window pooling, adaptive bins can overlap at
    non-divisible sizes; full coverage is not a lossless representation.
    """

    MODES = frozenset({'fixed_avg', 'adaptive_avg', 'adaptive_max', 'fixed_mix', 'learned_mix'})

    def __init__(self, dim: int, window_size: int = 7, mode: str = 'learned_mix'):
        super().__init__()
        if not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0:
            raise ValueError('dim must be a positive integer')
        if (not isinstance(window_size, Integral) or isinstance(window_size, bool)
                or window_size <= 0):
            raise ValueError('window_size must be a positive integer')
        if mode not in self.MODES:
            raise ValueError(f'Unsupported compression mode: {mode!r}')
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.mode = mode
        if mode == 'learned_mix':
            # Constant initialization consumes no additional random draws.
            self.mix_avg = nn.Parameter(torch.zeros(1, self.dim, 1, 1))
            self.mix_max = nn.Parameter(torch.zeros(1, self.dim, 1, 1))
            self.mix_bias = nn.Parameter(torch.zeros(1, self.dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f'Expected a BCHW tensor with {self.dim} channels')
        height, width = x.shape[-2:]
        if height <= 0 or width <= 0:
            raise ValueError('Spatial dimensions must be nonempty')
        if self.mode == 'fixed_avg':
            if height < self.window_size or width < self.window_size:
                raise ValueError('fixed_avg requires both spatial dimensions >= window_size')
            return F.avg_pool2d(x, kernel_size=self.window_size, stride=self.window_size)

        grid = (max(1, height // self.window_size), max(1, width // self.window_size))
        if self.mode == 'adaptive_avg':
            return F.adaptive_avg_pool2d(x, grid)
        if self.mode == 'adaptive_max':
            return F.adaptive_max_pool2d(x, grid)
        average = F.adaptive_avg_pool2d(x, grid)
        maximum = F.adaptive_max_pool2d(x, grid)
        if self.mode == 'fixed_mix':
            return 0.5 * average + 0.5 * maximum

        # Respect the activation dtype under mixed precision.  The casts are
        # differentiable and preserve the registered trainable parameters.
        avg_scale = self.mix_avg.to(dtype=average.dtype)
        max_scale = self.mix_max.to(dtype=average.dtype)
        bias = self.mix_bias.to(dtype=average.dtype)
        weight = torch.sigmoid(avg_scale * average + max_scale * maximum + bias)
        return (1 - weight) * average + weight * maximum

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, mode={self.mode!r}'


class SCSARoad(SCSA):
    """SCSA with an opt-in compression mode and effective head selection.

    ``head_num=None`` reproduces the local legacy dim//64 rule.  A positive
    explicit head count is honored, unlike the local legacy constructor.
    ``fixed_avg`` adds no parameters and accepts the legacy state dict
    strictly.  All other SCSA operations are the inherited implementation.
    """

    def __init__(self, dim: int, head_num: Optional[int] = None, window_size: int = 7,
                 compression_mode: str = 'learned_mix', **scsa_kwargs):
        if not isinstance(dim, Integral) or isinstance(dim, bool) or dim < 4 or dim % 4:
            raise ValueError('SCSA dim must be a positive multiple of four')
        selected_heads = max(1, dim // 64) if head_num is None else head_num
        if (not isinstance(selected_heads, Integral) or isinstance(selected_heads, bool)
                or selected_heads <= 0 or dim % selected_heads):
            raise ValueError('head_num must be positive and divide dim')
        if 'down_sample_mode' in scsa_kwargs:
            raise ValueError('Use compression_mode to select SCSARoad compression')
        # Validate before constructing any randomly initialized parent layers.
        compression = RoadCompression(dim, window_size, compression_mode)
        super().__init__(dim=int(dim), head_num=int(selected_heads),
                         window_size=int(window_size), down_sample_mode='avg_pool', **scsa_kwargs)
        self.head_num = int(selected_heads)
        self.head_dim = int(dim) // self.head_num
        self.scaler = self.head_dim ** -0.5
        self.compression_mode = compression_mode
        self.down_func = compression


class BasicBlock_SCSARoad(BasicBlock_SCSA):
    """Legacy residual block with only its SCSA implementation opted in."""

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', att=False,
                 compression_mode='learned_mix', head_num=None, window_size=7):
        # Build convolutions once, in the old order, without first constructing
        # and discarding a legacy attention module (which changes the RNG).
        super().__init__(ch_in, ch_out, stride, shortcut, act=act, variant=variant, att=False)
        self.att = bool(att)
        if self.att:
            self.se = SCSARoad(ch_out, head_num=head_num, window_size=window_size,
                               compression_mode=compression_mode)


class BlocksSCSARoad(nn.Module):
    """Stage wrapper that explicitly forwards new SCSA experiment settings."""

    def __init__(self, ch_in, ch_out, count, stage_num, att=True, variant='b',
                 compression_mode='learned_mix', head_num=None, window_size=7):
        super().__init__()
        if not isinstance(count, Integral) or isinstance(count, bool) or count < 1:
            raise ValueError('count must be a positive integer')
        if stage_num not in (2, 3, 4, 5):
            raise ValueError('stage_num must be one of 2, 3, 4, 5')
        self.blocks = nn.ModuleList()
        for index in range(count):
            self.blocks.append(BasicBlock_SCSARoad(
                ch_in, ch_out,
                stride=2 if index == 0 and stage_num != 2 else 1,
                shortcut=index != 0, variant=variant, att=att,
                compression_mode=compression_mode, head_num=head_num, window_size=window_size))
            ch_in = ch_out * BasicBlock_SCSARoad.expansion

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
