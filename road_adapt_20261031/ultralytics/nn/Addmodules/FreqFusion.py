"""P3 FreqFusion adapter with a fixed, dependency-free PyTorch backend.

Reference: Chen et al., Frequency-aware Feature Fusion for Dense Image Prediction,
TPAMI 2024, https://github.com/Linwei-Chen/FreqFusion
Reference revision: 3fb0c70637a3c194fb74294d3ce4681958b26241 (FreqFusion.py).
See docs/experiments/freqfusion.md for settings, attribution and local adaptations.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['FreqFusionConcat']


def carafe(x, mask, kernel_size, group=1, up=1):
    """Content-aware reassembly matching the upstream reflection-padding fallback.

    Contract low-resolution patches with subpixel masks directly, avoiding a
    materialized nearest-upsampled B*C*K*K*H*W patch tensor. No MMCV auto-switch.
    """
    b, c, h, w = x.shape
    if group != 1 or up not in (1, 2) or kernel_size % 2 != 1:
        raise ValueError('This FreqFusion backend requires group=1, up=1/2 and an odd kernel')
    if mask.shape != (b, kernel_size**2, h*up, w*up):
        raise ValueError('Reassembly mask does not match the input and upsampling factor')
    pad = kernel_size // 2
    patches = F.unfold(F.pad(x, (pad, pad, pad, pad), mode='reflect'), kernel_size)
    patches = patches.reshape(b, c, kernel_size**2, h, w)
    weights = mask.reshape(b, kernel_size**2, h, up, w, up)
    return torch.einsum('bckhw,bkhuwv->bchuwv', patches, weights).reshape(b, c, h*up, w*up)


def compute_similarity(x):
    """Eight cosine similarities in a dilated 3x3 neighborhood (upstream defaults)."""
    b, c, h, w = x.shape
    patches = F.unfold(x, 3, padding=2, dilation=2).reshape(b, c, 9, h, w)
    similarities = F.cosine_similarity(patches[:, :, 4:5], patches, dim=1)
    return torch.cat((similarities[:, :4], similarities[:, 5:]), dim=1)


class LocalSimGuidedSampler(nn.Module):
    """Two-times, four-group offset resampling with local similarity guidance."""

    def __init__(self, channels, groups=4):
        super().__init__()
        if channels % 8 or channels % groups:
            raise ValueError('Compressed channels must be divisible by 8 and sampler groups')
        self.groups = groups
        self.norm_hr = nn.GroupNorm(channels // 8, channels)
        self.norm_lr = nn.GroupNorm(channels // 8, channels)
        self.offset = nn.Conv2d(channels + 8, 8*groups, 3, padding=1)
        self.hr_offset = nn.Conv2d(channels + 8, 2*groups, 3, padding=1)
        self.direct_scale = nn.Conv2d(channels + 8, 8*groups, 3, padding=1)
        self.hr_direct_scale = nn.Conv2d(channels + 8, 2*groups, 3, padding=1)
        pos = torch.tensor([-.25, .25])
        initial = torch.stack(torch.meshgrid(pos, pos, indexing='ij')).transpose(1, 2)
        self.register_buffer('init_pos', initial.repeat(1, groups, 1).reshape(1, -1, 1, 1))

    def forward(self, hr_x, lr_x, feat2sample):
        hr_x, lr_x = self.norm_hr(hr_x), self.norm_lr(lr_x)
        hr = torch.cat((hr_x, compute_similarity(hr_x)), dim=1)
        lr = torch.cat((lr_x, compute_similarity(lr_x)), dim=1)
        offset = self.offset(lr) + F.pixel_unshuffle(self.hr_offset(hr), 2)
        scale = (self.direct_scale(lr) + F.pixel_unshuffle(self.hr_direct_scale(hr), 2)).sigmoid()
        offset = offset * scale + self.init_pos
        b, _, h, w = offset.shape
        offset = offset.reshape(b, 2, -1, h, w)
        yy, xx = torch.meshgrid(torch.arange(h, device=lr.device, dtype=lr.dtype) + .5,
                                torch.arange(w, device=lr.device, dtype=lr.dtype) + .5, indexing='ij')
        coords = torch.stack((xx, yy)).reshape(1, 2, 1, h, w)
        normalizer = lr.new_tensor([w, h]).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), 2)
        coords = coords.reshape(b, 2, self.groups, h*2, w*2).permute(0, 2, 3, 4, 1)
        sampled = F.grid_sample(feat2sample.reshape(b*self.groups, -1, h*2, w*2),
                                coords.reshape(b*self.groups, h*2, w*2, 2),
                                mode='bilinear', padding_mode='border', align_corners=False)
        return sampled.reshape(b, -1, h*2, w*2)


class FreqFusion(nn.Module):
    """Fixed full ALPF+AHPF+offset variant; HR resolution must be twice LR."""

    def __init__(self, hr_channels, lr_channels, compressed_channels=64, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.feature_resample = True
        self.use_high_pass = self.use_low_pass = True
        self.hr_channel_compressor = nn.Conv2d(hr_channels, compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, compressed_channels, 1)
        self.content_encoder = nn.Conv2d(compressed_channels, 25, 3, padding=1)
        self.content_encoder2 = nn.Conv2d(compressed_channels, 9, 3, padding=1)
        self.dysampler = LocalSimGuidedSampler(compressed_channels)
        for name, size in [('lowpass', 5), ('highpass', 3)]:
            window = torch.hamming_window(size, periodic=False)
            self.register_buffer('hamming_' + name, torch.outer(window, window)[None, None])
        # Match the reference parent initialization, including the offset layers.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.content_encoder.weight, std=.001)
        nn.init.normal_(self.content_encoder2.weight, std=.001)

    @staticmethod
    def normalize_kernel(mask, window):
        weights = mask.softmax(dim=1) * window.reshape(1, -1, 1, 1)
        return weights / weights.sum(dim=1, keepdim=True)

    def forward(self, hr_feat, lr_feat):
        if hr_feat.shape[-2:] != (2*lr_feat.shape[-2], 2*lr_feat.shape[-1]):
            raise ValueError('FreqFusion requires HR height/width exactly twice LR')
        if self.training and self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(self._forward, hr_feat, lr_feat, use_reentrant=False)
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        hr = self.hr_channel_compressor(hr_feat)
        lr = self.lr_channel_compressor(lr_feat)
        high_hr_logits = self.content_encoder2(hr)
        high_initial = self.normalize_kernel(high_hr_logits, self.hamming_highpass)
        hr = 2*hr - carafe(hr, high_initial, 3)
        low_hr_logits = self.content_encoder(hr)
        low_initial = self.normalize_kernel(low_hr_logits, self.hamming_lowpass)
        low_logits = low_hr_logits + carafe(self.content_encoder(lr), low_initial, 5, up=2)
        low_mask = self.normalize_kernel(low_logits, self.hamming_lowpass)
        high_logits = high_hr_logits + carafe(self.content_encoder2(lr), low_mask, 5, up=2)
        high_mask = self.normalize_kernel(high_logits, self.hamming_highpass)
        low_out = carafe(lr_feat, low_mask, 5, up=2)
        high_out = 2*hr_feat - carafe(hr_feat, high_mask, 3)
        low_out = self.dysampler(hr, lr, low_out)
        return low_mask, high_out, low_out


class FreqFusionConcat(nn.Module):
    """YAML adapter: [projected C3 (HR), fused P4 (LR)] -> [LR-up, HR]."""

    def __init__(self, hr_channels, lr_channels, compressed_channels=64):
        super().__init__()
        self.fusion = FreqFusion(hr_channels, lr_channels, compressed_channels)

    def forward(self, inputs):
        _, high, low = self.fusion(inputs[0], inputs[1])
        return torch.cat((low, high), dim=1)
