"""CDRB-inspired ResNet block for RT-DETR-R18.

The traffic-sign paper places a dilated reparameterization block before the
window attention of an inverted residual mobile block. This adaptation keeps
the repository's existing DRB/iRMB code and the ResNet stage interface.
"""

import torch.nn as nn
import torch.nn.functional as F

from .iRMB import iRMB
from .SCSA import SCSA
from .UniRepLknet import DilatedReparamBlock


class BasicBlock_CDRB(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', att=False):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError('CDRB supports only stride 1 or 2')
        self.shortcut = shortcut
        self.branch2a = (nn.Identity() if ch_in == ch_out and stride == 1 else
                         nn.Sequential(nn.Conv2d(ch_in, ch_out, 1, stride, bias=False),
                                       nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True)))
        self.drb = DilatedReparamBlock(ch_out, kernel_size=9, deploy=False,
                                      attempt_use_lk_impl=False)
        self.irmb = iRMB(ch_out, has_skip=False, exp_ratio=2.0,
                         dim_head=8, window_size=7, attn_s=True, dw_ks=3)
        if not shortcut:
            self.short = nn.Sequential(nn.Conv2d(ch_in, ch_out, 1, stride, bias=False),
                                       nn.BatchNorm2d(ch_out))
        self.att = att
        if att:
            self.se = SCSA(ch_out)

    def forward(self, x):
        out = self.irmb(self.drb(self.branch2a(x)))
        if self.att:
            out = self.se(out)
        return F.relu(out + (x if self.shortcut else self.short(x)))
