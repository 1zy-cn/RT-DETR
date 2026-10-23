"""Three-convolution reparameterizable unit for RT-DETR P3 feature fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import RepC3
from .UniRepLknet import DilatedReparamBlock


class Conv3XC(nn.Module):
    """1x1-3x3-1x1 path plus a 1x1 skip; fuse explicitly for deployment."""

    def __init__(self, c_in, c_out, gain=2, act=False):
        super().__init__()
        hidden_in, hidden_out = c_in * gain, c_out * gain
        self.conv1 = nn.Conv2d(c_in, hidden_in, 1)
        self.conv2 = nn.Conv2d(hidden_in, hidden_out, 3, padding=0)
        self.conv3 = nn.Conv2d(hidden_out, c_out, 1)
        self.skip = nn.Conv2d(c_in, c_out, 1)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        if hasattr(self, 'reparam'):
            return self.act(self.reparam(x))
        out = self.conv3(self.conv2(self.conv1(F.pad(x, (1, 1, 1, 1))))) + self.skip(x)
        return self.act(out)

    @torch.no_grad()
    def switch_to_deploy(self):
        """Replace the linear branches by an exactly equivalent padded 3x3."""
        if hasattr(self, 'reparam'):
            return
        first = self.conv1.weight[:, :, 0, 0]
        middle = self.conv2.weight
        last = self.conv3.weight[:, :, 0, 0]
        kernel = torch.einsum('om,mnhw,ni->oihw', last, middle, first)
        kernel[:, :, 1, 1] += self.skip.weight[:, :, 0, 0]
        middle_bias = self.conv2.bias + torch.einsum('mnhw,n->m', middle, self.conv1.bias)
        bias = self.conv3.bias + torch.mv(last, middle_bias) + self.skip.bias
        fused = nn.Conv2d(self.conv1.in_channels, self.conv3.out_channels, 3, padding=1,
                          device=kernel.device, dtype=kernel.dtype)
        fused.weight.copy_(kernel)
        fused.bias.copy_(bias)
        self.reparam = fused
        del self.conv1, self.conv2, self.conv3, self.skip


class RepC3_Conv3XC(RepC3):
    """Keep RepC3 fusion routes, replacing only its internal RepConv units."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__(c1, c2, n, e)
        hidden = int(c2 * e)
        # RepConv applies SiLU after its summed branches. Preserve that
        # nonlinearity when replacing RepConv inside RT-DETR's RepC3.
        self.m = nn.Sequential(*(Conv3XC(hidden, hidden, act=True) for _ in range(n)))


@torch.no_grad()
def reparameterize_cdrb_conv3xc(model):
    """Fuse the candidate branches in an eval-mode PyTorch detector in place."""
    if model.training:
        raise RuntimeError('Call model.eval() before reparameterizing DRB and Conv3XC')
    for module in list(model.modules()):
        if isinstance(module, DilatedReparamBlock):
            module.merge_dilated_branches()
        elif isinstance(module, Conv3XC):
            module.switch_to_deploy()
    return model
