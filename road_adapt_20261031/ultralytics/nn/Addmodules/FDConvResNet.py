"""Apply the upstream FDConv to one ResNet C4 residual convolution."""

from .FDConv import FDConv
from .ResNet import Blocks

__all__ = ['BlocksFDConvC4']


class BlocksFDConvC4(Blocks):
    """Keep the C4 stage intact apart from its second block's last 3x3 conv."""

    def __init__(self, ch_in, ch_out, count, block, stage_num, att=False, variant='b'):
        if stage_num != 4 or count != 2:
            raise ValueError('BlocksFDConvC4 is defined for two-block ResNet C4 only')
        super().__init__(ch_in, ch_out, count, block, stage_num, att, variant)
        original = self.blocks[1].branch2b.conv
        self.blocks[1].branch2b.conv = FDConv(
            original.in_channels, original.out_channels, kernel_size=3, stride=1,
            padding=1, bias=original.bias is not None, kernel_num=4)
