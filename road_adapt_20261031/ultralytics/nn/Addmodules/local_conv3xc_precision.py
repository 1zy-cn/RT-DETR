"""Opt-in local AMP precision repair; no parameter or architecture changes."""
import torch
from .Conv3XC import RepC3_Conv3XC


class FP32RepC3Conv3XC(RepC3_Conv3XC):
    """Keep the complete P3 fusion in FP32 until its output normalization.

    FP32 only inside individual Conv3XC units is insufficient: an intermediate
    larger than 65504 would overflow again before the outer projection/BN.
    """
    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            return super().forward(x.float())


def enable_local_fp32_conv3xc(model,layer=19):
    block=model.model[layer]
    if not isinstance(block,RepC3_Conv3XC):
        raise RuntimeError(f'Expected RepC3_Conv3XC P3 fusion at layer{layer}')
    # Preserve the existing module, parameters, state keys, parser metadata and
    # RNG state. Both classes are ordinary Python nn.Modules with one __dict__.
    block.__class__=FP32RepC3Conv3XC
    return model
