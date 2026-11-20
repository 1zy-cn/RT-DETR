"""Explicit controls for the second, high-level HSFPN channel gate.

This module replaces only a ``multiply([g, E])`` node.  The caller still
performs the later addition of the upsampled high-level feature U.  E has
already passed the original low-level channel selection; the module does
not recover unscreened features or establish a detection-performance gain.
"""

import math
from numbers import Real

import torch
from torch import nn


__all__ = ["GateStrengthMultiply"]


class GateStrengthMultiply(nn.Module):
    """Apply a bounded strength to an existing sigmoid gate.

    Args:
        mode: ``original`` applies g*E; ``none`` returns E; ``fixed`` uses
            (1-alpha+alpha*g)*E. ``learned`` learns alpha=sigmoid(gate_logit).
            ``mean_gate`` uses the same learned alpha after averaging g
            over channels, as a control for overall low-level amplitude.
        alpha: Fixed strength or learned initial strength. Fixed strengths
            include 0 and 1; learned initial strengths must be in (0, 1).

    Inputs are a pair [g, E] of four-dimensional floating tensors on the
    same device and with the same dtype. g must broadcast to E without
    enlarging E; the intended HSFPN gate has shape B*C*1*1. The original
    CA sigmoid supplies g in [0, 1]. Its range is not checked or clipped
    here, avoiding value-dependent device synchronization in forward.

    P3 and P4 should own separate instances. For learned modes, callers
    must explicitly place ``gate_logit`` in a no-weight-decay optimizer
    group if that is the registered experimental protocol. No global
    optimizer or legacy module behavior is changed by this class.
    """

    MODES = ("original", "none", "fixed", "learned", "mean_gate")

    def __init__(self, mode="learned", alpha=0.5):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown HSFPN gate mode {mode!r}; expected one of {self.MODES}")
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a finite real number")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and within [0, 1]")
        if mode in ("learned", "mean_gate") and not 0.0 < alpha < 1.0:
            raise ValueError("learned alpha initialization must lie strictly within (0, 1)")

        self.mode = mode
        self.alpha_init = alpha
        if mode in ("learned", "mean_gate"):
            # Constant initialization neither consumes RNG nor couples scales.
            self.gate_logit = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))
        else:
            self.register_parameter("gate_logit", None)

    @staticmethod
    def _validate_inputs(inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("GateStrengthMultiply expects the pair [gate, features]")
        if len(inputs) != 2:
            raise ValueError("GateStrengthMultiply expects exactly [gate, features]")
        gate, features = inputs
        if not isinstance(gate, torch.Tensor) or not isinstance(features, torch.Tensor):
            raise TypeError("gate and features must both be tensors")
        if gate.ndim != 4 or features.ndim != 4:
            raise ValueError("gate and features must both have shape B*C*H*W")
        if not gate.is_floating_point() or not features.is_floating_point():
            raise TypeError("gate and features must be floating tensors")
        if gate.device != features.device:
            raise ValueError("gate and features must be on the same device")
        if gate.dtype != features.dtype:
            raise TypeError("gate and features must have the same dtype")
        if any(size <= 0 for size in features.shape):
            raise ValueError("features must have nonempty batch, channel, and spatial dimensions")
        if any(gs not in (1, es) for gs, es in zip(gate.shape, features.shape)):
            raise ValueError("gate must broadcast to the feature shape without expanding it")
        return gate, features

    def forward(self, inputs):
        gate, features = self._validate_inputs(inputs)
        if self.mode == "original":
            return gate * features
        if self.mode == "none":
            return features
        if self.mode == "mean_gate":
            gate = gate.mean(dim=1, keepdim=True)

        alpha = self.alpha_init
        if self.gate_logit is not None:
            # Preserve the activation dtype under AMP; the cast keeps gradients.
            alpha = self.gate_logit.sigmoid().to(dtype=gate.dtype)
        return ((1.0 - alpha) + alpha * gate) * features

    def extra_repr(self):
        return f"mode={self.mode!r}, alpha_init={self.alpha_init}"
