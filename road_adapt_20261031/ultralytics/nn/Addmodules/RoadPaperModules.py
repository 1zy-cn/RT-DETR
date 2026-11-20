"""Paper names for the existing road adaptations, with checkpoint-safe aliases.

LMC-SA (Local Mixed-Compression Synergistic Attention) retains the original
SCSA SMSA/PCSA operations in ``SCSA.py``. Only the PCSA descriptor compression
is adapted by ``SCSARoad.py``. With its default ``learned_mix`` setting, A and
M are adaptive average/maximum descriptors on the same
max(1, H//7) by max(1, W//7) grid. For each channel and descriptor cell:

    r = sigmoid(mix_avg * A + mix_max * M + mix_bias)
    Z = (1 - r) * A + r * M

The three channel-wise parameters start at zero. SMSA and PCSA are inherited
from SCSA; average/max pooling and the surrounding attention are not claimed
as new inventions. The window size remains configurable through the old API.

BSS-FPN (Bounded Secondary-Screening Feature Pyramid Network) retains the
HS-FPN topology and CA operations in ``HSFPN.py``. ``BSSGate`` names only its
adapted second-screening node, implemented in ``HSFPNRoad.py``:

    alpha = sigmoid(gate_logit)
    E_out = ((1 - alpha) + alpha * g) * E

E has already undergone low-level channel selection. The existing graph
adds U after this node and then fuses with RepC3. Independent P3/P4 scalars
start at alpha=0.5; g is the inherited input-dependent sigmoid CA gate. The
bounded factor lies in [1-alpha, 1] when g is in [0, 1].

The names below are aliases, not subclasses or duplicate implementations.
They preserve state-dict keys, optimizer type checks, class/pickle paths and
the legacy parser's identity branches. Existing experimental controls remain
available through the original classes. The named YAML selects the two
learned variants; names alone introduce no additional operator or parameter
and provide no detection-accuracy or efficiency evidence.
"""

from .HSFPNRoad import GateStrengthMultiply
from .SCSARoad import BlocksSCSARoad, SCSARoad


__all__ = ['LMCSA', 'BlocksLMCSA', 'BSSGate']

LMCSA = SCSARoad
BlocksLMCSA = BlocksSCSARoad
BSSGate = GateStrengthMultiply
