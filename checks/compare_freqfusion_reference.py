"""Compare outputs/gradients to the pinned upstream standalone FreqFusion.py.

Usage: python checks/compare_freqfusion_reference.py /path/to/upstream/FreqFusion.py
This optional audit reads an already-downloaded reference; it never downloads code.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.nn.Addmodules.FreqFusion import FreqFusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    options = parser.parse_args()
    digest = hashlib.sha256(options.reference.read_bytes()).hexdigest()
    assert digest == '5852efce406681ce6f3f7763ff8e034cab44f93cb0fdda0114130bc4c8c97e42', 'Unexpected upstream source'
    spec = importlib.util.spec_from_file_location('freqfusion_reference', options.reference)
    reference = importlib.util.module_from_spec(spec)
    # Deliberately compare to the reference torch/reflect backend, even if MMCV exists.
    with patch.dict(sys.modules, {'mmcv': None}):
        spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    expected_model = reference.FreqFusion(16, 24, feature_resample=True).double().train()
    actual_model = FreqFusion(16, 24).double().train()
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    expected_inputs = [torch.randn(1, 16, 12, 16, dtype=torch.double, requires_grad=True),
                       torch.randn(1, 24, 6, 8, dtype=torch.double, requires_grad=True)]
    actual_inputs = [x.detach().clone().requires_grad_() for x in expected_inputs]
    with contextlib.redirect_stdout(io.StringIO()):
        expected = expected_model(*expected_inputs)
    actual = actual_model(*actual_inputs)
    errors = []
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-7, atol=1e-9)
        errors.append((a-e).abs().max().item())
    sum(x.square().mean() for x in actual).backward()
    sum(x.square().mean() for x in expected).backward()
    gradient_errors = []
    pairs = list(zip(actual_inputs, expected_inputs))
    expected_params = dict(expected_model.named_parameters())
    pairs += [(parameter, expected_params[name]) for name, parameter in actual_model.named_parameters()]
    for a, e in pairs:
        assert a.grad is not None and e.grad is not None
        torch.testing.assert_close(a.grad, e.grad, rtol=1e-6, atol=1e-9)
        gradient_errors.append((a.grad-e.grad).abs().max().item())
    print(json.dumps({'reference_commit': '3fb0c70637a3c194fb74294d3ce4681958b26241',
                      'reference_sha256': digest, 'backend': 'torch_reflect', 'torch': str(torch.__version__),
                      'output_max_absolute_errors': errors, 'gradient_max_absolute_error': max(gradient_errors),
                      'gradient_tensors_checked': len(pairs), 'passed': True}, indent=2))


if __name__ == '__main__':
    main()
