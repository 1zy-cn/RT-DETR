"""Offline CPU checks for SCSA activation in the ResNet18 experiments.

Run from the repository root: python -m pytest checks/test_scsa.py -v -s -o addopts=
Uses synthetic data and random weights; no dataset downloads or training loop.
"""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.Addmodules.SCSA import BasicBlock_SCSA, SCSA
from ultralytics.nn.tasks import RTDETRDetectionModel


MODEL_DIR = Path(__file__).resolve().parents[1] / 'ultralytics/cfg/models/rt-detr'
ENABLED_CONFIGS = ['rtdetr-resnet18-SCSA.yaml', 'resnet18-HSFPN+SCSA.yaml']


@pytest.fixture(autouse=True)
def cpu_seed():
    """Bound CPU usage and make the synthetic checks repeatable."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(old_threads)


def build_model(config):
    model = RTDETRDetectionModel(str(MODEL_DIR / config), nc=7, verbose=False).cpu()
    # DetectionTrainer.set_model_attributes normally supplies this to the loss.
    model.nc = 7
    return model


@pytest.mark.parametrize('config, expected', [
    ('rtdetr-resnet18.yaml', 0),
    ('rtdetr-resnet18-HSFPN.yaml', 0),
    ('rtdetr-resnet18-SCSA.yaml', 8),
    ('resnet18-HSFPN+SCSA.yaml', 8),
])
def test_config_builds_expected_attention(config, expected, record_property):
    """Catch disabled YAML attention flags and accidental changes to controls."""
    model = build_model(config)
    attention = [(name, module) for name, module in model.named_modules() if isinstance(module, SCSA)]
    assert len(attention) == expected, f'{config}: expected {expected} active SCSA modules, found {len(attention)}'
    if expected:
        assert [module.dim for _, module in attention] == [64, 64, 128, 128, 256, 256, 512, 512]
        assert all(block.att and hasattr(block, 'se') for block in model.modules()
                   if isinstance(block, BasicBlock_SCSA))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    record_property('parameters', parameters)
    record_property('scsa_modules', len(attention))
    print(f'{config}: SCSA={len(attention)}, parameters={parameters}')


@pytest.mark.parametrize('config', ENABLED_CONFIGS)
def test_attention_executes_changes_output_and_backpropagates(config, record_property):
    """Catch bypassed forward branches, detached attention, and absent gradients."""
    model = build_model(config)
    attention = dict((name, module) for name, module in model.named_modules() if isinstance(module, SCSA))
    assert len(attention) == 8, 'No valid attention path to verify'
    image = torch.rand(1, 3, 256, 256)
    calls = dict.fromkeys(attention, 0)

    def count_call(name):
        def hook(module, inputs, output):
            calls[name] += 1
        return hook

    handles = [module.register_forward_hook(count_call(name)) for name, module in attention.items()]
    try:
        model.eval()
        with torch.inference_mode():
            enabled_output = model(image)[0]
        assert enabled_output.shape == (1, 300, 11)
        assert torch.isfinite(enabled_output).all()
        assert all(value == 1 for value in calls.values()), calls

        # Use identical weights and input; change only the existing attention switch.
        blocks = [module for module in model.modules() if isinstance(module, BasicBlock_SCSA)]
        try:
            for block in blocks:
                block.att = False
            with torch.inference_mode():
                disabled_output = model(image)[0]
        finally:
            for block in blocks:
                block.att = True
        assert all(value == 1 for value in calls.values()), 'Disabled attention was still called'
        assert not torch.allclose(enabled_output, disabled_output, rtol=1e-5, atol=1e-7)

        model.train()
        batch = {'img': image, 'batch_idx': torch.tensor([0, 0]),
                 'cls': torch.tensor([[0.], [3.]]),
                 'bboxes': torch.tensor([[0.35, 0.4, 0.2, 0.15], [0.7, 0.65, 0.15, 0.2]])}
        loss, _ = model(batch)
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        assert all(value == 2 for value in calls.values()), calls
        branch_norms = {}
        for name, module in attention.items():
            for parameter_name, parameter in module.named_parameters():
                assert parameter.grad is not None, f'{name}.{parameter_name}: missing gradient'
                assert torch.isfinite(parameter.grad).all(), f'{name}.{parameter_name}: nonfinite gradient'
            spatial = sum(parameter.grad.abs().sum().item() for key, parameter in module.named_parameters()
                          if 'dwc' in key)
            channel = sum(parameter.grad.abs().sum().item() for key, parameter in module.named_parameters()
                          if key.startswith(('q.', 'k.', 'v.')))
            assert spatial > 0 and channel > 0, f'{name}: inactive spatial/channel gradient'
            branch_norms[name] = {'spatial': spatial, 'channel': channel}

        before = {name: module.local_dwc.weight.detach().clone() for name, module in attention.items()}
        torch.optim.SGD(model.parameters(), lr=0.001).step()
        assert all(not torch.equal(before[name], module.local_dwc.weight) for name, module in attention.items())
        record_property('synthetic_detection_loss', loss.item())
        record_property('attention_branch_gradients', str(branch_norms))
        print(f'{config}: all 8 modules executed, changed predictions, received gradients and updated weights; '
              f'synthetic loss={loss.item():.6f}')
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('config', ENABLED_CONFIGS)
def test_full_resolution_cpu_inference(config):
    """Verify the experiment's 640-pixel input through the real detector."""
    model = build_model(config).eval()
    with torch.inference_mode():
        predictions = model(torch.rand(1, 3, 640, 640))[0]
    assert predictions.shape == (1, 300, 11)
    assert torch.isfinite(predictions).all()
