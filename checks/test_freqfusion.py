"""CPU correctness and actual detector gradient checks for P3 FreqFusion."""
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.Addmodules.SCSA import SCSA

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [('rtdetr-resnet18-FreqFusion-P3.yaml', 0), ('rtdetr-resnet18-SCSA-FreqFusion-P3.yaml', 8)]


@pytest.fixture(autouse=True)
def cpu_seed():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(previous)


def implementation():
    name = 'ultralytics.nn.Addmodules.FreqFusion'
    assert importlib.util.find_spec(name), 'P3 FreqFusion implementation is missing'
    return importlib.import_module(name)


@pytest.mark.parametrize('up', [1, 2])
def test_reassembly_matches_explicit_reflection_reference_and_gradients(up):
    carafe = implementation().carafe
    x = torch.randn(1, 3, 5, 6, dtype=torch.double, requires_grad=True)
    mask = torch.randn(1, 9, 5*up, 6*up, dtype=torch.double, requires_grad=True).softmax(1)
    padded = F.pad(x, [1]*4, mode='reflect')
    expected = sum(F.interpolate(padded[:, :, i:i+5, j:j+6], scale_factor=up, mode='nearest')
                   * mask[:, i*3+j:i*3+j+1] for i in range(3) for j in range(3))
    actual = carafe(x, mask, 3, 1, up)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_grads = torch.autograd.grad(actual.square().sum(), (x, mask), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), (x, mask))
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('config, scsa_count', CONFIGS)
def test_p3_detector_real_loss_and_full_resolution(config, scsa_count, record_property):
    module = implementation()
    path = ROOT / 'ultralytics/cfg/models/rt-detr' / config
    assert path.exists(), 'FreqFusion experiment YAML missing'
    model = RTDETRDetectionModel(str(path), nc=7, verbose=False).cpu()
    model.nc = 7
    fusion = [m for m in model.modules() if isinstance(m, module.FreqFusionConcat)]
    assert len(fusion) == 1
    assert len([m for m in model.modules() if isinstance(m, SCSA)]) == scsa_count
    assert model.model[17].f == [16, 15]
    assert model.model[-1].f == [18, 21, 24]  # P3/P4/P5 after retained bottom-up path
    controls = yaml.safe_load((path.parent / ('rtdetr-resnet18-SCSA.yaml' if scsa_count else 'rtdetr-resnet18.yaml')).read_text(encoding='utf-8'))
    assert model.yaml['backbone'] == controls['backbone']
    calls = []
    handle = fusion[0].register_forward_hook(lambda _m, _i, _o: calls.append(1))
    image = torch.rand(1, 3, 256, 256)
    batch = {'img': image, 'batch_idx': torch.tensor([0, 0]), 'cls': torch.tensor([[0.], [3.]]),
             'bboxes': torch.tensor([[.35, .4, .2, .15], [.7, .65, .15, .2]])}
    model.train()
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert len(calls) == 1
    for name, param in fusion[0].named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    for branch in ['content_encoder', 'content_encoder2', 'dysampler']:
        params = list(getattr(fusion[0].fusion, branch).parameters())
        assert sum(p.grad.abs().sum().item() for p in params) > 0, branch
    for attention in [m for m in model.modules() if isinstance(m, SCSA)]:
        assert attention.local_dwc.weight.grad.abs().sum() > 0
        assert attention.q.weight.grad.abs().sum() > 0
    before = fusion[0].fusion.content_encoder.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=.001).step()
    assert not torch.equal(before, fusion[0].fusion.content_encoder.weight)
    handle.remove()
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 640, 640))[0]
    assert output.shape == (1, 300, 11) and torch.isfinite(output).all()
    record_property('parameters', sum(p.numel() for p in model.parameters()))
    record_property('scsa_modules', scsa_count)
    record_property('freqfusion_modules', 1)
    record_property('synthetic_detection_loss', loss.item())
