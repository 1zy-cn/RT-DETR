"""H/F factor contracts: shared sampling, real frequency bands and checkpoint-safe graph edits."""
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from ultralytics.nn.tasks import RTDETRDetectionModel

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'ultralytics/cfg/models/rt-detr'
FACTORS = ('baseline', 'h', 'f', 'hf', 'c', 'hc', 'fc', 'hfc')


@pytest.fixture(autouse=True)
def cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(23)
    yield
    torch.set_num_threads(previous)


def implementation():
    name = 'ultralytics.nn.Addmodules.WeatherHFC'
    assert importlib.util.find_spec(name), 'Weather H/F structural implementation is missing'
    return importlib.import_module(name)


def config(name):
    path = MODELS / 'weather-hfc' / f'{name}.yaml'
    assert path.is_file(), f'Weather factor configuration is missing: {name}'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def test_haar_known_subbands_and_odd_zero_padding():
    haar = implementation().HaarDownsample()
    # Top-left 2x2 cell [[1,2],[4,5]]; remaining cells use right/bottom zero padding.
    x = torch.arange(1., 10.).reshape(1, 1, 3, 3).requires_grad_()
    actual = haar(x)
    expected = torch.tensor([[[[6., 4.5], [7.5, 4.5]],
                              [[-3., -1.5], [7.5, 4.5]],
                              [[-1., 4.5], [-.5, 4.5]],
                              [[0., -1.5], [-.5, 4.5]]]])
    torch.testing.assert_close(actual, expected)
    # An orthonormal, four-band transform keeps padded-input energy before projection.
    torch.testing.assert_close(actual.square().sum(), x.square().sum())
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())


@pytest.mark.parametrize('size', [(160, 160), (159, 161)])
def test_haar_residual_branches_share_one_transform_and_preserve_reused_keys(size):
    stage = implementation().HaarResNetStage(64, 128, 2, 'BasicBlock', 3, False).eval()
    block = stage.blocks[0]
    seen = {'haar': 0}
    inputs = {}
    handles = [block.haar.register_forward_hook(lambda m, i, o: seen.__setitem__('haar', seen['haar'] + 1))]
    handles += [getattr(block, name).register_forward_pre_hook(
        lambda m, i, key=name: inputs.__setitem__(key, i[0])) for name in ('branch2a', 'short')]
    with torch.no_grad():
        output = stage(torch.randn(1, 64, *size))
    for handle in handles:
        handle.remove()
    assert seen['haar'] == 1
    assert inputs['branch2a'].data_ptr() == inputs['short'].data_ptr()
    assert output.shape == (1, 128, (size[0] + 1) // 2, (size[1] + 1) // 2)
    assert block.branch2a.conv.kernel_size == block.short.conv.kernel_size == (1, 1)
    assert block.branch2a.act == 'relu' and block.short.act is None
    assert block.branch2b.conv.kernel_size == (3, 3)
    keys = stage.state_dict()
    assert 'blocks.0.branch2b.conv.weight' in keys
    assert 'blocks.1.branch2a.conv.weight' in keys


@pytest.mark.parametrize('shape', [(2, 8, 40, 40), (1, 8, 39, 41), (1, 8, 1, 3)])
def test_freq_bands_sum_to_input_identity_init_and_finite_backward(shape):
    module = implementation().WeatherFrequencySelection(shape[1])
    x = torch.randn(*shape, requires_grad=True)
    bands = module.decompose(x)
    assert len(bands) == 4 and all(b.shape == x.shape for b in bands)
    torch.testing.assert_close(sum(bands), x, atol=5e-7, rtol=5e-6)
    y = module(x)
    torch.testing.assert_close(y, x, atol=5e-7, rtol=5e-6)
    (y * torch.randn_like(y)).sum().backward()
    assert torch.isfinite(x.grad).all()
    for gate in module.freq_weight_conv_list:
        assert torch.isfinite(gate.weight.grad).all()
    if min(shape[-2:]) > 3:
        assert all(gate.weight.grad.abs().sum() > 0 for gate in module.freq_weight_conv_list)


@pytest.mark.parametrize('height,width', [(40, 40), (39, 41), (1, 3)])
def test_freq_masks_are_nested_rectangles_with_real_ifft(height, width):
    module = implementation().WeatherFrequencySelection(1)
    masks = module.lowpass_masks(height, width, torch.device('cpu'))
    opposite_h = (-torch.arange(height)) % height
    opposite_w = (-torch.arange(width)) % width
    spectrum = torch.fft.fft2(torch.randn(1, 1, height, width), norm='ortho')
    for index, mask in enumerate(masks):
        assert mask.shape == (height, width)
        assert mask[0, 0]
        assert torch.equal(mask, mask[opposite_h][:, opposite_w])
        if index:
            assert not (mask & ~masks[index-1]).any()
        restored = torch.fft.ifft2(spectrum * mask, norm='ortho')
        assert restored.imag.abs().max() < 1e-6
    if height == width == 40:
        assert masks[0][10, 10] and not masks[0][11, 0]
        assert masks[1][5, 5] and not masks[1][6, 0]
        assert masks[2][2, 2] and not masks[2][3, 0]


def test_freq_selection_changes_only_the_selected_band_with_sigmoid_times_two():
    module = implementation().WeatherFrequencySelection(1)
    coordinate = torch.arange(40).float()
    waves = [torch.cos(2 * torch.pi * k * coordinate / 40).view(1, 1, 1, 40).expand(1, 1, 40, 40)
             for k in (13, 7, 3)]
    dc = torch.ones(1, 1, 40, 40)
    x = dc + sum(waves)
    for actual, expected in zip(module.decompose(x), [*waves, dc]):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=1e-5)
    with torch.no_grad():
        module.freq_weight_conv_list[1].bias.fill_(torch.log(torch.tensor(1 / 3)))  # 2*sigmoid=0.5
    torch.testing.assert_close(module(x), dc + waves[0] + .5 * waves[1] + waves[2], atol=5e-6, rtol=1e-5)


def test_freq_fp32_fft_under_cpu_autocast_and_half_input():
    module = implementation().WeatherFrequencySelection(8)
    x = torch.randn(1, 8, 39, 41, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = module(x)
        loss = y.square().mean()
    loss.backward()
    assert y.dtype == x.dtype and torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    half_module = implementation().WeatherFrequencySelection(8).half()
    half_x = torch.randn(1, 8, 39, 41).half().requires_grad_()
    half_y = half_module(half_x)
    assert half_y.dtype == torch.float16
    half_y.float().square().mean().backward()
    assert torch.isfinite(half_x.grad).all()


@pytest.mark.parametrize('name', FACTORS)
def test_factor_graph_has_only_declared_edits_and_safe_model_indices(name):
    module = implementation()
    expected = yaml.safe_load((MODELS / 'rtdetr-resnet18.yaml').read_text(encoding='utf-8'))
    expected['nc'] = 7
    flags = {'haar': 'h' in name, 'freq': 'f' in name, 'consistency': 'c' in name}
    if name == 'baseline':
        flags = dict.fromkeys(flags, False)
    if flags['haar']:
        expected['backbone'][5][2] = 'HaarResNetStage'
    if flags['freq']:
        expected['head'][4][2] = 'WeatherFreqSelectConv'
    actual = deepcopy(config(name))
    assert actual.pop('weather_hfc') == flags
    assert actual == expected
    model = RTDETRDetectionModel(config(name), nc=7, verbose=False)
    assert len(model.model) == 27 and model.model[-1].f == [19, 22, 25]
    assert model.model[12].f == 6 and model.model[17].f == 5
    assert isinstance(model.model[5], module.HaarResNetStage) == flags['haar']
    assert isinstance(model.model[12], module.WeatherFreqSelectConv) == flags['freq']
    assert model.model[-1].nc == 7


def test_baseline_exact_weights_and_freq_checkpoint_transfer_keep_prediction():
    module = implementation()
    baseline = RTDETRDetectionModel(config('baseline'), nc=7, verbose=False).eval()
    reference = yaml.safe_load((MODELS / 'rtdetr-resnet18.yaml').read_text(encoding='utf-8'))
    reference['nc'] = 7
    original = RTDETRDetectionModel(reference, nc=7, verbose=False).eval()
    assert baseline.state_dict().keys() == original.state_dict().keys()
    original.load_state_dict(baseline.state_dict())
    freq = RTDETRDetectionModel(config('f'), nc=7, verbose=False).eval()
    missing, unexpected = freq.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected and missing
    assert all(key.startswith('model.12.freq_select.') for key in missing)
    assert isinstance(freq.model[12], module.WeatherFreqSelectConv)
    image = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        prediction = baseline(image)[0]
        torch.testing.assert_close(original(image)[0], prediction, atol=0, rtol=0)
        torch.testing.assert_close(freq(image)[0], prediction, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize('name', ['h', 'hf'])
def test_haar_checkpoint_transfer_preserves_remaining_blocks(name):
    baseline = RTDETRDetectionModel(config('baseline'), nc=7, verbose=False)
    altered = RTDETRDetectionModel(config(name), nc=7, verbose=False)
    # Native .load intersects shape-compatible keys. Changed projections are newly initialized.
    altered.load(baseline, verbose=False)
    original = baseline.state_dict()
    updated = altered.state_dict()
    for key, value in original.items():
        if key in updated and value.shape == updated[key].shape:
            torch.testing.assert_close(updated[key], value, atol=0, rtol=0)
    assert updated['model.5.blocks.0.branch2a.conv.weight'].shape == (128, 256, 1, 1)


@pytest.mark.parametrize('name', ['baseline', 'h', 'f', 'hf'])
def test_structural_factors_full_resolution_forward_and_real_detection_backward(name):
    module = implementation()
    model = RTDETRDetectionModel(config(name), nc=7, verbose=False)
    model.nc = 7
    model.train()
    batch = dict(img=torch.rand(1, 3, 256, 256), batch_idx=torch.tensor([0, 0]),
                 cls=torch.tensor([[0.], [3.]]),
                 bboxes=torch.tensor([[.3, .4, .15, .2], [.7, .6, .2, .1]]))
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    if 'h' in name and name != 'baseline':
        block = model.model[5].blocks[0]
        for branch in (block.branch2a, block.short):
            assert torch.isfinite(branch.conv.weight.grad).all() and branch.conv.weight.grad.abs().sum() > 0
    if 'f' in name:
        for gate in model.model[12].freq_select.freq_weight_conv_list:
            assert torch.isfinite(gate.weight.grad).all() and gate.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.eval()
    shapes = {}
    handles = [model.model[index].register_forward_hook(
        lambda m, i, o, key=index: shapes.__setitem__(key, tuple(o.shape))) for index in (5, 12, 19)]
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 640, 640))[0]
    for handle in handles:
        handle.remove()
    assert output.shape == (1, 300, 11) and torch.isfinite(output).all()
    assert shapes == {5: (1, 128, 80, 80), 12: (1, 256, 40, 40), 19: (1, 256, 80, 80)}
    assert len([m for m in model.modules() if isinstance(m, module.WeatherFrequencySelection)]) == ('f' in name)


def test_freq_projection_retains_gate_after_deployment_fusion():
    model = RTDETRDetectionModel(config('f'), nc=7, verbose=False).eval()
    with torch.no_grad():
        model.model[12].freq_select.freq_weight_conv_list[0].weight.normal_(std=.02)
    image = torch.rand(1, 3, 256, 256)
    with torch.inference_mode():
        before = model(image)[0]
        model.fuse(verbose=False)
        after = model(image)[0]
    torch.testing.assert_close(after, before, atol=3e-5, rtol=3e-4)
