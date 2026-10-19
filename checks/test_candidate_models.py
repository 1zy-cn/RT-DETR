"""Checks for the two three-factor RT-DETR candidate families."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules.transformer import MSDeformAttn
from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'ultralytics/cfg/models/rt-detr'


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_scale_aware_uniform_points_match_existing_decoder():
    """A list of equal point counts must preserve the baseline computation."""
    old = MSDeformAttn(d_model=32, n_levels=3, n_heads=4, n_points=4)
    new = MSDeformAttn(d_model=32, n_levels=3, n_heads=4, n_points=[4, 4, 4])
    new.load_state_dict(old.state_dict())
    query = torch.randn(1, 5, 32)
    reference = torch.rand(1, 5, 1, 4)
    value = torch.randn(1, 21, 32)
    shapes = [(4, 4), (2, 2), (1, 1)]
    torch.testing.assert_close(new(query, reference, value, shapes), old(query, reference, value, shapes),
                               rtol=1e-5, atol=1e-6)


def test_p3_biased_decoder_points_backpropagate():
    """Unequal points must run all three feature levels and learn offsets."""
    attention = MSDeformAttn(d_model=32, n_levels=3, n_heads=4, n_points=[6, 4, 2])
    query = torch.randn(1, 5, 32, requires_grad=True)
    value = torch.randn(1, 21, 32, requires_grad=True)
    reference = torch.rand(1, 5, 1, 4)
    result = attention(query, reference, value, [(4, 4), (2, 2), (1, 1)])
    assert result.shape == (1, 5, 32) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert attention.sampling_offsets.weight.grad.abs().sum() > 0
    assert value.grad.abs().sum() > 0


@pytest.mark.parametrize('name,factors', [
    ('baseline', (0, 0, 0, 0, 0)),
    ('scsa', (8, 0, 0, 0, 0)),
    ('freqfusion', (0, 1, 0, 0, 0)),
    ('scale-decoder', (0, 0, 1, 0, 0)),
    ('scsa-freqfusion', (8, 1, 0, 0, 0)),
    ('scsa-scale-decoder', (8, 0, 1, 0, 0)),
    ('freqfusion-scale-decoder', (0, 1, 1, 0, 0)),
    ('scsa-freqfusion-scale-decoder', (8, 1, 1, 0, 0)),
    ('dysample', (0, 0, 0, 2, 0)),
    ('fdconv-c4', (0, 0, 0, 0, 1)),
    ('scsa-dysample', (8, 0, 0, 2, 0)),
    ('scsa-fdconv-c4', (8, 0, 0, 0, 1)),
    ('dysample-fdconv-c4', (0, 0, 0, 2, 1)),
    ('scsa-dysample-fdconv-c4', (8, 0, 0, 2, 1)),
])
def test_candidate_matrix_builds_exact_factors(name, factors):
    """A mislabeled YAML must not silently train a different ablation."""
    from train import EXPERIMENTS, verify_model
    from ultralytics.nn.Addmodules.Dysample import Dy_Sample
    from ultralytics.nn.Addmodules.FDConv import FDConv
    from ultralytics.nn.Addmodules.FreqFusion import FreqFusionConcat
    from ultralytics.nn.Addmodules.SCSA import SCSA

    config = EXPERIMENTS[name]['model']
    assert config.is_file()
    model = RTDETRDetectionModel(str(config), nc=7, verbose=False)
    modules = list(model.modules())
    point_setting = model.model[-1].decoder.layers[0].cross_attn.n_points
    decoder_points = (point_setting,) * 3 if isinstance(point_setting, int) else tuple(point_setting)
    actual = (sum(isinstance(m, SCSA) for m in modules),
              sum(isinstance(m, FreqFusionConcat) for m in modules),
              int(decoder_points == (6, 4, 2)),
              sum(isinstance(m, Dy_Sample) for m in modules),
              sum(isinstance(m, FDConv) for m in modules))
    assert actual == factors, name
    assert verify_model(model, name)['scsa_modules'] == factors[0]


@pytest.mark.parametrize('name', ['scsa-freqfusion-scale-decoder', 'scsa-dysample-fdconv-c4'])
def test_full_combinations_train_with_real_detection_loss(name):
    """Both three-factor models must compute finite loss and update new weights."""
    from train import EXPERIMENTS
    from ultralytics.nn.Addmodules.FDConv import FDConv

    model = RTDETRDetectionModel(str(EXPERIMENTS[name]['model']), nc=7, verbose=False).cpu()
    model.nc = 7
    model.train()
    image = torch.rand(2, 3, 256, 256)
    batch = {'img': image, 'batch_idx': torch.tensor([0, 1]), 'cls': torch.tensor([[0.], [3.]]),
             'bboxes': torch.tensor([[.5, .5, .2, .1], [.4, .4, .1, .2]])}
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    if 'fdconv' in name:
        fd = next(m for m in model.modules() if isinstance(m, FDConv))
        assert fd.dft_weight.grad.abs().sum() > 0
    else:
        cross = model.model[-1].decoder.layers[0].cross_attn
        assert cross.sampling_offsets.weight.grad.abs().sum() > 0


def test_two_epoch_local_smoke_overrides_preserve_full_run_defaults(tmp_path):
    """Smoke arguments must not change the 300-epoch comparable protocol."""
    from train import training_args

    full = training_args('scsa-freqfusion-scale-decoder')
    smoke = training_args('scsa-freqfusion-scale-decoder', project=tmp_path,
                          overrides={'epochs': 2, 'batch': 1, 'imgsz': 320, 'workers': 0,
                                     'warmup_epochs': 0, 'plots': False})
    assert (full['epochs'], full['batch'], full['imgsz']) == (300, 8, 640)
    assert (smoke['epochs'], smoke['batch'], smoke['imgsz'], smoke['workers'], smoke['warmup_epochs']) == (2, 1, 320, 0, 0)
    assert smoke['project'] == str(tmp_path)
    assert smoke['plots'] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for half-precision validator path')
def test_fdconv_combination_supports_half_precision_validation():
    """Validation must not feed float output into a half BatchNorm layer."""
    from train import EXPERIMENTS

    model = RTDETRDetectionModel(str(EXPERIMENTS['scsa-dysample-fdconv-c4']['model']),
                                 nc=7, verbose=False).cuda().half().eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 256, 256, device='cuda', dtype=torch.float16))[0]
    assert output.shape == (1, 300, 11)
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
