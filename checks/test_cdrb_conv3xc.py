"""Structural and numerical checks for the chapter-3 CDRB/Conv3XC candidates."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_cdrb_shape_gradient_and_drb_deploy_equivalence():
    from ultralytics.nn.Addmodules.CDRB import BasicBlock_CDRB

    block = BasicBlock_CDRB(64, 128, stride=2, shortcut=False)
    assert block.drb.lk_origin.groups == 128
    assert block.irmb.v.conv.out_channels > 128  # actual inverted expansion
    assert block.irmb.conv_local.conv.groups == block.irmb.v.conv.out_channels
    with torch.no_grad():
        block.train()(torch.randn(4, 64, 32, 32))  # non-default BN statistics
    block.eval()
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    y = block(x)
    assert y.shape == (2, 128, 16, 16)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block.drb.lk_origin.weight.grad.abs().sum() > 0
    before = block(x.detach())
    block.drb.merge_dilated_branches()
    after = block(x.detach())
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('act', [False, True])
def test_conv3xc_explicit_deploy_matches_train_graph(act):
    from ultralytics.nn.Addmodules.Conv3XC import Conv3XC

    module = Conv3XC(8, 8, gain=2, act=act).eval()
    x = torch.randn(2, 8, 17, 19, requires_grad=True)
    before = module(x)
    before.square().mean().backward()
    assert x.grad is not None and module.conv1.weight.grad.abs().sum() > 0
    module.switch_to_deploy()
    after = module(x.detach())
    torch.testing.assert_close(after, before.detach(), rtol=1e-4, atol=1e-5)
    assert not hasattr(module, 'conv1')


def test_conv3xc_replacement_preserves_repconv_activation():
    """Replacing RepConv must not remove every nonlinearity inside RepC3.m."""
    from ultralytics.nn.Addmodules.Conv3XC import Conv3XC, RepC3_Conv3XC

    fusion = RepC3_Conv3XC(512, 256, n=3, e=0.5)
    assert len(fusion.m) == 3
    for unit in fusion.m:
        assert isinstance(unit, Conv3XC)
        assert isinstance(unit.act, torch.nn.SiLU)


@pytest.mark.parametrize('name,expected', [
    ('svrdd-baseline', (0, 0, 0)),
    ('svrdd-scsa', (8, 0, 0)),
    ('svrdd-cdrb', (0, 8, 0)),
    ('svrdd-conv3xc', (0, 0, 3)),
    ('svrdd-scsa-cdrb', (8, 8, 0)),
    ('svrdd-scsa-conv3xc', (8, 0, 3)),
    ('svrdd-cdrb-conv3xc', (0, 8, 3)),
    ('svrdd-scsa-cdrb-conv3xc', (8, 8, 3)),
])
def test_eight_ablation_models_have_exact_modules_and_routes(name, expected):
    from train import EXPERIMENTS, verify_model
    from ultralytics.nn.Addmodules.CDRB import BasicBlock_CDRB
    from ultralytics.nn.Addmodules.Conv3XC import Conv3XC, RepC3_Conv3XC
    from ultralytics.nn.Addmodules.SCSA import SCSA

    config = EXPERIMENTS[name]['model']
    assert config.is_file()
    model = RTDETRDetectionModel(str(config), nc=7, verbose=False)
    modules = list(model.modules())
    counts = (sum(isinstance(m, SCSA) for m in modules),
              sum(isinstance(m, BasicBlock_CDRB) for m in modules),
              sum(isinstance(m, Conv3XC) for m in modules))
    assert counts == expected
    assert isinstance(model.model[19], RepC3_Conv3XC) == ('conv3xc' in name)
    assert model.model[-1].f == [19, 22, 25]
    assert model.model[-1].nc == 7
    assert verify_model(model, name)['scsa_modules'] == expected[0]


def test_three_factor_model_trains_with_real_detection_loss():
    from train import EXPERIMENTS
    from ultralytics.nn.Addmodules.CDRB import BasicBlock_CDRB
    from ultralytics.nn.Addmodules.Conv3XC import Conv3XC

    model = RTDETRDetectionModel(str(EXPERIMENTS['svrdd-scsa-cdrb-conv3xc']['model']), nc=7, verbose=False).cpu()
    model.nc = 7  # The trainer sets this after model construction in this fork.
    model.train()
    batch = {'img': torch.rand(2, 3, 256, 256),
             'batch_idx': torch.tensor([0, 1]), 'cls': torch.tensor([[0.], [3.]]),
             'bboxes': torch.tensor([[.5, .5, .2, .1], [.4, .4, .1, .2]])}
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    block = next(m for m in model.modules() if isinstance(m, BasicBlock_CDRB))
    fusion = next(m for m in model.modules() if isinstance(m, Conv3XC))
    assert block.drb.lk_origin.weight.grad.abs().sum() > 0
    assert fusion.conv1.weight.grad.abs().sum() > 0


def test_whole_model_reparameterization_preserves_fused_features():
    from train import EXPERIMENTS
    from ultralytics.nn.Addmodules.Conv3XC import reparameterize_cdrb_conv3xc

    model = RTDETRDetectionModel(str(EXPERIMENTS['svrdd-scsa-cdrb-conv3xc']['model']),
                                 nc=7, verbose=False).eval()
    image = torch.randn(1, 3, 256, 256)
    features = {}
    hooks = [model.model[index].register_forward_hook(
        lambda _m, _in, out, key=index: features.__setitem__(key, out.detach().clone()))
        for index in (4, 5, 6, 7, 19, 22, 25)]
    with torch.inference_mode():
        before = model(image)[0]
        before_features = {key: value.clone() for key, value in features.items()}
        reparameterize_cdrb_conv3xc(model)
        after = model(image)[0]
    for hook in hooks:
        hook.remove()
    for key, value in before_features.items():
        torch.testing.assert_close(features[key], value, rtol=2e-4, atol=2e-5)
    # Decoder top-k selection can reorder near-tied random queries after tiny
    # floating-point differences; compare the continuous encoder features.
    assert after.shape == before.shape == (1, 300, 11)
    assert torch.isfinite(after).all()


def test_reparameterization_rejects_training_mode():
    from ultralytics.nn.Addmodules.Conv3XC import reparameterize_cdrb_conv3xc

    with pytest.raises(RuntimeError, match='eval'):
        reparameterize_cdrb_conv3xc(torch.nn.Sequential().train())


def test_combined_model_executes_all_factors_at_expected_feature_scales():
    from train import EXPERIMENTS
    from ultralytics.nn.Addmodules.CDRB import BasicBlock_CDRB
    from ultralytics.nn.Addmodules.Conv3XC import Conv3XC
    from ultralytics.nn.Addmodules.SCSA import SCSA

    model = RTDETRDetectionModel(str(EXPERIMENTS['svrdd-scsa-cdrb-conv3xc']['model']),
                                 nc=7, verbose=False).eval()
    seen = {'cdrb': 0, 'scsa': 0, 'conv3xc': 0}
    shapes = {}
    hooks = []
    for name, cls in (('cdrb', BasicBlock_CDRB), ('scsa', SCSA), ('conv3xc', Conv3XC)):
        for module in model.modules():
            if isinstance(module, cls):
                hooks.append(module.register_forward_hook(
                    lambda _m, _in, _out, key=name: seen.__setitem__(key, seen[key] + 1)))
    for index in (4, 5, 6, 7, 19, 22, 25):
        hooks.append(model.model[index].register_forward_hook(
            lambda _m, _in, out, key=index: shapes.__setitem__(key, tuple(out.shape))))
    with torch.inference_mode():
        prediction = model(torch.rand(1, 3, 256, 256))[0]
    for hook in hooks:
        hook.remove()
    assert seen == {'cdrb': 8, 'scsa': 8, 'conv3xc': 3}
    assert shapes == {4: (1, 64, 64, 64), 5: (1, 128, 32, 32),
                      6: (1, 256, 16, 16), 7: (1, 512, 8, 8),
                      19: (1, 256, 32, 32), 22: (1, 256, 16, 16),
                      25: (1, 256, 8, 8)}
    assert prediction.shape == (1, 300, 11)
