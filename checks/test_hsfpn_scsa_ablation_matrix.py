"""Keep the two-factor HSFPN ablation models and training protocol aligned."""

import pytest
import torch

from ultralytics.nn.Addmodules.Dysample import Dy_Sample
from ultralytics.nn.tasks import RTDETRDetectionModel


MATRIX = (
    ('hsfpn-scsa', 5, 2, 0, [26, 19, 12]),
    ('hsfpn-scsa-dysample', 5, 0, 2, [26, 19, 12]),
    ('hsfpn-scsa-p2', 6, 2, 0, [36, 19, 12]),
    ('hsfpn-scsa-dysample-p2', 6, 0, 3, [36, 19, 12]),
)


@pytest.mark.parametrize('name,ca,transpose,dysample,decoder_from', MATRIX)
def test_ablation_builds_with_expected_routes(name, ca, transpose, dysample, decoder_from):
    from train import EXPERIMENTS, training_args, verify_model

    torch.set_num_threads(2)
    model = RTDETRDetectionModel(str(EXPERIMENTS[name]['model']), nc=7, verbose=False)
    counts = verify_model(model, name)
    assert (counts['scsa_modules'], counts['hsfpn_ca_modules'],
            counts['transpose_convolutions'], counts['dysample_modules']) == (8, ca, transpose, dysample)
    assert model.model[-1].f == decoder_from
    assert model.model[-1].nl == 3
    assert all(layer.cross_attn.point_counts == (4, 4, 4)
               for layer in model.model[-1].decoder.layers)
    assert all(model.model[i].offset.weight.requires_grad
               for i in (15, 22) if isinstance(model.model[i], Dy_Sample))
    args = training_args(name)
    assert (args['epochs'], args['batch'], args['imgsz'], args['seed'], args['amp']) == (300, 8, 640, 0, False)


def test_ablation_training_args_are_identical_except_run_name():
    from train import training_args

    args = [training_args(name) for name, *_ in MATRIX]
    assert len({arg['name'] for arg in args}) == len(MATRIX)
    assert all({k: v for k, v in arg.items() if k != 'name'} ==
               {k: v for k, v in args[0].items() if k != 'name'} for arg in args)


@pytest.mark.parametrize('name', ('hsfpn-scsa-dysample', 'hsfpn-scsa-p2'))
def test_single_factor_variants_backward(name):
    from train import EXPERIMENTS

    torch.set_num_threads(2)
    model = RTDETRDetectionModel(str(EXPERIMENTS[name]['model']), nc=7, verbose=False)
    model.nc = 7
    model.train()
    batch = {'img': torch.rand(1, 3, 256, 256), 'batch_idx': torch.tensor([0]),
             'cls': torch.tensor([[0.]]), 'bboxes': torch.tensor([[.5, .5, .2, .1]])}
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    if name.endswith('dysample'):
        assert all(model.model[i].offset.weight.grad is not None and
                   model.model[i].offset.weight.grad.abs().sum() > 0 for i in (15, 22))
    else:
        assert model.model[29].weight.grad is not None
        assert model.model[29].weight.grad.abs().sum() > 0
