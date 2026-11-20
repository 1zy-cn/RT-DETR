"""Offline Weather AMP startup checks with real tiny CPU/CUDA forwards."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.engine import trainer as trainer_module
from ultralytics.engine.trainer import BaseTrainer
from weather_hfc_training import WeatherHFCTrainer


class TinyPredictionModel(torch.nn.Module):
    """Real convolution/BN prediction path with RT-DETR's (prediction, aux) form."""

    def __init__(self, mode='finite'):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=32, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.mode = mode
        self.calls = []

    def forward(self, images):
        mixed_precision = torch.is_autocast_enabled()
        self.calls.append(dict(shape=tuple(images.shape), device=images.device,
                               grad=torch.is_grad_enabled(), autocast=mixed_precision,
                               training=[module.training for module in self.modules()]))
        # Verify restoration even when model inference consumes CPU/CUDA RNG.
        torch.rand(())
        torch.rand((), device=images.device)
        if self.mode == 'runtime_error':
            raise RuntimeError('synthetic CUDA inference failure')
        prediction = self.bn(self.conv(images)).mean(dim=(2, 3)).unsqueeze(1)
        if self.mode == 'nonfinite_fp32' and not mixed_precision:
            prediction = prediction * float('nan')
        if self.mode == 'nonfinite_amp' and mixed_precision:
            prediction = prediction * float('nan')
        if self.mode == 'shape_change' and mixed_precision:
            prediction = prediction[..., :-1]
        if self.mode == 'invalid_prediction':
            return None, prediction
        # Aux tensors are irrelevant to inference finite/shape checks.
        return prediction, prediction.new_full((1,), float('nan'))


def forbid_default_checker(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Weather AMP must not invoke YOLO downloads or asset checks')

    monkeypatch.setattr(trainer_module, 'check_amp', forbidden)


def snapshot(model):
    return dict(training=[module.training for module in model.modules()],
                state=deepcopy(model.state_dict()), cpu_rng=torch.get_rng_state().clone(),
                cuda_rng=[state.clone() for state in torch.cuda.get_rng_state_all()])


def assert_unchanged(model, before):
    assert [module.training for module in model.modules()] == before['training']
    assert torch.equal(torch.get_rng_state(), before['cpu_rng'])
    assert all(torch.equal(left, right) for left, right in
               zip(torch.cuda.get_rng_state_all(), before['cuda_rng']))
    for name, value in model.state_dict().items():
        assert torch.equal(value, before['state'][name]), name


@pytest.mark.parametrize('supported', [False, True])
def test_base_amp_check_keeps_legacy_checker(monkeypatch, supported):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = TinyPredictionModel()

    def legacy_checker(model):
        assert model is trainer.model
        return supported

    monkeypatch.setattr(trainer_module, 'check_amp', legacy_checker)
    assert trainer.check_amp_support() is supported


def test_cpu_weather_amp_is_disabled_without_forward_or_default_checker(monkeypatch):
    forbid_default_checker(monkeypatch)
    trainer = WeatherHFCTrainer.__new__(WeatherHFCTrainer)
    trainer.model = TinyPredictionModel().cpu().train()
    assert trainer.check_amp_support() is False
    assert trainer.model.calls == []


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires local CUDA')


@CUDA
def test_weather_amp_runs_local_cuda_prediction_and_preserves_state(monkeypatch):
    forbid_default_checker(monkeypatch)
    trainer = WeatherHFCTrainer.__new__(WeatherHFCTrainer)
    trainer.model = TinyPredictionModel().cuda().train()
    trainer.model.bn.eval()  # Preserve intentionally mixed child training flags.
    before = snapshot(trainer.model)
    assert trainer.check_amp_support() is True
    assert_unchanged(trainer.model, before)
    assert len(trainer.model.calls) == 2
    assert [call['autocast'] for call in trainer.model.calls] == [False, True]
    assert all(call['shape'] == (1, 3, 640, 640) and call['device'].type == 'cuda'
               and not call['grad'] and not any(call['training']) for call in trainer.model.calls)


@CUDA
@pytest.mark.parametrize('mode', ['nonfinite_fp32', 'nonfinite_amp', 'shape_change',
                                 'invalid_prediction', 'runtime_error'])
def test_weather_amp_disables_invalid_local_predictions_and_restores_state(monkeypatch, mode):
    forbid_default_checker(monkeypatch)
    from ultralytics.utils import LOGGER

    warnings = []
    monkeypatch.setattr(LOGGER, 'warning', warnings.append)
    trainer = WeatherHFCTrainer.__new__(WeatherHFCTrainer)
    trainer.model = TinyPredictionModel(mode).cuda().train()
    trainer.model.conv.eval()
    before = snapshot(trainer.model)
    assert trainer.check_amp_support() is False
    assert_unchanged(trainer.model, before)
    assert warnings


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=CUDA)])
def test_setup_train_uses_weather_amp_override_before_dataset_work(monkeypatch, device):
    forbid_default_checker(monkeypatch)
    monkeypatch.setattr(trainer_module, 'RANK', -1)
    trainer = WeatherHFCTrainer.__new__(WeatherHFCTrainer)
    trainer.device = torch.device(device)
    trainer.model = TinyPredictionModel().to(trainer.device)
    trainer.args = SimpleNamespace(freeze=0, amp=True, imgsz=640)
    trainer.batch_size = 2
    trainer.trainset = 'unread-training-data'
    trainer.run_callbacks = lambda event: None
    trainer.setup_model = lambda: None
    trainer.set_model_attributes = lambda: None

    class ReachedDatasetBoundary(Exception):
        pass

    def stop_before_data(*args, **kwargs):
        raise ReachedDatasetBoundary

    trainer.get_dataloader = stop_before_data
    with pytest.raises(ReachedDatasetBoundary):
        trainer._setup_train(world_size=1)
    assert trainer.amp is (trainer.device.type == 'cuda')
    assert trainer.scaler.is_enabled() is trainer.amp
    assert len(trainer.model.calls) == (2 if trainer.amp else 0)
