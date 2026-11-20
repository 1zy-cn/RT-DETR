"""Trusted full-model checkpoints remain usable with PyTorch 2.6 defaults.

All fixtures are synthetic CPU modules saved inside pytest's temporary directory.
Only the default at the torch.load boundary is changed; deserialization is real.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weather_hfc_training import WeatherHFCModel, WeatherHFCTrainer


class TinyCheckpointModel(WeatherHFCModel):
    """Serializable weather student without a detector graph or GPU allocation."""

    def __init__(self, value):
        torch.nn.Module.__init__(self)
        self.marker = torch.nn.Parameter(torch.tensor([float(value)]))
        self.model = torch.nn.ModuleList([torch.nn.Identity()])
        self.nc = 7
        self.names = {index: str(index) for index in range(7)}
        self.yaml = dict(nc=7, weather_hfc=dict(haar=False, freq=False, consistency=False))
        self.consistency_enabled = False
        self.consistency_weight = 0.1
        self.args = {}


@pytest.fixture
def pytorch_26_load(monkeypatch):
    """Use real weights-only unpickling when a caller omits its explicit mode."""
    real_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault('weights_only', True)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', load)
    return real_load


@pytest.mark.parametrize('retry_missing_dependency', [False, True])
def test_torch_safe_load_restores_full_module_with_new_default(
        tmp_path, monkeypatch, pytorch_26_load, retry_missing_dependency):
    from ultralytics.nn import tasks

    checkpoint_path = tmp_path/'trusted.pt'
    module = torch.nn.Linear(2, 1)
    with torch.no_grad():
        module.weight.fill_(2.5)
        module.bias.fill_(0.5)
    torch.save({'model': module}, checkpoint_path)
    if retry_missing_dependency:
        delegated_load = torch.load
        first_attempt = True

        def missing_once(*args, **kwargs):
            nonlocal first_attempt
            if first_attempt:
                first_attempt = False
                raise ModuleNotFoundError('fixture dependency', name='checkpoint_fixture_dependency')
            return delegated_load(*args, **kwargs)

        monkeypatch.setattr(torch, 'load', missing_once)
        # Keep the legacy retry path offline; its second load still unpickles
        # the same real checkpoint under the simulated 2.6 default.
        monkeypatch.setattr(tasks, 'check_requirements', lambda *args, **kwargs: None)
    checkpoint, filename = tasks.torch_safe_load(str(checkpoint_path))
    assert Path(filename) == checkpoint_path
    assert isinstance(checkpoint['model'], torch.nn.Linear)
    torch.testing.assert_close(checkpoint['model'](torch.ones(1, 2)), torch.tensor([[5.5]]))


def test_strip_optimizer_preserves_ema_model_with_new_default(tmp_path, pytorch_26_load):
    from ultralytics.utils.torch_utils import strip_optimizer

    checkpoint_path = tmp_path/'best.pt'
    student = torch.nn.Linear(1, 1, bias=False)
    ema = deepcopy(student)
    with torch.no_grad():
        student.weight.fill_(1)
        ema.weight.fill_(9)
    torch.save(dict(model=student, ema=ema, optimizer={'state': {}}, epoch=3,
                    train_args=dict(half=False), best_fitness=0.4, updates=17), checkpoint_path)
    strip_optimizer(checkpoint_path)
    checkpoint = pytorch_26_load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['optimizer'] is None and checkpoint['ema'] is None
    assert checkpoint['epoch'] == -1
    assert checkpoint['model'].weight.dtype == torch.float32
    assert not checkpoint['model'].weight.requires_grad
    torch.testing.assert_close(checkpoint['model'](torch.ones(1, 1)), torch.tensor([[9.0]]))


def save_weather_checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = dict(version=1, amp=False, consistency_weight=0.1, scaler={},
                   stopper=dict(patience=20), scheduler={}, rng={}, loaders={})
    torch.save(dict(model=TinyCheckpointModel(1), ema=TinyCheckpointModel(9),
                    optimizer={}, epoch=2, extra_state=dict(weather_hfc_runtime=runtime)), path)


def test_weather_setup_model_restores_raw_student_with_new_default(tmp_path, pytorch_26_load):
    checkpoint_path = tmp_path/'last.pt'
    save_weather_checkpoint(checkpoint_path)
    trainer = WeatherHFCTrainer.__new__(WeatherHFCTrainer)
    trainer.resume = True
    trainer.model = str(checkpoint_path)
    trainer.epochs = 10
    trainer.args = SimpleNamespace(amp=False, patience=20)
    trainer.consistency_weight = 0.1
    trainer.data = dict(nc=7, names={index: str(index) for index in range(7)})
    checkpoint = trainer.setup_model()
    assert isinstance(trainer.model, TinyCheckpointModel)
    torch.testing.assert_close(trainer.model.marker, torch.tensor([1.0]))
    torch.testing.assert_close(checkpoint['ema'].marker, torch.tensor([9.0]))
    assert trainer.teacher is None


def test_resume_entrypoint_reads_full_checkpoint_before_trainer_with_new_default(
        tmp_path, monkeypatch, pytorch_26_load):
    import train_weather_hfc
    import weather_hfc_training
    from weather_hfc_protocol import sha256

    run_dir = tmp_path/'runs'/'baseline_seed0'
    checkpoint_path = run_dir/'weights'/'last.pt'
    save_weather_checkpoint(checkpoint_path)
    record_path = run_dir/'weather_protocol.json'
    protocol = dict(configuration='baseline',
                    model=str(ROOT/'ultralytics/cfg/models/rt-detr/weather-hfc/baseline.yaml'),
                    data=str(tmp_path/'data.yaml'), manifest=str(tmp_path/'manifest.jsonl'),
                    weights=str(checkpoint_path), teacher_checkpoint=None, consistency_weight=0.1,
                    train_args=dict(project=str(run_dir.parent), name=run_dir.name,
                                    epochs=10, amp=False, patience=20))
    record_path.write_text(json.dumps(dict(protocol=protocol, status='failed',
        source_sha256={str(checkpoint_path.resolve()): sha256(checkpoint_path)})), encoding='utf-8')

    class NoTraining(WeatherHFCTrainer):
        # The real CLI validates the loaded raw/EMA checkpoint and writes its
        # protocol. Replace only construction/train to avoid optimizer updates.
        def __init__(self, *, overrides, manifest, teacher_checkpoint, consistency_weight):
            self.save_dir = Path(overrides['project'])/overrides['name']

        def train(self):
            pass

    monkeypatch.setattr(weather_hfc_training, 'WeatherHFCTrainer', NoTraining)
    train_weather_hfc.main(['--resume', str(checkpoint_path), '--train'])
    record = json.loads(record_path.read_text(encoding='utf-8'))
    assert record['status'] == 'completed'
    assert record['resume_history'][0]['checkpoint_sha256'] == sha256(checkpoint_path)
    assert record['protocol']['resume_checkpoint'] == str(checkpoint_path.resolve())


def test_evaluation_entrypoint_restores_ema_with_new_default(
        tmp_path, monkeypatch, pytorch_26_load):
    import evaluate_weather_hfc
    import weather_hfc_training

    checkpoint_path = tmp_path/'best.pt'
    save_weather_checkpoint(checkpoint_path)
    test_list = tmp_path/'clear.txt'
    test_list.write_text(''.join(f'fixture_{index}.jpg\n' for index in range(1000)), encoding='utf-8')
    data_path = tmp_path/'test_clear.yaml'
    data_path.write_text(yaml.safe_dump(dict(path=str(tmp_path), test=str(test_list), nc=7,
        names={index: str(index) for index in range(7)})), encoding='utf-8')
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text('', encoding='utf-8')
    output = tmp_path/'evaluation'/'baseline_clear'

    class NoPixelEvaluation:
        def __init__(self, *, manifest, args):
            self.save_dir = Path(args['project'])/args['name']
            self.save_dir.mkdir(parents=True)

        def __call__(self, *, model):
            assert isinstance(model, TinyCheckpointModel)
            torch.testing.assert_close(model.marker, torch.tensor([9.0]))
            assert not model.training and not model.marker.requires_grad
            return {'fixture_metric': 0.25}

    monkeypatch.setattr(weather_hfc_training, 'WeatherHFCValidator', NoPixelEvaluation)
    evaluate_weather_hfc.main(['--weights', str(checkpoint_path), '--data-dir', str(tmp_path),
        '--manifest', str(manifest), '--group', 'clear', '--device', 'cpu',
        '--project', str(output.parent), '--name', output.name, '--evaluate'])
    record = json.loads((output/'evaluation_protocol.json').read_text(encoding='utf-8'))
    assert record['status'] == 'completed'
    assert record['metrics'] == {'fixture_metric': 0.25}
