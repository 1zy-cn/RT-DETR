"""Prevent a long combination run from silently using a control model."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def entrypoint():
    spec = importlib.util.spec_from_file_location('experiment_train', ROOT / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('config', ['rtdetr-resnet18.yaml', 'rtdetr-resnet18-SCSA.yaml'])
def test_rejects_control_models(entrypoint, config):
    verify = getattr(entrypoint, 'verify_model', None)
    assert callable(verify), 'Training entrypoint must reject a baseline or SCSA-only model'
    model = RTDETRDetectionModel(str(ROOT / 'ultralytics/cfg/models/rt-detr' / config), nc=7, verbose=False)
    with pytest.raises(RuntimeError, match='HSFPN.*SCSA'):
        verify(model)


def test_snapshots_the_actual_training_model(entrypoint, tmp_path):
    callback = getattr(entrypoint, 'on_train_start', None)
    assert callable(callback), 'Training must retain its source/configuration provenance'
    model = RTDETRDetectionModel(
        str(ROOT / 'ultralytics/cfg/models/rt-detr/resnet18-HSFPN+SCSA.yaml'), nc=7, verbose=False)
    trainer = SimpleNamespace(model=model, save_dir=tmp_path, args=SimpleNamespace(seed=0, resume=False))
    callback(trainer)
    snapshot = tmp_path / 'source_snapshot'
    assert (snapshot / 'train.py').read_bytes() == (ROOT / 'train.py').read_bytes()
    assert (snapshot / 'SVRDD.yaml').read_bytes() == (ROOT / 'SVRDD.yaml').read_bytes()
    metadata = json.loads((snapshot / 'provenance.json').read_text(encoding='utf-8'))
    assert metadata['model_yaml']['nc'] == 7
    assert metadata['model_check']['scsa_modules'] == 8
    assert metadata['model_check']['hsfpn_ca_modules'] == 5
    assert metadata['training_args']['seed'] == 0
    assert metadata['initialization'] == 'random_from_yaml; no checkpoint loaded'
    # Simulate a bypassed attention branch in the actual trainer model.
    model.model[4].blocks[0].att = False
    with pytest.raises(RuntimeError, match='HSFPN.*SCSA'):
        entrypoint.verify_model(model)
