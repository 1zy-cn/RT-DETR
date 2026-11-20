"""Server preflight rejects bad inputs before writing local configs or training."""
import hashlib
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def server():
    return importlib.import_module('chapter4_server')


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    module = server()
    root = tmp_path/'weather'
    root.mkdir()
    weights = tmp_path/'original-best.pt'
    weights.write_bytes(b'approved-checkpoint')
    monkeypatch.setattr(module, 'APPROVED_WEIGHT_SHA256', hashlib.sha256(weights.read_bytes()).hexdigest())
    (root/'manifest.jsonl').write_text('fixture')
    (root/'data_weather_train.yaml').write_text('fixture')
    rows = [dict(output_image='train/images/sample__clear.jpg', output_label='train/labels/sample__clear.txt')]
    for field in ('output_image', 'output_label'):
        path = root/rows[0][field]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    monkeypatch.setattr(module.configuration, 'load_manifest', lambda path: (rows, {}, {}))
    monkeypatch.setattr(module.configuration, 'read_names', lambda path: dict(enumerate(module.CLASS_NAMES)))
    calls = []
    def configure(data_root, output, force=False):
        calls.append((data_root, output, force))
        output.mkdir(parents=True, exist_ok=True)
        (output/'dataset.yaml').write_text('generated')
    monkeypatch.setattr(module.configuration, 'configure', configure)
    return module, root, weights, tmp_path/'project/configs/chapter4', calls


def test_server_preflight_checks_then_generates_local_config(inputs):
    module, root, weights, output, calls = inputs
    assert module.prepare_inputs(root, weights, output) == output/'dataset.yaml'
    assert calls == [(root, output, True)]
    assert (root/'manifest.jsonl').read_text() == 'fixture'


def test_wrong_initializer_rejected_before_config_write(inputs):
    module, root, weights, output, calls = inputs
    weights.write_bytes(b'different-or-weather-adapted-weights')
    with pytest.raises(ValueError, match='checkpoint|权重'):
        module.prepare_inputs(root, weights, output)
    assert not calls and not output.exists()


@pytest.mark.parametrize('field', ['output_image', 'output_label'])
def test_missing_dataset_file_rejected_before_config_write(inputs, field):
    module, root, weights, output, calls = inputs
    path = root/('train/images/sample__clear.jpg' if field == 'output_image' else 'train/labels/sample__clear.txt')
    path.unlink()
    with pytest.raises(FileNotFoundError, match='missing|缺失'):
        module.prepare_inputs(root, weights, output)
    assert not calls and not output.exists()


def test_wrong_class_names_rejected(inputs, monkeypatch):
    module, root, weights, output, calls = inputs
    monkeypatch.setattr(module.configuration, 'read_names', lambda path: {i:str(i) for i in range(7)})
    with pytest.raises(ValueError, match='class|类别'):
        module.prepare_inputs(root, weights, output)
    assert not calls


def test_source_root_cannot_be_config_output(inputs):
    module, root, weights, output, calls = inputs
    with pytest.raises(ValueError, match='outside|数据根'):
        module.prepare_inputs(root, weights, root/'configs')
    assert not calls


def test_changed_manifest_is_not_silently_accepted(tmp_path, monkeypatch):
    module = server()
    root = tmp_path/'weather'
    root.mkdir()
    weights = tmp_path/'best.pt'
    weights.write_bytes(b'approved')
    monkeypatch.setattr(module, 'APPROVED_WEIGHT_SHA256', hashlib.sha256(weights.read_bytes()).hexdigest())
    (root/'manifest.jsonl').write_text('{"unexpected":true}')
    (root/'data_weather_train.yaml').write_text('nc: 7\nnames:\n'+''.join(f'  {i}: "{n}"\n' for i,n in enumerate(module.CLASS_NAMES)), encoding='utf-8')
    with pytest.raises(ValueError, match='Manifest|manifest'):
        module.prepare_inputs(root, weights, tmp_path/'configs')
