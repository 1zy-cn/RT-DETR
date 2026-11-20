"""Chapter-four entrypoint contracts without a training loop or dataset pixels."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def runner():
    spec = importlib.util.spec_from_file_location('weather_project_runner', ROOT/'train_weather_hfc.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_runner_uses_reviewed_project_dataset():
    args = runner().make_parser().parse_args([])
    assert args.data == ROOT/'configs/chapter4/dataset.yaml'


def test_protocol_early_stopping_and_ap_match_chapter(tmp_path):
    from test_weather_runner import arguments
    m = runner()
    args, configs = arguments(m, tmp_path)
    p = m.build_protocol(args, config_dir=configs)
    assert p['train_args']['patience'] == 20
    assert p['train_args']['conf'] == 0.001
    assert p['train_args']['max_det'] == 300


def test_resume_refuses_changed_list_before_creating_output(tmp_path):
    from weather_hfc_protocol import sha256, validate_resume_record
    items = tmp_path/'valid.txt'
    items.write_text('original.jpg\n', encoding='utf-8')
    record = tmp_path/'weather_protocol.json'
    record.write_text(json.dumps({'protocol': {}, 'source_sha256': {str(items): sha256(items)}}))
    items.write_text('different.jpg\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        validate_resume_record(record)


def test_snapshot_includes_data_lists(tmp_path):
    from weather_hfc_protocol import freeze_inputs
    for name in ['manifest.jsonl', 'weights.pt', 'model.yaml', 'train.txt', 'valid.txt', 'test.txt', 'weather_amp.py']:
        (tmp_path/name).write_text('fixture\n', encoding='utf-8')
    data = tmp_path/'data.yaml'
    data.write_text(yaml.safe_dump({'path': str(tmp_path), 'train': 'train.txt', 'val': 'valid.txt', 'test': 'test.txt'}))
    p = {k: str(tmp_path/v) for k,v in [('manifest','manifest.jsonl'),('weights','weights.pt'),('model','model.yaml')]}
    p.update(data=str(data), teacher_checkpoint=None)
    snapshot = freeze_inputs(p, code_root=tmp_path)
    assert all(str(tmp_path/name) in snapshot for name in ['train.txt','valid.txt','test.txt'])
    assert str(tmp_path/'weather_amp.py') in snapshot, 'resume must freeze its local runtime AMP checker'


def test_evaluation_preview_targets_real_test_split_and_group(tmp_path):
    from evaluate_weather_hfc import make_parser, build_protocol
    weights = tmp_path/'best.pt'
    weights.write_text('fixture')
    args = make_parser().parse_args(['--weights',str(weights),'--group','fog','--project',str(tmp_path/'runs')])
    p = build_protocol(args)
    assert p['eval_args']['split'] == 'test'
    assert p['eval_args']['conf'] == 0.001
    assert p['eval_args']['max_det'] == 300
    assert Path(p['data']).name == 'test_fog.yaml'
    assert p['images'] == 2000
    assert not (tmp_path/'runs').exists()


@pytest.mark.parametrize('group', ['primary', 'clear', 'severe', 'fog', 'rain', 'lowlight'])
def test_evaluation_resolved_defaults_satisfy_paired_geometry_contract(tmp_path, group):
    from evaluate_weather_hfc import make_parser, build_protocol
    from ultralytics.cfg import get_cfg
    from weather_hfc_training import validate_fixed_geometry
    weights = tmp_path/'best.pt'
    weights.write_text('fixture')
    args = make_parser().parse_args(['--weights', str(weights), '--group', group,
                                    '--project', str(tmp_path/'runs')])
    protocol = build_protocol(args)
    # The validator merges library defaults before constructing the paired dataset.
    # Previewing split/conf alone does not exercise this boundary.
    resolved = get_cfg(overrides=protocol['eval_args'])
    validate_fixed_geometry(resolved)
    assert resolved.augment is False
    assert not (tmp_path/'runs').exists()


def test_resume_preflight_rejects_stopped_checkpoint_before_trainer_init():
    m = runner()
    runtime = dict(version=1, amp=False, consistency_weight=.1, scaler={}, stopper={},
                   scheduler={}, rng={}, loaders={}, stop=True)
    checkpoint = dict(epoch=2, optimizer={}, extra_state={'weather_hfc_runtime':runtime})
    protocol = dict(consistency_weight=.1, train_args=dict(epochs=300,amp=False,patience=20))
    with pytest.raises(ValueError, match='early stopping'):
        m.validate_resume_before_init(checkpoint, protocol)


def test_resume_flags_cannot_silently_change_experiment(capsys):
    m = runner()
    with pytest.raises(SystemExit) as error:
        m.main(['--resume','last.pt','--seed','9'])
    assert error.value.code == 2
    assert 'remove fresh settings' in capsys.readouterr().err
