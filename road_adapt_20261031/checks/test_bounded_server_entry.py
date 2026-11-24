"""Server launcher must reproduce the qualified baseline and bound its cost."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
import yaml

ENTRY = Path(os.environ.get('ENTRYPOINT_UNDER_TEST', Path(__file__).resolve().parents[1] / 'train.py'))
sys.path.insert(0, str(ENTRY.parent))


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('bounded_server_launcher', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = {i: str(i) for i in range(7)}
    for split in ('train', 'val', 'test'):
        (tmp_path / f'{split}.txt').write_text('', encoding='utf-8')
    data = tmp_path / 'dataset.yaml'
    data.write_text(yaml.safe_dump(dict(nc=7, names=names, path=str(tmp_path),
        **{key: str(tmp_path / f'{key}.txt') for key in ('train', 'val', 'test')})), encoding='utf-8')
    weights = tmp_path / 'original.pt'
    weights.write_bytes(b'preview-must-not-load-this')
    (tmp_path / 'manifest.jsonl').write_text('{}\n', encoding='utf-8')
    prepared = []
    server = ModuleType('chapter4_server')
    server.prepare_inputs = lambda *args: prepared.append(args) or data
    monkeypatch.setitem(sys.modules, 'chapter4_server', server)
    module.DATA_ROOT = tmp_path
    module.ORIGINAL_WEIGHTS = weights
    module.PROJECT = tmp_path / 'runs'
    return module, prepared


@pytest.fixture
def child_calls(monkeypatch):
    calls = []
    for name, profile in (('train_weather_hfc', 'legacy_hfc'),
                          ('train_weather_finetune', 'baseline_finetune_v2')):
        child = ModuleType(name)
        def main(argv, profile=profile):
            calls.append((profile, argv))
            return profile, argv
        child.main = main
        monkeypatch.setitem(sys.modules, name, child)
    return calls


def option(argv, name):
    return argv[argv.index(name) + 1]


def test_default_baseline_uses_qualified_profile_and_one_epoch_budget(launcher, child_calls):
    module, prepared = launcher
    profile, argv = module.main(['--check'])
    assert profile == 'baseline_finetune_v2', 'Default baseline must not use legacy LR1e-4/warmup3'
    assert option(argv, '--budget-epochs') == '1'
    assert float(option(argv, '--lr0')) == 1e-5
    assert float(option(argv, '--lrf')) == 1.
    assert option(argv, '--epochs') == '30'
    assert '--configuration' not in argv and '--train' not in argv
    assert len(prepared) == 1


def test_training_forwards_the_same_explicit_bounded_settings(launcher, child_calls):
    module, _ = launcher
    module.LR0 = 1e-6
    module.LRF = .2
    module.COS_LR = True
    profile, argv = module.main([])
    assert profile == 'baseline_finetune_v2' and '--train' in argv
    assert float(option(argv, '--lr0')) == 1e-6 and float(option(argv, '--lrf')) == .2
    assert '--cos-lr' in argv and option(argv, '--budget-epochs') == '1'


def test_nonbaseline_cannot_silently_mix_qualified_and_legacy_profiles(launcher, child_calls):
    module, prepared = launcher
    module.CONFIGURATION = 'hfc'
    with pytest.raises(ValueError, match='baseline|统一'):
        module.main(['--check'])
    assert not prepared and not child_calls


def test_legacy_profile_remains_explicitly_available(launcher, child_calls):
    module, _ = launcher
    module.TRAINING_PROFILE = 'legacy_hfc'
    module.CONFIGURATION = 'hfc'
    profile, argv = module.main(['--check'])
    assert profile == 'legacy_hfc' and option(argv, '--configuration') == 'hfc'
    assert option(argv, '--epochs') == '300', 'Selecting legacy must retain its original schedule horizon'
    assert '--budget-epochs' not in argv and '--lr0' not in argv


def test_unknown_profile_fails_before_any_dataset_rewrite(launcher, child_calls):
    module, prepared = launcher
    module.TRAINING_PROFILE = 'typo'
    with pytest.raises(ValueError, match='profile|PROFILE'):
        module.main(['--check'])
    assert not prepared and not child_calls


@pytest.mark.parametrize('recorded,expected', [('baseline_finetune_v2', 'baseline_finetune_v2'), (None, 'legacy_hfc')])
def test_resume_routes_from_frozen_record_and_ignores_top_level_profile(launcher, child_calls, recorded, expected):
    module, prepared = launcher
    run = module.DATA_ROOT / 'original_run'
    weights = run / 'weights'
    weights.mkdir(parents=True)
    checkpoint = weights / 'last.pt'
    checkpoint.write_bytes(b'resume-child-will-validate')
    (run / 'weather_protocol.json').write_text(json.dumps(dict(protocol=dict(training_profile=recorded))), encoding='utf-8')
    module.RESUME_CHECKPOINT = checkpoint
    module.TRAINING_PROFILE = 'ignored-invalid-top-level'
    module.CONFIGURATION = 'ignored-invalid-top-level'
    module.BUDGET_EPOCHS = 2
    module.LR0 = -1
    profile, argv = module.main(['--check'])
    assert profile == expected and not prepared
    assert option(argv, '--resume') == str(checkpoint)
    assert '--lr0' not in argv and '--configuration' not in argv
    if expected == 'baseline_finetune_v2':
        assert option(argv, '--budget-epochs') == '2'
    else:
        assert '--budget-epochs' not in argv


def test_real_baseline_preview_resolves_protocol_without_loading_weights_or_creating_run(launcher, capsys):
    module, _ = launcher
    result = module.main(['--check'])
    assert result.get('training_profile') == 'baseline_finetune_v2'
    assert result['train_args']['lr0'] == 1e-5 and result['train_args']['lrf'] == 1.
    assert result['train_args']['warmup_epochs'] == 0. and result['train_args']['epochs'] == 30
    assert result['execution_budget_epochs'] == 1
    assert result['train_args']['patience'] == 20
    assert result['weights'] == str(module.ORIGINAL_WEIGHTS.resolve())
    assert not module.PROJECT.exists()
    assert 'Preview only' in capsys.readouterr().out
