"""CPU entrypoint/provenance regressions; no detector or GPU training."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import yaml

HERE=Path(__file__).resolve().parents[1]
PROJECT=HERE
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(HERE))

def entrypoint():
    filename=HERE/'train_weather_finetune.py'
    assert filename.is_file(),'The bounded baseline entrypoint is not implemented'
    spec=importlib.util.spec_from_file_location('tested_finetune_entrypoint',filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@pytest.fixture
def inputs(tmp_path):
    names={i:str(i) for i in range(7)}
    for key in ('train','val','test'):(tmp_path/f'{key}.txt').write_text('')
    data=tmp_path/'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(tmp_path),nc=7,names=names,
        **{key:str(tmp_path/f'{key}.txt') for key in ('train','val','test')})))
    weights=tmp_path/'original.pt';weights.write_bytes(b'fixture-not-loaded')
    manifest=tmp_path/'manifest.jsonl';manifest.write_text('{}\n')
    return ['--data',str(data),'--weights',str(weights),'--manifest',str(manifest),
            '--project',str(tmp_path/'runs'),'--name','trial','--amp']

def test_fresh_profile_records_actual_finetune_settings(inputs):
    module=entrypoint()
    protocol=module.build_protocol(module.make_parser().parse_args(inputs))
    assert protocol['training_profile']=='baseline_finetune_v2'
    assert protocol['configuration']=='baseline'
    assert protocol['train_args']['lr0']==1e-5
    assert protocol['train_args']['lrf']==1.0
    assert protocol['train_args']['warmup_epochs']==0
    assert protocol['train_args']['batch']==8 and protocol['train_args']['epochs']==300
    assert protocol['train_args']['patience']==20
    assert protocol['finetune_policy']==dict(version=1,warmup_updates=64,initial_amp_scale=64.,validation_batch=4)
    assert protocol['execution_budget_epochs']==1
    assert protocol['validation_interval_updates']==1000

@pytest.mark.parametrize('budget',['0','-1','1.5'])
def test_budget_rejects_invalid_values_before_outputs(inputs,budget):
    module=entrypoint()
    with pytest.raises((SystemExit,ValueError)):
        module.build_protocol(module.make_parser().parse_args(inputs+['--budget-epochs',budget]))

def test_preview_does_not_create_run_or_load_weights(inputs,capsys):
    module=entrypoint()
    module.main(inputs)
    assert 'Preview' in capsys.readouterr().out
    assert not Path(inputs[inputs.index('--project')+1]).exists()

def make_resume(module,inputs):
    protocol=module.build_protocol(module.make_parser().parse_args(inputs))
    run=Path(protocol['train_args']['project'])/protocol['train_args']['name']
    weights=run/'weights';weights.mkdir(parents=True)
    checkpoint=weights/'last.pt';checkpoint.write_bytes(b'checkpoint-not-loaded-during-preview')
    fingerprint={protocol['manifest']:module.sha256(protocol['manifest'])}
    (run/'weather_protocol.json').write_text(json.dumps(dict(protocol=protocol,source_sha256=fingerprint,status='paused_budget')))
    return checkpoint,protocol

def test_resume_changes_operational_budget_only(inputs):
    module=entrypoint();checkpoint,original=make_resume(module,inputs)
    protocol=module.build_protocol(module.make_parser().parse_args(['--resume',str(checkpoint),'--budget-epochs','2']))
    assert protocol['execution_budget_epochs']==2
    assert protocol['train_args']['lr0']==original['train_args']['lr0']
    assert protocol['train_args']['epochs']==300
    assert protocol['resume_checkpoint']==str(checkpoint.resolve())

def test_resume_rejects_changed_frozen_source(inputs):
    module=entrypoint();checkpoint,original=make_resume(module,inputs)
    Path(original['manifest']).write_text('changed\n')
    with pytest.raises(ValueError,match='changed|missing'):
        module.build_protocol(module.make_parser().parse_args(['--resume',str(checkpoint)]))

def test_resume_rejects_training_overrides(inputs):
    module=entrypoint();checkpoint,_=make_resume(module,inputs)
    with pytest.raises(SystemExit):
        module.main(['--resume',str(checkpoint),'--epochs','2'])

def test_budget_pause_is_not_recorded_as_failure(inputs,monkeypatch):
    module=entrypoint()
    class Pause(Exception):pass
    class FakeTrainer:
        def __init__(self,overrides,**kwargs):
            self.save_dir=Path(overrides['project'])/overrides['name'];self.save_dir.mkdir(parents=True)
            self.epoch=0;self.start_epoch=0;self.last=self.save_dir/'weights/last.pt'
            self.last.parent.mkdir();self.last.write_bytes(b'full-checkpoint-owned-by-trainer')
        def train(self):raise Pause('complete epoch budget')
    fake=types.ModuleType('weather_finetune')
    fake.BaselineFinetuneTrainer=FakeTrainer;fake.BudgetPause=Pause
    monkeypatch.setitem(sys.modules,'weather_finetune',fake)
    monkeypatch.setattr(module,'freeze_candidate_inputs',lambda protocol:{protocol['manifest']:module.sha256(protocol['manifest'])})
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    record=module.main(inputs+['--train'])
    assert record['status']=='paused_budget' and 'error' not in record
    assert record['completed_epochs_this_invocation']==1
    assert Path(record['last_checkpoint']).is_file()
