"""CPU protocol tests for explicit fresh-run learning-rate options."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

HERE=Path(__file__).resolve().parents[1]
PROJECT=HERE
sys.path.insert(0,str(PROJECT))

def module_at(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def module():
    return module_at(HERE/'train_weather_finetune.py','tested_lr_candidate')

@pytest.fixture
def inputs(tmp_path):
    for key in ('train','val','test'):(tmp_path/f'{key}.txt').write_text('')
    data=tmp_path/'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(tmp_path),nc=7,names={i:str(i) for i in range(7)},
        **{key:str(tmp_path/f'{key}.txt') for key in ('train','val','test')})))
    weights=tmp_path/'original.pt';weights.write_bytes(b'fixture-not-loaded')
    manifest=tmp_path/'manifest.jsonl';manifest.write_text('{}\n')
    return ['--data',str(data),'--weights',str(weights),'--manifest',str(manifest),
            '--project',str(tmp_path/'runs'),'--name','trial','--amp']

def protocol(module,argv):
    return module.build_protocol(module.make_parser().parse_args(argv))

def test_explicit_rates_and_cosine_are_used_in_actual_protocol(module,inputs):
    actual=protocol(module,inputs+['--lr0','4e-6','--lrf','0.05','--cos-lr'])
    assert actual['train_args']['lr0']==4e-6
    assert actual['train_args']['lrf']==.05
    assert actual['train_args']['cos_lr'] is True
    assert actual['finetune_policy']==module.POLICY
    assert actual['train_args']['patience']==20
    assert actual['train_args']['batch']==8

@pytest.mark.parametrize('lrf',['0','1'])
def test_inclusive_lrf_boundaries_and_linear_default(module,inputs,lrf):
    actual=protocol(module,inputs+['--lrf',lrf])
    assert actual['train_args']['lrf']==float(lrf)
    assert actual['train_args']['cos_lr'] is False

@pytest.mark.parametrize('value',['0','-1','nan','inf','-inf'])
def test_invalid_initial_lr_rejected_before_run_creation(module,inputs,value):
    args=module.make_parser().parse_args(inputs+['--lr0='+value])
    with pytest.raises(ValueError,match='lr0'):
        module.build_protocol(args)
    assert not Path(args.project).exists()

@pytest.mark.parametrize('value',['-0.001','1.001','nan','inf','-inf'])
def test_invalid_final_lr_fraction_rejected_before_run_creation(module,inputs,value):
    args=module.make_parser().parse_args(inputs+['--lrf='+value])
    with pytest.raises(ValueError,match='lrf'):
        module.build_protocol(args)
    assert not Path(args.project).exists()

def make_resume(module,inputs):
    frozen=protocol(module,inputs+['--lr0','4e-6','--lrf','.05','--cos-lr'])
    run=Path(frozen['train_args']['project'])/frozen['train_args']['name']
    (run/'weights').mkdir(parents=True)
    checkpoint=run/'weights/last.pt';checkpoint.write_bytes(b'fixture-preview-only')
    (run/'weather_protocol.json').write_text(json.dumps(dict(protocol=frozen,
        source_sha256={frozen['manifest']:module.sha256(frozen['manifest'])},status='paused_budget')))
    return checkpoint,frozen

def test_resume_restores_original_rate_schedule_over_parser_defaults(module,inputs):
    checkpoint,frozen=make_resume(module,inputs)
    resumed=protocol(module,['--resume',str(checkpoint),'--budget-epochs','2'])
    for key in ('lr0','lrf','cos_lr'):
        assert resumed['train_args'][key]==frozen['train_args'][key]
    assert resumed['execution_budget_epochs']==2

@pytest.mark.parametrize('override',[
    ['--lr0','1e-6'],['--lrf','.1'],['--cos-lr']])
def test_resume_cli_rejects_any_new_schedule_override(module,inputs,override):
    checkpoint,_=make_resume(module,inputs)
    with pytest.raises(SystemExit):
        module.main(['--resume',str(checkpoint)]+override)

def test_new_schedule_preview_loads_no_weights_and_creates_no_run(module,inputs,capsys):
    actual=module.main(inputs+['--lr0','3e-6','--lrf','.1','--cos-lr'])
    assert actual['train_args']['lr0']==3e-6
    assert 'Preview only' in capsys.readouterr().out
    assert not Path(actual['train_args']['project']).exists()
