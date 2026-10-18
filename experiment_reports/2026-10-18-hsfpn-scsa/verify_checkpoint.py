"""Verify only trusted, hash-matched archived checkpoints on CPU.

From the repository root:
python experiment_reports/2026-10-18-hsfpn-scsa/verify_checkpoint.py /path/to/weights
The folder must contain this experiment's best.pt and last.pt.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('weights', type=Path)
source = parser.parse_args().weights
report = Path(__file__).resolve().parent
manifest = json.loads((report/'source_manifest.json').read_text(encoding='utf-8'))
for name, meta in manifest['raw_files'].items():
    payload = (report/name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == meta['sha256'] and len(payload) == meta['bytes']
raw_rows = [{k.strip():float(v) for k,v in r.items()} for r in csv.DictReader((report/'run/results.csv').read_text().splitlines())]
sys.path.insert(0, str(report.parents[1]))
import torch
import yaml
from ultralytics.nn.Addmodules.SCSA import BasicBlock_SCSA, SCSA
from ultralytics.nn.Addmodules.HSFPN import CA
torch.set_num_threads(2)
torch.manual_seed(0)
evidence = {'runtime':{'torch':str(torch.__version__),'device':'cpu'},'checkpoints':{}}
config = yaml.safe_load((report/'source_snapshot/model.yaml').read_text(encoding='utf-8'))
for name in ['best.pt','last.pt']:
    path = source/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['weights'][name]['sha256']
    ckpt = torch.load(path,map_location='cpu')
    model = ckpt['model'] if ckpt.get('model') is not None else ckpt['ema']
    model.float().cpu().eval()
    groups = {'scsa':{n:m for n,m in model.named_modules() if isinstance(m,SCSA)},
              'hsfpn_ca':{n:m for n,m in model.named_modules() if isinstance(m,CA)},
              'upsample':{n:m for n,m in model.named_modules() if isinstance(m,torch.nn.ConvTranspose2d)}}
    assert {k:len(v) for k,v in groups.items()} == {'scsa':8,'hsfpn_ca':5,'upsample':2}
    blocks = [m for m in model.modules() if isinstance(m,BasicBlock_SCSA)]
    assert len(blocks) == 8 and all(b.att and hasattr(b,'se') for b in blocks)
    assert model.yaml['backbone'] == config['backbone'] and model.yaml['head'] == config['head']
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for key,vals in ckpt['train_results'].items():
        assert len(vals) == len(raw_rows) == 300
        assert vals == [r[key] for r in raw_rows[:len(vals)]]
    info = {'sha256':manifest['weights'][name]['sha256'],'stored_epoch':ckpt.get('epoch'),
        'optimizer_present':ckpt.get('optimizer') is not None,'date':ckpt.get('date'),'version':ckpt.get('version'),
        'module_counts':{k:len(v) for k,v in groups.items()},'active_scsa_blocks':len(blocks),
        'parameters':sum(p.numel() for p in model.parameters()),'all_parameters_finite':True,
        'train_results_match_csv':True,'train_results_epochs':len(next(iter(ckpt['train_results'].values()))),
        'train_args':ckpt['train_args'],'train_metrics':ckpt['train_metrics'],
        'model_yaml':model.yaml,'class_names':model.names,'structure_matches_archived_yaml':True}
    if name == 'best.pt':
        tracked = {n:m for g in groups.values() for n,m in g.items()}
        calls = dict.fromkeys(tracked,0)
        def hook_for(n):
            def hook(_m,_i,_o):
                calls[n] += 1
            return hook
        handles = [m.register_forward_hook(hook_for(n)) for n,m in tracked.items()]
        image = torch.rand(1,3,640,640)
        with torch.inference_mode():
            prediction = model(image)[0]
        assert prediction.shape == (1,300,11) and torch.isfinite(prediction).all()
        assert all(v == 1 for v in calls.values())
        for h in handles:
            h.remove()
        for b in blocks:
            b.att = False
        with torch.inference_mode():
            disabled = model(image)[0]
        assert torch.isfinite(disabled).all() and not torch.allclose(prediction,disabled,atol=1e-7,rtol=1e-5)
        for b in blocks:
            b.att = True
        info['runtime_640'] = {'shape':list(prediction.shape),'finite':True,'module_calls':calls,
                              'disabled_scsa_max_output_difference':(prediction-disabled).abs().max().item()}
        (report/'model_from_checkpoint.yaml').write_text(yaml.safe_dump(model.yaml,allow_unicode=True,sort_keys=False),encoding='utf-8')
    evidence['checkpoints'][name] = info
    del model,ckpt
(report/'checkpoint_verification.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({n:{'module_counts':i['module_counts'],'parameters':i['parameters'],'epochs':i['train_results_epochs'],
    'actual_batch':i['train_args']['batch'],'actual_nbs':i['train_args']['nbs'],'metrics':i['train_metrics'],
    'runtime_640':i.get('runtime_640')} for n,i in evidence['checkpoints'].items()},indent=2))
