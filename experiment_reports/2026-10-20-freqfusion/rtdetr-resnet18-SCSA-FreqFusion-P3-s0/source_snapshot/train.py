"""Fresh SVRDD ResNet18 experiments with explicit model selection (seed 0).

python train.py --experiment freqfusion
python train.py --experiment scsa-freqfusion --check
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from ultralytics import RTDETR
from ultralytics.nn.Addmodules.HSFPN import CA
from ultralytics.nn.Addmodules.SCSA import BasicBlock_SCSA, SCSA
from ultralytics.nn.Addmodules.FreqFusion import FreqFusionConcat
from ultralytics.utils.torch_utils import de_parallel


ROOT = Path(__file__).resolve().parent
MODEL_YAML = ROOT / 'ultralytics/cfg/models/rt-detr/resnet18-HSFPN+SCSA.yaml'
DATA_YAML = ROOT / 'SVRDD.yaml'
EXPERIMENTS = {
    'freqfusion': {
        'model': MODEL_YAML.parent / 'rtdetr-resnet18-FreqFusion-P3.yaml',
        'name': 'rtdetr-resnet18-FreqFusion-P3-s0', 'scsa': 0,
        'project': '/root/autodl-fs/RT-DETR-FreqFusion-runs',
    },
    'scsa-freqfusion': {
        'model': MODEL_YAML.parent / 'rtdetr-resnet18-SCSA-FreqFusion-P3.yaml',
        'name': 'rtdetr-resnet18-SCSA-FreqFusion-P3-s0', 'scsa': 8,
        'project': '/root/autodl-fs/RT-DETR-FreqFusion-runs',
    },
    'hsfpn-scsa': {
        'model': MODEL_YAML, 'name': 'rtdetr-resnet18-HSFPN-SCSA-enabled-s0', 'scsa': 8,
        'project': '/root/autodl-fs/RT-DETR-HSFPN-SCSA-runs',
    },
}
TRAIN_ARGS = dict(
    data=str(DATA_YAML),
    cache=False,
    imgsz=640,
    epochs=300,
    batch=8,
    workers=4,
    device='0',
    project='/root/autodl-fs/RT-DETR-HSFPN-SCSA-runs',
    name='rtdetr-resnet18-HSFPN-SCSA-enabled-s0',
    exist_ok=False,
    resume=False,
    amp=False,
    optimizer='AdamW',
    lr0=0.0001,
    lrf=1.0,
    cos_lr=False,
    weight_decay=0.0001,
    warmup_epochs=2000,  # This fork interprets the value as warmup iterations.
    close_mosaic=10,
    seed=0,
    deterministic=True,
)


def training_args(experiment, data=None, project=None, device='0'):
    """One shared protocol; only the model-specific run name changes for FreqFusion."""
    settings = EXPERIMENTS[experiment]
    return {**TRAIN_ARGS, 'data': str(data if data is not None else DATA_YAML),
            'project': str(project if project is not None else settings['project']),
            'name': settings['name'], 'device': device}


def verify_model(model, experiment='hsfpn-scsa'):
    """Reject wrong experiments, bypassed SCSA and incomplete FreqFusion variants."""
    settings = EXPERIMENTS[experiment]
    model = de_parallel(model)
    modules = list(model.modules())
    blocks = [m for m in modules if isinstance(m, BasicBlock_SCSA)]
    counts = {
        'scsa_modules': sum(isinstance(m, SCSA) for m in modules),
        'hsfpn_ca_modules': sum(isinstance(m, CA) for m in modules),
        'transpose_convolutions': sum(isinstance(m, torch.nn.ConvTranspose2d) for m in modules),
        'freqfusion_modules': sum(isinstance(m, FreqFusionConcat) for m in modules),
    }
    legacy = experiment == 'hsfpn-scsa'
    expected = {'scsa_modules': settings['scsa'], 'hsfpn_ca_modules': 5 if legacy else 0,
                'transpose_convolutions': 2 if legacy else 0, 'freqfusion_modules': 0 if legacy else 1}
    valid = (counts == expected and len(blocks) == settings['scsa']
             and all(b.att and hasattr(b, 'se') for b in blocks))
    if not legacy:
        fusion = [m.fusion for m in modules if isinstance(m, FreqFusionConcat)]
        valid = valid and all(m.feature_resample and m.use_high_pass and m.use_low_pass for m in fusion)
        valid = valid and model.model[17].f == [16, 15] and model.model[-1].f == [18, 21, 24]
    if not valid:
        label = 'HSFPN + enabled SCSA' if legacy else experiment
        raise RuntimeError(f'Expected {label}: {expected}; found {counts}. Check attention flags and P3 wiring.')
    return counts


def git_info(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=5,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def on_train_start(trainer, experiment='hsfpn-scsa'):
    """Check the rebuilt, dataset-specific model and preserve experiment sources."""
    model = de_parallel(trainer.model)
    counts = verify_model(model, experiment)
    snapshot = Path(trainer.save_dir) / 'source_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(getattr(trainer.args, 'data', DATA_YAML))
    sources = [Path(__file__).resolve(), EXPERIMENTS[experiment]['model'], data_yaml,
               ROOT / 'ultralytics/nn/tasks.py', ROOT / 'ultralytics/engine/trainer.py',
               ROOT / 'ultralytics/nn/Addmodules/ResNet.py', ROOT / 'ultralytics/nn/Addmodules/SCSA.py']
    if experiment != 'hsfpn-scsa':
        sources.append(ROOT / 'ultralytics/nn/Addmodules/FreqFusion.py')
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    metadata = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_info('rev-parse', 'HEAD'),
        'git_status': git_info('status', '--porcelain=v1'),
        'initialization': 'random_from_yaml; no checkpoint loaded',
        'experiment': experiment,
        'freqfusion_backend': 'torch_reflect' if experiment != 'hsfpn-scsa' else None,
        'source_sha256': {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
        'model_check': counts,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'model_yaml': model.yaml,
        'training_args': vars(trainer.args),
        'python': sys.version,
        'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
    }
    (snapshot / 'provenance.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(f'{experiment} verified: {counts}; sources saved to {snapshot}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=EXPERIMENTS,
                        help='Explicitly select the model; there is no implicit training default')
    parser.add_argument('--check', action='store_true', help='Check model structure only; do not train')
    parser.add_argument('--data', type=Path, default=DATA_YAML, help='Dataset YAML (default: repository SVRDD.yaml)')
    parser.add_argument('--project', type=Path, help='Override shared-disk output parent directory')
    parser.add_argument('--device', default='0', help='Training device, e.g. 0 or 1')
    options = parser.parse_args()
    args = training_args(options.experiment, data=options.data.expanduser().resolve(),
                         project=options.project.expanduser().resolve() if options.project else None,
                         device=options.device)
    model_yaml = EXPERIMENTS[options.experiment]['model']
    os.chdir(ROOT)
    # Build from YAML without model.load(): do not resume or fine-tune old experiments.
    model = RTDETR(str(model_yaml))
    counts = verify_model(model.model, options.experiment)
    print(f'{options.experiment} model check passed: {counts}', flush=True)
    if options.check:
        print(json.dumps({'experiment': options.experiment, 'model': str(model_yaml), 'training': args}, indent=2))
        print('Check only: no training, dataset validation, or GPU check was performed.')
        return
    model.add_callback('on_train_start', partial(on_train_start, experiment=options.experiment))
    torch.cuda.empty_cache()
    model.train(**args)


if __name__ == '__main__':
    main()
