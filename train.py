"""SVRDD ResNet18 experiments with explicit model selection (seed 0).

Fixed-IoU, batch-8 three-stage comparison (all P3/P4/P5; no P2 branch):
python train.py --experiment svrdd-baseline --check
python train.py --experiment svrdd-hsfpn-scsa --check
python train.py --experiment svrdd-hsfpn-scsa-dysample --check
python train.py --experiment svrdd-baseline
python train.py --experiment svrdd-hsfpn-scsa
python train.py --experiment svrdd-hsfpn-scsa-dysample

python train.py --experiment freqfusion
python train.py --experiment scsa-freqfusion --check
python train.py --experiment hsfpn-scsa --check
python train.py --experiment hsfpn-scsa-dysample --check
python train.py --experiment hsfpn-scsa-p2 --check
python train.py --experiment hsfpn-scsa-p3decoder --check
python train.py --experiment hsfpn-scsa-dysample-p2 --check
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import yaml

from ultralytics import RTDETR
from scripts.prepare_svrdd_official_split import DEFAULT_MANIFESTS
from scripts.audit_svrdd import audit_dataset
from ultralytics.nn.Addmodules.HSFPN import CA
from ultralytics.nn.Addmodules.SlimNeck import GSConv
from ultralytics.nn.Addmodules.SCSA import BasicBlock_SCSA, SCSA
from ultralytics.nn.Addmodules.FreqFusion import FreqFusionConcat
from ultralytics.nn.Addmodules.Dysample import Dy_Sample
from ultralytics.nn.Addmodules.FDConv import FDConv
from ultralytics.nn.Addmodules.CDRB import BasicBlock_CDRB
from ultralytics.nn.Addmodules.Conv3XC import Conv3XC, RepC3_Conv3XC
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.torch_utils import de_parallel


ROOT = Path(__file__).resolve().parent
MODEL_YAML = ROOT / 'ultralytics/cfg/models/rt-detr/resnet18-HSFPN+SCSA.yaml'
DATA_YAML = ROOT / 'SVRDD.yaml'
RUNS_DIR = ROOT / 'runs'
EXPERIMENTS = {
    'svrdd-baseline': {
        'model': MODEL_YAML.parent / 'resnet18-SVRDD-baseline.yaml',
        'name': 'svrdd-r18-baseline-fixediou-officialsplit-b8-s0', 'scsa': 0,
        'project': str(RUNS_DIR), 'factors': (),
    },
    'svrdd-hsfpn-scsa': {
        'model': MODEL_YAML,
        'name': 'svrdd-r18-hsfpn-scsa-fixediou-officialsplit-b8-s0', 'scsa': 8,
        'project': str(RUNS_DIR), 'factors': ('scsa', 'hsfpn'),
    },
    'svrdd-hsfpn-scsa-dysample': {
        'model': MODEL_YAML.parent / 'resnet18-HSFPN+SCSA-DySample.yaml',
        'name': 'svrdd-r18-hsfpn-scsa-dysample-fixediou-officialsplit-b8-s0', 'scsa': 8,
        'project': str(RUNS_DIR), 'factors': ('scsa', 'hsfpn', 'dysample'),
    },
    'freqfusion': {
        'model': MODEL_YAML.parent / 'rtdetr-resnet18-FreqFusion-P3.yaml',
        'name': 'rtdetr-resnet18-FreqFusion-P3-s0', 'scsa': 0,
        'project': str(RUNS_DIR),
    },
    'scsa-freqfusion': {
        'model': MODEL_YAML.parent / 'rtdetr-resnet18-SCSA-FreqFusion-P3.yaml',
        'name': 'rtdetr-resnet18-SCSA-FreqFusion-P3-s0', 'scsa': 8,
        'project': str(RUNS_DIR),
    },
    'hsfpn': {
        'model': MODEL_YAML.parent / 'rtdetr-resnet18-HSFPN.yaml',
        'name': 'rtdetr-resnet18-HSFPN-b8-s0', 'scsa': 0,
        'project': str(RUNS_DIR), 'factors': ('hsfpn',),
    },
    'hsfpn-scsa': {
        'model': MODEL_YAML, 'name': 'rtdetr-resnet18-HSFPN-SCSA-b8-s0', 'scsa': 8,
        'project': str(RUNS_DIR), 'factors': ('scsa', 'hsfpn'),
    },
    'hsfpn-scsa-dysample': {
        'model': MODEL_YAML.parent / 'resnet18-HSFPN+SCSA-DySample.yaml',
        'name': 'rtdetr-resnet18-HSFPN-SCSA-DySample-b8-s0', 'scsa': 8,
        'project': str(RUNS_DIR), 'factors': ('scsa', 'hsfpn', 'dysample'),
    },
    'hsfpn-scsa-p2': {
        'model': MODEL_YAML.parent / 'resnet18-HSFPN+SCSA-P2.yaml',
        'name': 'rtdetr-resnet18-HSFPN-SCSA-P2-b8-s0', 'scsa': 8,
        'project': str(RUNS_DIR), 'factors': ('scsa', 'hsfpn', 'p2'),
    },
    'hsfpn-scsa-p3decoder': {
        'model': MODEL_YAML.parent / 'resnet18-HSFPN+SCSA-P3Decoder.yaml',
        'name': 'rtdetr-resnet18-HSFPN-SCSA-P3Decoder-b8-s0', 'scsa': 8,
        'project': str(RUNS_DIR), 'factors': ('scsa', 'hsfpn', 'scale-decoder'),
    },
    'hsfpn-scsa-dysample-p2': {
        'model': MODEL_YAML.parent / 'resnet18-HSFPN+SCSA-DySample-P2.yaml',
        'name': 'rtdetr-resnet18-HSFPN-SCSA-DySample-P2-b8-s0', 'scsa': 8,
        'project': str(RUNS_DIR), 'factors': ('scsa', 'hsfpn', 'dysample', 'p2'),
    },
}
FIXED_IOU_GROUPS = frozenset((
    'svrdd-baseline', 'svrdd-hsfpn-scsa', 'svrdd-hsfpn-scsa-dysample',
))
_CANDIDATE_FACTORS = {
    'baseline': (),
    'scsa': ('scsa',),
    'freqfusion': ('freqfusion',),
    'scale-decoder': ('scale-decoder',),
    'scsa-freqfusion': ('scsa', 'freqfusion'),
    'scsa-scale-decoder': ('scsa', 'scale-decoder'),
    'freqfusion-scale-decoder': ('freqfusion', 'scale-decoder'),
    'scsa-freqfusion-scale-decoder': ('scsa', 'freqfusion', 'scale-decoder'),
    'dysample': ('dysample',),
    'fdconv-c4': ('fdconv-c4',),
    'scsa-dysample': ('scsa', 'dysample'),
    'scsa-fdconv-c4': ('scsa', 'fdconv-c4'),
    'dysample-fdconv-c4': ('dysample', 'fdconv-c4'),
    'scsa-dysample-fdconv-c4': ('scsa', 'dysample', 'fdconv-c4'),
}
_EXISTING_YAMLS = {
    'baseline': 'rtdetr-resnet18.yaml',
    'scsa': 'rtdetr-resnet18-SCSA.yaml',
    'freqfusion': 'rtdetr-resnet18-FreqFusion-P3.yaml',
    'scsa-freqfusion': 'rtdetr-resnet18-SCSA-FreqFusion-P3.yaml',
    'dysample': 'rtdetr-resnet18-DySample.yaml',
}
for _name, _factors in _CANDIDATE_FACTORS.items():
    if _name not in EXPERIMENTS:
        EXPERIMENTS[_name] = {
            'model': MODEL_YAML.parent / _EXISTING_YAMLS.get(_name, f'rtdetr-resnet18-{_name}.yaml'),
            'name': f'rtdetr-resnet18-{_name}-s0',
            'scsa': 8 if 'scsa' in _factors else 0,
            'project': str(RUNS_DIR),
        }
    EXPERIMENTS[_name]['factors'] = _factors
_CH3_CANDIDATE_FACTORS = {
    'svrdd-scsa': ('scsa',),
    'svrdd-cdrb': ('cdrb',),
    'svrdd-conv3xc': ('conv3xc',),
    'svrdd-scsa-cdrb': ('scsa', 'cdrb'),
    'svrdd-scsa-conv3xc': ('scsa', 'conv3xc'),
    'svrdd-cdrb-conv3xc': ('cdrb', 'conv3xc'),
    'svrdd-scsa-cdrb-conv3xc': ('scsa', 'cdrb', 'conv3xc'),
}
for _name, _factors in _CH3_CANDIDATE_FACTORS.items():
    EXPERIMENTS[_name] = {
        'model': MODEL_YAML.parent / f'rtdetr-resnet18-{_name}.yaml',
        'name': f'{_name}-fixediou-officialsplit-b8-s0',
        'scsa': 8 if 'scsa' in _factors else 0,
        'project': str(RUNS_DIR),
        'factors': _factors,
    }
FIXED_IOU_GROUPS = FIXED_IOU_GROUPS | frozenset(_CH3_CANDIDATE_FACTORS)
CH3_GROUPS = ('svrdd-baseline', *_CH3_CANDIDATE_FACTORS)
TRAIN_ARGS = dict(
    data=str(DATA_YAML),
    cache=False,
    imgsz=640,
    epochs=300,
    batch=8,
    workers=4,
    device='0',
    project=str(RUNS_DIR),
    name='rtdetr-resnet18-HSFPN-SCSA-b8-s0',
    exist_ok=False,
    resume=False,
    amp=False,
    optimizer='AdamW',
    lr0=0.0001,
    lrf=1.0,
    cos_lr=False,
    weight_decay=0.0001,
    warmup_epochs=2000,  # This fork interprets the value as warmup iterations.
    warmup_bias_lr=0.0,  # AdamW biases must not start at the YOLO default 0.1.
    close_mosaic=10,
    seed=0,
    deterministic=True,
)


def training_args(experiment, data=None, project=None, device='0', overrides=None, phase='formal', seed=0):
    """Keep one full-training protocol across the candidate ablation matrix."""
    settings = EXPERIMENTS[experiment]
    args = {**TRAIN_ARGS, 'data': str(data if data is not None else DATA_YAML),
            'project': str(project if project is not None else settings['project']),
            'name': settings['name'], 'device': device}
    if phase not in ('formal', 'screen') or seed < 0:
        raise ValueError('phase must be formal/screen and seed must be nonnegative')
    if phase == 'screen' and experiment not in CH3_GROUPS:
        raise ValueError('The screening protocol is only defined for the eight chapter-3 models')
    args['seed'] = seed
    if experiment in FIXED_IOU_GROUPS:
        # Explicit protocol v2: no early stop; FP32 training AND validation.
        args.update(patience=0, half=False, pretrained=False, nbs=8, momentum=0.9,
                    val=True, split='val', conf=0.001, iou=0.7, max_det=300,
                    single_cls=False, fraction=1.0, freeze=None, multi_scale=False,
                    mosaic=1.0, mixup=0.0, copy_paste=0.0,
                    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0,
                    translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
                    flipud=0.0, fliplr=0.5)
        args['name'] = f'{settings["name"].rsplit("-s0", 1)[0]}-v2-{phase}-s{seed}'
        if phase == 'screen':
            args.update(epochs=80, close_mosaic=0)
            args['name'] = args['name'].replace('-screen-', '-screen80-')
    elif seed != 0:
        args['name'] = f'{settings["name"]}-seed{seed}'
    allowed = {'epochs', 'batch', 'imgsz', 'workers', 'warmup_epochs', 'plots'}
    if overrides:
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f'Unsupported training overrides: {sorted(unknown)}')
        args.update(overrides)
    return args


def verify_official_svrdd_split(data_yaml, manifest_dir=DEFAULT_MANIFESTS):
    """Reject a mismatched or overlapping SVRDD split before a formal rerun."""
    return audit_dataset(data_yaml, manifest_dir)['counts']


def verify_standard_iou():
    """Fail before training if an older Inner-IoU implementation is still installed."""
    box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    shifted = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
    identical_iou = bbox_iou(box, box, xywh=False)
    partial_iou = bbox_iou(box, shifted, xywh=False)
    # For disjoint unit boxes, enclosing area is 3 and GIoU is -1/3.
    distant = torch.tensor([[2.0, 0.0, 3.0, 1.0]])
    giou = bbox_iou(box, distant, xywh=False, GIoU=True)
    xywh_giou = bbox_iou(torch.tensor([[.5, .5, 1., 1.]]),
                         torch.tensor([[2.5, .5, 1., 1.]]), xywh=True, GIoU=True)
    if not (torch.allclose(identical_iou, torch.ones_like(identical_iou), atol=1e-6)
            and torch.allclose(partial_iou, torch.full_like(partial_iou, 1 / 7), atol=1e-6)
            and torch.allclose(giou, torch.full_like(giou, -1 / 3), atol=1e-6)
            and torch.allclose(xywh_giou, giou, atol=1e-6)):
        raise RuntimeError(
            f'Fixed-IoU SVRDD experiments require standard bbox_iou; '
            f'found identical={identical_iou.tolist()}, partial={partial_iou.tolist()}, GIoU={giou.tolist()}')


def verify_model(model, experiment='hsfpn-scsa'):
    """Reject wrong experiments, bypassed SCSA and incomplete neck variants."""
    settings = EXPERIMENTS[experiment]
    model = de_parallel(model)
    modules = list(model.modules())
    attention_blocks = [m for m in modules if isinstance(m, (BasicBlock_SCSA, BasicBlock_CDRB))
                        and m.att]
    decoder = model.model[-1]
    points = decoder.decoder.layers[0].cross_attn.n_points
    points = (points,) * decoder.nl if isinstance(points, int) else tuple(points)
    counts = {
        'scsa_modules': sum(isinstance(m, SCSA) for m in modules),
        'hsfpn_ca_modules': sum(isinstance(m, CA) for m in modules),
        'transpose_convolutions': sum(isinstance(m, torch.nn.ConvTranspose2d) for m in modules),
        'freqfusion_modules': sum(isinstance(m, FreqFusionConcat) for m in modules),
        'dysample_modules': sum(isinstance(m, Dy_Sample) for m in modules),
        'fdconv_modules': sum(isinstance(m, FDConv) for m in modules),
        'cdrb_blocks': sum(isinstance(m, BasicBlock_CDRB) for m in modules),
        'conv3xc_modules': sum(isinstance(m, Conv3XC) for m in modules),
        'decoder_points': points,
    }
    factors = settings.get('factors', ())
    hsfpn_experiment = 'hsfpn' in factors
    p2_experiment = hsfpn_experiment and 'p2' in factors
    hsfpn_dysample = hsfpn_experiment and 'dysample' in factors
    expected = {
        'scsa_modules': 8 if 'scsa' in factors else 0,
        'hsfpn_ca_modules': 6 if p2_experiment else 5 if hsfpn_experiment else 0,
        'transpose_convolutions': 0 if hsfpn_dysample else 2 if hsfpn_experiment else 0,
        'freqfusion_modules': int('freqfusion' in factors),
        'dysample_modules': (3 if p2_experiment else 2) if hsfpn_dysample else
                            2 if 'dysample' in factors else 0,
        'fdconv_modules': int('fdconv-c4' in factors),
        'cdrb_blocks': 8 if 'cdrb' in factors else 0,
        'conv3xc_modules': 3 if 'conv3xc' in factors else 0,
        'decoder_points': (6, 4, 2) if 'scale-decoder' in factors else (4,) * decoder.nl,
    }
    valid = (counts == expected and len(attention_blocks) == settings['scsa']
             and all(hasattr(b, 'se') for b in attention_blocks))
    if hsfpn_experiment:
        # Counts alone would accept a neck that bypasses the intended fusion paths.
        routes = {13: 6, 15: 11, 17: [-1, 14], 18: [-1, 15],
                  20: 5, 22: 15, 24: [-1, 21], 25: [-1, 22]}
        if p2_experiment:
            routes.update({27: 26, 28: 27, 29: 4, 30: 28, 31: [-1, 29],
                           32: [-1, 28], 34: -1, 35: [-1, 26], 37: [36, 19, 12]})
        else:
            routes[27] = [26, 19, 12]
        ca_flags = {10: True, 13: True, 16: False, 20: True, 23: False}
        if p2_experiment:
            ca_flags[30] = False
        projection = model.model[8]
        valid = (valid and isinstance(projection, GSConv)
                 and isinstance(projection.cv1.act, torch.nn.SiLU)
                 and isinstance(projection.cv2.act, torch.nn.SiLU)
                 and all(model.model[i].f == source for i, source in routes.items())
                 and all(isinstance(model.model[i], CA) and model.model[i].flag == flag
                         for i, flag in ca_flags.items())
                 and all(isinstance(model.model[i], Dy_Sample if hsfpn_dysample
                                    else torch.nn.ConvTranspose2d) for i in (15, 22)))
        if p2_experiment:
            valid = valid and isinstance(model.model[27], Dy_Sample if hsfpn_dysample
                                         else torch.nn.Upsample)
    if hsfpn_experiment:
        valid = valid and model.model[-1].nc == 7
    if 'freqfusion' in factors:
        fusion = [m.fusion for m in modules if isinstance(m, FreqFusionConcat)]
        valid = valid and all(m.feature_resample and m.use_high_pass and m.use_low_pass for m in fusion)
        valid = valid and model.model[17].f == [16, 15] and model.model[-1].f == [18, 21, 24]
    if 'fdconv-c4' in factors:
        valid = valid and isinstance(model.model[6].blocks[1].branch2b.conv, FDConv)
    if 'cdrb' in factors:
        valid = valid and all(all(isinstance(block, BasicBlock_CDRB) and
                                  block.drb.kernel_sizes == [5, 5, 3, 3] and
                                  block.drb.dilates == [1, 2, 3, 4] and
                                  block.drb.lk_origin.kernel_size == (9, 9) and
                                  block.irmb.attn_s and
                                  block.irmb.v.conv.out_channels ==
                                  2 * block.drb.lk_origin.out_channels and
                                  block.irmb.conv_local.conv.groups ==
                                  block.irmb.v.conv.out_channels
                                  for block in model.model[stage].blocks)
                              for stage in (4, 5, 6, 7))
    if 'conv3xc' in factors:
        valid = valid and isinstance(model.model[19], RepC3_Conv3XC) and len(model.model[19].m) == 3
        valid = valid and all(isinstance(unit, Conv3XC) and
                              isinstance(unit.act, torch.nn.SiLU) and
                              unit.conv1.out_channels == 2 * unit.conv1.in_channels
                              for unit in model.model[19].m)
        valid = valid and all(isinstance(model.model[i], torch.nn.Module) and
                              not isinstance(model.model[i], RepC3_Conv3XC) for i in (14, 22, 25))
        valid = valid and decoder.f == [19, 22, 25]
    if 'dysample' in factors and not hsfpn_experiment:
        valid = valid and model.model[-1].f == [19, 22, 25]
    if experiment in FIXED_IOU_GROUPS:
        expected_inputs = [26, 19, 12] if hsfpn_experiment else [19, 22, 25]
        valid = valid and decoder.nc == 7 and decoder.nl == 3 and decoder.f == expected_inputs
    if not valid:
        label = 'HSFPN + enabled SCSA' if hsfpn_experiment else experiment
        raise RuntimeError(f'Expected {label}: {expected}; found {counts}. Check attention flags and P3 wiring.')
    return counts


def git_info(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=5,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def on_train_start(trainer, experiment='hsfpn-scsa', data_audit=None):
    """Check the rebuilt, dataset-specific model and preserve experiment sources."""
    model = de_parallel(trainer.model)
    counts = verify_model(model, experiment)
    snapshot = Path(trainer.save_dir) / 'source_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(getattr(trainer.args, 'data', DATA_YAML))
    factors = EXPERIMENTS[experiment].get('factors', ())
    sources = [Path(__file__).resolve(), EXPERIMENTS[experiment]['model'], data_yaml,
               ROOT / 'ultralytics/nn/tasks.py', ROOT / 'ultralytics/engine/trainer.py',
               ROOT / 'ultralytics/engine/validator.py',
               ROOT / 'ultralytics/utils/torch_utils.py',
               ROOT / 'ultralytics/data/augment.py', ROOT / 'ultralytics/cfg/default.yaml',
               ROOT / 'scripts/audit_svrdd.py', ROOT / 'scripts/prepare_svrdd_official_split.py',
               ROOT / 'ultralytics/utils/metrics.py',
               ROOT / 'ultralytics/models/utils/loss.py', ROOT / 'ultralytics/models/utils/ops.py',
               ROOT / 'ultralytics/nn/modules/transformer.py', ROOT / 'ultralytics/nn/modules/utils.py',
               ROOT / 'ultralytics/nn/Addmodules/__init__.py',
               ROOT / 'ultralytics/nn/Addmodules/ResNet.py', ROOT / 'ultralytics/nn/Addmodules/SCSA.py']
    if 'freqfusion' in factors:
        sources.append(ROOT / 'ultralytics/nn/Addmodules/FreqFusion.py')
    if 'dysample' in factors:
        sources.append(ROOT / 'ultralytics/nn/Addmodules/Dysample.py')
    if 'fdconv-c4' in factors:
        sources.extend([ROOT / 'ultralytics/nn/Addmodules/FDConv.py',
                        ROOT / 'ultralytics/nn/Addmodules/FDConvResNet.py',
                        ROOT / 'ultralytics/nn/Addmodules/FDConv-LICENSE.txt'])
    if 'cdrb' in factors:
        sources.extend([ROOT / 'ultralytics/nn/Addmodules/CDRB.py',
                        ROOT / 'ultralytics/nn/Addmodules/UniRepLknet.py',
                        ROOT / 'ultralytics/nn/Addmodules/iRMB.py'])
    if 'conv3xc' in factors:
        sources.extend([ROOT / 'ultralytics/nn/Addmodules/Conv3XC.py',
                        ROOT / 'ultralytics/nn/modules/block.py'])
    if 'hsfpn' in factors:
        sources.extend([ROOT / 'ultralytics/nn/Addmodules/HSFPN.py',
                        ROOT / 'ultralytics/nn/Addmodules/SlimNeck.py',
                        ROOT / 'ultralytics/nn/Addmodules/ASFYOLO.py'])
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    packages = {}
    for package in ('numpy', 'torchvision', 'timm', 'einops', 'albumentations', 'opencv-python', 'scipy'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    metadata = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_info('rev-parse', 'HEAD'),
        'git_status': git_info('status', '--porcelain=v1'),
        'initialization': 'random_from_yaml; no checkpoint loaded',
        'experiment': experiment,
        'freqfusion_backend': 'torch_reflect' if 'freqfusion' in factors else None,
        'source_sha256': {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
        'model_check': counts,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'model_yaml': model.yaml,
        'training_args': vars(trainer.args),
        'python': sys.version,
        'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
        'packages': packages,
        'dataset_audit': data_audit,
    }
    (snapshot / 'provenance.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(f'{experiment} verified: {counts}; sources saved to {snapshot}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=EXPERIMENTS,
                        help='Explicitly select the model; there is no implicit training default')
    parser.add_argument('--check', action='store_true', help='Check model structure only; do not train')
    parser.add_argument('--preflight', action='store_true', help='Deep dataset audit plus selected model check; do not train')
    parser.add_argument('--phase', choices=('formal', 'screen'), default='formal',
                        help='formal: 300 epochs; screen: 80 epochs, no mosaic shutdown, separate outputs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data', type=Path, default=DATA_YAML, help='Dataset YAML (default: repository SVRDD.yaml)')
    parser.add_argument('--project', type=Path, help='Override shared-disk output parent directory')
    parser.add_argument('--device', default='0', help='Training device, e.g. 0 or 1')
    parser.add_argument('--epochs', type=int, help='Override epochs for a local smoke run')
    parser.add_argument('--batch', type=int, help='Override batch for a local smoke run')
    parser.add_argument('--imgsz', type=int, help='Override image size for a local smoke run')
    parser.add_argument('--workers', type=int, help='Override dataloader workers for a local smoke run')
    parser.add_argument('--warmup-iterations', type=int,
                        help='Override warmup iterations (this fork stores them as warmup_epochs)')
    parser.add_argument('--no-plots', action='store_true', help='Disable matplotlib plots for a smoke run')
    options = parser.parse_args()
    options.data = options.data.expanduser().resolve()
    overrides = {key: value for key, value in {
        'epochs': options.epochs, 'batch': options.batch, 'imgsz': options.imgsz,
        'workers': options.workers, 'warmup_epochs': options.warmup_iterations,
        'plots': False if options.no_plots else None}.items()
        if value is not None}
    args = training_args(options.experiment, data=options.data.expanduser().resolve(),
                         project=options.project.expanduser().resolve() if options.project else None,
                         device=options.device, overrides=overrides, phase=options.phase, seed=options.seed)
    if (options.experiment in FIXED_IOU_GROUPS or
            'hsfpn' in EXPERIMENTS[options.experiment].get('factors', ())) and not options.check:
        required = {'batch': 8, 'epochs': 80 if options.phase == 'screen' else 300, 'imgsz': 640,
                    'warmup_epochs': 2000, 'warmup_bias_lr': 0.0}
        different = {key: (args[key], value) for key, value in required.items() if args[key] != value}
        if different:
            raise ValueError(f'Matched SVRDD rerun requires {required}; found {different}')
    model_yaml = EXPERIMENTS[options.experiment]['model']
    os.chdir(ROOT)
    if options.experiment in FIXED_IOU_GROUPS:
        verify_standard_iou()
    # Build from YAML without model.load(): do not resume or fine-tune old experiments.
    model = RTDETR(str(model_yaml))
    counts = verify_model(model.model, options.experiment)
    print(f'{options.experiment} model check passed: {counts}', flush=True)
    if options.check and not options.preflight:
        print(json.dumps({'experiment': options.experiment, 'model': str(model_yaml), 'training': args}, indent=2))
        print('Check only: no training, dataset validation, or GPU check was performed.')
        return
    data_audit = None
    if options.experiment in FIXED_IOU_GROUPS or options.preflight:
        print('Auditing every image and label; checking cross-split content duplicates...', flush=True)
        data_audit = audit_dataset(options.data, deep=True)
        print(json.dumps(data_audit, ensure_ascii=False, indent=2), flush=True)
    if options.preflight:
        print(json.dumps({'experiment': options.experiment, 'training': args}, indent=2))
        print('Preflight passed; no training or server GPU-capacity test was performed.')
        return
    # The framework label cache hashes file sizes, not label content. Rebuild
    # only these derived cache files after validating the complete source data.
    if data_audit is not None:
        cfg = yaml.safe_load(options.data.read_text(encoding='utf-8-sig'))
        root = Path(cfg['path'])
        if not root.is_absolute():
            root = options.data.resolve().parent / root
        for key in ('train', 'val', 'test'):
            cache = (root / cfg[key]).resolve().parent / 'labels.cache'
            if cache.is_file():
                cache.unlink()
    model.add_callback('on_train_start', partial(on_train_start, experiment=options.experiment, data_audit=data_audit))
    torch.cuda.empty_cache()
    model.train(**args)


if __name__ == '__main__':
    main()
