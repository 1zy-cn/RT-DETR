"""Fresh SVRDD ResNet18 + HSFPN + enabled SCSA experiment (seed 0).

python train.py          Start training on CUDA device 0.
python train.py --check  Check the model/configuration on CPU without training.
"""
import argparse
from datetime import datetime, timezone
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
from ultralytics.utils.torch_utils import de_parallel


ROOT = Path(__file__).resolve().parent
MODEL_YAML = ROOT / 'ultralytics/cfg/models/rt-detr/resnet18-HSFPN+SCSA.yaml'
DATA_YAML = ROOT / 'SVRDD.yaml'
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


def verify_model(model):
    """Reject disabled SCSA and accidental baseline/SCSA-only configurations."""
    model = de_parallel(model)
    modules = list(model.modules())
    blocks = [m for m in modules if isinstance(m, BasicBlock_SCSA)]
    counts = {
        'scsa_modules': sum(isinstance(m, SCSA) for m in modules),
        'hsfpn_ca_modules': sum(isinstance(m, CA) for m in modules),
        'transpose_convolutions': sum(isinstance(m, torch.nn.ConvTranspose2d) for m in modules),
    }
    if (counts != {'scsa_modules': 8, 'hsfpn_ca_modules': 5, 'transpose_convolutions': 2}
            or len(blocks) != 8 or not all(b.att and hasattr(b, 'se') for b in blocks)):
        raise RuntimeError(f'Expected HSFPN + 8 enabled SCSA modules; found {counts}')
    return counts


def git_info(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=5,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def on_train_start(trainer):
    """Check the rebuilt, dataset-specific model and preserve experiment sources."""
    model = de_parallel(trainer.model)
    counts = verify_model(model)
    snapshot = Path(trainer.save_dir) / 'source_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in [Path(__file__).resolve(), MODEL_YAML, DATA_YAML]:
        shutil.copy2(source, snapshot / source.name)
    metadata = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_info('rev-parse', 'HEAD'),
        'git_status': git_info('status', '--porcelain=v1'),
        'initialization': 'random_from_yaml; no checkpoint loaded',
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
    print(f'HSFPN + SCSA verified: {counts}; sources saved to {snapshot}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check model structure only; do not train')
    options = parser.parse_args()
    os.chdir(ROOT)
    # Build from YAML without model.load(): do not resume or fine-tune old experiments.
    model = RTDETR(str(MODEL_YAML))
    counts = verify_model(model.model)
    print(f'HSFPN + SCSA model check passed: {counts}', flush=True)
    if options.check:
        print(json.dumps({'model': str(MODEL_YAML), 'training': TRAIN_ARGS}, indent=2))
        print('Check only: no training, dataset validation, or GPU check was performed.')
        return
    model.add_callback('on_train_start', on_train_start)
    torch.cuda.empty_cache()
    model.train(**TRAIN_ARGS)


if __name__ == '__main__':
    main()
