"""Two real training batches, validation and checkpoint reload before renting a long GPU run.

Run from the repository root: python -m checks.smoke_ch3_training --data SVRDD.yaml
Metrics from this deliberately tiny subset are NOT research results.
"""
import argparse
from functools import partial
from pathlib import Path
import tempfile

import torch
import yaml

from train import CH3_GROUPS, EXPERIMENTS, ROOT, on_train_start, training_args, verify_model, verify_standard_iou
from ultralytics import RTDETR
from ultralytics.utils.files import increment_path


def reset_memory_stats(trainer):
    if trainer.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(trainer.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=CH3_GROUPS, default='svrdd-scsa-cdrb-conv3xc')
    parser.add_argument('--data', type=Path, default=ROOT / 'SVRDD.yaml')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    options = parser.parse_args()
    if options.batch < 2:
        raise ValueError('Use batch >= 2 for BatchNorm training')
    if ',' in options.device:
        raise ValueError('This smoke check supports one GPU or cpu')
    torch.set_num_threads(2)
    verify_standard_iou()
    cfg = yaml.safe_load(options.data.read_text(encoding='utf-8-sig'))
    root = Path(cfg['path'])
    root = root if root.is_absolute() else options.data.resolve().parent / root
    destination = increment_path(ROOT / 'runs/server-smoke' / options.experiment, mkdir=True)
    with tempfile.TemporaryDirectory(prefix='smoke-inputs-', dir=destination) as directory:
        directory = Path(directory)
        subset = {key: cfg[key] for key in ('nc', 'names')}
        subset['path'] = str(root.resolve())
        for split, count in (('train', options.batch * 2), ('val', options.batch)):
            images = sorted((root / cfg[split]).glob('*.jpg'))[:count]
            if len(images) != count:
                raise ValueError(f'Not enough {split} images: expected {count}')
            listing = directory / f'{split}.txt'
            listing.write_text(''.join(f'{image.resolve().as_posix()}\n' for image in images), encoding='utf-8')
            subset[split] = str(listing.resolve())
        data = directory / 'smoke.yaml'
        data.write_text(yaml.safe_dump(subset, allow_unicode=True), encoding='utf-8')
        args = training_args(options.experiment, data=data, project=destination, device=options.device,
            overrides={'epochs': 1, 'batch': options.batch, 'imgsz': options.imgsz, 'workers': 0, 'plots': False})
        args.update(name='two-batch-check', close_mosaic=0)
        model = RTDETR(str(EXPERIMENTS[options.experiment]['model']))
        model.add_callback('on_train_start', partial(on_train_start, experiment=options.experiment))
        model.add_callback('on_train_start', reset_memory_stats)
        model.train(**args)
        saved = model.trainer.best
        loaded = RTDETR(str(saved))
        verify_model(loaded.model, options.experiment)
        if next(loaded.model.parameters()).dtype != torch.float32:
            raise RuntimeError('Expected FP32 checkpoint')
        # train() already reloads/evaluates best.pt via final_eval on the tiny val set.
        print(f'SMOKE PASSED: two training batches + validation + FP32 checkpoint reload: {saved}')
        if model.trainer.device.type == 'cuda':
            print(f'CUDA peak allocated GiB: {torch.cuda.max_memory_allocated(model.trainer.device) / 2**30:.3f}')
        print('Tiny-subset metrics are not mAP evidence. Full preflight and sustained server monitoring are still required.')


if __name__ == '__main__':
    main()
