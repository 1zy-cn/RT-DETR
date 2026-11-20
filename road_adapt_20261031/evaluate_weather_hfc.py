"""Preview a Chapter 4 test group; --evaluate explicitly executes evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

from weather_hfc_protocol import sha256

ROOT = Path(__file__).resolve().parent
GROUPS = dict(primary=6000, clear=1000, severe=3000, fog=2000, rain=2000, lowlight=2000)


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--group', choices=tuple(GROUPS), default='primary')
    p.add_argument('--data-dir', type=Path, default=ROOT/'configs/chapter4')
    p.add_argument('--manifest', type=Path, default=Path('D:/Users/86186/ResearchData/SVRDD-weather-v1/manifest.jsonl'))
    p.add_argument('--project', type=Path, default=ROOT/'runs/weather_hfc_eval')
    p.add_argument('--name')
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--evaluate', action='store_true')
    return p


def build_protocol(args):
    weights = args.weights.resolve()
    data = (args.data_dir/f'test_{args.group}.yaml').resolve()
    manifest = args.manifest.resolve()
    for path in (weights, data, manifest):
        if not path.is_file(): raise FileNotFoundError(path)
    if args.batch < 1 or args.workers < 0:
        raise ValueError('batch must be positive and workers nonnegative')
    if args.device != 'cpu' and not args.device.isdecimal():
        raise ValueError('Only cpu or one GPU index is supported')
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    test_list = Path(config['test'])
    if not test_list.is_absolute(): test_list = Path(config['path'])/test_list
    images = len([s for s in test_list.read_text(encoding='utf-8').splitlines() if s.strip()])
    if images != GROUPS[args.group]: raise ValueError(f'Unexpected {args.group} image count: {images}')
    name = args.name or f'{weights.parent.parent.name}_{args.group}'
    if Path(name).name != name or '/' in name or '\\' in name or name in ('.','..'):
        raise ValueError('name must be one directory name')
    if (args.project.resolve()/name).exists(): raise FileExistsError(args.project/name)
    eval_args = dict(model=str(weights), data=str(data), split='test', imgsz=640, batch=args.batch,
                     workers=args.workers, device=args.device, half=False, conf=0.001, max_det=300,
                     # BaseValidator merges training defaults too. The paired dataset
                     # requires this contract even when augment=False at evaluation.
                     augment=False, multi_scale=False, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                     degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                     fliplr=0.0, flipud=0.0,
                     plots=False, save_json=True, project=str(args.project.resolve()), name=name, exist_ok=False)
    return dict(weights=str(weights), data=str(data), manifest=str(manifest), test_list=str(test_list),
                group=args.group, images=images, eval_args=eval_args)


def main(argv=None):
    p = make_parser(); args = p.parse_args(argv)
    try: protocol = build_protocol(args)
    except (ValueError,FileNotFoundError,FileExistsError) as exc: p.error(str(exc))
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    if not args.evaluate:
        print('Preview only. Add --evaluate to execute this test group; no evaluation has run.')
        return
    import torch
    from weather_hfc_training import WeatherHFCValidator
    checkpoint = torch.load(protocol['weights'], map_location='cpu', weights_only=False)
    model = (checkpoint.get('ema') or checkpoint.get('model')) if isinstance(checkpoint, dict) else checkpoint
    names = yaml.safe_load(Path(protocol['data']).read_text(encoding='utf-8'))['names']
    names = dict(enumerate(names)) if isinstance(names, list) else names
    actual = model.names
    actual = dict(enumerate(actual)) if isinstance(actual, list) else actual
    if actual != names:
        raise ValueError('Checkpoint class order/names differ from the seven-class dataset')
    model = model.float().eval().requires_grad_(False)
    validator = WeatherHFCValidator(manifest=protocol['manifest'], args=protocol['eval_args'])
    record = dict(protocol=protocol, python=sys.executable, torch=torch.__version__,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), status='starting',
                  source_sha256={protocol[k]:sha256(protocol[k]) for k in ('weights','data','manifest','test_list')})
    record_path = validator.save_dir/'evaluation_protocol.json'
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        metrics = validator(model=model)
        record.update(status='completed', metrics={k:float(v) for k,v in metrics.items()})
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
