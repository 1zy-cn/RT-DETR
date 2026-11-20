"""Preview or run the opt-in Chapter 4 weather H/F/C training protocol."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml
from weather_hfc_protocol import freeze_inputs, resolve_resume, sha256

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("D:/Users/86186/ResearchData/SVRDD-weather-v1")
BASELINE_WEIGHTS = Path("D:/Users/86186/ResearchData/SVRDD-local-runs/svrdd-r18-baseline-officialsplit-local-amp-b8-p20-s0/weights/best.pt")
CONFIG_DIR = ROOT / "ultralytics/cfg/models/rt-detr/weather-hfc"
CONFIGURATIONS = ("baseline", "h", "f", "hf", "c", "hc", "fc", "hfc")

def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configuration", choices=CONFIGURATIONS, default="hfc")
    p.add_argument("--data", type=Path, default=ROOT / "configs/chapter4/dataset.yaml")
    p.add_argument("--manifest", type=Path, default=DATA_ROOT / "manifest.jsonl")
    p.add_argument("--weights", type=Path, default=BASELINE_WEIGHTS)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--consistency-weight", type=float, default=0.1)
    p.add_argument("--project", type=Path, default=ROOT / "runs/weather_hfc")
    p.add_argument("--name", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="0")
    p.add_argument("--amp", action="store_true", help="Enable AMP; FP32 is the initial verification default.")
    p.add_argument("--resume", type=Path, help="Resume with the original frozen weather_protocol.json.")
    p.add_argument("--train", action="store_true", help="Start training. Otherwise only print the resolved protocol.")
    return p

def build_protocol(args, config_dir=None):
    if getattr(args, "resume", None) is not None:
        return resolve_resume(args)
    config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    if args.batch < 1 or args.epochs < 1 or args.workers < 0:
        raise ValueError("batch/epochs must be positive and workers nonnegative")
    if args.imgsz < 128 or args.imgsz % 32:
        raise ValueError("imgsz must be a multiple of 32 and at least 128 for the fixed 300-query decoder")
    if not 0 <= args.consistency_weight < float("inf"):
        raise ValueError("consistency-weight must be finite and nonnegative")
    if args.device != "cpu" and not args.device.isdecimal():
        raise ValueError("Only cpu or one local GPU index is supported by this paired trainer")
    name = args.name or f"{args.configuration}_seed{args.seed}"
    if Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("name must be one directory name")
    paths = dict(model=(config_dir / f"{args.configuration}.yaml").resolve(),
                 data=args.data.resolve(), manifest=args.manifest.resolve(), weights=args.weights.resolve())
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{key} does not exist: {path}")
    cfg = yaml.safe_load(paths["model"].read_text(encoding="utf-8"))
    expected = dict(haar="h" in args.configuration, freq="f" in args.configuration,
                    consistency="c" in args.configuration)
    if cfg.get("nc") != 7 or cfg.get("weather_hfc") != expected:
        raise ValueError(f"Model factor metadata or class count is inconsistent: {paths['model']}")
    data_cfg = yaml.safe_load(paths["data"].read_text(encoding="utf-8"))
    names = data_cfg.get("names", {}) if isinstance(data_cfg, dict) else {}
    if (not isinstance(data_cfg, dict) or data_cfg.get("nc") != 7
            or not isinstance(names, (dict, list)) or len(names) != 7
            or (isinstance(names, dict) and set(names) != set(range(7)))):
        raise ValueError("Weather HFC data must declare nc=7 and seven names with IDs 0..6")
    teacher = (args.teacher or args.weights).resolve() if expected["consistency"] else None
    if teacher is not None and not teacher.is_file():
        raise FileNotFoundError(f"teacher checkpoint does not exist: {teacher}")
    project = args.project.resolve()
    if (project / name).exists():
        raise FileExistsError(f"Run directory exists; choose a new --name: {project / name}")
    train_args = dict(
        model=str(paths["model"]), data=str(paths["data"]), pretrained=str(paths["weights"]),
        project=str(project), name=name, imgsz=args.imgsz, batch=args.batch,
        epochs=args.epochs, workers=args.workers, device=args.device, seed=args.seed,
        optimizer="AdamW", lr0=1e-4, lrf=0.1, weight_decay=1e-4, momentum=0.9,
        warmup_epochs=3.0, warmup_bias_lr=0.0, patience=20, nbs=args.batch,
        amp=args.amp, half=False, deterministic=False, multi_scale=False, rect=False,
        cache=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
        conf=0.001, max_det=300, cos_lr=False,
        plots=False, save=True, val=True, exist_ok=False,
    )
    return dict(configuration=args.configuration, **{k:str(v) for k,v in paths.items()},
                teacher_checkpoint=str(teacher) if teacher else None,
                consistency_weight=args.consistency_weight, train_args=train_args)

def validate_resume_before_init(checkpoint, protocol):
    """Apply trainer state checks before its constructor can write args.yaml."""
    from types import SimpleNamespace
    from weather_hfc_training import WeatherHFCModel, WeatherHFCTrainer
    probe = WeatherHFCTrainer.__new__(WeatherHFCTrainer)
    probe.args = SimpleNamespace(**protocol['train_args'])
    probe.epochs = probe.args.epochs
    probe.consistency_weight = protocol['consistency_weight']
    probe._resume_runtime(checkpoint)
    model = checkpoint.get('model')
    if not isinstance(model, WeatherHFCModel) or model.nc != 7:
        raise ValueError('Resume requires a raw seven-class WeatherHFCModel')
    if not isinstance(checkpoint.get('ema'), WeatherHFCModel):
        raise ValueError('Resume requires the separate student EMA')
    if float(model.consistency_weight) != probe.consistency_weight:
        raise ValueError('Resume raw student consistency weight differs from the saved protocol')
    expected = yaml.safe_load(Path(protocol['model']).read_text(encoding='utf-8'))['weather_hfc']
    if model.yaml.get('weather_hfc') != expected:
        raise ValueError('Resume student configuration differs from the saved model factors')


def main(argv=None):
    parser = make_parser()
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.resume:
        forbidden = [s for s in argv if s.startswith('--') and s.split('=')[0] not in ('--resume','--train')]
        if forbidden:
            parser.error('Resume restores the saved experiment; remove fresh settings: '+', '.join(forbidden))
    try:
        protocol = build_protocol(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    if not args.train:
        print("Preview only. Add --train to start this protocol; full experiments have not been launched.")
        return
    import torch
    from weather_hfc_training import WeatherHFCTrainer
    if args.resume:
        checkpoint = torch.load(protocol['resume_checkpoint'], map_location='cpu', weights_only=False)
        validate_resume_before_init(checkpoint, protocol)
        del checkpoint
        record = json.loads(Path(protocol['resume_record']).read_text(encoding='utf-8'))
        record.setdefault('resume_history', []).append(dict(
            checkpoint=protocol['resume_checkpoint'], checkpoint_sha256=sha256(protocol['resume_checkpoint']),
            resumed_at_utc=datetime.now(timezone.utc).isoformat()))
        record.update(protocol=protocol, status='starting')
    else:
        record = dict(protocol=protocol, source_sha256=freeze_inputs(protocol),
                      python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                      created_at_utc=datetime.now(timezone.utc).isoformat(),
                      scope='Chapter4 configured protocol; not experimental efficacy evidence',
                      status='starting', consistency_severities=[1,2], random_geometry=False)
    trainer = WeatherHFCTrainer(
        overrides=protocol["train_args"], manifest=Path(protocol["manifest"]),
        teacher_checkpoint=Path(protocol["teacher_checkpoint"]) if protocol["teacher_checkpoint"] else None,
        consistency_weight=protocol["consistency_weight"],
    )
    record_path = trainer.save_dir / "weather_protocol.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        trainer.train()
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        record["status"] = "completed"
    finally:
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

