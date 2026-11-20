"""Isolated, real-train-pair, one-update H/F/C technical readiness probe.

This is deliberately not a trainer.train() run or an efficacy experiment. Each
configuration starts anew from the same explicit original-image checkpoint.
Only copies of two eligible training pairs are scanned; validation/test samples,
metrics, formal run directories and checkpoint saving are never used.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_fixture(source, manifest, train_list, data_yaml, destination):
    """Copy two distinct nonempty-GT training groups, one severity 1 and one 2."""
    import yaml

    allowed = {str(Path(line.strip()).resolve()).casefold()
               for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()}
    clears, candidates = {}, []
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != "train":
                continue
            if row["condition"] == "clear":
                clears[row["original_id"]] = row
            elif int(row["severity"]) in (1, 2):
                image = source / row["output_image"]
                if str(image.resolve()).casefold() in allowed:
                    candidates.append(row)
    chosen, identities = [], set()
    for severity in (1, 2):
        for row in candidates:
            if int(row["severity"]) != severity or row["original_id"] in identities:
                continue
            label = source / row["output_label"]
            if row["original_id"] in clears and label.read_text(encoding="utf-8").strip():
                chosen.append(row)
                identities.add(row["original_id"])
                break
        else:
            raise ValueError(f"No distinct nonempty severity-{severity} pair in the training list")
    rows = [item for weather in chosen for item in (clears[weather["original_id"]], weather)]
    destination.mkdir(parents=True, exist_ok=False)
    fingerprints = []
    for row in rows:
        for key, expected_key in (("output_image", "output_sha256"), ("output_label", "label_sha256")):
            origin = (source / row[key]).resolve()
            target = (destination / row[key]).resolve()
            if not origin.is_relative_to(source) or not target.is_relative_to(destination):
                raise ValueError("Manifest output path escapes its declared root")
            before = digest(origin)
            if before != row[expected_key]:
                raise AssertionError(f"Manifest fingerprint mismatch: {origin}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            if digest(target) != before:
                raise AssertionError(f"Fixture copy mismatch: {target}")
            fingerprints.append(dict(path=str(origin), sha256=before, copy=str(target)))
    fixture_manifest = destination / "manifest.jsonl"
    fixture_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    fixture_list = destination / "train.txt"
    fixture_list.write_text("".join(str(destination / row["output_image"]) + "\n" for row in chosen), encoding="utf-8")
    original_data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data = dict(path=str(destination), train=str(fixture_list), nc=7, names=original_data["names"])
    fixture_yaml = destination / "data.yaml"
    fixture_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dict(data=data, data_yaml=fixture_yaml, manifest=fixture_manifest, train_list=fixture_list,
                selected=chosen, source_fingerprints=fingerprints)


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_configuration(name, args, fixture, device):
    """Exercise native model loading, paired transforms, teacher, loss and AdamW."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.models.rtdetr.val import RTDETRDataset
    from ultralytics.nn.Addmodules.WeatherHFC import HaarResNetStage, WeatherFrequencySelection
    from train_weather_hfc import CONFIG_DIR, build_protocol, make_parser
    from weather_hfc_training import PairedWeatherDataset, WeatherHFCTrainer, regional_cosine_loss

    class IsolatedPairDataset(PairedWeatherDataset):
        # Native paired cache misses normally use ROOT/runs/weather_hfc/cache_scan.
        # Here every image is a verified copy, so native scanning can safely write
        # its cache beside those copies, entirely inside this technical output.
        def cache_labels(self, path):
            if not Path(path).resolve().is_relative_to(args.work_dir / "train_pair_fixture"):
                raise ValueError("Probe cache escaped the isolated fixture")
            return RTDETRDataset.cache_labels(self, path)

    seed_everything(args.seed)
    started = time.perf_counter()
    namespace = make_parser().parse_args([
        "--configuration", name, "--data", str(fixture["data_yaml"]),
        "--manifest", str(fixture["manifest"]), "--weights", str(args.weights),
        "--project", str(args.work_dir / "unused_training_project"), "--name", name,
        "--epochs", "1", "--batch", "2", "--imgsz", "640", "--workers", "0",
        "--seed", str(args.seed), "--device", args.device,
    ])
    protocol = build_protocol(namespace)
    # Construct only the attributes required by the native methods below. The
    # trainer constructor/_setup_train would write run files and create a val
    # loader; neither is needed for this bounded forward/backward/update check.
    trainer = WeatherHFCTrainer.__new__(WeatherHFCTrainer)
    trainer.args = get_cfg(overrides=protocol["train_args"])
    trainer.data = fixture["data"]
    trainer.device = device
    trainer.manifest = fixture["manifest"]
    trainer.teacher_checkpoint = Path(protocol["teacher_checkpoint"]) if protocol["teacher_checkpoint"] else None
    trainer.consistency_weight = protocol["consistency_weight"]
    trainer.teacher = None
    source = trainer._load_local_model(args.weights)
    model = trainer.get_model(cfg=str(CONFIG_DIR / f"{name}.yaml"), weights=source, verbose=False)
    state = model.state_dict()
    common = [(key, tensor) for key, tensor in source.state_dict().items()
              if key in state and tensor.shape == state[key].shape]
    assert len(common) == model.initialization_transferred_keys
    assert all(torch.equal(state[key], value) for key, value in common)
    transferred = len(common)
    del source, state, common
    trainer.model = model.to(device=device, dtype=torch.float32).train()
    trainer.set_model_attributes()
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    dataset = IsolatedPairDataset(manifest=trainer.manifest, split="train", load_reference=model.consistency_enabled,
        img_path=str(fixture["train_list"]), imgsz=640, batch_size=2, augment=True, hyp=trainer.args,
        rect=False, cache=None, prefix=f"readiness/{name}: ", data=trainer.data)
    assert len(dataset) == 2
    raw = dataset.collate_fn([dataset[i] for i in range(2)])
    assert raw["consistency_eligible"].all() and set(raw["source_split"]) == {"train"}
    assert set(int(value) for value in raw["severity"]) == {1, 2}
    image_sha = hashlib.sha256(raw["img"].numpy().tobytes()).hexdigest()
    gt_counts = [(raw["batch_idx"] == index).sum().item() for index in range(2)]
    assert all(count > 0 for count in gt_counts)
    teacher = trainer.teacher
    teacher_state = {key: tensor.detach().cpu().clone() for key, tensor in teacher.state_dict().items()} if teacher else None
    torch.cuda.reset_peak_memory_stats(device)
    batch = trainer.preprocess_batch(raw)
    assert batch["img"].shape == (2, 3, 640, 640) and batch["img"].dtype == torch.float32
    if model.consistency_enabled:
        assert teacher is not None and not teacher.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
        assert batch["reference_features"].shape == (2, 256, 80, 80)
        assert not batch["reference_features"].requires_grad
    else:
        assert teacher is None and "reference_features" not in batch
    optimizer = trainer.build_optimizer(model, name="AdamW", lr=1e-4, momentum=0.9, decay=1e-4)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert teacher is None or optimizer_ids.isdisjoint(id(parameter) for parameter in teacher.parameters())
    assert not any("teacher" in key or "reference" in key for key in model.state_dict())
    selected = {"backbone": next(model.parameters())}
    haar = next((module for module in model.modules() if isinstance(module, HaarResNetStage)), None)
    frequency = next((module for module in model.modules() if isinstance(module, WeatherFrequencySelection)), None)
    if haar is not None:
        selected["haar"] = next(haar.parameters())
    if frequency is not None:
        selected["frequency"] = frequency.freq_weight_conv_list[0].weight
    before = {key: parameter.detach().clone() for key, parameter in selected.items()}
    capture = {}
    hook = model.model[model.model[-1].f[0]].register_forward_hook(
        lambda module, inputs, output: capture.update(p3=output)) if model.consistency_enabled else None
    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            loss, items = model(batch)
    finally:
        if hook is not None:
            hook.remove()
    assert loss.ndim == 0 and torch.isfinite(loss) and items.numel() == 4 and torch.isfinite(items).all()
    region_info = None
    if model.consistency_enabled:
        region = regional_cosine_loss(capture["p3"], batch["reference_features"], batch["bboxes"],
                                      batch["batch_idx"], eligible=batch["consistency_eligible"])
        assert torch.isfinite(region) and region > 0 and items[-1] > 0
        torch.testing.assert_close(region.detach() * trainer.consistency_weight, items[-1])
        region_gradients = torch.autograd.grad(region, tuple(selected.values()), retain_graph=True)
        region_norms = {key: float(gradient.norm().item()) for key, gradient in zip(selected, region_gradients)}
        assert all(torch.isfinite(gradient).all() and gradient.norm() > 0 for gradient in region_gradients)
        region_info = dict(unweighted_loss=float(region.detach()), weighted_loss=float(items[-1]),
                           gradient_norms=region_norms, fused_p3_shape=list(capture["p3"].shape),
                           teacher_reference_detached=True)
        del region_gradients, region
    else:
        assert float(items[-1]) == 0.0
    loss.backward()
    with_grad = [parameter for parameter in model.parameters() if parameter.grad is not None]
    assert with_grad and all(torch.isfinite(parameter.grad).all() for parameter in with_grad)
    branch_gradients = {key: float(parameter.grad.norm()) for key, parameter in selected.items()}
    assert all(value > 0 for value in branch_gradients.values())
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    assert torch.isfinite(total_norm)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    updates = {key: float((parameter.detach() - before[key]).abs().max()) for key, parameter in selected.items()}
    assert all(value > 0 for value in updates.values())
    if teacher is not None:
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())
        assert all(torch.equal(value, teacher.state_dict()[key].detach().cpu()) for key, value in teacher_state.items())
    torch.cuda.synchronize(device)
    result = dict(configuration=name, passed=True, initialization_transferred_keys=transferred,
        all_matching_initialization_tensors_equal=True, parameters=sum(parameter.numel() for parameter in model.parameters()),
        student_batch_sha256=image_sha, gt_per_image=gt_counts, total_loss=float(loss.detach()),
        detection_total_loss=float(loss.detach() - items[-1]),
        logged_loss_items=dict(zip(("giou", "classification", "l1", "weighted_region"), items.detach().cpu().tolist())),
        consistency=region_info, finite_gradient_tensors=len(with_grad), gradient_norm_before_clip=float(total_norm),
        selected_gradient_norms=branch_gradients, selected_parameter_max_abs_updates=updates,
        optimizer="AdamW", optimizer_steps=1, optimizer_learning_rate=1e-4, optimizer_updated=True,
        teacher_frozen=True if teacher is not None else None, teacher_all_state_unchanged=True if teacher is not None else None,
        teacher_excluded_from_optimizer_and_student=True, precision="float32", amp=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        elapsed_seconds=time.perf_counter() - started)
    del optimizer, model, trainer, dataset, batch, raw, teacher, teacher_state, capture, selected, before, with_grad
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv=None):
    from train_weather_hfc import BASELINE_WEIGHTS, CONFIGURATIONS, DATA_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="New isolated technical output directory")
    parser.add_argument("--source-data", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--train-list", type=Path, default=ROOT / "configs/chapter4/train_mixed.txt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/chapter4/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=BASELINE_WEIGHTS)
    parser.add_argument("--device", default="0", help="Single CUDA device index; FP32 only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    args = parser.parse_args(argv)
    for key in ("work_dir", "source_data", "train_list", "data", "weights"):
        setattr(args, key, getattr(args, key).resolve())
    args.manifest = args.manifest.resolve() if args.manifest else args.source_data / "manifest.jsonl"
    if not args.device.isdecimal():
        parser.error("This readiness probe requires a single CUDA device index")
    if args.work_dir.is_relative_to(ROOT / "runs") or args.work_dir.is_relative_to(args.source_data):
        parser.error("work-dir must be outside project runs and source data")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this readiness probe")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    protected = [args.weights, args.manifest, args.train_list, args.data,
                 Path(__file__).resolve(), ROOT / "weather_hfc_training.py", ROOT / "train_weather_hfc.py",
                 ROOT / "ultralytics/nn/Addmodules/WeatherHFC.py"]
    protected += [ROOT / f"ultralytics/cfg/models/rt-detr/weather-hfc/{name}.yaml" for name in args.configurations]
    source_hashes = {str(path): digest(path) for path in protected}
    fixture = prepare_fixture(args.source_data, args.manifest, args.train_list, args.data, args.work_dir / "train_pair_fixture")
    result = dict(scope="Technical single-update readiness only; not a formal experiment or efficacy evidence",
        created_at_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable, torch=torch.__version__,
        cuda=torch.version.cuda, device=torch.cuda.get_device_name(device), batch=2, imgsz=640, precision="float32",
        amp=False, seed=args.seed, tf32=False, full_train_called=False, validation_or_test_evaluated=False,
        checkpoint_saved=False, source_fingerprints=source_hashes, selected_train_pairs=fixture["selected"],
        source_copy_provenance=fixture["source_fingerprints"], configurations=[], all_passed=False)
    output = args.work_dir / "readiness.json"
    save_json(output, result)
    for name in args.configurations:
        try:
            row = one_configuration(name, args, fixture, device)
        except Exception as exc:
            row = dict(configuration=name, passed=False, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            gc.collect()
            torch.cuda.empty_cache()
        result["configurations"].append(row)
        save_json(output, result)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result["protected_inputs_unchanged"] = all(digest(path) == value for path, value in source_hashes.items())
    result["source_pair_files_unchanged"] = all(digest(row["path"]) == row["sha256"] for row in fixture["source_fingerprints"])
    result["same_student_batch_all_configurations"] = len({row.get("student_batch_sha256") for row in result["configurations"] if row["passed"]}) == 1
    result["all_passed"] = all(row["passed"] for row in result["configurations"]) and result["protected_inputs_unchanged"] and result["source_pair_files_unchanged"] and result["same_student_batch_all_configurations"]
    save_json(output, result)
    print(f"READINESS_RESULT {output} all_passed={result['all_passed']}", flush=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
