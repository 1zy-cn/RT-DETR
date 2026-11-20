"""Weather pairs and teacher supervision: CPU contracts, not accuracy claims."""
import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import RTDETRDetectionModel

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


def api():
    assert (ROOT / "weather_hfc_training.py").is_file(), "weather pairing/training implementation is missing"
    return importlib.import_module("weather_hfc_training")


def fixed_hyp(**updates):
    settings = {key: 0.0 for key in ("mosaic", "mixup", "copy_paste", "degrees", "translate", "scale",
                                         "shear", "perspective", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v")}
    settings.update(updates)
    return get_cfg(overrides=settings)


@pytest.fixture
def pair_files(tmp_path):
    records = []
    # Distinct colours prove that reference comes from manifest clear output,
    # even though source_image is deliberately absent from this dataset.
    for stem, condition, severity, colour, empty in [
        ("a_clear", "clear", 0, (11, 37, 203), False),
        ("a_rain", "rain", 1, (151, 23, 19), False),
        ("a_heavy", "rain", 3, (71, 33, 24), False),
        ("b_clear", "clear", 0, (12, 15, 110), True),
        ("b_fog", "fog", 2, (70, 80, 90), True),
    ]:
        image = tmp_path / "train/images" / f"{stem}.png"
        label = tmp_path / "train/labels" / f"{stem}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image), np.full((40, 80, 3), colour, dtype=np.uint8))
        label.write_text("" if empty else "0 0.25 0.75 0.2 0.3\n", encoding="utf-8")
        records.append(dict(original_id=stem[0], split="train", condition=condition, severity=severity,
                            output_image=image.relative_to(tmp_path).as_posix(),
                            source_image=f"external-original/{stem[0]}.png", width=80, height=40))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    return manifest, records


def make_dataset(pair_files, **kwargs):
    manifest, _ = pair_files
    return api().PairedWeatherDataset(manifest=manifest, img_path=str(manifest.parent / "train/images"),
                                     imgsz=64, batch_size=2, augment=True, hyp=fixed_hyp(), rect=False,
                                     cache=False, data={"names": {0: "target"}, "nc": 1}, **kwargs)


def test_manifest_clear_pair_rgb_stretch_geometry_and_empty_collate(pair_files):
    dataset = make_dataset(pair_files)
    weather = dataset[dataset.im_files.index(str(pair_files[0].parent / "train/images/a_rain.png"))]
    assert weather["original_id"] == "a" and weather["source_split"] == "train"
    assert weather["img"].shape == weather["reference_img"].shape == (3, 64, 64)
    assert weather["img"][:, 20, 20].tolist() == [19, 23, 151]
    assert weather["reference_img"][:, 20, 20].tolist() == [203, 37, 11]
    assert torch.allclose(weather["bboxes"], torch.tensor([[0.25, 0.75, 0.2, 0.3]]))
    empty = dataset[dataset.im_files.index(str(pair_files[0].parent / "train/images/b_fog.png"))]
    batch = dataset.collate_fn([weather, empty])
    assert batch["reference_img"].shape == (2, 3, 64, 64)
    assert batch["consistency_eligible"].tolist() == [True, True]
    assert batch["batch_idx"].tolist() == [0]
    assert batch["cls"].shape == (1, 1)
    assert not list(pair_files[0].parent.rglob("*.cache")), "pair loading must not modify dataset caches"


def test_clear_and_severity_three_remain_detection_only(pair_files):
    dataset = make_dataset(pair_files)
    eligibility = {Path(dataset.im_files[i]).stem: dataset[i]["consistency_eligible"] for i in range(len(dataset))}
    assert eligibility == {"a_clear": False, "a_heavy": False, "a_rain": True, "b_clear": False, "b_fog": True}


def test_student_hsv_preserves_unaugmented_native_image_cache(pair_files):
    dataset = make_dataset(pair_files)
    dataset.transforms = dataset.build_transforms(fixed_hyp(hsv_v=0.9))
    index = dataset.im_files.index(str(pair_files[0].parent / "train/images/a_rain.png"))
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        sample = dataset[index]
    finally:
        np.random.set_state(rng_state)
    assert sample["img"][:, 20, 20].tolist() != [19, 23, 151]
    assert dataset.ims[index][20, 20].tolist() == [151, 23, 19]
    assert sample["reference_img"][:, 20, 20].tolist() == [203, 37, 11]


@pytest.mark.parametrize("setting", ["mosaic", "mixup", "copy_paste", "degrees", "translate", "scale",
                                    "shear", "perspective", "fliplr", "flipud", "multi_scale"])
def test_geometry_that_breaks_pairs_is_rejected(setting):
    with pytest.raises(ValueError, match=setting):
        api().validate_fixed_geometry(fixed_hyp(**{setting: True if setting == "multi_scale" else 0.1}))


def test_pair_cannot_fall_back_to_another_split(pair_files):
    manifest, records = pair_files
    records[0]["split"] = "val"
    manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="clear|split"):
        make_dataset(pair_files)


def test_unknown_dataset_image_fails_closed(pair_files):
    manifest, _ = pair_files
    cv2.imwrite(str(manifest.parent / "train/images/unknown.png"), np.zeros((40, 80, 3), np.uint8))
    (manifest.parent / "train/labels/unknown.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        make_dataset(pair_files)


def test_corrupt_jpeg_is_rejected_without_native_in_place_repair(pair_files):
    manifest, rows = pair_files
    image = manifest.parent / "train/images/a_broken.jpg"
    cv2.imwrite(str(image), np.full((40, 80, 3), 70, np.uint8))
    truncated = image.read_bytes()[:-2]
    image.write_bytes(truncated)
    (manifest.parent / "train/labels/a_broken.txt").write_text("0 0.25 0.75 0.2 0.3\n", encoding="utf-8")
    rows.append({**rows[1], "output_image": "train/images/a_broken.jpg", "severity": 2})
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    rejected = False
    try:
        make_dataset(pair_files)
    except ValueError:
        rejected = True
    assert image.read_bytes() == truncated, "native verification must never repair source JPEG bytes"
    assert rejected


def test_region_cosine_uses_normalized_gt_roi_and_detaches_teacher():
    # Top-left and bottom-right have opposing two-channel descriptors. Only
    # the top-left quarter is a GT region; global pooling would give nonzero loss.
    student = torch.zeros(1, 2, 8, 12, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    with torch.no_grad():
        student[:, 0] = 1
        teacher[:, 0, :4, :6] = 1
        teacher[:, 1, 4:, 6:] = 1
    boxes = torch.tensor([[0.25, 0.25, 0.4, 0.4]])
    value = api().regional_cosine_loss(student, teacher, boxes, torch.tensor([0]), torch.tensor([True]))
    assert value.item() == pytest.approx(0.0, abs=1e-6)
    different = api().regional_cosine_loss(student, -teacher, boxes, torch.tensor([0]), torch.tensor([True]))
    assert different.item() == pytest.approx(2.0, abs=1e-6)
    different.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


@pytest.mark.parametrize("boxes,indices,eligible", [
    ([], [], [True]),
    ([[0.5, 0.5, 0.5, 0.5]], [0], [False]),
    ([[0.5, 0.5, 0.0, 0.5]], [0], [True]),
])
def test_no_effective_regions_returns_differentiable_zero(boxes, indices, eligible):
    student = torch.randn(1, 4, 8, 8, requires_grad=True)
    loss = api().regional_cosine_loss(student, torch.randn_like(student), torch.tensor(boxes).reshape(-1, 4),
                                      torch.tensor(indices), torch.tensor(eligible))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def basic_config(consistency=True):
    cfg = yaml.safe_load((ROOT / "ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml").read_text(encoding="utf-8"))
    cfg["nc"] = 7
    cfg["head"][-1][3][2] = 20  # 64px fixture has fewer than 300 feature positions.
    cfg["weather_hfc"] = dict(haar=False, freq=False, consistency=consistency)
    return cfg


def target_batch():
    return dict(img=torch.rand(2, 3, 64, 64), bboxes=torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
                cls=torch.tensor([[0.]]), batch_idx=torch.tensor([0]), consistency_eligible=torch.tensor([True, False]))


def test_partial_teacher_stops_before_pan_and_matches_decoder_p3():
    model = RTDETRDetectionModel(basic_config(False), nc=7, verbose=False).eval()
    image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected = api().fused_p3_features(model, image)
        full = []
        h = model.model[model.model[-1].f[0]].register_forward_hook(lambda m, x, y: full.append(y))
        model(image)
        h.remove()
    assert torch.equal(expected, full[0])
    # A raising PAN hook proves partial teacher forwarding never reaches PAN.
    def forbidden(*args):
        raise AssertionError("teacher ran beyond fused P3")
    h = model.model[model.model[-1].f[0] + 1].register_forward_hook(forbidden)
    try:
        with torch.no_grad():
            actual = api().fused_p3_features(model, image)
        assert actual.shape == (2, 256, 8, 8)
    finally:
        h.remove()


def test_native_detection_loss_four_items_backward_validation_and_no_persistent_hook():
    model = api().WeatherHFCModel(basic_config(), nc=7, verbose=False).train()
    batch = target_batch()
    batch["reference_features"] = torch.rand(2, 256, 8, 8, requires_grad=True)
    feature_node = model.model[model.model[-1].f[0]]
    before = len(feature_node._forward_hooks)
    total, items = model(batch)
    assert items.shape == (4,) and items[3].item() > 0 and torch.isfinite(total)
    total.backward()
    assert batch["reference_features"].grad is None
    assert len(feature_node._forward_hooks) == before
    assert model.model[17].conv.weight.grad.abs().sum() > 0
    model.eval()
    batch.pop("reference_features")
    with torch.no_grad():
        preds = model(batch["img"])
        loss, items = model.loss(batch, preds)
    assert torch.isfinite(loss) and items.shape == (4,) and items[3].item() == 0
    # Empty GT is a valid native detector batch and still emits four loss items.
    batch.update(bboxes=torch.zeros(0, 4), cls=torch.zeros(0, 1), batch_idx=torch.zeros(0))
    with torch.no_grad():
        loss, items = model.loss(batch)
    assert torch.isfinite(loss) and items.shape == (4,)


def test_trainer_initialization_reference_freeze_and_student_checkpoint(tmp_path):
    teacher = RTDETRDetectionModel(basic_config(False), nc=7, verbose=False).eval()
    checkpoint = tmp_path / "original.pt"
    torch.save({"model": teacher}, checkpoint)
    # Real get_model/preprocess_batch methods with only native trainer directory
    # and dataset setup omitted; these methods own all teacher lifecycle effects.
    trainer = api().WeatherHFCTrainer.__new__(api().WeatherHFCTrainer)
    trainer.teacher_checkpoint = checkpoint
    trainer.consistency_weight = 0.1
    trainer.teacher = None
    trainer.args = fixed_hyp()
    trainer.args.pretrained = str(checkpoint)
    trainer.data = {"nc": 7, "names": [str(i) for i in range(7)]}
    trainer.device = torch.device("cpu")
    trainer.model = trainer.get_model(basic_config(), verbose=False).train()
    batch = target_batch()
    batch["img"] = (batch["img"] * 255).byte()
    batch["reference_img"] = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    result = trainer.preprocess_batch(batch)
    assert result["reference_features"].shape == (2, 256, 8, 8)
    assert not result["reference_features"].requires_grad
    assert not trainer.teacher.training
    assert all(not p.requires_grad and p.grad is None for p in trainer.teacher.parameters())
    assert not ({id(p) for p in trainer.teacher.parameters()} & {id(p) for p in trainer.model.parameters()})
    assert torch.equal(trainer.model.model[0].conv.weight, teacher.model[0].conv.weight)
    clone = deepcopy(trainer.model)
    assert not any("teacher" in key for key in clone.state_dict())
    assert not any(m is trainer.teacher for m in clone.modules())


def test_consistency_requires_explicit_checkpoint_without_random_fallback():
    trainer = api().WeatherHFCTrainer.__new__(api().WeatherHFCTrainer)
    trainer.teacher_checkpoint = None
    trainer.consistency_weight = 0.1
    trainer.teacher = None
    trainer.args = SimpleNamespace(pretrained=False)
    trainer.data = {"nc": 7, "names": [str(i) for i in range(7)]}
    with pytest.raises(ValueError, match="teacher|checkpoint"):
        trainer.get_model(basic_config(), verbose=False)


def test_training_cannot_silently_drop_eligible_consistency_and_hooks_cleanup_on_error():
    model = api().WeatherHFCModel(basic_config(), nc=7, verbose=False).train()
    batch = target_batch()
    with pytest.raises(ValueError, match="reference"):
        model(batch)
    feature_node = model.model[model.model[-1].f[0]]
    before = len(feature_node._forward_hooks)
    batch["reference_features"] = torch.rand(2, 256, 7, 7)
    with pytest.raises(ValueError, match="identical"):
        model(batch)
    assert len(feature_node._forward_hooks) == before


def test_final_validator_loader_preserves_manifest_and_source_dataset(pair_files):
    manifest, records = pair_files
    for row in records:
        row["split"] = "valid"
    manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    args = vars(fixed_hyp()).copy()
    args["imgsz"] = 64
    validator = api().WeatherHFCValidator(manifest=manifest, args=args)
    validator.data = {"names": {0: "target"}, "nc": 1}
    dataset = validator.build_dataset(str(manifest.parent / "train/images"), batch=2)
    assert isinstance(dataset, api().PairedWeatherDataset)
    assert dataset[0]["source_split"] == "valid"
    assert "reference_img" not in dataset[0]
    assert not list(manifest.parent.rglob("*.cache"))


def test_native_trainer_val_factory_maps_to_manifest_valid(pair_files):
    manifest, records = pair_files
    for row in records:
        row["split"] = "valid"
    manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    trainer = api().WeatherHFCTrainer.__new__(api().WeatherHFCTrainer)
    trainer.manifest = manifest
    trainer.args = fixed_hyp()
    trainer.args.imgsz = 64
    trainer.model = SimpleNamespace(consistency_enabled=False)
    trainer.data = {"nc": 7, "names": [str(i) for i in range(7)]}
    dataset = trainer.build_dataset(str(manifest.parent / "train/images"), mode="val", batch=2)
    assert dataset[0]["source_split"] == "valid"


@pytest.mark.parametrize("nc,names", [(6, ["target"] * 6), (7, ["target"] * 6), (6, ["target"] * 7)])
@pytest.mark.parametrize("factory", ["get_model", "build_dataset"])
def test_direct_trainer_factories_reject_nonseven_class_data(nc, names, factory):
    trainer = api().WeatherHFCTrainer.__new__(api().WeatherHFCTrainer)
    trainer.data = {"nc": nc, "names": names}
    trainer.args = fixed_hyp()
    trainer.consistency_weight = 0.1
    trainer.teacher_checkpoint = None
    trainer.manifest = Path("unused.jsonl")
    trainer.model = SimpleNamespace(consistency_enabled=False)
    with pytest.raises(ValueError, match="seven|7"):
        if factory == "get_model":
            trainer.get_model(basic_config(False), verbose=False)
        else:
            trainer.build_dataset("unused", batch=2)


def test_existing_numpy_sidecar_is_preserved_and_rejected(pair_files):
    manifest, _ = pair_files
    sidecar = manifest.parent / "train/images/a_rain.npy"
    original = b"invalid numpy cache must never be deleted"
    sidecar.write_bytes(original)
    with pytest.raises(ValueError, match="npy|sidecar"):
        dataset = make_dataset(pair_files)
        dataset[dataset.im_files.index(str(sidecar.with_suffix(".png")))]
    assert sidecar.read_bytes() == original
