"""Opt-in paired-weather RT-DETR training; checkpoints contain only the student.

Regional cosine supervision adapts the same-source idea of WARLearn (WACV
2025), rather than reproducing its Barlow Twins objective. Geometry is fixed
for the initial experiment; teacher inputs receive no HSV augmentation.
"""
import json
import math
import os
import random
import tempfile
from copy import copy, deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, distributed
from torchvision.ops import roi_align

from ultralytics.data.augment import Compose, Format, RandomHSV
from ultralytics.data.build import seed_worker
from ultralytics.data.utils import PIN_MEMORY, check_det_dataset
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel, attempt_load_one_weight
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


GEOMETRY_SETTINGS = ("mosaic", "mixup", "copy_paste", "degrees", "translate", "scale",
                     "shear", "perspective", "fliplr", "flipud")


def validate_fixed_geometry(hyp):
    """Reject any transformation not supported by the initial pair protocol."""
    for key in (*GEOMETRY_SETTINGS, "multi_scale"):
        value = hyp.get(key, 0) if isinstance(hyp, dict) else getattr(hyp, key, 0)
        if value != 0:
            raise ValueError(f"Weather HFC requires {key}=0 for identical fixed pair geometry; got {value!r}")


def _path_key(path):
    return str(Path(path).resolve()).casefold()


def _validate_seven_classes(data):
    names = data.get("names")
    if data.get("nc") != 7 or not isinstance(names, (list, tuple, dict)) or len(names) != 7:
        raise ValueError("Weather HFC requires nc=7 and exactly seven class names")


class PairedWeatherDataset(RTDETRDataset):
    """Native square-stretched RGB samples paired by manifest identity and split.

    Optional transforms are intentionally limited to student HSV and native
    Format: even installed Albumentations packages cannot change this protocol.
    Cache misses are scanned through a temporary cache without editing data.
    """

    def __init__(self, *args, manifest, split=None, load_reference=True, **kwargs):
        self.manifest = Path(manifest).resolve()
        self.manifest_root = self.manifest.parent
        self.source_split = split
        self.load_reference = bool(load_reference)
        if kwargs.get("rect", False):
            raise ValueError("PairedWeatherDataset requires rect=False and square-stretched geometry")
        if kwargs.get("cache") == "disk":
            raise ValueError("Weather HFC disallows cache='disk' to preserve the source dataset")
        validate_fixed_geometry(kwargs.get("hyp", {}))
        self.manifest_rows, self.clear_rows = {}, {}
        with self.manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = {"output_image", "original_id", "split", "condition", "severity", "source_image"}
                if required - row.keys():
                    raise ValueError(f"Manifest line {line_number} lacks {sorted(required - row.keys())}")
                row = dict(row)
                row["severity"] = int(row["severity"])
                identity = (row["split"], row["original_id"])
                key = _path_key(self.manifest_root / row["output_image"])
                if key in self.manifest_rows:
                    raise ValueError(f"Duplicate output_image in manifest: {row['output_image']}")
                self.manifest_rows[key] = row
                if row["condition"] == "clear":
                    if row["severity"] != 0 or identity in self.clear_rows:
                        raise ValueError(f"Invalid or duplicate clear manifest entry: {identity}")
                    self.clear_rows[identity] = row
        super().__init__(*args, **kwargs)
        self.pair_rows = [self._pair_for(path) for path in self.im_files]

    def get_img_files(self, img_path):
        files = super().get_img_files(img_path)
        # Fail before any label/cache work for unknown files or missing pairs.
        for path in files:
            self._pair_for(path)
        return files

    def _pair_for(self, image_path):
        row = self.manifest_rows.get(_path_key(image_path))
        if row is None:
            raise ValueError(f"Dataset image is absent from manifest: {image_path}")
        if self.source_split is not None and row["split"] != self.source_split:
            raise ValueError(f"Manifest split {row['split']!r} does not match {self.source_split!r}: {image_path}")
        identity = (row["split"], row["original_id"])
        clear = self.clear_rows.get(identity)
        if clear is None:
            raise ValueError(f"Missing same-split clear manifest pair for {identity}")
        for key in ("source_image", "width", "height", "label_sha256"):
            if key in row and key in clear and row[key] != clear[key]:
                raise ValueError(f"Manifest pair {identity} disagrees on {key}")
        reference = self.manifest_root / clear["output_image"]
        if self.load_reference and not reference.is_file():
            raise FileNotFoundError(f"Clear manifest reference does not exist: {reference}")
        return row, clear

    def cache_labels(self, path=Path("./labels.cache")):
        # Native verify_image_label may repair JPEGs in place. Refuse that path
        # before scanning: dataset bytes belong to the fixed input protocol.
        for filename in self.im_files:
            if Path(filename).suffix.lower() in (".jpg", ".jpeg"):
                with open(filename, "rb") as image:
                    image.seek(-2, 2)
                    if image.read() != b"\xff\xd9":
                        raise ValueError(f"Corrupt JPEG requires separate repair; source left unchanged: {filename}")
        cache_root = Path(__file__).resolve().parent / "runs/weather_hfc/cache_scan"
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="labels-", dir=cache_root) as directory:
            return super().cache_labels(Path(directory) / "labels.cache")

    def build_transforms(self, hyp=None):
        validate_fixed_geometry(hyp)
        transforms = []
        if self.augment:
            transforms.append(RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v))
        transforms.append(Format(bbox_format="xywh", normalize=True, batch_idx=True))
        return Compose(transforms)

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        # RandomHSV writes in-place; keep native cached bytes pristine across
        # epochs when fixed geometry bypasses RandomPerspective's image copy.
        label["img"] = label["img"].copy()
        return label

    def load_image(self, i, rect_mode=False):
        sidecar = Path(self.im_files[i]).with_suffix(".npy")
        if sidecar.exists():
            raise ValueError(f"Existing .npy sidecar is unsupported and left unchanged: {sidecar}")
        return super().load_image(i=i, rect_mode=rect_mode)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row, clear = self.pair_rows[index]
        reference_path = self.manifest_root / clear["output_image"]
        sample.update(original_id=row["original_id"], condition=row["condition"], severity=row["severity"],
                      source_split=row["split"], reference_file=str(reference_path),
                      consistency_eligible=row["condition"] != "clear" and row["severity"] in (1, 2))
        # Native Format emits shape (0,) for empty cls, whereas nonempty cls is
        # (N,1). Normalize at our boundary so mixed background batches collate.
        sample["cls"] = sample["cls"].reshape(-1, 1)
        if self.load_reference:
            reference = cv2.imread(str(reference_path))
            if reference is None:
                raise FileNotFoundError(f"Cannot read clear reference: {reference_path}")
            if reference.shape[:2] != tuple(sample["ori_shape"]):
                raise ValueError(f"Weather/clear source dimensions differ: {row['original_id']}")
            reference = cv2.resize(reference, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            sample["reference_img"] = torch.from_numpy(np.ascontiguousarray(reference.transpose(2, 0, 1)[::-1]))
        return sample

    @staticmethod
    def collate_fn(batch):
        result = RTDETRDataset.collate_fn(batch)
        if "reference_img" in result:
            result["reference_img"] = torch.stack(result["reference_img"])
        result["consistency_eligible"] = torch.tensor(result["consistency_eligible"], dtype=torch.bool)
        return result


def fused_p3_features(model, images):
    """Forward only through the fused P3 node consumed by the decoder.

    This uses graph edges, not a fixed layer number. Teacher PAN and decoder
    never execute, reducing the temporary activation footprint.
    """
    model = de_parallel(model)
    node = model.model[-1].f[0]
    if not isinstance(node, int) or node < 0:
        raise ValueError("Expected an explicit fused-P3 decoder source index")
    outputs = []
    x = images
    for module in model.model[:-1]:
        if module.f != -1:
            x = outputs[module.f] if isinstance(module.f, int) else [x if j == -1 else outputs[j] for j in module.f]
        if hasattr(module, "backbone"):
            raise ValueError("Weather HFC requires the explicit RT-DETR-R18 graph")
        x = module(x)
        outputs.append(x if module.i in model.save else None)
        if module.i == node:
            if x.ndim != 4 or x.shape[1] != 256:
                raise ValueError(f"Fused P3 must be BCHW with 256 channels; got {tuple(x.shape)}")
            return x
    raise ValueError(f"Decoder fused-P3 source {node} was not found")


def regional_cosine_loss(student_features, reference_features, bboxes, batch_idx, eligible=None, output_size=3):
    """Average 1-cosine for GT xywh ROIs, with a detached reference and 3x3 grid."""
    if student_features.ndim != 4 or student_features.shape != reference_features.shape:
        raise ValueError("Student/reference fused features must have identical BCHW shapes")
    device = student_features.device
    boxes = bboxes.to(device=device, dtype=torch.float32).reshape(-1, 4)
    indices = batch_idx.to(device=device, dtype=torch.long).reshape(-1)
    if len(boxes) != len(indices):
        raise ValueError("One batch index is required for every GT box")
    if ((indices < 0) | (indices >= len(student_features))).any():
        raise ValueError("GT batch index lies outside feature batch")
    valid = torch.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
    if eligible is not None:
        eligible = torch.as_tensor(eligible, device=device, dtype=torch.bool).reshape(-1)
        if len(eligible) != len(student_features):
            raise ValueError("One eligibility flag is required for every image")
        valid &= eligible[indices]
    corners = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1).clamp(0, 1)
    valid &= (corners[:, 2:] > corners[:, :2]).all(1)
    if not valid.any():
        return student_features.sum() * 0.0
    height, width = student_features.shape[-2:]
    scaled = corners[valid] * corners.new_tensor([width, height, width, height])
    rois = torch.cat((indices[valid, None].float(), scaled), dim=1)
    # FP32 ROI/cosine avoids low-precision cancellation and works under AMP.
    student = roi_align(student_features.float(), rois, output_size=output_size, aligned=True).flatten(1)
    reference = roi_align(reference_features.detach().to(device=device, dtype=torch.float32), rois,
                          output_size=output_size, aligned=True).flatten(1)
    return (1.0 - F.cosine_similarity(student, reference, dim=1, eps=1e-8)).mean()


class WeatherHFCModel(RTDETRDetectionModel):
    """Native detection criterion plus a train-only GT-region cosine objective."""

    def __init__(self, cfg, nc=7, verbose=False):
        super().__init__(cfg=cfg, nc=nc, verbose=verbose)
        self.nc = self.yaml["nc"]
        self.consistency_enabled = bool(self.yaml.get("weather_hfc", {}).get("consistency", False))
        self.consistency_weight = 0.1

    def loss(self, batch, preds=None):
        captured = []
        reference = batch.get("reference_features")
        if self.training and self.consistency_enabled and preds is None:
            if "consistency_eligible" not in batch:
                raise ValueError("Consistency training requires manifest eligibility metadata")
            eligible = torch.as_tensor(batch["consistency_eligible"], dtype=torch.bool)
            indices = batch["batch_idx"].to(device=eligible.device, dtype=torch.long)
            valid_boxes = torch.isfinite(batch["bboxes"]).all(1) & (batch["bboxes"][:, 2:] > 0).all(1)
            needs_reference = (eligible[indices] & valid_boxes.to(eligible.device)).any()
            if needs_reference and reference is None:
                raise ValueError("Eligible consistency targets require detached teacher reference_features")
        use_region = self.training and self.consistency_enabled and reference is not None and preds is None
        hook = None
        if use_region:
            node = self.model[-1].f[0]
            hook = self.model[node].register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            detection, items = super().loss(batch, preds)
            region = detection.new_zeros(())
            if use_region:
                if not captured or captured[0].shape[1] != 256:
                    raise ValueError("Student fused-P3 feature capture failed")
                region = regional_cosine_loss(captured[0], reference, batch["bboxes"], batch["batch_idx"],
                                              batch.get("consistency_eligible")) * self.consistency_weight
            return detection + region, torch.cat((items, region.detach().reshape(1)))
        finally:
            if hook is not None:
                hook.remove()


class WeatherHFCValidator(RTDETRValidator):
    """Keep final standalone validation on the same read-only manifest loader."""

    def check_detection_data(self, data):
        return check_det_dataset(data, autodownload=False, check_fonts=bool(self.args.plots))

    def __init__(self, *args, manifest, **kwargs):
        self.manifest = Path(manifest).resolve()
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="val", batch=None):
        if self.training:
            split = "valid"
        else:
            requested = self.args.split
            if requested not in ("val", "test"):
                raise ValueError(f"Weather evaluation requires split='val' or 'test'; got {requested!r}")
            split = "valid" if requested == "val" else "test"
        return PairedWeatherDataset(manifest=self.manifest, split=split,
                                    load_reference=False, img_path=img_path, imgsz=self.args.imgsz,
                                    batch_size=batch, augment=False, hyp=self.args, rect=False,
                                    cache=self.args.cache or None, prefix=colorstr(f"{mode}: "), data=self.data)

    def postprocess(self, preds):
        """Keep native DETR boxes, with the frozen score floor and count cap."""
        outputs = super().postprocess(preds)
        threshold = self.args.conf if self.args.conf is not None else 0.001
        maximum = int(self.args.max_det)
        if maximum < 1:
            raise ValueError("Weather evaluation max_det must be positive")
        return [prediction[prediction[:, 4] >= threshold][:maximum] for prediction in outputs]


class WeatherHFCTrainer(RTDETRTrainer):
    """Trainer owns the frozen teacher; student/EMA/optimizer cannot register it."""

    def check_detection_data(self, data):
        return check_det_dataset(data, autodownload=False, check_fonts=bool(self.args.plots))

    def check_amp_support(self):
        from weather_amp import check_weather_amp
        return check_weather_amp(self.model)

    def __init__(self, *args, manifest, teacher_checkpoint=None, consistency_weight=0.1, **kwargs):
        self.manifest = Path(manifest).resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self.teacher_checkpoint = Path(teacher_checkpoint).resolve() if teacher_checkpoint is not None else None
        if self.teacher_checkpoint is not None and not self.teacher_checkpoint.is_file():
            raise FileNotFoundError(self.teacher_checkpoint)
        self.consistency_weight = float(consistency_weight)
        if not math.isfinite(self.consistency_weight) or self.consistency_weight < 0:
            raise ValueError("consistency_weight must be finite and nonnegative")
        self.teacher = None
        super().__init__(*args, **kwargs)
        validate_fixed_geometry(self.args)

    def get_warmup_iterations(self, nb):
        """Weather protocols specify warmup in epochs, not optimizer updates."""
        epochs = float(self.args.warmup_epochs)
        if not math.isfinite(epochs) or epochs < 0 or nb < 1:
            raise ValueError("Weather warmup needs finite nonnegative epochs and a nonempty loader")
        return round(epochs * nb) if epochs > 0 else -1

    def _initialize_teacher(self, enabled):
        self.teacher = None
        if enabled:
            if self.teacher_checkpoint is None:
                raise ValueError("Consistency requires an explicit trained original-image teacher checkpoint")
            self.teacher = self._load_local_model(self.teacher_checkpoint)
            if any(self.teacher.yaml.get("weather_hfc", {}).values()):
                raise ValueError("Consistency teacher must be an original-image baseline without H/F/C")
            self.teacher.requires_grad_(False).eval()

    def _resume_runtime(self, ckpt):
        """Reject stripped/legacy checkpoints before constructing training state."""
        if not isinstance(ckpt, dict) or not isinstance(ckpt.get("optimizer"), dict):
            raise ValueError("Weather resume checkpoint requires a recoverable optimizer")
        epoch = ckpt.get("epoch")
        if type(epoch) is not int or epoch < 0 or epoch + 1 >= self.epochs:
            raise ValueError("Weather resume checkpoint epoch is invalid or its training budget is complete")
        runtime = ckpt.get("extra_state", {}).get("weather_hfc_runtime")
        required = {"version", "amp", "consistency_weight", "scaler", "stopper", "scheduler", "rng", "loaders"}
        if not isinstance(runtime, dict) or required - runtime.keys() or runtime["version"] != 1:
            raise ValueError("Weather resume checkpoint lacks supported runtime state")
        if runtime.get("stop", False):
            raise ValueError("Weather resume checkpoint has already reached early stopping")
        if bool(runtime.get("requested_amp", runtime["amp"])) != bool(self.args.amp):
            raise ValueError("Weather resume cannot change the checkpoint precision mode")
        if float(runtime["consistency_weight"]) != self.consistency_weight:
            raise ValueError("Weather resume cannot change the checkpoint consistency weight")
        if runtime["stopper"]["patience"] != self.args.patience:
            raise ValueError("Weather resume cannot change the checkpoint early-stopping patience")
        return runtime

    def setup_model(self):
        if not self.resume:
            return super().setup_model()
        path = Path(self.model)
        if not path.is_file():
            raise FileNotFoundError(f"Explicit Weather resume checkpoint does not exist: {path}")
        # Inference loading prefers EMA. Optimizer continuation must use raw model.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        runtime = self._resume_runtime(ckpt)
        student = ckpt.get("model")
        if not isinstance(student, WeatherHFCModel) or student.nc != 7:
            raise ValueError("Weather resume checkpoint requires its raw seven-class WeatherHFCModel")
        _validate_seven_classes(self.data)
        if float(student.consistency_weight) != self.consistency_weight:
            raise ValueError("Weather resume raw student and protocol consistency weights disagree")
        if not isinstance(ckpt.get("ema"), WeatherHFCModel):
            raise ValueError("Weather resume checkpoint requires its separate student EMA")
        self.model = student.float()
        self._weather_resume_runtime = runtime
        self._initialize_teacher(self.model.consistency_enabled)
        return ckpt

    def checkpoint_extra_state(self):
        """Versioned epoch-boundary state; the teacher is intentionally excluded."""
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        loaders = {}
        for name in ("train_loader", "test_loader"):
            generator = getattr(getattr(self, name, None), "generator", None)
            if generator is not None:
                loaders[name] = generator.get_state()
        return {"weather_hfc_runtime": {
            "version": 1, "amp": bool(self.amp), "requested_amp": bool(self.args.amp),
            "consistency_weight": self.consistency_weight,
            "scaler": deepcopy(self.scaler.state_dict()), "scheduler": deepcopy(self.scheduler.state_dict()),
            "stopper": {key: getattr(self.stopper, key) for key in
                        ("best_epoch", "best_fitness", "patience", "possible_stop")},
            "stop": bool(self.stop), "rng": rng, "loaders": loaders}}

    def _restore_weather_runtime(self, runtime):
        self.scaler.load_state_dict(runtime["scaler"])
        self.scheduler.load_state_dict(runtime["scheduler"])
        for key, value in runtime["stopper"].items():
            setattr(self.stopper, key, value)
        self.stop = bool(runtime.get("stop", False))
        rng = runtime["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("Weather resume cannot restore the checkpoint CUDA random state")
            torch.cuda.set_rng_state_all(rng["cuda"])
        for name, state in runtime["loaders"].items():
            generator = getattr(getattr(self, name, None), "generator", None)
            if generator is None:
                raise ValueError(f"Weather resume loader generator is unavailable: {name}")
            generator.set_state(state)

    def resume_training(self, ckpt):
        if not self.resume:
            return super().resume_training(ckpt)
        runtime = self._resume_runtime(ckpt)
        if bool(self.amp) != bool(runtime["amp"]):
            raise ValueError("Weather resume effective AMP mode differs from the checkpoint")
        super().resume_training(ckpt)
        self._weather_resume_runtime = runtime
        self._restore_weather_runtime(runtime)

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # Base setup adjusts scheduler.last_epoch and executes setup callbacks
        # after resume_training. Reapply state after that construction finishes.
        if self.resume:
            self._restore_weather_runtime(self._weather_resume_runtime)

    def get_model(self, cfg=None, weights=None, verbose=True):
        _validate_seven_classes(self.data)
        model = WeatherHFCModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        model.consistency_weight = self.consistency_weight
        if model.consistency_enabled and self.teacher_checkpoint is None:
            raise ValueError("Consistency requires an explicit trained original-image teacher checkpoint")
        source = weights
        if source is None:
            pretrained = getattr(self.args, "pretrained", None)
            source = pretrained if isinstance(pretrained, (str, Path)) else self.teacher_checkpoint
        if source is None:
            raise ValueError("Every weather configuration requires the same explicit original-image baseline weights")
        if isinstance(source, (str, Path)):
            source = self._load_local_model(source)
        model.load(source, verbose=verbose)
        source_model = source["model"] if isinstance(source, dict) else source
        student_state = model.state_dict()
        model.initialization_transferred_keys = sum(key in student_state and value.shape == student_state[key].shape
                                                     for key, value in source_model.state_dict().items())
        LOGGER.info(f"Weather HFC: initialized {model.initialization_transferred_keys} matching student state entries")
        self._initialize_teacher(model.consistency_enabled)
        return model

    @staticmethod
    def _load_local_model(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        model, _ = attempt_load_one_weight(str(path), device="cpu", fuse=False)
        if not isinstance(model, RTDETRDetectionModel):
            raise ValueError("Weather HFC requires an RT-DETR baseline checkpoint")
        return model

    def build_dataset(self, img_path, mode="val", batch=None):
        _validate_seven_classes(self.data)
        validate_fixed_geometry(self.args)
        model = de_parallel(self.model)
        enabled = bool(getattr(model, "consistency_enabled", False))
        return PairedWeatherDataset(manifest=self.manifest, split="valid" if mode == "val" else mode,
                                    load_reference=mode == "train" and enabled,
                                    img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                                    augment=mode == "train", hyp=self.args, rect=False, cache=self.args.cache or None,
                                    prefix=colorstr(f"{mode}: "), data=self.data)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Finish each epoch's sampler so saved generator state resumes its order.

        Keep native loader settings, but do not carry InfiniteDataLoader's
        suspended sampler or prefetched batches across checkpoint boundaries.
        The fixed weather protocol has close_mosaic=0 and needs no reset().
        """
        assert mode in ("train", "val")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        batch_size = min(batch_size, len(dataset))
        workers = min(os.cpu_count() // max(torch.cuda.device_count(), 1), batch_size, workers)
        sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
        generator = torch.Generator().manual_seed(6148914691236517205 + RANK)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                          num_workers=workers, sampler=sampler, pin_memory=PIN_MEMORY,
                          collate_fn=getattr(dataset, "collate_fn", None), worker_init_fn=seed_worker,
                          generator=generator)

    def get_validator(self):
        self.loss_names = "giou_loss", "cls_loss", "l1_loss", "region_loss"
        return WeatherHFCValidator(self.test_loader, manifest=self.manifest, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        model = de_parallel(self.model)
        if not model.training or not model.consistency_enabled:
            return batch
        eligible = batch.get("consistency_eligible")
        if eligible is None:
            raise ValueError("Consistency training requires manifest eligibility metadata")
        if not torch.as_tensor(eligible).any() or not len(batch["bboxes"]):
            return batch
        if self.teacher is None:
            raise ValueError("Consistency teacher was not initialized from its checkpoint")
        if "reference_img" not in batch:
            raise ValueError("Consistency training requires paired original images")
        self.teacher.to(self.device).requires_grad_(False).eval()
        references = batch.pop("reference_img").to(self.device, non_blocking=True).float() / 255.0
        if references.shape != batch["img"].shape:
            raise ValueError("Student/reference input geometry differs")
        with torch.no_grad():
            batch["reference_features"] = fused_p3_features(self.teacher, references).detach()
        return batch
