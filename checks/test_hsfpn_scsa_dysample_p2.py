"""The borrowed DySample + P2 branch must enrich the P3 decoder input."""

from pathlib import Path
from types import SimpleNamespace
import json

import torch

from ultralytics.nn.Addmodules.Dysample import Dy_Sample
from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[1]


def test_hsfpn_scsa_dysample_p2_builds_and_trains(tmp_path):
    from train import EXPERIMENTS, on_train_start, training_args, verify_model

    name = 'hsfpn-scsa-dysample-p2'
    config = EXPERIMENTS[name]['model']
    torch.set_num_threads(2)
    model = RTDETRDetectionModel(str(config), nc=7, verbose=False)
    counts = verify_model(model, name)
    assert counts['scsa_modules'] == 8
    assert counts['hsfpn_ca_modules'] == 6
    assert counts['dysample_modules'] == 3
    assert counts['transpose_convolutions'] == 0
    assert sum(isinstance(m, Dy_Sample) for m in model.modules()) == 3
    decoder = model.model[-1]
    assert decoder.nl == 3
    assert decoder.f == [36, 19, 12]
    assert all(layer.cross_attn.point_counts == (4, 4, 4) for layer in decoder.decoder.layers)
    assert (training_args(name)['epochs'], training_args(name)['batch'],
            training_args(name)['imgsz']) == (300, 8, 640)

    model.nc = 7
    model.train()
    batch = {'img': torch.rand(1, 3, 256, 256), 'batch_idx': torch.tensor([0]),
             'cls': torch.tensor([[0.]]), 'bboxes': torch.tensor([[.5, .5, .2, .1]])}
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for index in (15, 22, 27):
        grad = model.model[index].offset.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

    trainer = SimpleNamespace(model=model, save_dir=tmp_path,
                              args=SimpleNamespace(data=str(ROOT / 'SVRDD.yaml'), seed=0))
    on_train_start(trainer, experiment=name)
    snapshot = tmp_path / 'source_snapshot'
    metadata = json.loads((snapshot / 'provenance.json').read_text(encoding='utf-8'))
    assert metadata['model_check']['dysample_modules'] == 3
    assert (snapshot / config.name).read_bytes() == config.read_bytes()
    assert (snapshot / 'Dysample.py').is_file()
    assert (snapshot / 'HSFPN.py').is_file()
