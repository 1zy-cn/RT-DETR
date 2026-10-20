"""Guard the proposed P3-biased decoder against silent configuration drift."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'ultralytics/cfg/models/rt-detr'
BASE = MODELS / 'resnet18-HSFPN+SCSA.yaml'
CANDIDATE = MODELS / 'resnet18-HSFPN+SCSA-P3Decoder.yaml'


def test_p3_decoder_changes_only_sampling_budget_and_runs_real_loss(tmp_path):
    from train import EXPERIMENTS, on_train_start, training_args, verify_model

    baseline = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding='utf-8'))
    expected = deepcopy(baseline)
    assert expected['head'][-1][3][3] == 4
    expected['head'][-1][3][3] = [6, 4, 2]
    assert candidate == expected
    assert EXPERIMENTS['hsfpn-scsa-p3decoder']['model'] == CANDIDATE

    torch.set_num_threads(2)
    base_model = RTDETRDetectionModel(str(BASE), nc=7, verbose=False)
    model = RTDETRDetectionModel(str(CANDIDATE), nc=7, verbose=False)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in base_model.parameters())
    del base_model
    counts = verify_model(model, 'hsfpn-scsa-p3decoder')
    assert counts['decoder_points'] == (6, 4, 2)
    assert counts['scsa_modules'] == 8 and counts['hsfpn_ca_modules'] == 5
    assert counts['transpose_convolutions'] == 2
    decoder = model.model[-1]
    assert decoder.f == [26, 19, 12]
    assert all(layer.cross_attn.scale_aware for layer in decoder.decoder.layers)
    assert all(layer.cross_attn.point_counts == (6, 4, 2) for layer in decoder.decoder.layers)

    args = training_args('hsfpn-scsa-p3decoder')
    assert (args['epochs'], args['batch'], args['imgsz'], args['seed']) == (300, 8, 640, 0)

    model.nc = 7
    model.train()
    image = torch.rand(1, 3, 256, 256)
    batch = {'img': image, 'batch_idx': torch.tensor([0]),
             'cls': torch.tensor([[0.]]), 'bboxes': torch.tensor([[.5, .5, .2, .1]])}
    loss, _ = model(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    cross = decoder.decoder.layers[0].cross_attn
    assert cross.sampling_offsets.weight.grad is not None
    assert torch.isfinite(cross.sampling_offsets.weight.grad).all()
    assert cross.sampling_offsets.weight.grad.abs().sum() > 0

    trainer = SimpleNamespace(model=model, save_dir=tmp_path,
                              args=SimpleNamespace(data=str(ROOT / 'SVRDD.yaml'), seed=0))
    on_train_start(trainer, experiment='hsfpn-scsa-p3decoder')
    snapshot = tmp_path / 'source_snapshot'
    metadata = json.loads((snapshot / 'provenance.json').read_text(encoding='utf-8'))
    assert metadata['model_check']['decoder_points'] == [6, 4, 2]
    assert (snapshot / CANDIDATE.name).read_bytes() == CANDIDATE.read_bytes()
    assert (snapshot / 'HSFPN.py').is_file()
