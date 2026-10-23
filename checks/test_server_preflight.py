"""Prevent expensive runs on inconsistent data or evaluation protocols."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image


@pytest.fixture
def tiny_svrdd(tmp_path):
    from scripts.audit_svrdd import CLASS_NAMES
    manifests = tmp_path / 'manifests'
    manifests.mkdir()
    for index, (split, manifest) in enumerate((('train', 'train'), ('valid', 'val'), ('test', 'test'))):
        images, labels = tmp_path / split / 'images', tmp_path / split / 'labels'
        images.mkdir(parents=True)
        labels.mkdir()
        Image.new('RGB', (20, 20), (index * 60, 20, 40)).save(images / f'{split}.jpg')
        (labels / f'{split}.txt').write_text('0 0.5 0.5 0.2 0.1\n')
        (manifests / f'{manifest}.txt').write_text(f'{split}/images/{split}.jpg\n')
    config = tmp_path / 'data.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(tmp_path), train='train/images',
        val='valid/images', test='test/images', nc=7, names=CLASS_NAMES)), encoding='utf-8')
    return config, manifests


def test_deep_dataset_audit_and_fingerprint(tiny_svrdd):
    from scripts.audit_svrdd import audit_dataset
    config, manifests = tiny_svrdd
    report = audit_dataset(config, manifests, deep=True)
    assert report['counts'] == {'train': 1, 'valid': 1, 'test': 1}
    assert report['splits']['train']['boxes_per_class'] == [1, 0, 0, 0, 0, 0, 0]
    assert len(report['dataset_sha256']) == 64
    assert not report['cross_split_duplicates']


@pytest.mark.parametrize('bad', ['7 .5 .5 .2 .1', '0 nan .5 .2 .1',
                               '0 .5 .5 0 .1', '0 .5 .5 .2 .1 .3', '0.5 .5 .5 .2 .1'])
def test_bad_annotations_rejected(tiny_svrdd, bad):
    from scripts.audit_svrdd import audit_dataset
    config, manifests = tiny_svrdd
    (config.parent / 'train/labels/train.txt').write_text(bad)
    with pytest.raises(ValueError, match='label'):
        audit_dataset(config, manifests)


def test_cross_split_content_leak_rejected(tiny_svrdd):
    from scripts.audit_svrdd import audit_dataset
    config, manifests = tiny_svrdd
    (config.parent / 'test/images/test.jpg').write_bytes((config.parent / 'train/images/train.jpg').read_bytes())
    with pytest.raises(ValueError, match='cross-split'):
        audit_dataset(config, manifests, deep=True)


def test_wrong_class_order_rejected(tiny_svrdd):
    from scripts.audit_svrdd import audit_dataset
    config, manifests = tiny_svrdd
    data = yaml.safe_load(config.read_text())
    data['names'][4], data['names'][5] = data['names'][5], data['names'][4]
    config.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(ValueError, match='class'):
        audit_dataset(config, manifests)


def test_all_eight_groups_share_explicit_protocol():
    from train import CH3_GROUPS, training_args
    configs = [training_args(name) for name in CH3_GROUPS]
    assert len(configs) == 8
    for args in configs:
        assert args['patience'] == 0 and args['half'] is False
        assert args['pretrained'] is False and args['amp'] is False
        assert args['batch'] == args['nbs'] == 8
        assert args['epochs'] == 300 and args['seed'] == 0
        assert {k: v for k, v in args.items() if k != 'name'} == {
            k: v for k, v in configs[0].items() if k != 'name'}


def test_screen_protocol_has_distinct_name_and_schedule():
    from train import training_args
    full = training_args('svrdd-baseline')
    screen = training_args('svrdd-baseline', phase='screen', seed=1)
    assert screen['epochs'] == 80 and screen['close_mosaic'] == 0
    assert screen['seed'] == 1 and 'screen80' in screen['name']
    assert full['name'] != screen['name']


def test_iou_preflight_covers_giou(monkeypatch):
    import train
    train.verify_standard_iou()
    original = train.bbox_iou
    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        return result + .1 if kwargs.get('GIoU') else result
    monkeypatch.setattr(train, 'bbox_iou', broken)
    with pytest.raises(RuntimeError, match='IoU'):
        train.verify_standard_iou()


def test_training_validation_honors_fp32_before_first_batch(tmp_path):
    """Exercise the actual validator setup; no CUDA allocation is needed."""
    from ultralytics.models.rtdetr.val import RTDETRValidator
    class AtBatchBoundary(Exception):
        pass
    model = torch.nn.Linear(2, 2)
    trainer = SimpleNamespace(device=torch.device('cuda'), data={},
        ema=SimpleNamespace(ema=model), model=model,
        loss_items=torch.zeros(3), stopper=SimpleNamespace(possible_stop=False), epoch=0, epochs=300)
    validator = RTDETRValidator(save_dir=tmp_path, args=dict(half=False, plots=False))
    # Stop at loss allocation after the real precision-selection branch.
    from unittest.mock import patch
    with patch('ultralytics.engine.validator.torch.zeros_like', side_effect=AtBatchBoundary):
        with pytest.raises(AtBatchBoundary):
            validator(trainer=trainer)
    assert model.weight.dtype == torch.float32
    assert validator.args.half is False


def test_fp32_checkpoint_not_quantized_on_save_or_strip(tmp_path):
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils.torch_utils import strip_optimizer
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.metrics, trainer.fitness, trainer.best_fitness, trainer.epoch = {}, .1, .1, 0
    trainer.csv = tmp_path / 'results.csv'
    trainer.csv.write_text('epoch,loss\n1,1.0\n')
    trainer.model = torch.nn.Linear(2, 2)
    trainer.ema = SimpleNamespace(ema=trainer.model, updates=1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.args = SimpleNamespace(half=False)
    trainer.last, trainer.best = tmp_path / 'last.pt', tmp_path / 'best.pt'
    trainer.save_period = -1
    expected = trainer.model.weight.detach().clone()
    trainer.save_model()
    ckpt = torch.load(trainer.best, map_location='cpu')
    assert ckpt['ema'].weight.dtype == torch.float32
    assert torch.equal(ckpt['ema'].weight, expected)
    strip_optimizer(trainer.best)
    final = torch.load(trainer.best, map_location='cpu')['model']
    assert final.weight.dtype == torch.float32
    assert torch.equal(final.weight, expected)


def test_smoke_memory_reset_uses_trainer_logical_device(monkeypatch):
    from checks.smoke_ch3_training import reset_memory_stats
    seen = []
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', seen.append)
    reset_memory_stats(SimpleNamespace(device=torch.device('cpu')))
    assert seen == []
    reset_memory_stats(SimpleNamespace(device=torch.device('cuda:0')))
    assert seen == [torch.device('cuda:0')]
