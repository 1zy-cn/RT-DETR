"""Weather runtime regressions; synthetic CPU state only, never a training run."""
import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

from ultralytics.cfg import get_cfg
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import EarlyStopping
from weather_hfc_training import WeatherHFCModel, WeatherHFCTrainer, WeatherHFCValidator

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


class TinyWeatherModel(WeatherHFCModel):
    """Real serializable module, with no detector forward or dataset dependency."""
    def __init__(self, value=1.0):
        torch.nn.Module.__init__(self)
        self.marker = torch.nn.Parameter(torch.tensor([float(value)]))
        self.model = torch.nn.ModuleList([torch.nn.Identity()])
        self.yaml = yaml.safe_load((ROOT/'ultralytics/cfg/models/rt-detr/weather-hfc/baseline.yaml').read_text())
        self.consistency_enabled = False
        self.consistency_weight = 0.1
        self.nc = 7
        self.names = {i: str(i) for i in range(7)}
        self.args = {}
        self.stride = torch.tensor([32.0])


def bare_trainer(tmp_path, trainer_class=WeatherHFCTrainer):
    trainer = trainer_class.__new__(trainer_class)
    trainer.args = SimpleNamespace(warmup_epochs=3.0, epochs=10, close_mosaic=0, amp=False,
                                  patience=20, model=str(tmp_path/'last.pt'))
    trainer.model = TinyWeatherModel(1.0)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.003)
    for parameter in trainer.model.parameters():
        trainer.optimizer.state[parameter] = dict(step=torch.tensor(7.0),
            exp_avg=torch.full_like(parameter, 0.2), exp_avg_sq=torch.full_like(parameter, 0.3))
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda epoch: 1-epoch/20)
    trainer.ema = SimpleNamespace(ema=TinyWeatherModel(9.0), updates=17)
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
    trainer.amp = False
    trainer.stopper = EarlyStopping(patience=20)
    trainer.stopper(2, 0.5)
    trainer.stop = False
    trainer.epoch = 3
    trainer.best_fitness = 0.5
    trainer.fitness = 0.4
    trainer.train_loader = SimpleNamespace(generator=torch.Generator().manual_seed(91))
    trainer.test_loader = SimpleNamespace(generator=torch.Generator().manual_seed(92))
    trainer.teacher = None
    trainer.teacher_checkpoint = None
    trainer.consistency_weight = 0.1
    trainer.data = {'nc':7, 'names': {i:str(i) for i in range(7)}}
    trainer.resume = True
    trainer.epochs = 10
    return trainer


def checkpoint(trainer):
    extra = trainer.checkpoint_extra_state() if hasattr(trainer, 'checkpoint_extra_state') else {}
    return {'model': deepcopy(trainer.model), 'ema': deepcopy(trainer.ema.ema), 'epoch':3,
            'optimizer': trainer.optimizer.state_dict(), 'best_fitness':0.5, 'updates':17,
            'train_args':vars(trainer.args), 'extra_state':extra}


class TensorWeatherTrainer(WeatherHFCTrainer):
    """Keep the actual Weather loader/runtime; substitute only image I/O."""

    def build_dataset(self, img_path, mode='val', batch=None):
        return torch.utils.data.TensorDataset(torch.arange(self.tensor_count))


class TensorNativeTrainer(RTDETRTrainer):
    def build_dataset(self, img_path, mode='val', batch=None):
        return torch.utils.data.TensorDataset(torch.arange(self.tensor_count))


def epoch_order(loader):
    return torch.cat([batch[0] for batch in loader]).tolist()


@pytest.mark.parametrize('count', [11, 12])
def test_weather_resume_replays_uninterrupted_epoch_orders(tmp_path, count):
    def make_trainer():
        trainer = bare_trainer(tmp_path, TensorWeatherTrainer)
        trainer.tensor_count = count
        trainer.args.workers = 0
        trainer.train_loader = trainer.get_dataloader('unused', batch_size=3, rank=-1, mode='train')
        trainer.test_loader = trainer.get_dataloader('unused', batch_size=6, rank=-1, mode='val')
        return trainer

    original = make_trainer()
    assert sorted(epoch_order(original.train_loader)) == list(range(count))
    assert epoch_order(original.test_loader) == list(range(count))
    saved = checkpoint(original)
    expected = [epoch_order(original.train_loader) for _ in range(2)]

    restored = make_trainer()
    restored.resume_training(saved)
    actual = [epoch_order(restored.train_loader) for _ in range(2)]
    assert actual == expected, 'resume must preserve the next complete epoch sample permutations'
    assert epoch_order(restored.test_loader) == list(range(count))


def test_weather_loader_preserves_native_first_epoch_and_generic_loader(tmp_path):
    weather = bare_trainer(tmp_path, TensorWeatherTrainer)
    native = TensorNativeTrainer.__new__(TensorNativeTrainer)
    for trainer in (weather, native):
        trainer.args = SimpleNamespace(workers=0)
        trainer.tensor_count = 12
    weather_loader = weather.get_dataloader('unused', batch_size=3, rank=-1, mode='train')
    native_loader = native.get_dataloader('unused', batch_size=3, rank=-1, mode='train')
    assert isinstance(native_loader, InfiniteDataLoader)
    assert epoch_order(weather_loader) == epoch_order(native_loader)
    assert sorted(epoch_order(native_loader)) == list(range(12))


@pytest.mark.parametrize('epochs,nb,want', [(3,1,3),(3,12000,36000),(0,12000,-1),(0.5,10,5)])
def test_weather_warmup_uses_batches_per_epoch_without_rewriting_args(tmp_path, epochs, nb, want):
    trainer = bare_trainer(tmp_path)
    trainer.args.warmup_epochs = epochs
    assert hasattr(trainer, 'get_warmup_iterations'), 'weather warmup resolver is missing'
    assert trainer.get_warmup_iterations(nb) == want
    assert trainer.args.warmup_epochs == epochs


def test_generic_trainer_warmup_retains_existing_semantics():
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(warmup_epochs=3)
    assert hasattr(trainer, 'get_warmup_iterations'), 'base warmup seam is missing'
    assert trainer.get_warmup_iterations(12000) == 3


@pytest.fixture
def validation_files(tmp_path):
    rows = []
    for split in ('valid','test'):
        image = tmp_path/split/'images'/f'{split}__clear.png'
        label = tmp_path/split/'labels'/f'{split}__clear.txt'
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        cv2.imwrite(str(image), np.zeros((24,32,3),dtype=np.uint8))
        label.write_text('0 0.5 0.5 0.4 0.4\n')
        rows.append(dict(split=split, original_id=split, condition='clear', severity=0,
            source_image=f'{split}/external.png', output_image=image.relative_to(tmp_path).as_posix()))
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in rows))
    return manifest


@pytest.mark.parametrize('requested,want', [('val','valid'),('test','test')])
def test_standalone_validator_uses_requested_manifest_split(validation_files, requested, want):
    validator = WeatherHFCValidator.__new__(WeatherHFCValidator)
    validator.manifest = validation_files
    settings = {name:0 for name in ('mosaic','mixup','copy_paste','degrees','translate','scale','shear',
                                   'perspective','fliplr','flipud','multi_scale')}
    settings.update(split=requested, imgsz=64, cache=False, multi_scale=False)
    validator.args = get_cfg(overrides=settings)
    validator.training = False
    validator.data = {'nc':7,'names':{i:str(i) for i in range(7)}}
    dataset = validator.build_dataset(str(validation_files.parent/want/'images'), mode='val', batch=1)
    assert dataset[0]['source_split'] == want
    assert 'reference_img' not in dataset[0]


def test_training_validator_keeps_validation_split_even_if_args_test(validation_files):
    validator = WeatherHFCValidator.__new__(WeatherHFCValidator)
    validator.manifest = validation_files
    settings = {name:0 for name in ('mosaic','mixup','copy_paste','degrees','translate','scale','shear',
                                   'perspective','fliplr','flipud','multi_scale')}
    settings.update(split='test', imgsz=64, cache=False, multi_scale=False)
    validator.args = get_cfg(overrides=settings)
    validator.training = True
    validator.data = {'nc':7,'names':{i:str(i) for i in range(7)}}
    dataset = validator.build_dataset(str(validation_files.parent/'valid/images'), batch=1)
    assert dataset[0]['source_split'] == 'valid'


def test_weather_postprocess_honors_score_floor_and_top_k_without_nms():
    validator = WeatherHFCValidator.__new__(WeatherHFCValidator)
    validator.args = SimpleNamespace(imgsz=64, conf=0.001, max_det=2)
    # Identical boxes must both survive: DETR should not gain an NMS step.
    values = torch.tensor([[[.5,.5,.2,.2,.0001,.0002], [.5,.5,.2,.2,.9,.1],
                            [.5,.5,.2,.2,.7,.2], [.5,.5,.2,.2,.3,.4]]])
    output = validator.postprocess((values.clone(),))[0]
    assert output.shape == (2,6)
    assert torch.allclose(output[:,4], torch.tensor([.9,.7]))
    validator.args.conf = 1.0
    assert validator.postprocess((values.clone(),))[0].shape == (0,6)


def test_resume_recovers_raw_student_instead_of_ema(tmp_path):
    trainer = bare_trainer(tmp_path)
    path = tmp_path/'last.pt'
    torch.save(checkpoint(trainer),path)
    trainer.model = str(path)
    loaded = trainer.setup_model()
    assert isinstance(trainer.model, TinyWeatherModel), 'resume replaced raw student with a new/EMA detector'
    assert torch.equal(trainer.model.marker, torch.tensor([1.0]))
    assert torch.equal(loaded['ema'].marker, torch.tensor([9.0]))
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(),lr=.5)
    trainer.ema = SimpleNamespace(ema=TinyWeatherModel(-1),updates=0)
    trainer.resume_training(loaded)
    assert trainer.start_epoch == 4 and trainer.ema.updates == 17
    assert torch.equal(trainer.ema.ema.marker, torch.tensor([9.0]))
    assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(.003)


@pytest.mark.parametrize('damage', ['optimizer','epoch','runtime'])
def test_resume_rejects_stripped_or_untracked_state_before_loading_model(tmp_path, damage):
    trainer = bare_trainer(tmp_path)
    saved = checkpoint(trainer)
    if damage == 'optimizer': saved['optimizer'] = None
    if damage == 'epoch': saved['epoch'] = -1
    if damage == 'runtime': saved['extra_state'] = {}
    path = tmp_path/'invalid.pt'
    torch.save(saved,path)
    trainer.model = str(path)
    with pytest.raises(ValueError,match='resume|checkpoint|runtime|optimizer|epoch'):
        trainer.setup_model()


def test_resume_restores_early_stopping_and_random_states(tmp_path):
    trainer = bare_trainer(tmp_path)
    assert hasattr(trainer, 'checkpoint_extra_state'), 'runtime checkpoint seam is missing'
    random.seed(41); np.random.seed(42); torch.manual_seed(43)
    saved = checkpoint(trainer)
    expected = (random.random(), np.random.rand(), torch.rand(3),
                torch.rand(3,generator=trainer.train_loader.generator))
    random.seed(51); np.random.seed(52); torch.manual_seed(53)
    trainer.train_loader.generator.manual_seed(54)
    trainer.stopper = EarlyStopping(patience=20)
    trainer.resume_training(saved)
    assert trainer.stopper.best_epoch == 2 and trainer.stopper.best_fitness == .5
    assert (random.random(),np.random.rand()) == expected[:2]
    assert torch.equal(torch.rand(3),expected[2])
    assert torch.equal(torch.rand(3,generator=trainer.train_loader.generator),expected[3])


def test_resume_preserves_requested_amp_when_native_check_fell_back_to_fp32(tmp_path):
    trainer = bare_trainer(tmp_path)
    trainer.args.amp = True
    trainer.amp = False  # A native AMP check can disable a requested AMP mode.
    saved = checkpoint(trainer)
    trainer.resume_training(saved)
    assert trainer.amp is False and trainer.start_epoch == 4


def test_base_checkpoint_writer_persists_subclass_extra_state(tmp_path):
    class WithExtra(BaseTrainer):
        def checkpoint_extra_state(self): return {'runtime_marker':17}
    trainer = WithExtra.__new__(WithExtra)
    trainer.model = TinyWeatherModel()
    trainer.ema = SimpleNamespace(ema=TinyWeatherModel(9),updates=2)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.args = SimpleNamespace(half=False)
    trainer.metrics = {}; trainer.fitness = .3; trainer.best_fitness = .3; trainer.epoch = 0
    trainer.csv = tmp_path/'results.csv'; trainer.csv.write_text('epoch,loss\n0,1\n')
    trainer.last = tmp_path/'last.pt'; trainer.best = tmp_path/'best.pt'; trainer.save_period = -1
    trainer.save_model()
    assert torch.load(trainer.last,map_location='cpu').get('extra_state') == {'runtime_marker':17}
