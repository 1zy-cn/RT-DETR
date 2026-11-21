"""CPU state regressions for bounded baseline fine-tuning; never detector training."""
import importlib
import importlib.util
import json
import random
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA
from weather_hfc_training import WeatherHFCModel

torch.set_num_threads(2)
POLICY = dict(version=1, warmup_updates=64, initial_amp_scale=64., validation_batch=4)


class TinyBaseline(WeatherHFCModel):
    def __init__(self, marker=1.):
        torch.nn.Module.__init__(self)
        self.marker = torch.nn.Parameter(torch.tensor([float(marker)]))
        self.model = torch.nn.ModuleList([torch.nn.Identity()])
        self.yaml = {'nc': 7, 'weather_hfc': dict(haar=False, freq=False, consistency=False)}
        self.nc = 7
        self.names = {i: str(i) for i in range(7)}
        self.stride = torch.tensor([32.])
        self.args = {}
        self.consistency_enabled = False
        self.consistency_weight = .1


def trainer_class():
    return importlib.import_module('weather_finetune').BaselineFinetuneTrainer


def bare(tmp_path, cls=None):
    cls = cls or trainer_class()
    trainer = cls.__new__(cls)
    trainer.finetune_policy = deepcopy(POLICY)
    trainer.execution_budget_epochs = 1
    trainer.finetune_attempts = trainer.finetune_successes = trainer.finetune_skips = 0
    trainer._last_saved_epoch = None
    trainer.args = SimpleNamespace(warmup_epochs=3., epochs=10, close_mosaic=0, amp=False,
                                  patience=20, model=str(tmp_path/'weights/last.pt'), half=False,
                                  plots=False, workers=0, save=True)
    trainer.device = torch.device('cpu')
    trainer.model = TinyBaseline(1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.003)
    trainer.model.marker.square().sum().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda epoch: 1-epoch/20)
    trainer.ema = ModelEMA(trainer.model)
    trainer.ema.ema.marker.data.fill_(9)
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
    trainer.amp = False
    trainer.stopper = EarlyStopping(patience=20)
    trainer.stopper(0, .5)
    trainer.stop = False
    trainer.epoch = trainer.start_epoch = 0
    trainer.epochs = 10
    trainer.best_fitness = .5
    trainer.fitness = .4
    trainer.metrics = {}
    trainer.train_loader = SimpleNamespace(generator=torch.Generator().manual_seed(91))
    trainer.test_loader = SimpleNamespace(generator=torch.Generator().manual_seed(92))
    trainer.teacher = None
    trainer.teacher_checkpoint = None
    trainer.consistency_weight = .1
    trainer.data = {'nc': 7, 'names': trainer.model.names}
    trainer.resume = False
    trainer.save_dir = tmp_path
    trainer.wdir = tmp_path/'weights'
    trainer.wdir.mkdir(exist_ok=True)
    trainer.last = trainer.wdir/'last.pt'
    trainer.best = trainer.wdir/'best.pt'
    trainer.csv = tmp_path/'results.csv'
    trainer.csv.write_text('epoch,loss\n1,1\n')
    trainer.save_period = -1
    trainer.callbacks = {}
    return trainer


def test_finetune_candidate_module_exists():
    assert importlib.util.find_spec('weather_finetune') is not None, 'bounded fine-tuning candidate is missing'


def test_warmup_is_update_count_not_dataset_size(tmp_path):
    trainer = bare(tmp_path)
    assert trainer.get_warmup_iterations(3) == 64
    assert trainer.get_warmup_iterations(3000) == 64
    assert trainer.args.warmup_epochs == 3.


def test_validation_loader_uses_safe_batch_without_changing_training_batch(tmp_path):
    class TensorTrainer(trainer_class()):
        def build_dataset(self, img_path, mode='val', batch=None):
            return torch.utils.data.TensorDataset(torch.arange(12))
    trainer = bare(tmp_path, TensorTrainer)
    assert trainer.get_dataloader('unused', batch_size=16, rank=-1, mode='val').batch_size == 4
    assert trainer.get_dataloader('unused', batch_size=8, rank=-1, mode='train').batch_size == 8


def test_budget_pause_preserves_actual_complete_checkpoint(tmp_path):
    module = importlib.import_module('weather_finetune')
    trainer = bare(tmp_path)
    trainer.save_model()
    before = trainer.last.read_bytes()
    with pytest.raises(module.BudgetPause):
        module.pause_at_epoch_budget(trainer)
    saved = torch.load(trainer.last, map_location='cpu', weights_only=False)
    assert trainer.last.read_bytes() == before
    assert saved['epoch'] == 0 and saved['optimizer']['state']
    assert saved['extra_state']['weather_hfc_runtime']['stop'] is False
    assert saved['model'].marker.item() != saved['ema'].marker.item()
    assert trainer.stop is False


@pytest.mark.parametrize('reason', ['early_stop', 'final_epoch', 'under_budget'])
def test_budget_does_not_disguise_other_termination(tmp_path, reason):
    module = importlib.import_module('weather_finetune')
    trainer = bare(tmp_path)
    if reason == 'early_stop': trainer.stop = True
    elif reason == 'final_epoch': trainer.epoch = trainer.epochs-1
    else: trainer.execution_budget_epochs = 2
    module.pause_at_epoch_budget(trainer)
    assert not trainer.last.exists(), 'non-budget path should not invent a checkpoint'


def test_finetune_policy_and_counters_survive_resume(tmp_path):
    original = bare(tmp_path)
    original.finetune_attempts, original.finetune_successes, original.finetune_skips = 4, 3, 1
    original.ema.updates = 3
    original.save_model()
    saved = torch.load(original.last, map_location='cpu', weights_only=False)
    restored = bare(tmp_path)
    restored.resume = True
    restored.resume_training(saved)
    assert (restored.finetune_attempts, restored.finetune_successes, restored.finetune_skips) == (4, 3, 1)
    assert restored.start_epoch == 1 and restored.ema.updates == 3
    assert restored.finetune_policy == POLICY


@pytest.mark.parametrize('damage', ['policy', 'counts', 'missing'])
def test_resume_rejects_incompatible_profile_state(tmp_path, damage):
    trainer = bare(tmp_path)
    trainer.save_model()
    saved = torch.load(trainer.last, map_location='cpu', weights_only=False)
    runtime = saved['extra_state']['finetune_runtime']
    if damage == 'policy': runtime['policy']['warmup_updates'] += 1
    elif damage == 'counts': runtime['successes'] += 1
    else: del saved['extra_state']['finetune_runtime']
    with pytest.raises(ValueError, match='finetune|policy|counter|runtime'):
        trainer._resume_runtime(saved)


class InitialValidator:
    condition_metrics = {'clear': {'metrics/mAP50-95(B)': .4}}
    def __call__(self, trainer=None, model=None):
        assert trainer is not None and model is None
        assert torch.is_tensor(trainer.loss) and trainer.loss_items.shape == (4,)
        assert trainer.epoch == -1
        random.random(); np.random.rand(); torch.rand(1)
        torch.rand(1, generator=trainer.test_loader.generator)
        return {'fitness': .46, 'metrics/mAP50(B)': .7, 'metrics/mAP50-95(B)': .433333,
                'val/giou_loss': .4, 'val/cls_loss': .5, 'val/l1_loss': .1, 'val/region_loss': 0.}


def test_initial_validation_is_safe_and_preserves_rng_and_original_best(tmp_path):
    trainer = bare(tmp_path)
    trainer.validator = InitialValidator()
    trainer.loss = None
    trainer.loss_items = None
    random.seed(11); np.random.seed(12); torch.manual_seed(13)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(), trainer.test_loader.generator.get_state())
    trainer._evaluate_initial()
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert torch.equal(trainer.test_loader.generator.get_state(), states[3])
    saved = torch.load(trainer.best, map_location='cpu', weights_only=False)
    assert saved['optimizer'] is None and saved['model'].marker.item() == 9
    assert trainer.best_fitness == .46 and trainer.stopper.best_epoch == 0
    record = json.loads((tmp_path/'finetune_validation.jsonl').read_text())
    assert record['stage'] == 'initial' and record['epoch'] == 0
    assert record['groups'] == InitialValidator.condition_metrics


class ControlledScaler:
    def __init__(self, skip): self.skip, self.scale = skip, 64.
    def unscale_(self, optimizer): pass
    def get_scale(self): return self.scale
    def step(self, optimizer):
        if not self.skip: optimizer.step()
    def update(self):
        if self.skip: self.scale /= 2


def test_amp_skip_does_not_advance_student_or_ema(tmp_path):
    trainer = bare(tmp_path)
    trainer.scaler = ControlledScaler(skip=True)
    before = trainer.model.marker.detach().clone()
    trainer.model.marker.grad = torch.ones_like(trainer.model.marker)
    trainer.optimizer_step()
    assert torch.equal(trainer.model.marker, before)
    assert trainer.ema.updates == 0
    assert (trainer.finetune_attempts, trainer.finetune_successes, trainer.finetune_skips) == (1, 0, 1)
    trainer.scaler.skip = False
    trainer.model.marker.grad = torch.ones_like(trainer.model.marker)
    trainer.optimizer_step()
    assert not torch.equal(trainer.model.marker, before)
    assert trainer.ema.updates == 1
    assert (trainer.finetune_attempts, trainer.finetune_successes, trainer.finetune_skips) == (2, 1, 1)


class TrainableTinyBaseline(TinyBaseline):
    def __init__(self):
        super().__init__()
        self.seen_examples = 0
    def forward(self, batch):
        self.seen_examples += len(batch['img'])
        loss = self.marker.square().sum()
        return loss, torch.stack((loss.detach(), loss.detach(), loss.detach(), loss.detach()*0))


def test_native_loop_stops_after_complete_epoch_and_keeps_recovery_state(tmp_path):
    module = importlib.import_module('weather_finetune')
    class Images(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, index):
            return dict(img=torch.zeros(3,4,4,dtype=torch.uint8), cls=torch.zeros(1),
                        bboxes=torch.zeros(4), batch_idx=torch.tensor(0))
    class Validator:
        condition_metrics = {}
        def __call__(self, trainer=None, model=None):
            assert trainer is not None
            return {'fitness': .3 if trainer.epoch == -1 else .6,
                    'metrics/mAP50(B)': .7, 'metrics/mAP50-95(B)': .5,
                    'val/giou_loss': .4, 'val/cls_loss': .5, 'val/l1_loss': .1, 'val/region_loss': 0.}
    class NativeLoop(trainer_class()):
        def _setup_train(self, world_size):
            self._evaluate_initial()
    trainer = bare(tmp_path, NativeLoop)
    trainer.model = TrainableTinyBaseline()
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.003)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda epoch: 1-epoch/20)
    trainer.lf = lambda epoch: 1-epoch/20
    trainer.ema = ModelEMA(trainer.model)
    trainer.train_loader = torch.utils.data.DataLoader(Images(),batch_size=2,num_workers=0,
        generator=torch.Generator().manual_seed(1))
    trainer.validator = Validator()
    trainer.args.nbs = 2
    trainer.args.warmup_bias_lr = 0.
    trainer.args.warmup_momentum = .8
    trainer.args.momentum = .9
    trainer.args.multi_scale = False
    trainer.args.imgsz = 4
    trainer.args.val = True
    trainer.args.batch = 2
    trainer.batch_size = 2
    trainer.loss_names = ('giou_loss','cls_loss','l1_loss','region_loss')
    trainer.callbacks = defaultdict(list)
    trainer.add_callback('on_fit_epoch_end',module.pause_at_epoch_budget)
    trainer.csv.unlink()
    with pytest.raises(module.BudgetPause):
        trainer._do_train(world_size=1)
    saved = torch.load(trainer.last,map_location='cpu',weights_only=False)
    assert trainer.model.seen_examples == saved['model'].seen_examples == 4
    assert saved['epoch'] == 0 and saved['optimizer']['state']
    assert saved['updates'] == saved['extra_state']['finetune_runtime']['successes'] == 2
    assert saved['extra_state']['weather_hfc_runtime']['stop'] is False
    assert not (trainer.wdir/'best_inference.pt').exists(), 'budget path must not run destructive finalization'


def test_genuine_finalization_exports_a_copy_and_preserves_raw_files(tmp_path):
    trainer = bare(tmp_path)
    trainer.epoch = trainer.epochs-1
    trainer.fitness = trainer.best_fitness = .5
    trainer.save_model()
    raw_last, raw_best = trainer.last.read_bytes(), trainer.best.read_bytes()
    class InferenceValidator:
        args = SimpleNamespace(plots=False)
        condition_metrics = {}
        def __call__(self, trainer=None, model=None):
            assert trainer is None and Path(model).name == 'best_inference.pt'
            saved = torch.load(model,map_location='cpu',weights_only=False)
            assert saved['optimizer'] is None and saved['epoch'] == -1
            return {'fitness': .5, 'metrics/mAP50(B)': .6, 'metrics/mAP50-95(B)': .49}
    trainer.validator = InferenceValidator()
    trainer.final_eval()
    assert trainer.last.read_bytes() == raw_last
    assert trainer.best.read_bytes() == raw_best
    assert torch.load(trainer.last,map_location='cpu',weights_only=False)['optimizer']['state']


class UpdateValidator:
    condition_metrics = {}
    def __init__(self): self.calls = 0
    def __call__(self, trainer=None, model=None):
        assert trainer is not None and model is None
        self.calls += 1
        trainer.model.eval()
        trainer.ema.ema.train()
        random.random();np.random.rand();torch.rand(1)
        return {'fitness': .6, 'metrics/mAP50(B)': .7, 'metrics/mAP50-95(B)': .588889,
                'val/giou_loss': .4, 'val/cls_loss': .5, 'val/l1_loss': .1, 'val/region_loss': 0.}


def mid_update_trainer(tmp_path, monkeypatch):
    trainer = bare(tmp_path)
    monkeypatch.setattr(trainer_class(),'VALIDATION_INTERVAL_UPDATES',2,raising=False)
    trainer.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(8)),
        batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(11))
    trainer.validator = UpdateValidator()
    trainer.loss = torch.zeros(())
    trainer.loss_items = torch.zeros(4)
    return trainer


def test_mid_update_validation_avoids_duplicate_epoch_boundary(tmp_path, monkeypatch):
    trainer = mid_update_trainer(tmp_path,monkeypatch)
    for _ in range(4):
        trainer.model.marker.grad = torch.ones_like(trainer.model.marker)
        trainer.optimizer_step()
    assert trainer.validator.calls == 1, 'step2 validates, step4 is a full-epoch boundary and must defer to native val'
    assert trainer.finetune_attempts == 4
    assert not trainer.last.exists(), 'mid-validation must not invent an epoch-boundary checkpoint'


def test_mid_update_checkpoint_is_inference_only_and_preserves_training_state(tmp_path, monkeypatch):
    trainer = mid_update_trainer(tmp_path,monkeypatch)
    before_stopper = deepcopy(trainer.stopper.__dict__)
    random.seed(101);np.random.seed(102);torch.manual_seed(103)
    before_rng = (random.getstate(),np.random.get_state(),torch.get_rng_state())
    for _ in range(2):
        trainer.model.marker.grad = torch.ones_like(trainer.model.marker)
        trainer.optimizer_step()
    assert trainer.best.is_file(), 'improved mid-epoch result must be retained'
    saved = torch.load(trainer.best,map_location='cpu',weights_only=False)
    assert saved['epoch'] == -1 and saved['optimizer'] is None
    with pytest.raises(ValueError,match='optimizer'):
        trainer._resume_runtime(saved)
    assert trainer.stopper.__dict__ == before_stopper and trainer.stop is False
    assert trainer.model.training and not trainer.ema.ema.training
    assert trainer.model.marker.requires_grad and trainer.model.marker.grad is None
    assert random.getstate() == before_rng[0]
    assert np.array_equal(np.random.get_state()[1],before_rng[1][1])
    assert torch.equal(torch.get_rng_state(),before_rng[2])
    record = json.loads((tmp_path/'finetune_validation.jsonl').read_text())
    assert record['stage'] == 'update' and record['attempts'] == 2
    assert trainer.best_fitness == .6
