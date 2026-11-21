"""Opt-in, bounded warm-start training for the Chapter 4 baseline only.

The legacy Weather trainer is unchanged. Resource pauses happen after a complete
epoch checkpoint has been saved and remain distinct from model early stopping.
"""
from contextlib import contextmanager
from copy import copy, deepcopy
from datetime import datetime, timezone
import json
import math
import random

import numpy as np
import torch

from ultralytics.utils.torch_utils import strip_optimizer
from weather_hfc_training import WeatherHFCTrainer


class BudgetPause(Exception):
    """A complete epoch boundary reached the invocation's resource budget."""


def validate_finetune_policy(policy):
    """Return a canonical policy before a trainer can create output files."""
    required = {'version', 'warmup_updates', 'initial_amp_scale', 'validation_batch'}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError('finetune policy requires exactly version/warmup_updates/initial_amp_scale/validation_batch')
    if type(policy['version']) is not int or policy['version'] != 1:
        raise ValueError('Unsupported finetune policy version')
    for key in ('warmup_updates', 'validation_batch'):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f'finetune policy {key} must be a positive integer')
    if isinstance(policy['initial_amp_scale'], bool):
        raise ValueError('finetune initial_amp_scale must be finite and positive')
    try:
        scale = float(policy['initial_amp_scale'])
    except (TypeError, ValueError) as exc:
        raise ValueError('finetune initial_amp_scale must be finite and positive') from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('finetune initial_amp_scale must be finite and positive')
    return {**policy, 'initial_amp_scale': scale}


def pause_at_epoch_budget(trainer):
    """Top-level callback so checkpoints need not pickle a bound local closure."""
    if trainer.stop or trainer.epoch + 1 >= trainer.epochs:
        return
    completed = trainer.epoch - trainer.start_epoch + 1
    if completed < trainer.execution_budget_epochs:
        return
    # Native on_fit_epoch_end is after save_model. Also protect manual callers
    # and save=False configurations from accidentally pausing without recovery.
    if getattr(trainer, '_last_saved_epoch', None) != trainer.epoch:
        trainer.save_model()
    if not trainer.last.is_file():
        raise RuntimeError('finetune budget pause requires the current raw last.pt checkpoint')
    trainer.paused_by_budget = True
    raise BudgetPause(f'Paused after {completed} complete epoch(s); raw checkpoint retained: {trainer.last}')


@contextmanager
def _preserve_validation_state(trainer):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generators = []
    for name in ('train_loader', 'test_loader'):
        generator = getattr(getattr(trainer, name, None), 'generator', None)
        if generator is not None:
            generators.append((generator, generator.get_state()))
    modes = {id(module): (module, module.training)
             for model in (trainer.model, trainer.ema.ema) for module in model.modules()}
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for generator, state in generators:
            generator.set_state(state)
        for module, training in modes.values():
            module.training = training


class BaselineFinetuneTrainer(WeatherHFCTrainer):
    """Baseline candidate with explicit warmup, initial evaluation and budget pauses."""

    VALIDATION_INTERVAL_UPDATES = 1000

    def __init__(self, *args, finetune_policy, execution_budget_epochs=1, **kwargs):
        self.finetune_policy = validate_finetune_policy(finetune_policy)
        if type(execution_budget_epochs) is not int or execution_budget_epochs < 1:
            raise ValueError('execution_budget_epochs must be a positive integer')
        self.execution_budget_epochs = execution_budget_epochs
        self.finetune_attempts = self.finetune_successes = self.finetune_skips = 0
        self._last_saved_epoch = None
        self.paused_by_budget = False
        super().__init__(*args, **kwargs)
        self.add_callback('on_fit_epoch_end', pause_at_epoch_budget)

    @staticmethod
    def _require_baseline(model):
        expected = {'haar': False, 'freq': False, 'consistency': False}
        if model.yaml.get('weather_hfc') != expected or getattr(model, 'consistency_enabled', False):
            raise ValueError('Baseline finetune profile requires all H/F/C factors disabled')

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self._require_baseline(model)
        return model

    def get_warmup_iterations(self, nb):
        return self.finetune_policy['warmup_updates']

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode == 'val':
            batch_size = self.finetune_policy['validation_batch']
        return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

    def get_validator(self):
        from weather_validation_groups import ConditionWeatherValidator
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss', 'region_loss'
        settings = copy(self.args)
        settings.batch = self.finetune_policy['validation_batch']
        return ConditionWeatherValidator(self.test_loader, manifest=self.manifest,
                                         save_dir=self.save_dir, args=settings)

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self._require_baseline(self.model)
        if not self.resume:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp,
                init_scale=self.finetune_policy['initial_amp_scale'])
            self._evaluate_initial()

    def _call_native_validation(self):
        # Do not use standalone AutoBackend on the live EMA: it can fuse it.
        with _preserve_validation_state(self):
            metrics = self.validator(trainer=self)
        metrics = {key: float(value) for key, value in metrics.items()}
        if 'fitness' not in metrics or not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError('finetune validation requires finite metrics and fitness')
        return metrics

    def _record_validation(self, stage, metrics):
        record = dict(stage=stage, epoch=self.epoch + 1, attempts=self.finetune_attempts,
            successes=self.finetune_successes, skips=self.finetune_skips,
            metrics=metrics, groups=deepcopy(getattr(self.validator, 'condition_metrics', {})),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        with (self.save_dir/'finetune_validation.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def _evaluate_initial(self):
        self.epoch = -1
        self.loss = torch.zeros((), device=self.device)
        self.loss_items = torch.zeros(4, device=self.device)
        metrics = self._call_native_validation()
        self.fitness = self.best_fitness = metrics['fitness']
        self.metrics = {key: value for key, value in metrics.items() if key != 'fitness'}
        self.stopper(0, self.fitness)
        self._record_validation('initial', metrics)
        # An unfavourable first training epoch must not overwrite a stronger
        # initialization just because no pre-update reference was recorded.
        torch.save(dict(model=deepcopy(self.ema.ema).cpu().float(), ema=None,
            optimizer=None, epoch=-1, updates=0, best_fitness=self.fitness,
            train_args=vars(self.args), train_metrics=metrics,
            scope='Initial baseline reference, inference only; not a resumable checkpoint'), self.best)

    def validate(self):
        metrics = self._call_native_validation()
        fitness = metrics['fitness']
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
        self._record_validation('epoch', metrics)
        return {key: value for key, value in metrics.items() if key != 'fitness'}, fitness

    def optimizer_step(self):
        before = self.scaler.get_scale()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=.1)
        if not self.amp and not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite FP32 gradient; optimizer was not updated')
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < before
        self.optimizer.zero_grad(set_to_none=True)
        self.finetune_attempts += 1
        if skipped:
            self.finetune_skips += 1
        else:
            self.finetune_successes += 1
            if self.ema:
                self.ema.update(self.model)
        if self.finetune_attempts % 100 == 0:
            print('FINETUNE_PROGRESS ' + json.dumps(dict(attempts=self.finetune_attempts,
                successes=self.finetune_successes, skips=self.finetune_skips,
                lr=[group['lr'] for group in self.optimizer.param_groups],
                amp_scale=self.scaler.get_scale()), allow_nan=False), flush=True)
        if (self.finetune_attempts % self.VALIDATION_INTERVAL_UPDATES == 0
                and self.finetune_attempts % len(self.train_loader) != 0):
            self._evaluate_update()

    def _evaluate_update(self):
        """Observe full validation between epochs without advancing epoch patience."""
        metrics = self._call_native_validation()
        self._record_validation('update', metrics)
        if self.best_fitness is None or metrics['fitness'] > self.best_fitness:
            self.best_fitness = metrics['fitness']
            # This file is only a selected inference model. The complete raw
            # last.pt remains exclusively an epoch-boundary continuation point.
            # Native save_model may choose the latest epoch model at equal fitness.
            torch.save(dict(model=deepcopy(self.ema.ema).cpu().float(), ema=None,
                optimizer=None, epoch=-1, updates=self.finetune_successes,
                best_fitness=self.best_fitness, train_args=vars(self.args), train_metrics=metrics,
                validation_attempt=self.finetune_attempts,
                scope='Mid-epoch validation reference, inference only; resume from raw last.pt',
                equal_fitness_tiebreak='Native epoch-end checkpoint may replace an equal-fitness reference'), self.best)

    def checkpoint_extra_state(self):
        extra = super().checkpoint_extra_state()
        extra['finetune_runtime'] = dict(version=1, policy=deepcopy(self.finetune_policy),
            attempts=self.finetune_attempts, successes=self.finetune_successes, skips=self.finetune_skips)
        return extra

    def _resume_runtime(self, ckpt):
        weather_runtime = super()._resume_runtime(ckpt)
        runtime = ckpt.get('extra_state', {}).get('finetune_runtime')
        if not isinstance(runtime, dict) or runtime.get('version') != 1:
            raise ValueError('finetune checkpoint lacks supported runtime state')
        policy = validate_finetune_policy(runtime.get('policy'))
        if policy != self.finetune_policy:
            raise ValueError('finetune resume policy differs from the checkpoint')
        counts = [runtime.get(key) for key in ('attempts', 'successes', 'skips')]
        if (any(type(value) is not int or value < 0 for value in counts)
                or counts[1] + counts[2] != counts[0] or ckpt.get('updates') != counts[1]):
            raise ValueError('finetune checkpoint has inconsistent update counters')
        return weather_runtime

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        if self.resume:
            runtime = ckpt['extra_state']['finetune_runtime']
            self.finetune_attempts = runtime['attempts']
            self.finetune_successes = runtime['successes']
            self.finetune_skips = runtime['skips']

    def save_model(self):
        super().save_model()
        self._last_saved_epoch = self.epoch

    def final_eval(self):
        # Keep last/best raw states recoverable/inspectable. Only the separately
        # named inference export is stripped, even after a genuine completion.
        if self.best.is_file():
            inference = self.wdir/'best_inference.pt'
            strip_optimizer(self.best, inference)
            self.validator.args.plots = self.args.plots
            metrics = self.validator(model=str(inference))
            self._record_validation('final', {key: float(value) for key, value in metrics.items()})
            self.metrics = {key: value for key, value in metrics.items() if key != 'fitness'}
