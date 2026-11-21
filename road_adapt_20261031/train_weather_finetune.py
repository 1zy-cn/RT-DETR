"""Preview/run the opt-in baseline finetuning candidate with complete-epoch budgets.

Examples: python train_weather_finetune.py --amp --train
          python train_weather_finetune.py --resume RUN/weights/last.pt --budget-epochs 1 --train

Budget pauses preserve full optimizer/runtime state. They do not mean the
candidate passed scientific validation. Legacy weather training is unchanged.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
from types import SimpleNamespace

from train_weather_hfc import build_protocol as build_legacy_protocol, validate_resume_before_init
from weather_hfc_protocol import freeze_inputs, sha256

ROOT=Path(__file__).resolve().parent
if (ROOT/'weather_hfc_training.py').is_file():
    PROJECT=ROOT
else:
    _source=importlib.util.find_spec('weather_hfc_training')
    if _source is None or not _source.origin:
        raise RuntimeError('Run from the RT-DETR checkout or put it on PYTHONPATH')
    PROJECT=Path(_source.origin).resolve().parent

POLICY=dict(version=1,warmup_updates=64,initial_amp_scale=64.,validation_batch=4)
PROFILE='baseline_finetune_v2'
VALIDATION_INTERVAL_UPDATES=1000
DEFAULT_DATA_ROOT=Path(r'D:\Users\86186\ResearchData\SVRDD-weather-v1')
DEFAULT_WEIGHTS=Path(r'D:\Users\86186\ResearchData\SVRDD-local-runs\svrdd-r18-baseline-officialsplit-local-amp-b8-p20-s0\weights\best.pt')

def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(configuration='baseline',teacher=None,consistency_weight=.1)
    parser.add_argument('--data',type=Path,default=PROJECT/'configs/chapter4/dataset.yaml')
    parser.add_argument('--manifest',type=Path,default=DEFAULT_DATA_ROOT/'manifest.jsonl')
    parser.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument('--project',type=Path,default=PROJECT/'runs/weather_finetune')
    parser.add_argument('--name')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='0')
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--budget-epochs',type=int,default=1,
        help='Operational limit of complete epochs for this invocation, separate from patience')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--train',action='store_true')
    return parser

def build_protocol(args):
    if type(args.budget_epochs) is not int or args.budget_epochs<1:
        raise ValueError('budget-epochs must be a positive integer')
    if args.resume is None and args.name is None:
        precision='amp' if args.amp else 'fp32'
        args.name=f'baseline_finetune_v2_seed{args.seed}_b{args.batch}_{precision}'
    protocol=build_legacy_protocol(args,config_dir=PROJECT/'ultralytics/cfg/models/rt-detr/weather-hfc')
    if args.resume is not None:
        if (protocol.get('training_profile')!=PROFILE or protocol.get('finetune_policy')!=POLICY
                or protocol.get('validation_interval_updates')!=VALIDATION_INTERVAL_UPDATES):
            raise ValueError('Resume requires the same baseline_finetune_v2 policy')
        if protocol.get('configuration')!='baseline':
            raise ValueError('This entrypoint only supports the baseline')
    else:
        protocol['train_args'].update(lr0=1e-5,lrf=1.,warmup_epochs=0.)
        protocol.update(training_profile=PROFILE,finetune_policy=dict(POLICY),
                        validation_interval_updates=VALIDATION_INTERVAL_UPDATES)
    protocol['execution_budget_epochs']=args.budget_epochs
    return protocol

def freeze_candidate_inputs(protocol):
    frozen=freeze_inputs(protocol,code_root=PROJECT)
    for filename in ('train_weather_finetune.py','weather_finetune.py','weather_validation_groups.py'):
        path=ROOT/filename
        if not path.is_file():raise FileNotFoundError(path)
        frozen[str(path.resolve())]=sha256(path)
    return frozen

def _prevalidate_resume(checkpoint,protocol,trainer_class):
    validate_resume_before_init(checkpoint,protocol)
    probe=trainer_class.__new__(trainer_class)
    probe.args=SimpleNamespace(**protocol['train_args'])
    probe.epochs=probe.args.epochs
    probe.consistency_weight=protocol['consistency_weight']
    probe.finetune_policy=protocol['finetune_policy']
    probe._resume_runtime(checkpoint)

def main(argv=None):
    import sys
    parser=make_parser()
    argv=sys.argv[1:] if argv is None else argv
    args=parser.parse_args(argv)
    if args.resume:
        allowed={'--resume','--train','--budget-epochs'}
        forbidden=[token for token in argv if token.startswith('--') and token.split('=')[0] not in allowed]
        if forbidden:parser.error('Resume restores frozen training settings; remove '+', '.join(forbidden))
    try:protocol=build_protocol(args)
    except (ValueError,FileNotFoundError,FileExistsError) as exc:parser.error(str(exc))
    print(json.dumps(protocol,ensure_ascii=False,indent=2),flush=True)
    if not args.train:
        print('Preview only. Add --train to run a bounded complete-epoch stage.')
        return protocol

    import torch
    from weather_finetune import BaselineFinetuneTrainer,BudgetPause
    if args.resume:
        checkpoint=torch.load(protocol['resume_checkpoint'],map_location='cpu',weights_only=False)
        _prevalidate_resume(checkpoint,protocol,BaselineFinetuneTrainer)
        del checkpoint
        record=json.loads(Path(protocol['resume_record']).read_text(encoding='utf-8'))
        record.setdefault('resume_history',[]).append(dict(checkpoint=protocol['resume_checkpoint'],
            checkpoint_sha256=sha256(protocol['resume_checkpoint']),
            resumed_at_utc=datetime.now(timezone.utc).isoformat(),budget_epochs=args.budget_epochs))
        record.update(protocol=protocol,status='starting')
        for key in ('error','error_type'):record.pop(key,None)
    else:
        record=dict(protocol=protocol,source_sha256=freeze_candidate_inputs(protocol),
            python=sys.executable,python_version=platform.python_version(),torch=str(torch.__version__),
            cuda=torch.version.cuda,created_at_utc=datetime.now(timezone.utc).isoformat(),
            scope='Baseline candidate, full-epoch qualification; budget pause is not efficacy acceptance',
            status='starting',consistency_severities=[1,2],random_geometry=False)
    trainer=BaselineFinetuneTrainer(overrides=protocol['train_args'],manifest=Path(protocol['manifest']),
        teacher_checkpoint=None,consistency_weight=protocol['consistency_weight'],
        finetune_policy=protocol['finetune_policy'],execution_budget_epochs=args.budget_epochs)
    print(f"Finetune stage: at most {args.budget_epochs} complete epoch(s) this invocation; "
          f"full validation every {VALIDATION_INTERVAL_UPDATES} updates. "
          "Overall epoch ceiling and patience remain separate.",flush=True)
    record_path=trainer.save_dir/'weather_protocol.json'
    record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        trainer.train()
    except BudgetPause:
        record.update(status='paused_budget',last_checkpoint=str(trainer.last),
            completed_epochs_this_invocation=trainer.epoch-trainer.start_epoch+1,
            last_saved_epoch=trainer.epoch+1)
        print('BUDGET_PAUSED: raw last.pt retained. Resume with the same entrypoint and --resume.',flush=True)
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    else:
        record.update(status='completed',last_checkpoint=str(trainer.last),
            completed_epochs_this_invocation=trainer.epoch-trainer.start_epoch+1,
            last_saved_epoch=trainer.epoch+1)
    finally:
        record['updated_at_utc']=datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record

if __name__=='__main__':main()
