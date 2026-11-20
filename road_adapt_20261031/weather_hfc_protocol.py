"""Local provenance checks shared by the Chapter 4 entrypoints (no model imports)."""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_inputs(protocol, code_root=ROOT):
    """Include executable dependencies, original weights and all YAML-referenced lists."""
    code_root = Path(code_root)
    paths = [Path(protocol[k]) for k in ('model','data','manifest','weights') if protocol.get(k)]
    if protocol.get('teacher_checkpoint'):
        paths.append(Path(protocol['teacher_checkpoint']))
    paths += list((code_root/'ultralytics').rglob('*.py'))
    paths += [p for p in [code_root/'train_weather_hfc.py', code_root/'weather_hfc_training.py',
                         code_root/'weather_hfc_protocol.py', code_root/'weather_amp.py',
                         code_root/'ultralytics/cfg/default.yaml'] if p.is_file()]
    data = yaml.safe_load(Path(protocol['data']).read_text(encoding='utf-8'))
    root = Path(data.get('path', Path(protocol['data']).parent))
    for key in ('train','val','test'):
        values = data.get(key, [])
        values = values if isinstance(values, list) else [values]
        for value in values:
            path = Path(value)
            path = path if path.is_absolute() else root/path
            if path.suffix == '.txt':
                if not path.is_file():
                    raise FileNotFoundError(f'Dataset list missing: {path}')
                paths.append(path)
    return {str(p.resolve()): sha256(p) for p in dict.fromkeys(paths)}


def validate_resume_record(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Resume requires the original weather_protocol.json: {path}')
    record = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record.get('protocol'), dict) or not record.get('source_sha256'):
        raise ValueError('Resume record lacks frozen protocol/input fingerprints')
    for filename, expected in record['source_sha256'].items():
        source = Path(filename)
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f'Resume input/code changed or is missing: {source}')
    if record.get('status') == 'completed':
        raise ValueError('This run completed; its optimizer may be stripped. Start a new named run instead.')
    return record


def resolve_resume(args):
    checkpoint = args.resume.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    record_path = checkpoint.parent.parent/'weather_protocol.json'
    record = validate_resume_record(record_path)
    protocol = json.loads(json.dumps(record['protocol']))
    required = ('configuration','model','data','manifest','weights','train_args','consistency_weight')
    if any(k not in protocol for k in required):
        raise ValueError('Resume record is not a complete Chapter 4 training protocol')
    run_dir = (Path(protocol['train_args']['project'])/protocol['train_args']['name']).resolve()
    if run_dir != checkpoint.parent.parent:
        raise ValueError('Resume checkpoint is outside the recorded run directory')
    protocol['train_args'].update(model=str(checkpoint), resume=str(checkpoint), exist_ok=True)
    protocol['resume_checkpoint'] = str(checkpoint)
    protocol['resume_record'] = str(record_path)
    return protocol
