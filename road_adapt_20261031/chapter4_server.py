"""Prepare server-local Chapter 4 inputs without copying or changing the dataset."""
import hashlib
from pathlib import Path

from scripts import configure_chapter4_data as configuration

APPROVED_WEIGHT_SHA256 = 'faf76f76820986b8affaa2de9d91c0c7c3a76c105de3814430d87735d1bcd69b'
CLASS_NAMES = ('D00-纵向裂缝', 'D10-横向裂缝', 'D20-网状裂缝', 'D40-坑槽',
               '井盖', '纵向修补', '横向修补')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_inputs(data_root, original_weights, output):
    """Fail before training on missing/mixed versions; regenerate only local configs.

    The approved manifest proves the planned source-level split and memberships.
    File existence here checks transfer completeness, not pixel/label hashes or
    human visibility. Full original/output label audit is recorded separately.
    The old data-root YAML is read only for class names, never for its old lists.
    """
    root, weights, output = (Path(p).expanduser().resolve()
                              for p in (data_root, original_weights, output))
    if output == root or root in output.parents:
        raise ValueError('Generated configs must stay outside the read-only dataset root / 数据根')
    if not weights.is_file():
        raise FileNotFoundError(f'Original-image baseline checkpoint does not exist / 权重缺失: {weights}')
    if _sha256(weights) != APPROVED_WEIGHT_SHA256:
        raise ValueError('Original baseline checkpoint / 原图初始化权重 differs from the approved version; '
                         f'expected SHA256 {APPROVED_WEIGHT_SHA256}. '
                         'Use the shared original-image RT-DETR-R18, not weather-adapted or Chapter 3 improved weights.')
    for name in ('manifest.jsonl', 'data_weather_train.yaml'):
        if not (root/name).is_file():
            raise FileNotFoundError(f'Weather dataset file missing / 缺失: {root/name}. '
                                    'git pull transfers code, not the 44,000-image weather dataset.')
    names = configuration.read_names(root/'data_weather_train.yaml')
    if names != dict(enumerate(CLASS_NAMES)):
        raise ValueError('The dataset class / 类别 names and ID order differ from the approved seven classes')
    rows, _, _ = configuration.load_manifest(root)
    missing = sum(not (root/row[field]).is_file()
                  for row in rows for field in ('output_image', 'output_label'))
    if missing:
        raise FileNotFoundError(f'Weather dataset has {missing} missing / 缺失 image or label files; '
                                'finish copying the existing weather dataset before training.')
    # All generated files are owned by this project and intentionally untracked.
    # Regeneration updates old Windows paths after a server move; data stays read-only.
    configuration.configure(root, output, force=True)
    return output/'dataset.yaml'
