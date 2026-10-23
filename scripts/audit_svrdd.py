"""Read-only SVRDD integrity audit. Run from the repository: python -m scripts.audit_svrdd --data ..."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.prepare_svrdd_official_split import DEFAULT_MANIFESTS, MANIFEST_FILES, read_manifests

CLASS_NAMES = ['D00-纵向裂缝', 'D10-横向裂缝', 'D20-网状裂缝', 'D40-坑槽', '井盖', '纵向修补', '横向修补']


def _image_digest(image):
    import cv2
    import numpy as np
    image_bytes = image.read_bytes()
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or min(decoded.shape[:2]) < 10:
        raise ValueError(f'Corrupt or too small image: {image}')
    return image.name, hashlib.sha256(image_bytes).digest()


def audit_dataset(data_yaml, manifest_dir=DEFAULT_MANIFESTS, deep=False):
    """Check filenames, labels and class order; deep adds decode and content-leak checks.

    Content hashes detect byte-identical images, not near-duplicate street scenes.
    No annotations or images are repaired or silently discarded.
    """
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8-sig'))
    names = cfg.get('names')
    if isinstance(names, dict):
        names = [names.get(i) for i in range(7)] if set(names) == set(range(7)) else None
    if cfg.get('nc') != 7 or names != CLASS_NAMES:
        raise ValueError('SVRDD class order must be D00,D10,D20,D40,manhole,longitudinal repair,transverse repair; '
                         'use the canonical names from SVRDD.yaml')
    root = Path(cfg['path'])
    root = (root if root.is_absolute() else data_yaml.parent / root).resolve()
    expected = read_manifests(manifest_dir)
    if Path(manifest_dir).resolve() == DEFAULT_MANIFESTS.resolve() and list(map(len, expected.values())) != [6000, 1000, 1000]:
        raise ValueError('Official SVRDD manifests must have 6000/1000/1000 entries')
    report = dict(counts={s: len(v) for s, v in expected.items()}, class_names=names,
                  deep=deep, splits={}, cross_split_duplicates=[], within_split_duplicates=[],
                  manifest_sha256={}, warnings=[])
    for split, file in MANIFEST_FILES.items():
        # Canonical LF allows the same manifest fingerprint after a Windows Git checkout.
        content = (Path(manifest_dir) / file).read_text(encoding='utf-8-sig')
        report['manifest_sha256'][split] = hashlib.sha256(content.encode()).hexdigest()
    dataset_hash, image_hashes = hashlib.sha256(), defaultdict(list)
    for split, key in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
        image_dir = (root / cfg[key]).resolve()
        label_dir = image_dir.parent / 'labels'
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f'Missing official split directories: {image_dir}, {label_dir}')
        images = {p.name for p in image_dir.iterdir() if p.is_file()}
        labels = {p.stem for p in label_dir.glob('*.txt')}
        if images != expected[split] or labels != {Path(n).stem for n in images}:
            raise ValueError(f'SVRDD {split} does not match official split manifest or paired labels')
        stats = dict(images=len(images), boxes_per_class=[0] * 7, effective_boxes_per_class=[0] * 7, empty_labels=0,
                     duplicate_label_rows=0, boundary_boxes=0, label_cache_present=(image_dir.parent / 'labels.cache').exists())
        for name in sorted(images):
            image, label = image_dir / name, label_dir / f'{Path(name).stem}.txt'
            raw = label.read_bytes()
            rows = raw.decode('utf-8-sig').splitlines()
            seen = set()
            nonempty = 0
            for number, line in enumerate(rows, 1):
                if not line.strip():
                    continue
                nonempty += 1
                try:
                    values = tuple(map(float, line.split()))
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError()
                    cls, x, y, w, h = values
                    if cls != int(cls) or not 0 <= cls < 7 or not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                        raise ValueError()
                except (ValueError, OverflowError) as error:
                    raise ValueError(f'Invalid YOLO detection label: {label}:{number}: {line}') from error
                if values in seen:
                    stats['duplicate_label_rows'] += 1
                else:
                    stats['effective_boxes_per_class'][int(cls)] += 1
                seen.add(values)
                stats['boxes_per_class'][int(cls)] += 1
                if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                    stats['boundary_boxes'] += 1
            stats['empty_labels'] += int(nonempty == 0)
            dataset_hash.update(f'{split}/{name}\0'.encode())
            dataset_hash.update(hashlib.sha256(raw).digest())
        if deep:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for name, digest in pool.map(_image_digest, (image_dir / n for n in sorted(images))):
                    image_hashes[digest.hex()].append((split, name))
                    dataset_hash.update(name.encode())
                    dataset_hash.update(digest)
        report['splits'][split] = stats
        if stats['duplicate_label_rows']:
            report['warnings'].append(f'{split}: {stats["duplicate_label_rows"]} exact duplicate label rows; '
                                      'Ultralytics removes duplicates in memory; source labels are retained')
        if stats['boundary_boxes']:
            report['warnings'].append(f'{split}: {stats["boundary_boxes"]} boxes cross image borders; inspect source labels')
        if not all(stats['boxes_per_class']):
            report['warnings'].append(f'{split}: one or more classes have no boxes')
    for locations in image_hashes.values():
        if len(locations) > 1:
            kind = 'cross_split_duplicates' if len({s for s, _ in locations}) > 1 else 'within_split_duplicates'
            report[kind].append(locations)
    if report['cross_split_duplicates']:
        raise ValueError(f'Byte-identical cross-split images: {report["cross_split_duplicates"][:5]}')
    report['dataset_sha256'] = dataset_hash.hexdigest()
    report['fingerprint_scope'] = 'paths+raw_labels+raw_images' if deep else 'paths+raw_labels'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    report = audit_dataset(options.data, deep=True)
    content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(content, encoding='utf-8')
    print(content)


if __name__ == '__main__':
    main()
