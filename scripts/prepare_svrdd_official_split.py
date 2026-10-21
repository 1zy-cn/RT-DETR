"""Rebuild disjoint SVRDD train/valid/test folders from the official Zenodo split lists.

The input is the previously downloaded, overlapping SVRDD folder. The script
never edits it. Lists come from SVRDD_YOLO.zip, Zenodo record 10100129.
"""

import argparse
import filecmp
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = ROOT / 'scripts/svrdd_official_manifest'
SPLITS = ('train', 'valid', 'test')
MANIFEST_FILES = {'train': 'train.txt', 'valid': 'val.txt', 'test': 'test.txt'}


def read_manifests(directory):
    """Return the official filename sets, rejecting repeats and path mistakes."""
    directory = Path(directory)
    groups = {}
    for split in SPLITS:
        paths = (directory / MANIFEST_FILES[split]).read_text(encoding='utf-8-sig').splitlines()
        names = []
        for item in paths:
            parts = item.replace('\\', '/').split('/')
            if len(parts) != 3 or parts[1] != 'images' or not parts[0] or not parts[2].lower().endswith('.jpg'):
                raise ValueError(f'Unexpected {split} manifest entry: {item}')
            names.append(parts[2])
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate filename within {split} manifest')
        groups[split] = set(names)
    if len(set.union(*groups.values())) != sum(map(len, groups.values())):
        raise ValueError('Official manifests overlap')
    return groups


def source_files(source):
    """Index one copy of each image and reject conflicting duplicates."""
    source = Path(source)
    indexed = {}
    total_images = 0
    for split in SPLITS:
        image_dir, label_dir = source / split / 'images', source / split / 'labels'
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f'Missing source split: {split}')
        for image in image_dir.iterdir():
            if not image.is_file() or image.suffix.lower() != '.jpg':
                continue
            label = label_dir / f'{image.stem}.txt'
            if not label.is_file():
                raise FileNotFoundError(f'Missing label for {image}')
            total_images += 1
            if image.name in indexed:
                original_image, original_label = indexed[image.name]
                if not filecmp.cmp(original_image, image, shallow=False) or not filecmp.cmp(original_label, label, shallow=False):
                    raise ValueError(f'Conflicting duplicate: {image.name}')
            else:
                indexed[image.name] = image, label
    return indexed, total_images


def build_dataset(source, destination, manifest_dir=DEFAULT_MANIFESTS, workers=8):
    if workers < 1:
        raise ValueError('workers must be at least 1')
    source, destination, manifest_dir = map(Path, (source, destination, manifest_dir))
    groups = read_manifests(manifest_dir)
    indexed, total_images = source_files(source)
    selected = set.union(*groups.values())
    if selected != set(indexed):
        raise ValueError(f'Manifest/source mismatch: missing={len(selected-set(indexed))}, extra={len(set(indexed)-selected)}')
    if destination.exists():
        raise FileExistsError(f'Destination already exists: {destination}')
    staging = destination.with_name(destination.name + '.building')
    if staging.exists():
        raise FileExistsError(f'Staging directory already exists: {staging}')
    staging.mkdir(parents=True)
    def copy_pair(item):
        name, image_dir, label_dir = item
        image, label = indexed[name]
        shutil.copy2(image, image_dir / name)
        shutil.copy2(label, label_dir / label.name)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split in SPLITS:
            image_dir, label_dir = staging / split / 'images', staging / split / 'labels'
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            tasks = ((name, image_dir, label_dir) for name in sorted(groups[split]))
            for _ in executor.map(copy_pair, tasks):
                pass
            print(f'{split}: {len(groups[split])} images and labels copied', flush=True)
    manifest_copy = staging / 'official_manifest'
    manifest_copy.mkdir()
    for filename in MANIFEST_FILES.values():
        shutil.copy2(manifest_dir / filename, manifest_copy / filename)
    report = {
        'official_source': 'https://zenodo.org/records/10100129 (SVRDD_YOLO.zip)',
        'source_directory': str(source),
        'destination_directory': str(destination),
        'counts': {split: len(groups[split]) for split in SPLITS},
        'source_file_instances': total_images,
        'source_unique_images': len(indexed),
        'source_duplicate_instances': total_images - len(indexed),
        'manifest_sha256': {
            filename: hashlib.sha256((manifest_dir / filename).read_bytes()).hexdigest()
            for filename in MANIFEST_FILES.values()
        },
    }
    (staging / 'split_provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    staging.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest-dir', type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument('--workers', type=int, default=8, help='Parallel file copies (default: 8)')
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.source, args.destination, args.manifest_dir, args.workers), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
