"""Fetch a tiny, attributed SVRDD sample for a *training pipeline* smoke test.

This sample is not the thesis training split and must never be used to report AP.
Run: python checks/download_svrdd_smoke.py
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'datasets/svrdd-smoke-hf'
REPO = 'ShuoZheLi/SVRDD_YOLO'
API = f'https://huggingface.co/api/datasets/{REPO}'
NAMES = ['longitudinal crack', 'transverse crack', 'alligator crack', 'pothole',
         'manhole cover', 'longitudinal patch', 'transverse patch']


def fetch_json(url):
    with urlopen(url, timeout=90) as response:
        return json.load(response)


def fetch_bytes(url):
    with urlopen(url, timeout=90) as response:
        return response.read()


def main():
    revision = fetch_json(API)['sha']
    base = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'
    manifest = {'source': f'https://huggingface.co/datasets/{REPO}', 'revision': revision,
                'license': 'CC BY 4.0', 'purpose': 'two-epoch training pipeline smoke test', 'files': []}
    jobs = []
    for source_split, local_split, count in [('train', 'train', 12), ('validation', 'valid', 4)]:
        rows = fetch_bytes(base + f'{source_split}/metadata.jsonl').decode('utf-8').splitlines()
        for line in rows[:count]:
            row = json.loads(line)
            image_path = DEST / local_split / 'images' / Path(row['file_name']).name
            label_path = DEST / local_split / 'labels' / (image_path.stem + '.txt')
            objects = row['objects']
            labels = []
            for cls, box in zip(objects['categories'], objects['bbox_yolo']):
                if not 0 <= cls < len(NAMES) or len(box) != 4 or not all(0 <= v <= 1 for v in box):
                    raise ValueError(f'Invalid object in {row["file_name"]}')
                labels.append(f'{cls} ' + ' '.join(f'{v:.6f}' for v in box))
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text('\n'.join(labels) + '\n', encoding='utf-8')
            jobs.append((row, image_path, base + quote(row['file_name'], safe='/'), local_split))

    def download(job):
        row, image_path, url, local_split = job
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            image_path.write_bytes(fetch_bytes(url))
        data = image_path.read_bytes()
        if len(data) < 1000 or data[:3] != b'\xff\xd8\xff':
            raise ValueError(f'Invalid JPEG: {image_path}')
        return {'split': local_split, 'source': row['file_name'], 'local': str(image_path.relative_to(DEST)),
                'image_sha256': hashlib.sha256(data).hexdigest(), 'objects': len(row['objects']['categories'])}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest['files'] = list(pool.map(download, jobs))
    config = {'path': str(DEST.resolve()), 'train': 'train/images', 'val': 'valid/images',
              'nc': 7, 'names': NAMES}
    (DEST / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                        encoding='utf-8')
    print(f'Downloaded {len(jobs)} real SVRDD images; data={DEST / "data.yaml"}')


if __name__ == '__main__':
    main()
