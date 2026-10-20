"""The official SVRDD split must be reproduced without image leakage."""

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.prepare_svrdd_official_split import build_dataset, read_manifests
from train import verify_official_svrdd_split


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / 'scripts/svrdd_official_manifest'


class OfficialSplitTest(unittest.TestCase):
    def test_official_manifest_lists_all_images_once(self):
        groups = read_manifests(MANIFESTS)
        self.assertEqual([len(groups[s]) for s in ('train', 'valid', 'test')], [6000, 1000, 1000])
        self.assertEqual(len(set().union(*groups.values())), 8000)
        self.assertFalse(groups['train'] & groups['valid'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['valid'] & groups['test'])

    def test_rebuild_uses_manifest_and_copies_each_image_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, dest, manifests = root / 'source', root / 'clean', root / 'manifest'
            manifests.mkdir()
            for split, names in {'train': ('a', 'b'), 'valid': ('c',), 'test': ('d',)}.items():
                (manifests / f"{'val' if split == 'valid' else split}.txt").write_text(
                    ''.join(f'District\\images\\{name}.jpg\n' for name in names), encoding='utf-8')
            for split, names in {'train': ('a', 'b', 'c'), 'valid': ('a', 'c', 'd'), 'test': ('b', 'd')}.items():
                for name in names:
                    images, labels = source / split / 'images', source / split / 'labels'
                    images.mkdir(parents=True, exist_ok=True)
                    labels.mkdir(parents=True, exist_ok=True)
                    (images / f'{name}.jpg').write_bytes(f'image-{name}'.encode())
                    (labels / f'{name}.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
            result = build_dataset(source, dest, manifests)
            self.assertEqual(result['counts'], {'train': 2, 'valid': 1, 'test': 1})
            for split, names in {'train': {'a', 'b'}, 'valid': {'c'}, 'test': {'d'}}.items():
                self.assertEqual({p.stem for p in (dest / split / 'images').iterdir()}, names)
                self.assertEqual({p.stem for p in (dest / split / 'labels').iterdir()}, names)
            self.assertEqual((dest / 'valid/images/c.jpg').read_bytes(), b'image-c')
            data_yaml = root / 'data.yaml'
            data_yaml.write_text(yaml.safe_dump({
                'path': str(dest), 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images'}), encoding='utf-8')
            verify_official_svrdd_split(data_yaml, manifests)
            (dest / 'valid/images/c.jpg').rename(dest / 'valid/images/a.jpg')
            with self.assertRaisesRegex(ValueError, 'official split'):
                verify_official_svrdd_split(data_yaml, manifests)

    def test_conflicting_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, dest, manifests = root / 'source', root / 'clean', root / 'manifest'
            manifests.mkdir()
            (manifests / 'train.txt').write_text('District\\images\\a.jpg\n', encoding='utf-8')
            (manifests / 'val.txt').write_text('', encoding='utf-8')
            (manifests / 'test.txt').write_text('', encoding='utf-8')
            for split, content in (('train', b'one'), ('valid', b'two')):
                images, labels = source / split / 'images', source / split / 'labels'
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / 'a.jpg').write_bytes(content)
                (labels / 'a.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
                build_dataset(source, dest, manifests)

    def test_training_config_uses_new_directory_and_run_names(self):
        data = yaml.safe_load((ROOT / 'SVRDD.yaml').read_text(encoding='utf-8'))
        self.assertTrue(data['path'].endswith('/SVRDD-official-split'))
        from train import EXPERIMENTS
        for experiment in ('svrdd-baseline', 'svrdd-hsfpn-scsa', 'svrdd-hsfpn-scsa-dysample'):
            self.assertIn('officialsplit', EXPERIMENTS[experiment]['name'])


if __name__ == '__main__':
    unittest.main()
