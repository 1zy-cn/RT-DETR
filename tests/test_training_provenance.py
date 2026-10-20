"""The saved run must include the code that defines its training objective."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from train import EXPERIMENTS, ROOT, on_train_start
from ultralytics import RTDETR


class TrainingProvenanceTest(unittest.TestCase):
    def test_snapshot_contains_iou_loss_and_matcher_sources(self):
        model = RTDETR(str(EXPERIMENTS['baseline']['model']))
        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(
                model=model.model,
                save_dir=Path(directory),
                args=SimpleNamespace(data=str(ROOT / 'SVRDD.yaml')),
            )
            on_train_start(trainer, experiment='baseline')
            snapshot = Path(directory) / 'source_snapshot'
            for filename in ('metrics.py', 'loss.py', 'ops.py'):
                with self.subTest(filename=filename):
                    self.assertTrue((snapshot / filename).is_file(), filename)


if __name__ == '__main__':
    unittest.main()
