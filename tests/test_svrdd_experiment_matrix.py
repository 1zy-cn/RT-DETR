"""The four core SVRDD ablations must be runnable through one entrypoint."""

import unittest

from train import EXPERIMENTS, verify_model
from ultralytics import RTDETR


class SvrddExperimentMatrixTest(unittest.TestCase):
    def test_hsfpn_only_control_is_registered_and_has_no_scsa(self):
        self.assertIn('hsfpn', EXPERIMENTS)
        settings = EXPERIMENTS['hsfpn']
        self.assertEqual(settings['factors'], ('hsfpn',))
        model = RTDETR(str(settings['model']))
        counts = verify_model(model.model, 'hsfpn')
        self.assertEqual(counts['scsa_modules'], 0)
        self.assertEqual(counts['hsfpn_ca_modules'], 5)


if __name__ == '__main__':
    unittest.main()
