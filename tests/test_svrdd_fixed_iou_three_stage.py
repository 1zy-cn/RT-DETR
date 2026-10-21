"""The fixed-IoU SVRDD comparison must isolate the intended model changes."""

import unittest
from unittest.mock import patch

import torch
import yaml
from ultralytics.cfg import get_cfg

from train import EXPERIMENTS, training_args, verify_model, verify_standard_iou
from ultralytics import RTDETR


GROUPS = (
    'svrdd-baseline',
    'svrdd-hsfpn-scsa',
    'svrdd-hsfpn-scsa-dysample',
)


class FixedIouThreeStageTest(unittest.TestCase):
    def test_three_groups_use_one_training_protocol(self):
        args = [training_args(group) for group in GROUPS]
        for arg in args[1:]:
            self.assertEqual(
                {key: value for key, value in args[0].items() if key not in ('name', 'project')},
                {key: value for key, value in arg.items() if key not in ('name', 'project')},
            )
        self.assertEqual(len({arg['name'] for arg in args}), 3)
        self.assertEqual((args[0]['batch'], args[0]['epochs'], args[0]['imgsz']), (8, 300, 640))
        self.assertEqual(args[0]['warmup_bias_lr'], 0.0)
        self.assertEqual(get_cfg(overrides=args[0]).warmup_bias_lr, 0.0)

    def test_models_have_seven_classes_and_only_requested_modules(self):
        expected = ((0, 0, 0), (8, 5, 0), (8, 5, 2))
        for group, (scsa, ca, dysample) in zip(GROUPS, expected):
            with self.subTest(group=group):
                model = RTDETR(str(EXPERIMENTS[group]['model']))
                counts = verify_model(model.model, group)
                self.assertEqual(model.model.model[-1].nc, 7)
                self.assertEqual(model.model.model[-1].nl, 3)
                self.assertEqual(counts['scsa_modules'], scsa)
                self.assertEqual(counts['hsfpn_ca_modules'], ca)
                self.assertEqual(counts['dysample_modules'], dysample)

    def test_training_iou_is_standard(self):
        verify_standard_iou()

    def test_old_inner_iou_is_rejected(self):
        with patch('train.bbox_iou', return_value=torch.tensor([[0.532]])):
            with self.assertRaisesRegex(RuntimeError, 'standard bbox_iou'):
                verify_standard_iou()

    def test_baseline_only_changes_original_class_count(self):
        original = yaml.safe_load((EXPERIMENTS['baseline']['model']).read_text(encoding='utf-8'))
        svrdd = yaml.safe_load((EXPERIMENTS['svrdd-baseline']['model']).read_text(encoding='utf-8'))
        original['nc'] = 7
        self.assertEqual(original, svrdd)

    def test_third_group_only_replaces_two_upsamplers(self):
        combo = yaml.safe_load(EXPERIMENTS['svrdd-hsfpn-scsa']['model'].read_text(encoding='utf-8'))
        dy = yaml.safe_load(EXPERIMENTS['svrdd-hsfpn-scsa-dysample']['model'].read_text(encoding='utf-8'))
        for head_index in (7, 14):  # Model layers 15 and 22.
            self.assertEqual(combo['head'][head_index][2], 'nn.ConvTranspose2d')
            self.assertEqual(dy['head'][head_index][2], 'Dy_Sample')
            dy['head'][head_index] = combo['head'][head_index]
        self.assertEqual(combo, dy)


if __name__ == '__main__':
    unittest.main()
