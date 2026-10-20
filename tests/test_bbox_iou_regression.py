"""Regression checks for box overlaps used by detection training."""

import unittest

import torch

from ultralytics.utils.metrics import bbox_iou


class BboxIouRegressionTest(unittest.TestCase):
    def test_identical_boxes_have_unit_overlap_for_all_variants(self):
        box = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
        for kwargs in ({}, {'GIoU': True}, {'DIoU': True}, {'CIoU': True}):
            with self.subTest(kwargs=kwargs):
                self.assertAlmostEqual(bbox_iou(box, box, **kwargs).item(), 1.0, delta=1e-4)

    def test_partial_and_disjoint_overlap_in_xyxy_coordinates(self):
        first = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        partial = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
        disjoint = torch.tensor([[3.0, 3.0, 4.0, 4.0]])
        self.assertAlmostEqual(bbox_iou(first, partial, xywh=False).item(), 1 / 3, places=5)
        self.assertAlmostEqual(bbox_iou(first, partial, xywh=False, GIoU=True).item(), 1 / 3, places=5)
        self.assertAlmostEqual(bbox_iou(first, disjoint, xywh=False).item(), 0.0, places=6)

    def test_broadcast_overlap_keeps_finite_gradients(self):
        predicted = torch.tensor([[[0.4, 0.5, 0.2, 0.1]], [[0.6, 0.5, 0.2, 0.1]]],
                                 requires_grad=True)
        target = torch.tensor([[[0.5, 0.5, 0.2, 0.1], [0.6, 0.5, 0.2, 0.1]]])
        overlaps = bbox_iou(predicted, target, GIoU=True)
        self.assertEqual(tuple(overlaps.shape), (2, 2, 1))
        overlaps.sum().backward()
        self.assertTrue(torch.isfinite(predicted.grad).all().item())


if __name__ == '__main__':
    unittest.main()
