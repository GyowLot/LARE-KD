import math
import unittest

import torch

from adaptive_temp import (
    adaptive_tskd_loss,
    adaptive_tskd_softmax,
    compute_entropy,
    compute_lambda_from_entropy,
    init_entropy_memory,
    update_entropy_memory,
)
from attention_enhance import generate_attention_map, get_attention_boxes


class AttentionEnhanceTests(unittest.TestCase):
    def test_generate_attention_map_normalizes_and_upsamples(self):
        feature = torch.tensor(
            [[[[0.0, 2.0], [4.0, 6.0]], [[1.0, 1.0], [1.0, 1.0]]]]
        )

        attn = generate_attention_map(feature, out_size=(4, 4))

        self.assertEqual(tuple(attn.shape), (1, 1, 4, 4))
        self.assertGreaterEqual(float(attn.min()), 0.0)
        self.assertLessEqual(float(attn.max()), 1.0)
        self.assertAlmostEqual(float(attn.max()), 1.0, places=6)

    def test_get_attention_boxes_uses_topk_and_minimum_box(self):
        attn = torch.zeros(1, 1, 8, 8)
        attn[:, :, 1:3, 5:7] = 10.0

        boxes = get_attention_boxes(attn, topk_ratio=0.1, min_box_ratio=0.5)

        self.assertEqual(len(boxes), 1)
        x1, y1, x2, y2 = boxes[0]
        self.assertGreaterEqual(x2 - x1, 4)
        self.assertGreaterEqual(y2 - y1, 4)
        self.assertTrue(0 <= x1 < x2 <= 8)
        self.assertTrue(0 <= y1 < y2 <= 8)
        self.assertLessEqual(x1, 5)
        self.assertGreaterEqual(x2, 7)


class AdaptiveTemperatureTests(unittest.TestCase):
    def test_adaptive_tskd_softmax_scales_target_and_non_target_differently(self):
        logits = torch.tensor([[2.0, 4.0, 6.0]])
        labels = torch.tensor([1])
        lambda_i = torch.tensor([2.0])

        prob = adaptive_tskd_softmax(logits, labels, tau=2.0, lambda_i=lambda_i)

        expected_scaled = torch.tensor([[0.5, 2.0, 1.5]])
        expected = torch.softmax(expected_scaled, dim=1)
        self.assertTrue(torch.allclose(prob, expected, atol=1e-6))

    def test_entropy_memory_updates_with_ema_and_lambda_clamps(self):
        memory = init_entropy_memory(4, num_classes=2, init="uniform")
        entropy = torch.tensor([0.0, math.log(2.0)])
        indices = torch.tensor([1, 3])

        updated = update_entropy_memory(memory, indices, entropy, momentum=0.5)
        lambda_i = compute_lambda_from_entropy(
            updated[indices], num_classes=2, lambda_min=0.5, lambda_max=2.0
        )

        self.assertAlmostEqual(float(updated[1]), math.log(2.0) * 0.5, places=6)
        self.assertAlmostEqual(float(updated[3]), math.log(2.0), places=6)
        self.assertTrue(torch.all(lambda_i >= 0.5))
        self.assertTrue(torch.all(lambda_i <= 2.0))

    def test_adaptive_tskd_loss_detaches_teacher_distribution(self):
        student = torch.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        lambda_i = torch.tensor([1.0, 1.5])

        loss = adaptive_tskd_loss(student, teacher, labels, tau=2.0, lambda_i=lambda_i)
        loss.backward()

        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)
        self.assertGreaterEqual(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
