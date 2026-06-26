import math
import unittest

import torch

from adaptive_temp import (
    adaptive_tskd_loss,
    adaptive_tskd_softmax,
    compute_entropy,
    compute_lambda_from_confidence,
    compute_lambda_from_entropy,
    init_entropy_memory,
    update_entropy_memory,
)
from attention_enhance import generate_attention_map, get_attention_boxes
from training_utils import (
    apply_logit_adjustment,
    build_class_balanced_weights,
    build_sample_weights,
    find_best_binary_threshold,
    init_class_prototypes,
    prototype_regularization_loss,
    tta_forward,
    update_class_prototypes,
)


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

    def test_confidence_inverse_lambda_gives_uncertain_samples_larger_lambda(self):
        confidence = torch.tensor([0.95, 0.50])

        lambda_i = compute_lambda_from_confidence(
            confidence, num_classes=2, lambda_min=0.8, lambda_max=1.2
        )

        self.assertLess(float(lambda_i[0]), float(lambda_i[1]))
        self.assertGreaterEqual(float(lambda_i.min()), 0.8)
        self.assertLessEqual(float(lambda_i.max()), 1.2)


class TrainingUtilityTests(unittest.TestCase):
    def test_class_balanced_weights_give_rare_classes_larger_weight(self):
        weights = build_class_balanced_weights(
            torch.tensor([90, 10]), beta=0.99, device=torch.device("cpu")
        )

        self.assertEqual(tuple(weights.shape), (2,))
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_sample_weights_follow_inverse_class_frequency(self):
        labels = torch.tensor([0, 0, 0, 1])

        sample_weights = build_sample_weights(labels, num_classes=2)

        self.assertEqual(tuple(sample_weights.shape), (4,))
        self.assertGreater(float(sample_weights[3]), float(sample_weights[0]))

    def test_logit_adjustment_lowers_minority_prior_more_during_training(self):
        logits = torch.zeros(1, 2)
        class_counts = torch.tensor([90, 10])

        adjusted = apply_logit_adjustment(logits, class_counts, tau=1.0)

        self.assertGreater(float(adjusted[0, 0]), float(adjusted[0, 1]))

    def test_binary_threshold_search_can_improve_accuracy(self):
        probs = torch.tensor([0.10, 0.20, 0.40, 0.45, 0.60])
        targets = torch.tensor([0, 0, 1, 1, 1])

        threshold, acc = find_best_binary_threshold(probs, targets)

        self.assertLess(float(threshold), 0.5)
        self.assertAlmostEqual(float(acc), 1.0, places=6)

    def test_tta_forward_averages_original_and_flipped_predictions(self):
        class MeanModel(torch.nn.Module):
            def forward(self, x):
                left = x[:, :, :, :2].mean(dim=(1, 2, 3))
                right = x[:, :, :, 2:].mean(dim=(1, 2, 3))
                return torch.stack([left, right], dim=1)

        images = torch.zeros(1, 1, 2, 4)
        images[:, :, :, :2] = 1.0

        logits = tta_forward(MeanModel(), images, tta_views=2)

        self.assertTrue(torch.allclose(logits, torch.tensor([[0.5, 0.5]]), atol=1e-6))

    def test_class_prototypes_initialize_with_expected_shapes(self):
        prototypes, counts = init_class_prototypes(
            num_classes=3,
            feature_dim=128,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(prototypes.shape), (3, 128))
        self.assertEqual(tuple(counts.shape), (3,))
        self.assertTrue(torch.allclose(prototypes, torch.zeros_like(prototypes)))
        self.assertTrue(torch.equal(counts, torch.zeros_like(counts)))

    def test_class_prototypes_update_only_observed_classes(self):
        prototypes, counts = init_class_prototypes(3, 2, device=torch.device("cpu"))
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
        labels = torch.tensor([0, 1, 1])

        update_class_prototypes(
            prototypes,
            counts,
            features,
            labels,
            momentum=0.9,
            min_count=1,
        )

        self.assertTrue(torch.allclose(prototypes[0], torch.tensor([1.0, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(prototypes[1], torch.tensor([0.0, 2.0]), atol=1e-6))
        self.assertTrue(torch.allclose(prototypes[2], torch.zeros(2), atol=1e-6))
        self.assertEqual(int(counts[0]), 1)
        self.assertEqual(int(counts[1]), 2)
        self.assertEqual(int(counts[2]), 0)

    def test_prototype_regularization_loss_backpropagates_to_features(self):
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        features = torch.tensor([[0.9, 0.1], [0.2, 0.8]], requires_grad=True)
        labels = torch.tensor([0, 1])

        loss = prototype_regularization_loss(
            features,
            labels,
            prototypes,
            temperature=0.2,
        )
        loss.backward()

        self.assertEqual(tuple(loss.shape), ())
        self.assertIsNotNone(features.grad)
        self.assertGreaterEqual(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
