"""Adaptive temperature scaling utilities for LARE-KD."""

import math

import torch
import torch.nn.functional as F


def adaptive_tskd_softmax(logits, labels, tau, lambda_i):
    """Apply target/non-target temperature scaling and return probabilities.

    Args:
        logits: torch.Tensor with shape [B, C].
        labels: torch.Tensor with shape [B].
        tau: target-class temperature.
        lambda_i: sample-wise non-target multiplier with shape [B].

    Returns:
        torch.Tensor with shape [B, C].
    """
    if logits.dim() != 2:
        raise ValueError("logits must have shape [B, C]")

    labels = labels.long().view(-1).to(logits.device)
    lambda_i = lambda_i.detach().to(logits.device, dtype=logits.dtype).view(-1)
    if lambda_i.numel() != logits.size(0):
        raise ValueError("lambda_i length must match batch size")

    tau_non_target = lambda_i.view(-1, 1) * float(tau)
    scaled_logits = logits / tau_non_target.clamp_min(1e-8)
    target_mask = F.one_hot(labels, num_classes=logits.size(1)).bool()
    scaled_logits = scaled_logits.clone()
    scaled_logits[target_mask] = logits[target_mask] / float(tau)
    return F.softmax(scaled_logits, dim=1)


def adaptive_tskd_loss(student_logits, teacher_logits, labels, tau, lambda_i, sample_weight=None, eps=1e-8):
    """Return KL loss with sample-wise adaptive TSKD distributions.

    Args:
        student_logits: student logits [B, C].
        teacher_logits: teacher logits [B, C].
        labels: ground-truth labels [B].
        tau: target-class temperature.
        lambda_i: sample-wise non-target multiplier [B].
        sample_weight: optional sample-wise KD weight with shape [B].
        eps: numerical stability constant for log.

    Returns:
        Scalar KL-divergence loss.
    """
    lambda_i = lambda_i.detach()
    teacher_prob = adaptive_tskd_softmax(teacher_logits, labels, tau, lambda_i).detach()
    student_prob = adaptive_tskd_softmax(student_logits, labels, tau, lambda_i)
    per_sample_kl = F.kl_div(
        torch.log(student_prob + eps),
        teacher_prob,
        reduction="none",
    ).sum(dim=1)

    if sample_weight is not None:
        sample_weight = sample_weight.detach().to(
            student_logits.device,
            dtype=student_logits.dtype,
        ).view(-1)
        if sample_weight.numel() != student_logits.size(0):
            raise ValueError("sample_weight length must match batch size")
        per_sample_kl = per_sample_kl * sample_weight

    return per_sample_kl.sum() / max(student_logits.size(0), 1)


def compute_entropy(prob, eps=1e-8):
    """Compute sample-wise entropy from class probabilities.

    Args:
        prob: torch.Tensor with shape [B, C].
        eps: numerical stability constant for log.

    Returns:
        torch.Tensor with shape [B].
    """
    return -(prob * torch.log(prob.clamp_min(eps))).sum(dim=1)


def init_entropy_memory(num_samples, num_classes, init="uniform", device=None):
    """Create per-sample entropy memory.

    Args:
        num_samples: number of training samples.
        num_classes: number of classes.
        init: "uniform" initializes to log(num_classes), "zero" to 0.
        device: optional torch device.

    Returns:
        torch.Tensor with shape [num_samples].
    """
    if init == "uniform":
        value = math.log(float(num_classes))
    elif init == "zero":
        value = 0.0
    else:
        raise ValueError("entropy_init must be 'uniform' or 'zero'")
    return torch.full((num_samples,), value, dtype=torch.float32, device=device)


def update_entropy_memory(memory, indices, entropy, momentum=0.9):
    """Update entropy memory with EMA and return the same memory tensor.

    Args:
        memory: torch.Tensor with shape [num_train_samples].
        indices: sample indices for this batch.
        entropy: current teacher entropy for the batch.
        momentum: EMA momentum.

    Returns:
        The updated memory tensor.
    """
    indices = indices.long().to(memory.device)
    entropy = entropy.detach().to(memory.device, dtype=memory.dtype)
    with torch.no_grad():
        memory[indices] = momentum * memory[indices] + (1.0 - momentum) * entropy
    return memory


def compute_lambda_from_entropy(entropy, num_classes, lambda_min=0.5, lambda_max=2.0):
    """Map entropy values to sample-wise lambda values.

    Args:
        entropy: torch.Tensor with shape [B].
        num_classes: number of classes.
        lambda_min: lower clamp.
        lambda_max: upper clamp.

    Returns:
        torch.Tensor with shape [B], detached and clamped.
    """
    if lambda_min > lambda_max:
        raise ValueError("lambda_min must be <= lambda_max")

    max_entropy = math.log(float(num_classes))
    norm_entropy = entropy.detach() / max(max_entropy, 1e-8)
    lambda_i = lambda_min + (lambda_max - lambda_min) * norm_entropy
    return lambda_i.clamp(lambda_min, lambda_max).detach()


def compute_lambda_from_confidence(confidence, num_classes, lambda_min=0.5, lambda_max=2.0):
    """Map teacher confidence to lambda using inverse confidence.

    High-confidence samples receive a lower lambda, while uncertain samples
    receive a higher lambda. Confidence is normalized from [1 / C, 1] to
    uncertainty [1, 0].

    Args:
        confidence: torch.Tensor with shape [B].
        num_classes: number of classes.
        lambda_min: lower clamp.
        lambda_max: upper clamp.

    Returns:
        torch.Tensor with shape [B], detached and clamped.
    """
    if lambda_min > lambda_max:
        raise ValueError("lambda_min must be <= lambda_max")

    min_conf = 1.0 / max(float(num_classes), 1.0)
    denom = max(1.0 - min_conf, 1e-8)
    norm_uncertainty = (1.0 - confidence.detach()) / denom
    lambda_i = lambda_min + (lambda_max - lambda_min) * norm_uncertainty
    return lambda_i.clamp(lambda_min, lambda_max).detach()
