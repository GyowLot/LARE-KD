"""Training utilities for stronger and more stable LARE-KD experiments."""

import copy

import torch


def build_class_balanced_weights(class_counts, beta=0.9999, device=None):
    """Build effective-number class weights normalized to mean 1.

    Args:
        class_counts: 1D tensor/list containing the number of samples per class.
        beta: effective-number smoothing factor. Larger values give stronger
            rare-class compensation.
        device: optional torch device for the returned tensor.

    Returns:
        torch.Tensor with shape [num_classes].
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")

    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    if counts.dim() != 1:
        raise ValueError("class_counts must be a 1D tensor or list")

    counts = counts.clamp_min(1.0)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, device=counts.device), counts)
    weights = (1.0 - beta) / effective_num.clamp_min(1e-8)
    return weights / weights.mean().clamp_min(1e-8)


def extract_class_counts(dataset, num_classes):
    """Extract class counts from MedMNIST or ImageFolder-like datasets.

    Args:
        dataset: dataset object with ``targets`` or ``labels`` attributes.
        num_classes: expected number of classes.

    Returns:
        torch.Tensor with shape [num_classes].
    """
    labels = None
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "dataset"):
        return extract_class_counts(dataset.dataset, num_classes)

    if labels is None:
        raise ValueError("Unable to infer class counts: dataset has no targets/labels attribute")

    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    counts = torch.bincount(labels, minlength=num_classes).float()
    if counts.numel() > num_classes:
        counts = counts[:num_classes]
    return counts


def create_ema_model(model):
    """Create a detached EMA copy of a model for mean-teacher distillation."""
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def update_ema_model(ema_model, model, decay=0.999):
    """Update EMA model parameters and buffers from the online model.

    Args:
        ema_model: detached EMA model.
        model: online student model.
        decay: EMA decay factor.

    Returns:
        The updated EMA model.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")

    model_state = model.state_dict()
    with torch.no_grad():
        for name, ema_value in ema_model.state_dict().items():
            model_value = model_state[name].detach().to(ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)
    ema_model.eval()
    return ema_model


def tta_forward(model, images, tta_views=1):
    """Forward a batch with deterministic flip-based test-time augmentation.

    Args:
        model: classification model.
        images: input tensor [B, C, H, W].
        tta_views: number of deterministic views. Supported values are 1-4:
            original, horizontal flip, vertical flip, and both flips.

    Returns:
        Averaged logits with shape [B, num_classes].
    """
    if tta_views < 1:
        raise ValueError("tta_views must be >= 1")

    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[2, 3]),
    ]
    logits = []
    for transform in transforms[: min(int(tta_views), len(transforms))]:
        logits.append(model(transform(images)))
    return torch.stack(logits, dim=0).mean(dim=0)
