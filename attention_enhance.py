"""Attention-guided local CLAHE enhancement for LARE-KD."""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only when OpenCV is absent.
    cv2 = None


def generate_attention_map(feature, out_size):
    """Return normalized attention maps from feature maps.

    Args:
        feature: torch.Tensor with shape [B, C, H, W].
        out_size: (height, width) for the returned attention map.

    Returns:
        torch.Tensor with shape [B, 1, out_height, out_width], normalized
        sample-wise to [0, 1].
    """
    if feature.dim() != 4:
        raise ValueError("feature must have shape [B, C, H, W]")

    attn = torch.relu(feature.mean(dim=1, keepdim=True))
    b = attn.size(0)
    flat = attn.view(b, -1)
    min_v = flat.min(dim=1)[0].view(b, 1, 1, 1)
    max_v = flat.max(dim=1)[0].view(b, 1, 1, 1)
    attn = (attn - min_v) / (max_v - min_v + 1e-8)
    return F.interpolate(attn, size=out_size, mode="bilinear", align_corners=False)


def get_attention_boxes(attn, topk_ratio=0.2, min_box_ratio=0.25):
    """Select bounding boxes from high-response attention regions.

    Args:
        attn: torch.Tensor with shape [B, 1, H, W].
        topk_ratio: fraction of highest-response pixels used for the mask.
        min_box_ratio: minimum box width/height as a ratio of image size.

    Returns:
        List of boxes [(x1, y1, x2, y2), ...] using exclusive x2/y2 coords.
    """
    if attn.dim() != 4 or attn.size(1) != 1:
        raise ValueError("attn must have shape [B, 1, H, W]")
    if not 0 < topk_ratio <= 1:
        raise ValueError("topk_ratio must be in (0, 1]")
    if not 0 < min_box_ratio <= 1:
        raise ValueError("min_box_ratio must be in (0, 1]")

    attn_cpu = attn.detach().cpu()
    b, _, h, w = attn_cpu.shape
    min_w = max(1, int(round(w * min_box_ratio)))
    min_h = max(1, int(round(h * min_box_ratio)))
    boxes = []

    for idx in range(b):
        amap = attn_cpu[idx, 0]
        flat = amap.reshape(-1)
        k = max(1, int(round(flat.numel() * topk_ratio)))
        threshold = torch.topk(flat, k).values.min()
        mask = amap >= threshold

        if mask.sum().item() == 0:
            x1, y1, x2, y2 = _center_box(w, h, min_w, min_h)
        else:
            ys, xs = torch.where(mask)
            x1 = int(xs.min().item())
            x2 = int(xs.max().item()) + 1
            y1 = int(ys.min().item())
            y2 = int(ys.max().item()) + 1
            x1, y1, x2, y2 = _expand_and_clip_box(x1, y1, x2, y2, w, h, min_w, min_h)
        boxes.append((x1, y1, x2, y2))

    return boxes


def apply_local_clahe_tensor(images, boxes, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply local CLAHE to selected boxes in a tensor batch.

    Args:
        images: torch.Tensor with shape [B, C, H, W], values in [0, 1].
        boxes: list of boxes [(x1, y1, x2, y2), ...].
        clip_limit: OpenCV CLAHE clip limit.
        tile_grid_size: OpenCV CLAHE tile grid size.

    Returns:
        Enhanced torch.Tensor with the same shape, dtype, and device as images.
    """
    if cv2 is None:
        raise ImportError("OpenCV is required for CLAHE. Install it with: pip install opencv-python")
    if images.dim() != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    if len(boxes) != images.size(0):
        raise ValueError("boxes length must match batch size")

    device = images.device
    dtype = images.dtype
    imgs = images.detach().clamp(0, 1).cpu()
    enhanced = imgs.clone()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue

        patch = imgs[i, :, y1:y2, x1:x2]
        np_patch = (patch.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        np_patch = _apply_clahe_np(np_patch, clahe)
        patch_tensor = torch.from_numpy(np_patch).float().permute(2, 0, 1) / 255.0
        enhanced[i, :, y1:y2, x1:x2] = patch_tensor

    return enhanced.to(device=device, dtype=dtype)


def _center_box(width, height, min_w, min_h):
    cx = width // 2
    cy = height // 2
    x1 = cx - min_w // 2
    y1 = cy - min_h // 2
    x2 = x1 + min_w
    y2 = y1 + min_h
    return _clip_box(x1, y1, x2, y2, width, height)


def _expand_and_clip_box(x1, y1, x2, y2, width, height, min_w, min_h):
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_w:
        pad = min_w - box_w
        x1 -= pad // 2
        x2 += pad - pad // 2
    if box_h < min_h:
        pad = min_h - box_h
        y1 -= pad // 2
        y2 += pad - pad // 2
    return _clip_box(x1, y1, x2, y2, width, height)


def _clip_box(x1, y1, x2, y2, width, height):
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    return int(x1), int(y1), int(x2), int(y2)


def _apply_clahe_np(np_patch, clahe):
    if np_patch.ndim == 2:
        return clahe.apply(np_patch)[:, :, None]
    if np_patch.shape[2] == 1:
        return clahe.apply(np_patch[:, :, 0])[:, :, None]
    if np_patch.shape[2] == 3:
        lab = cv2.cvtColor(np_patch, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    channels = [clahe.apply(np_patch[:, :, c]) for c in range(np_patch.shape[2])]
    return np.stack(channels, axis=2)
