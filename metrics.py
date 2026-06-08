"""Shared segmentation metric + image-panel helpers.

Lives in its own module (rather than train.py) so other entry points — e.g.
train_refiner.py — can reuse them without importing train.py, whose module body
parses argv and builds a Config() on import.
"""
import torch
import torch.nn.functional as F


# ImageNet normalize constants — used to denormalize tensors before logging them as images.
_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _denormalize(x):
    return (x.float().cpu() * _IMNET_STD + _IMNET_MEAN).clamp(0, 1)


def _to_3ch(x):
    x = x.float().cpu()
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def _overlay(image, mask, color=(1.0, 0.2, 0.2), alpha=0.5):
    mask = mask.float().cpu().clamp(0, 1)
    c = torch.tensor(color).view(1, 3, 1, 1)
    return (image * (1 - mask * alpha) + c * (mask * alpha)).clamp(0, 1)


def _ring(mask, r=2):
    """Dilation − erosion ⇒ a ring of width ~2r along the contour."""
    k = 2 * r + 1
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=r)
    ero = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=r)
    return (dil - ero).clamp(0, 1)


@torch.no_grad()
def _contour_miou(pred_prob, gt, r=2, thresh=0.5, eps=1e-6):
    """Boundary IoU: IoU on ring masks around the contour. Sensitive to edge sloppiness in a way
    plain IoU is not — interior pixels dominate plain IoU on large blobs."""
    pb = (pred_prob > thresh).float()
    gb = (gt > 0.5).float()
    pr_ring = _ring(pb, r)
    gt_ring = _ring(gb, r)
    inter = (pr_ring * gt_ring).flatten(1).sum(1)
    union = (pr_ring + gt_ring - pr_ring * gt_ring).flatten(1).sum(1)
    return ((inter + eps) / (union + eps)).mean().item()


def _binary_metrics(pred_prob, gt, eps=1e-7, contour_radius=2):
    """Per-batch mean IoU / F1 / MAE / Contour_mIoU on a foreground-binary mask.
    pred_prob in [0,1], gt in [0,1]."""
    p = (pred_prob > 0.5).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum(dim=(1, 2, 3))
    union = (p + g - p * g).sum(dim=(1, 2, 3))
    iou = (inter / (union + eps)).mean().item()
    tp = inter
    fp = (p * (1 - g)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * g).sum(dim=(1, 2, 3))
    f1 = ((2 * tp) / (2 * tp + fp + fn + eps)).mean().item()
    mae = (pred_prob - gt).abs().mean().item()
    contour = _contour_miou(pred_prob, gt, r=contour_radius)
    return {'iou': iou, 'f1': f1, 'mae': mae, 'contour_miou': contour}
