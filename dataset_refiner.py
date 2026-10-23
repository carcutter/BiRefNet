"""Dataset for the standalone UNet mask-refiner.

Each row of `data/processed/blob_crops.csv` (`image, mask, source_image`) yields a
4-channel input and a 1-channel target:

    input  = [ crop RGB (3) | degraded mask (1) ]   (normalized RGB, mask in [0,1])
    target = clean GT crop mask (1)                 in [0,1]

The degraded mask already carries the crop's spatial context, so the full source frame is
not used.

The "degraded mask" simulates a coarse-model prediction: the GT mask is corrupted with
random morphology (erode/dilate), resolution loss (downsample→upsample), affine jitter,
and light noise. Train degradation is random per call; val degradation is *deterministic*
per row (seeded by index) so the val metric is stable across epochs.

Reuses the deterministic CSV split convention from dataset.py:MyData (np.random.default_rng
shuffle + val_split slice, flipped is_train ⇒ disjoint subsets).
"""
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


# ImageNet stats — matches dataset.py:56-58 so the pretrained smp encoder sees familiar inputs.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# Default degradation knobs; overridden from configs/refiner.yaml via train_refiner.py.
DEFAULT_DEGRADE_CFG = {
    'morph_prob': 0.8,      # P(apply erosion or dilation)
    'max_radius': 6,        # max morphology kernel radius (px); kernel = 2r+1
    'downsample_prob': 0.8, # P(downsample→upsample blur)
    'max_down': 8,          # max downsample factor
    'affine_prob': 0.4,     # P(small affine jitter)
    'max_translate': 0.04,  # fraction of size
    'max_rotate': 5.0,      # degrees
    'scale_jitter': 0.05,   # ± fraction
    'noise_prob': 0.3,      # P(additive gaussian noise)
    'noise_std': 0.05,
}


def _resolve_factory(csv_path, csv_image_root=None):
    """Candidate-root path resolver, same strategy as dataset.py:69-79."""
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    candidate_roots = [r for r in (csv_image_root, os.getcwd(), csv_dir) if r]

    def _resolve(rel):
        if os.path.isabs(rel):
            return rel
        for root in candidate_roots:
            cand = os.path.join(root, rel)
            if os.path.exists(cand):
                return cand
        return os.path.join(candidate_roots[0], rel)
    return _resolve


def degrade_mask(gt, cfg, rng, gen=None):
    """Corrupt a clean GT mask to look like a coarse prediction.

    gt:  (1, H, W) float tensor in [0,1].
    cfg: dict of knobs (see DEFAULT_DEGRADE_CFG).
    rng: random.Random for scalar choices (seed it for deterministic val).
    gen: optional torch.Generator for noise (seed it for deterministic val).
    Returns a soft mask (1, H, W) in [0,1] (kept soft to mimic a probability map).
    """
    x = gt.unsqueeze(0)  # (1,1,H,W)
    H, W = x.shape[-2:]

    # 1. Affine jitter — simulates coarse-mask misalignment.
    if rng.random() < cfg['affine_prob']:
        angle = rng.uniform(-cfg['max_rotate'], cfg['max_rotate'])
        tx = rng.uniform(-cfg['max_translate'], cfg['max_translate']) * W
        ty = rng.uniform(-cfg['max_translate'], cfg['max_translate']) * H
        scale = 1.0 + rng.uniform(-cfg['scale_jitter'], cfg['scale_jitter'])
        x = TF.affine(x, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                      interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)

    # 2. Morphology — random erosion OR dilation via the max_pool2d trick (cf. metrics._ring).
    if rng.random() < cfg['morph_prob']:
        r = rng.randint(1, cfg['max_radius'])
        k = 2 * r + 1
        if rng.random() < 0.5:
            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=r)            # dilation
        else:
            x = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=r)          # erosion

    # 3. Resolution loss — downsample then upsample to soften/blur the boundary.
    if rng.random() < cfg['downsample_prob']:
        f = rng.randint(2, cfg['max_down'])
        h2, w2 = max(1, H // f), max(1, W // f)
        x = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

    # 4. Light additive noise.
    if rng.random() < cfg['noise_prob']:
        noise = torch.randn(x.shape, generator=gen) * cfg['noise_std']
        x = x + noise

    return x.squeeze(0).clamp(0, 1)


class RefinerData(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_size=512, is_train=True, val_split=0.2,
                 csv_split_seed=42, max_samples=0, csv_image_root=None, degrade_cfg=None,
                 use_mask_input=True):
        self.image_size = int(image_size)
        self.is_train = is_train
        self.use_mask_input = use_mask_input   # True ⇒ 4ch [crop|degraded mask]; False ⇒ 3ch [crop]
        self.cfg = dict(DEFAULT_DEGRADE_CFG, **(degrade_cfg or {}))

        resolve = _resolve_factory(csv_path, csv_image_root)
        rows = []  # (crop_abs, mask_abs)
        with open(csv_path, 'r', newline='') as f:
            for row in csv.DictReader(f):
                crop = (row.get('image') or row.get('image_path') or '').strip()
                mask = (row.get('mask') or row.get('mask_path') or '').strip()
                if not crop or not mask:
                    continue
                rows.append((resolve(crop), resolve(mask)))

        # Deterministic split — identical convention to dataset.py:92-100.
        if val_split and 0.0 < val_split < 1.0:
            rng = np.random.default_rng(csv_split_seed)
            idx = np.arange(len(rows))
            rng.shuffle(idx)
            n_val = int(round(len(rows) * val_split))
            val_idx = set(idx[:n_val].tolist())
            keep = (lambda i: i not in val_idx) if is_train else (lambda i: i in val_idx)
            rows = [r for i, r in enumerate(rows) if keep(i)]

        if max_samples and len(rows) > max_samples:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _load_rgb(self, path):
        img = Image.open(path).convert('RGB').resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        return TF.to_tensor(img)  # (3,H,W) in [0,1]

    def _load_mask(self, path):
        m = Image.open(path).convert('L').resize(
            (self.image_size, self.image_size), Image.NEAREST)
        return (TF.to_tensor(m) > 0.5).float()  # (1,H,W) binary

    def __getitem__(self, index):
        crop_path, mask_path = self.rows[index]
        crop = self._load_rgb(crop_path)
        gt = self._load_mask(mask_path)

        # --- Geometric aug (train only): applied jointly so crop+mask stay aligned. ---
        if self.is_train:
            if random.random() < 0.5:
                crop, gt = TF.hflip(crop), TF.hflip(gt)
            if random.random() < 0.2:
                angle = random.uniform(-15, 15)
                crop = TF.rotate(crop, angle, interpolation=TF.InterpolationMode.BILINEAR)
                gt = (TF.rotate(gt, angle, interpolation=TF.InterpolationMode.NEAREST) > 0.5).float()
            # Light color jitter on the crop.
            if random.random() < 0.5:
                b, c, s = (random.uniform(0.8, 1.2) for _ in range(3))
                crop = TF.adjust_brightness(TF.adjust_contrast(TF.adjust_saturation(crop, s), c), b)

        crop_n = TF.normalize(crop, _MEAN, _STD)
        if not self.use_mask_input:
            return crop_n, gt  # (3,H,W) — crop only; plain segmentation, no degraded-mask input

        # --- Degrade GT → coarse-like mask. Deterministic per-row in val. ---
        if self.is_train:
            rng, gen = random, None
        else:
            rng = random.Random(index)
            gen = torch.Generator().manual_seed(index)
        degraded = degrade_mask(gt, self.cfg, rng, gen)
        x = torch.cat([crop_n, degraded], dim=0)  # (4,H,W)
        return x, gt
