"""Resolution / aspect-ratio analysis for the interior-segmentation pipeline.

Two independent analyses (no GPU, header-only image reads):

  1. CROP DOWNSCALING — for every GT-extracted blob crop in data/processed/blob_crops.csv,
     read its native (bbox) resolution and decide whether the refiner's 512x512 resize
     scales it DOWN. The refiner resize is non-aspect-preserving (each axis to 512), so an
     axis is downscaled iff its native length > 512.

  2. ASPECT RATIO — for every full-frame interior image in data/interior_segmentation/index.csv,
     read its native (w,h) and aspect ratio (w/h), reported separately for the train and test
     (val) split using the repo's deterministic split (np.random.default_rng(seed) shuffle,
     first val_split fraction = val), matching dataset.py:MyData.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFINER_SIZE = 512
VAL_SPLIT = 0.2
SEED = 42

Image.MAX_IMAGE_PIXELS = None  # these are big frames; don't trip the decompression-bomb guard


def _resolve(rel, csv_dir):
    if os.path.isabs(rel):
        return rel
    for root in (REPO, os.getcwd(), csv_dir):
        cand = os.path.join(root, rel)
        if os.path.exists(cand):
            return cand
    return os.path.join(REPO, rel)


def _read_size(path):
    """Return (w, h) from the image header only, or None on failure."""
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


def _load_sizes(csv_path, img_col_candidates):
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    rows = []  # (abs_path, dataset)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rel = next((row[c].strip() for c in img_col_candidates if row.get(c)), "")
            if not rel:
                continue
            rows.append((_resolve(rel, csv_dir), row.get("dataset", "")))
    paths = [r[0] for r in rows]
    with ThreadPoolExecutor(max_workers=32) as ex:
        sizes = list(ex.map(_read_size, paths))
    out = []  # (w, h, dataset)
    for (_, ds), sz in zip(rows, sizes):
        if sz is not None:
            out.append((sz[0], sz[1], ds))
    n_fail = sum(1 for s in sizes if s is None)
    return out, n_fail


def _split_mask(n):
    """Boolean array of length n: True = val/test, matching dataset.py."""
    rng = np.random.default_rng(SEED)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * VAL_SPLIT))
    is_val = np.zeros(n, dtype=bool)
    is_val[idx[:n_val]] = True
    return is_val


def _pct(x):
    return f"{100 * x:6.2f}%"


def _describe(name, vals):
    a = np.asarray(vals, dtype=float)
    qs = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    print(f"  {name:>14}: mean={a.mean():8.2f}  "
          f"min={qs[0]:7.1f}  p5={qs[1]:7.1f}  p25={qs[2]:7.1f}  "
          f"median={qs[3]:7.1f}  p75={qs[4]:7.1f}  p95={qs[5]:7.1f}  max={qs[6]:7.1f}")


# ------------------------------------------------------------------ Task 1: crops
def analyze_crops():
    csv_path = os.path.join(REPO, "data/processed/blob_crops.csv")
    print("=" * 78)
    print(f"TASK 1 — CROP DOWNSCALING vs refiner input {REFINER_SIZE}x{REFINER_SIZE}")
    print(f"  source: {csv_path}")
    sizes, n_fail = _load_sizes(csv_path, ["image", "image_path"])
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])
    n = len(sizes)
    print(f"  crops read: {n}  (unreadable: {n_fail})\n")

    mx = np.maximum(w, h)
    mn = np.minimum(w, h)
    down_w = w > REFINER_SIZE
    down_h = h > REFINER_SIZE
    down_any = mx > REFINER_SIZE          # at least one axis shrunk
    down_both = mn > REFINER_SIZE         # both axes shrunk
    up_both = mx < REFINER_SIZE           # whole crop enlarged (both axes < 512)
    # area-equivalent linear scale for the square resize (geometric mean of per-axis scales)
    area_scale = np.sqrt((REFINER_SIZE / w) * (REFINER_SIZE / h))

    print("  Native crop dimensions (px):")
    _describe("width", w)
    _describe("height", h)
    _describe("max(w,h)", mx)
    _describe("min(w,h)", mn)
    print()
    print("  Downscaling breakdown (refiner resizes each axis independently to 512):")
    print(f"    width  > 512 (width shrunk) ............ {_pct(down_w.mean())}  ({down_w.sum()})")
    print(f"    height > 512 (height shrunk) ........... {_pct(down_h.mean())}  ({down_h.sum()})")
    print(f"    max(w,h) > 512 (>=1 axis shrunk) ....... {_pct(down_any.mean())}  ({down_any.sum()})  <-- 'resized down'")
    print(f"    min(w,h) > 512 (both axes shrunk) ...... {_pct(down_both.mean())}  ({down_both.sum()})")
    print(f"    max(w,h) < 512 (whole crop enlarged) ... {_pct(up_both.mean())}  ({up_both.sum()})")
    print(f"    area-eq linear scale < 1 (net shrink) .. {_pct((area_scale < 1).mean())}  ({(area_scale < 1).sum()})")
    print()
    # how far down do the shrunk ones go?
    shrunk = area_scale[area_scale < 1]
    if len(shrunk):
        print(f"  Among net-shrunk crops, area-eq linear scale: "
              f"median={np.median(shrunk):.2f}  p5={np.percentile(shrunk,5):.2f}  min={shrunk.min():.2f}")
    print()


# ------------------------------------------------------------------ Task 2: aspect ratio
def analyze_aspect():
    csv_path = os.path.join(REPO, "data/interior_segmentation/index.csv")
    print("=" * 78)
    print("TASK 2 — ASPECT RATIO of full-frame interior images (train vs test split)")
    print(f"  source: {csv_path}  (split: seed={SEED}, val_split={VAL_SPLIT})")
    sizes, n_fail = _load_sizes(csv_path, ["image_path", "image"])
    n = len(sizes)
    is_val = _split_mask(n)
    print(f"  images read: {n}  (unreadable: {n_fail})  "
          f"train={int((~is_val).sum())}  test={int(is_val.sum())}\n")

    w = np.array([s[0] for s in sizes], dtype=float)
    h = np.array([s[1] for s in sizes], dtype=float)
    ar = w / h

    def report(label, m):
        print(f"  [{label}]  n={int(m.sum())}")
        _describe("width", w[m])
        _describe("height", h[m])
        _describe("aspect w/h", ar[m])
        # bucket by common orientations
        b = Counter()
        for v in ar[m]:
            if v < 0.95:
                b["portrait (<0.95)"] += 1
            elif v <= 1.05:
                b["~square (0.95-1.05)"] += 1
            elif v < 1.5:
                b["landscape (1.05-1.5)"] += 1
            elif v < 1.85:
                b["wide (1.5-1.85)"] += 1
            else:
                b["ultra-wide (>=1.85)"] += 1
        tot = m.sum()
        for k in ["portrait (<0.95)", "~square (0.95-1.05)", "landscape (1.05-1.5)",
                  "wide (1.5-1.85)", "ultra-wide (>=1.85)"]:
            print(f"      {k:>24}: {_pct(b[k]/tot)}  ({b[k]})")
        # most common exact (w,h) resolutions
        res = Counter((int(a), int(bb)) for a, bb in zip(w[m], h[m]))
        print("      top resolutions:", ", ".join(f"{a}x{b}:{c}" for (a, b), c in res.most_common(5)))
        print()

    report("TRAIN", ~is_val)
    report("TEST", is_val)

    # per-dataset aspect overview (full set)
    by_ds = defaultdict(list)
    for ww, hh, ds in sizes:
        by_ds[ds].append(ww / hh)
    print("  Aspect ratio by dataset (full set):")
    for ds in sorted(by_ds):
        v = np.array(by_ds[ds])
        print(f"    {ds:>42}: n={len(v):5d}  median AR={np.median(v):.3f}  "
              f"range=[{v.min():.2f},{v.max():.2f}]")


if __name__ == "__main__":
    analyze_crops()
    analyze_aspect()
