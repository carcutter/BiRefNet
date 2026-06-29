"""Blob bounding-box resolution analysis (the real 'resized-down' question).

The stored crops in data/processed/blob_crops.csv were extracted as fixed 512x512
windows (crops.crop_size=512), so at the refiner's 512x512 input there is no resize.
But that fixed window TRUNCATES any blob whose native bbox exceeds 512 px. The
deployment-relevant question is therefore: in the original full-frame images, how big
is each background blob's bounding box, and what fraction exceed 512 px (i.e. cannot fit
the 512 window without losing resolution — truncated today, or downscaled under a
tight-bbox+resize scheme)?

We recompute blob bboxes directly from the GT masks using the exact same logic as the
extraction (interior_segmentation.crops.find_blobs: bg = black+blue, min_area=256, mask
resized to the image resolution), over the full interior dataset.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLING_SRC = "/home/renato/projects/interior-segmentation/interior-segmentation/src"
sys.path.insert(0, SIBLING_SRC)
from interior_segmentation.crops import find_blobs, iter_pairs  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CROP_SIZE = 512
MIN_AREA = 256
BG_COLORS = ((0, 0, 0), (0, 0, 255))
DATA_ROOT = os.path.join(REPO, "data/interior_segmentation")
IMAGE_SUBDIRS = ("raw", "raw_images")
MASK_SUBDIR = "masks"


def _blob_dims(pair):
    """Return list of (w, h) bbox sizes (in image-resolution px) for one (image, mask) pair."""
    img_path, mask_path = pair
    try:
        with Image.open(img_path) as im:
            iw, ih = im.size
        m = Image.open(mask_path)
        if m.size != (iw, ih):
            m = m.resize((iw, ih), Image.NEAREST)
        arr = np.array(m)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        blobs = find_blobs(arr, bg_colors=BG_COLORS, min_area=MIN_AREA)
        out = []
        for b in blobs:
            x0, y0, x1, y1 = b.bbox
            out.append((x1 - x0, y1 - y0))
        return out
    except Exception as e:  # noqa: BLE001
        return [("ERR", str(e))]


def main():
    datasets = sorted((d for d in os.scandir(DATA_ROOT) if d.is_dir()), key=lambda d: d.name)
    pairs = []  # (img, mask, dataset)
    for ds in datasets:
        mask_dir = os.path.join(ds.path, MASK_SUBDIR)
        image_dir = next((os.path.join(ds.path, s) for s in IMAGE_SUBDIRS
                          if os.path.isdir(os.path.join(ds.path, s))), None)
        if image_dir is None or not os.path.isdir(mask_dir):
            continue
        from pathlib import Path
        for img, mask in iter_pairs(Path(mask_dir), Path(image_dir)):
            pairs.append((img, mask, ds.name))

    print(f"pairs: {len(pairs)} across {len(set(p[2] for p in pairs))} datasets")
    with ThreadPoolExecutor(max_workers=16) as ex:
        per_pair = list(ex.map(_blob_dims, [(p[0], p[1]) for p in pairs]))

    ws, hs, ds_of_blob = [], [], []
    n_err = 0
    for (_, _, ds), dims in zip(pairs, per_pair):
        for d in dims:
            if d[0] == "ERR":
                n_err += 1
                continue
            ws.append(d[0]); hs.append(d[1]); ds_of_blob.append(ds)
    w = np.array(ws, dtype=float)
    h = np.array(hs, dtype=float)
    n = len(w)
    print(f"blobs (min_area>={MIN_AREA}): {n}   (pair errors: {n_err})\n")

    mx = np.maximum(w, h)
    mn = np.minimum(w, h)
    over_w = w > CROP_SIZE
    over_h = h > CROP_SIZE
    over_any = mx > CROP_SIZE
    over_both = mn > CROP_SIZE
    # tight-crop -> 512 resize: per-axis scale; net linear (area-eq) scale
    area_scale = np.sqrt((CROP_SIZE / np.maximum(w, 1)) * (CROP_SIZE / np.maximum(h, 1)))

    def pct(x):
        return f"{100*x:6.2f}%"

    def desc(name, a):
        q = np.percentile(a, [0, 5, 25, 50, 75, 90, 95, 99, 100])
        print(f"  {name:>12}: mean={a.mean():7.1f}  min={q[0]:6.0f}  p25={q[2]:6.0f}  "
              f"med={q[3]:6.0f}  p75={q[4]:6.0f}  p90={q[5]:6.0f}  p95={q[6]:6.0f}  "
              f"p99={q[7]:6.0f}  max={q[8]:6.0f}")

    print(f"Blob bbox dimensions (image-resolution px), vs crop window {CROP_SIZE}:")
    desc("width", w)
    desc("height", h)
    desc("max(w,h)", mx)
    desc("min(w,h)", mn)
    print()
    print("Fraction of blobs whose bbox exceeds the 512 window (= lose resolution):")
    print(f"  width  > 512 ........................... {pct(over_w.mean())}  ({over_w.sum()})")
    print(f"  height > 512 ........................... {pct(over_h.mean())}  ({over_h.sum()})")
    print(f"  max(w,h) > 512 (>=1 axis exceeds) ...... {pct(over_any.mean())}  ({over_any.sum()})  <-- 'resized down'")
    print(f"  min(w,h) > 512 (both axes exceed) ...... {pct(over_both.mean())}  ({over_both.sum()})")
    print()
    print("If instead crops were tight-bbox then resized to 512x512 (area-eq linear scale):")
    print(f"  scale < 1 (net downscale) .............. {pct((area_scale < 1).mean())}  ({(area_scale<1).sum()})")
    sh = area_scale[area_scale < 1]
    if len(sh):
        print(f"  among downscaled: median scale={np.median(sh):.2f}  p5={np.percentile(sh,5):.2f}  min={sh.min():.3f}")
    print()
    # per dataset
    print("max(w,h) > 512 by dataset:")
    by = defaultdict(lambda: [0, 0])
    for ww, hh, ds in zip(w, h, ds_of_blob):
        by[ds][1] += 1
        if max(ww, hh) > CROP_SIZE:
            by[ds][0] += 1
    for ds in sorted(by):
        over, tot = by[ds]
        print(f"  {ds:>42}: {pct(over/tot)}  ({over}/{tot})")


if __name__ == "__main__":
    main()
