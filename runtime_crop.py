"""Runtime crops for CSV rows that carry ``wheel_boxes`` / ``crop_box`` columns.

Datasets stored as full originals (interior-segmentation ``details/exterior_originals``) are cropped
at load time, on the full-resolution image and before the resize to ``config.size``:

- a row with ``crop_box`` (JSON [x0, y0, x1, y1]; the fixed test crops) is always cut to that box;
- otherwise a row with ``wheel_boxes`` (JSON {wheel: [x0, y0, x1, y1]}) gets a box from
  ``sample_crop_box``: a fresh draw each time in training, and a per-image seeded draw in validation
  so val metrics are comparable across epochs.

The sampler mirrors ``interior_segmentation/runtime_crops.py``. Keep the two in sync.
"""
import json
import random
import zlib

CROP_FRAC = (0.4, 0.7)
P_WHEEL = 0.5


def sample_crop_box(width, height, wheel_boxes, rng, frac=CROP_FRAC, p_wheel=P_WHEEL, area=False):
    """Crop of ``frac`` x the image size (aspect kept), inside the image. With probability
    ``p_wheel`` (if any wheel is present) it is centred on a random wheel's bbox centre, slid back
    inside near the edges; otherwise its position is uniform.

    ``area=False`` (default): ``frac`` is the fraction of each SIDE. ``area=True``: ``frac`` is the
    fraction of the image AREA, so each side scales by ``sqrt(frac)`` (aspect still kept)."""
    f = rng.uniform(*frac)
    if area:
        f = f ** 0.5
    cw, ch = max(1, round(f * width)), max(1, round(f * height))
    if wheel_boxes and rng.random() < p_wheel:
        x0, y0, x1, y1 = wheel_boxes[rng.choice(sorted(wheel_boxes))]
        left = min(max(round((x0 + x1) / 2 - cw / 2), 0), width - cw)
        top = min(max(round((y0 + y1) / 2 - ch / 2), 0), height - ch)
    else:
        left, top = rng.randint(0, width - cw), rng.randint(0, height - ch)
    return left, top, left + cw, top + ch


def parse_crop_meta(row):
    """-> None, or {'crop_box': tuple|None, 'wheel_boxes': dict} for a runtime-cropped CSV row."""
    box, wheels = (row.get('crop_box') or '').strip(), (row.get('wheel_boxes') or '').strip()
    if not box and not wheels:
        return None
    return {'crop_box': tuple(json.loads(box)) if box else None,
            'wheel_boxes': {k: tuple(v) for k, v in json.loads(wheels).items()} if wheels else {}}


def crop_box_for(meta, width, height, is_train, key, frac=CROP_FRAC, area=False):
    if meta['crop_box'] is not None:
        return meta['crop_box']
    rng = random if is_train else random.Random(zlib.crc32(key.encode()))
    return sample_crop_box(width, height, meta['wheel_boxes'], rng, frac=frac, area=area)
