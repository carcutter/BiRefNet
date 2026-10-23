"""Build a combined training CSV = original interior (train.csv) + rim/wheel detail crops
(detail_crops) + the copied refinement folder (kw2624_interior_segmentation_refinement).

Writes ABSOLUTE image_path/mask_path so MyData resolves them regardless of csv location.
Only pairs where both files exist on disk are kept (per-source counts are printed).
"""
import csv
import glob
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Drop 360/equirectangular panoramas (2:1 aspect ratio) — excluded from training for now; they
# need different handling than pinhole images. Set EXCLUDE_360 = False to include them again.
EXCLUDE_360 = True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../BiRefNet
SIB = os.path.join(os.path.dirname(REPO), 'interior-segmentation')          # sibling repo
OUT = os.path.join(REPO, 'data/interior_segmentation/train_interior_rims_refinement.csv')

rows = []          # (image_abs, mask_abs, dataset)
counts = {}


def add(img, mask, tag):
    if os.path.isfile(img) and os.path.isfile(mask):
        rows.append((img, mask, tag))
        counts[tag] = counts.get(tag, [0, 0]);  counts[tag][0] += 1
    else:
        counts.setdefault(tag, [0, 0]); counts[tag][1] += 1


# 1) Original interior — train.csv (paths relative to data/interior_segmentation).
tr = os.path.join(REPO, 'data/interior_segmentation/train.csv')
base = os.path.join(REPO, 'data/interior_segmentation')
with open(tr, newline='') as f:
    for r in csv.DictReader(f):
        img = r['image_path'].strip(); mask = r['mask_path'].strip()
        if img and mask:
            add(os.path.join(base, img), os.path.join(base, mask), 'interior_train')

# 1b) New interior dataset kw2622 (raw/*.jpg <-> masked/<stem>.png, sibling data dir).
kw2622 = os.path.join(SIB, 'data/interior_segmentation/kw2622_car_interior_segmentation')
_m2622 = {os.path.splitext(os.path.basename(p))[0]: p
          for p in glob.glob(os.path.join(kw2622, 'masked', '*.png'))}
for img in sorted(glob.glob(os.path.join(kw2622, 'raw', '*.jpg'))):
    stem = os.path.splitext(os.path.basename(img))[0]
    if stem in _m2622:
        add(img, _m2622[stem], 'kw2622_interior')
    else:
        counts.setdefault('kw2622_interior', [0, 0]); counts['kw2622_interior'][1] += 1

# 2) Rims — detail_crops.csv (image/mask relative to the sibling repo root).
dc = os.path.join(SIB, 'data/detail_crops/detail_crops.csv')
with open(dc, newline='') as f:
    for r in csv.DictReader(f):
        img = r['image'].strip(); mask = r['mask'].strip()
        if img and mask:
            add(os.path.join(SIB, img), os.path.join(SIB, mask), 'detail_crops_rims')

# 3) Refinement folder — pair raw/*.jpg with masked/<stem>.png by stem.
ref = os.path.join(REPO, 'data/kw2624_interior_segmentation_refinement')
masked = {os.path.splitext(os.path.basename(p))[0]: p
          for p in glob.glob(os.path.join(ref, 'masked', '*.png'))}
for img in sorted(glob.glob(os.path.join(ref, 'raw', '*.jpg'))):
    stem = os.path.splitext(os.path.basename(img))[0]
    if stem in masked:
        add(img, masked[stem], 'refinement_kw2624')
    else:
        counts.setdefault('refinement_kw2624', [0, 0]); counts['refinement_kw2624'][1] += 1

if EXCLUDE_360:
    def _ar(p):
        try:
            with Image.open(p) as im:
                w, h = im.size
                return w / h
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=32) as ex:
        ars = list(ex.map(_ar, [r[0] for r in rows]))
    kept = [r for r, ar in zip(rows, ars) if not (ar is not None and abs(ar - 2.0) < 0.05)]
    n_360 = len(rows) - len(kept)
    print('excluded {} 360/2:1 panorama images'.format(n_360))
    rows = kept

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image_path', 'mask_path', 'dataset'])
    w.writerows(rows)

print('wrote {}  ({} pairs)'.format(OUT, len(rows)))
for tag, (kept, dropped) in sorted(counts.items()):
    print('  {:>20}: kept={:5d}  dropped(missing)={}'.format(tag, kept, dropped))
