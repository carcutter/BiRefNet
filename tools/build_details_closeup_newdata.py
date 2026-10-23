"""Rebuild the decoder-closeup training CSV on the NEW data layout.

Recipe (per user): "same data as the old run — ALL details + 20% of the gold_dataset interior data".
Reads the new manifest data_link/train.csv (columns: dataset, csv_path, n_samples), where each row
points to a per-dataset CSV of image_path,mask_path pairs. Then:
  - details/*         -> take ALL rows
  - gold_dataset/*    -> take a stratified 20% (seed 42) of each dataset
Writes a flat image_path,mask_path,dataset CSV (existence-filtered) for MyData. Runtime-crop columns
(wheel_boxes, crop_box; see runtime_crop.py) are carried through for datasets stored as originals.
"""
import csv
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, 'data_link/train.csv')
OUT = os.path.join(REPO, 'data/interior_segmentation/details_closeup_newdata.csv')
GOLD_FRAC = 0.20
SEED = 42

rng = random.Random(SEED)
rows = []          # (image, mask, dataset, wheel_boxes, crop_box)
CROP_COLS = ('wheel_boxes', 'crop_box')
summary = {}


def read_pairs(csv_path):
    out = []
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            img = (r.get('image_path') or r.get('image') or '').strip()
            msk = (r.get('mask_path') or r.get('mask') or '').strip()
            if img and msk:
                out.append((img, msk) + tuple(r.get(c) or '' for c in CROP_COLS))
    return out


with open(MANIFEST, newline='') as f:
    manifest = list(csv.DictReader(f))

for m in manifest:
    ds = m['dataset']
    pairs = read_pairs(m['csv_path'])
    if ds.startswith('details/'):
        kept = pairs                                  # ALL details
    elif ds.startswith('gold_dataset/'):
        k = max(1, round(GOLD_FRAC * len(pairs)))
        kept = rng.sample(pairs, k)                   # 20% stratified per dataset
    else:
        kept = []
    present = [p for p in kept if os.path.isfile(p[0]) and os.path.isfile(p[1])]
    for i, mk, *crop in present:
        rows.append((i, mk, ds, *crop))
    summary[ds] = (len(pairs), len(kept), len(present))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image_path', 'mask_path', 'dataset', *CROP_COLS])
    w.writerows(rows)

print('wrote {}  ({} pairs)'.format(OUT, len(rows)))
det = sum(k for d, (_, _, k) in summary.items() if d.startswith('details/'))
gld = sum(k for d, (_, _, k) in summary.items() if d.startswith('gold_dataset/'))
print('  details (all): {}   gold (20%): {}'.format(det, gld))
for d in sorted(summary):
    total, kept, present = summary[d]
    print('  {:52s} total={:5d}  kept={:5d}  present={:5d}'.format(d, total, kept, present))
