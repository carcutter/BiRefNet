"""Build three per-bucket training CSVs for the decoder-only fine-tunes (shared frozen encoder).

Buckets (decided with the user):
  1. interior : interior_train + kw2622 + kw2624 (roofs/dashboards/consoles)   [full-frame + interior details]
  2. cubemap  : interior_360_notbg_cube (cube-face 360 tiles, sibling repo)
  3. details  : detail_crops rims/wheels                                        [exterior detail crops]

interior/details rows are LIFTED from the existing combined CSV (already absolute + existence-filtered),
so they stay identical to what prior runs saw. cubemap is paired fresh from images/ <-> masks/ by stem.
All outputs carry absolute image_path/mask_path + a dataset tag; only pairs where both files exist are kept.
"""
import csv
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # .../BiRefNet
SIB = os.path.join(os.path.dirname(REPO), 'interior-segmentation')                 # sibling repo
DATA = os.path.join(REPO, 'data/interior_segmentation')
COMBINED = os.path.join(DATA, 'train_interior_rims_refinement.csv')
# Cubemap inputs = the "not backgrounded" cube faces (original image, background intact); their
# foreground masks live in the sibling interior_360_cube/masks (paired by stem). notbg_cube has no
# masks/ of its own, so we cross-reference.
CUBE_IMAGES = os.path.join(SIB, 'data/processed/interior_360_notbg_cube/images')
CUBE_MASKS = os.path.join(SIB, 'data/processed/interior_360_cube/masks')

INTERIOR_TAGS = {'interior_train', 'kw2622_interior', 'refinement_kw2624'}
DETAILS_TAGS = {'detail_crops_rims'}


def _write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'mask_path', 'dataset'])
        w.writerows(rows)
    print('wrote {}  ({} pairs)'.format(path, len(rows)))


def _from_combined(tags):
    rows = []
    with open(COMBINED, newline='') as f:
        for r in csv.DictReader(f):
            if r['dataset'] in tags and os.path.isfile(r['image_path']) and os.path.isfile(r['mask_path']):
                rows.append((r['image_path'], r['mask_path'], r['dataset']))
    return rows


def _cubemap_rows():
    # Pair notbg image <stem>.<ext> with interior_360_cube mask <stem>.<any> by basename stem.
    masks = {}
    for p in glob.glob(os.path.join(CUBE_MASKS, '*')):
        masks.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    rows, missing = [], 0
    for img in sorted(glob.glob(os.path.join(CUBE_IMAGES, '*'))):
        stem = os.path.splitext(os.path.basename(img))[0]
        m = masks.get(stem)
        if m and os.path.isfile(img) and os.path.isfile(m):
            rows.append((os.path.abspath(img), os.path.abspath(m), 'interior_360_notbg_cube'))
        else:
            missing += 1
    if missing:
        print('  cubemap: {} images had no matching mask (skipped)'.format(missing))
    return rows


interior = _from_combined(INTERIOR_TAGS)
details = _from_combined(DETAILS_TAGS)
cubemap = _cubemap_rows()

_write(os.path.join(DATA, 'ft_decoder_interior.csv'), interior)
_write(os.path.join(DATA, 'ft_decoder_cubemap.csv'), cubemap)
_write(os.path.join(DATA, 'ft_decoder_details.csv'), details)

for name, rows in [('interior', interior), ('cubemap', cubemap), ('details', details)]:
    tags = {}
    for _, _, t in rows:
        tags[t] = tags.get(t, 0) + 1
    print('  {:>8}: {:5d}  {}'.format(name, len(rows), dict(sorted(tags.items()))))
