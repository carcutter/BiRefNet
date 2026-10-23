"""Read data/interior_segmentation/index.csv and write tools/viewer/manifest.json.

For each sample we compute the image size and the mask foreground area fraction
(`mask_area_frac`) so the viewer can flag empty / fully-foreground masks as
likely-bad data. Paths in the manifest are relative to the repo root (the
viewer is served from there).
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from PIL import Image


def _stable_id(image_path):
    parts = image_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return "{}/{}".format(parts[0], os.path.splitext(parts[-1])[0])
    return os.path.splitext(parts[-1])[0]


def _mask_area_frac(mask_path, fg_colors=None):
    """Foreground fraction of a mask. Supports color-coded masks (R/G fg) and
    pure binary masks."""
    try:
        with Image.open(mask_path) as im:
            arr = np.asarray(im.convert("RGB"))
    except Exception:
        return None
    if fg_colors:
        fg = np.zeros(arr.shape[:2], dtype=bool)
        for c in fg_colors:
            fg |= np.all(arr == np.array(c, dtype=arr.dtype), axis=-1)
    else:
        gray = arr.mean(axis=-1)
        fg = gray > 127
    return float(fg.mean())


def _image_size(image_path):
    try:
        with Image.open(image_path) as im:
            return [im.size[0], im.size[1]]
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/interior_segmentation/index.csv",
                   help="Input CSV (default: data/interior_segmentation/index.csv)")
    p.add_argument("--image_root", default="data_link",
                   help="Directory the in-CSV paths resolve against. Also the path prefix used in "
                        "the manifest so the static viewer can load images via <repo>/<image_root>/...")
    p.add_argument("--out", default="tools/viewer/manifest.json",
                   help="Output manifest (default: tools/viewer/manifest.json)")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, sample this many rows uniformly across datasets (for fast iteration)")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Restrict to these dataset names")
    p.add_argument("--fg_colors", default="255,0,0;0,255,0",
                   help="Semicolon-separated RGB triples treated as mask foreground. "
                        "Pass '' for greyscale-binary masks.")
    p.add_argument("--compute_stats", action="store_true", default=True,
                   help="Open each mask + image to compute mask_area_frac and image_size. "
                        "Slow but high-value for spotting empty masks.")
    p.add_argument("--no_compute_stats", dest="compute_stats", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.csv):
        print("ERROR: CSV not found: {}".format(args.csv), file=sys.stderr)
        sys.exit(2)
    fg_colors = []
    if args.fg_colors:
        for tok in args.fg_colors.split(";"):
            if not tok.strip():
                continue
            try:
                fg_colors.append(tuple(int(x) for x in tok.split(",")))
            except ValueError:
                print("WARN: skipping malformed --fg_colors token: {!r}".format(tok), file=sys.stderr)

    rows = []
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            if args.datasets and r.get("dataset") not in args.datasets:
                continue
            rows.append(r)

    if args.limit and len(rows) > args.limit:
        per_ds = defaultdict(list)
        for r in rows:
            per_ds[r.get("dataset", "_unknown")].append(r)
        # uniform stratified subsample
        n_per = max(1, args.limit // max(1, len(per_ds)))
        sampled = []
        for ds, ds_rows in per_ds.items():
            sampled.extend(ds_rows[:n_per])
        rows = sampled[:args.limit]

    samples = []
    t0 = time.time()
    for i, r in enumerate(rows):
        img_rel = r["image_path"]
        msk_rel = r["mask_path"]
        depth_rel = r.get("depth_path", "")
        dataset = r.get("dataset", "")

        img_abs = os.path.join(args.image_root, img_rel) if not os.path.isabs(img_rel) else img_rel
        msk_abs = os.path.join(args.image_root, msk_rel) if not os.path.isabs(msk_rel) else msk_rel
        depth_abs = os.path.join(args.image_root, depth_rel) if (depth_rel and not os.path.isabs(depth_rel)) else depth_rel

        entry = {
            "id": _stable_id(img_rel),
            "image": os.path.join(args.image_root, img_rel),
            "mask": os.path.join(args.image_root, msk_rel),
            "aux": (os.path.join(args.image_root, depth_rel) if depth_rel else ""),
            "dataset": dataset,
        }
        if args.compute_stats:
            entry["image_size"] = _image_size(img_abs)
            entry["mask_area_frac"] = _mask_area_frac(msk_abs, fg_colors=fg_colors)
        samples.append(entry)
        if args.compute_stats and (i + 1) % 200 == 0:
            print("  scanned {}/{}  ({:.1f}s)".format(i + 1, len(rows), time.time() - t0))

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_csv": args.csv,
        "image_root": args.image_root,
        "fg_colors": fg_colors,
        "samples": samples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote {} ({} samples, {:.1f}s)".format(args.out, len(samples), time.time() - t0))


if __name__ == "__main__":
    main()
