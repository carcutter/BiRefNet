"""Scan a dataset root, pair images with masks, and write a CSV index.

The output CSV has one row per training example with columns:
    image_path  — relative to --data_root
    mask_path   — relative to --data_root
    depth_path  — relative to --data_root, empty string when no depth dir exists
    dataset     — name of the immediate subdir under --data_root

By default the scanner walks every subdir under --data_root and looks for an
image dir (one of `raw`, `raw_images`, `images`) paired with a mask dir (one
of `masks`, `gt`). Pairs are matched on filename stem; multiple extensions are
tried for the mask file. Missing-pair counts are reported per-dataset so it
is obvious when an extraction step broke.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict


IMAGE_DIR_CANDIDATES = ("raw", "raw_images", "images", "im")
MASK_DIR_CANDIDATES = ("masks", "gt", "mask")
DEPTH_DIR_CANDIDATES = ("depth",)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _find_subdir(parent, candidates):
    for c in candidates:
        p = os.path.join(parent, c)
        if os.path.isdir(p):
            return c
    return None


def _index_dir_by_stem(path, allowed_exts):
    out = {}
    if not path or not os.path.isdir(path):
        return out
    for name in os.listdir(path):
        stem, ext = os.path.splitext(name)
        if ext in allowed_exts:
            out.setdefault(stem, name)
    return out


def build_index(data_root, datasets=None, include=None):
    """Return (rows, summary) for every dataset subdir under data_root."""
    rows = []
    summary = defaultdict(lambda: {"paired": 0, "no_mask": 0, "no_image_dir": 0, "no_mask_dir": 0})

    all_subdirs = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))
    for sub in all_subdirs:
        if datasets and sub not in datasets:
            continue
        if include and include not in sub:
            continue
        sub_path = os.path.join(data_root, sub)
        img_dir = _find_subdir(sub_path, IMAGE_DIR_CANDIDATES)
        mask_dir = _find_subdir(sub_path, MASK_DIR_CANDIDATES)
        depth_dir = _find_subdir(sub_path, DEPTH_DIR_CANDIDATES)

        if img_dir is None:
            summary[sub]["no_image_dir"] += 1
            continue
        if mask_dir is None:
            summary[sub]["no_mask_dir"] += 1
            continue

        image_stems = _index_dir_by_stem(os.path.join(sub_path, img_dir), IMAGE_EXTS)
        mask_stems = _index_dir_by_stem(os.path.join(sub_path, mask_dir), MASK_EXTS)
        depth_stems = _index_dir_by_stem(os.path.join(sub_path, depth_dir) if depth_dir else "", MASK_EXTS)

        for stem in sorted(image_stems):
            if stem not in mask_stems:
                summary[sub]["no_mask"] += 1
                continue
            img_rel = os.path.join(sub, img_dir, image_stems[stem])
            msk_rel = os.path.join(sub, mask_dir, mask_stems[stem])
            depth_rel = os.path.join(sub, depth_dir, depth_stems[stem]) if (depth_dir and stem in depth_stems) else ""
            rows.append({
                "image_path": img_rel,
                "mask_path": msk_rel,
                "depth_path": depth_rel,
                "dataset": sub,
            })
            summary[sub]["paired"] += 1
    return rows, summary


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "mask_path", "depth_path", "dataset"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="data_link",
                   help="Root directory whose immediate subdirs are datasets. Default: data_link")
    p.add_argument("--out", default="data/interior_segmentation/index.csv",
                   help="Output CSV path. Default: data/interior_segmentation/index.csv")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Restrict to these dataset subdir names (default: all)")
    p.add_argument("--include", default="interior",
                   help="Only include dataset subdir names that contain this substring. "
                        "Default: 'interior' (matches *_interior_* / *_interior_segmentation). "
                        "Pass --include '' to disable.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.data_root):
        print("ERROR: data_root does not exist: {}".format(args.data_root), file=sys.stderr)
        sys.exit(2)
    rows, summary = build_index(args.data_root, datasets=set(args.datasets) if args.datasets else None,
                                include=args.include or None)
    write_csv(rows, args.out)

    total = sum(s["paired"] for s in summary.values())
    print("wrote {} ({} pairs across {} datasets)".format(args.out, total, len(summary)))
    print()
    print("{:<48s} {:>8s} {:>8s} {:>10s} {:>10s}".format("dataset", "paired", "no_mask", "no_img_dir", "no_msk_dir"))
    for name in sorted(summary):
        s = summary[name]
        print("{:<48s} {:>8d} {:>8d} {:>10d} {:>10d}".format(
            name, s["paired"], s["no_mask"], s["no_image_dir"], s["no_mask_dir"]))


if __name__ == "__main__":
    main()
