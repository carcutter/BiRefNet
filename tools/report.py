"""Post-training HTML report.

Runs a checkpoint over the val split (same rng split as training) and emits a
self-contained report.html — inline CSS + base64-PNG thumbnails — under
reports/<model>_<ts>/. Computes F1, MAE, IoU, and Contour mIoU per-sample;
aggregates them overall and per-dataset; embeds an IoU histogram and best/worst
example tiles.
"""
import argparse
import base64
import csv
import io
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from config import Config
from dataset import MyData
from models.birefnet import BiRefNet
from utils import check_state_dict
from yaml_config import parse_args_with_yaml


# -------------------- metrics --------------------

def _ring(mask, r=2):
    k = 2 * r + 1
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=r)
    ero = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=r)
    return (dil - ero).clamp(0, 1)


@torch.no_grad()
def _contour_miou(pred_prob, gt, r=2, thresh=0.5, eps=1e-6):
    pb = (pred_prob > thresh).float()
    gb = (gt > 0.5).float()
    pr_ring = _ring(pb, r)
    gt_ring = _ring(gb, r)
    inter = (pr_ring * gt_ring).flatten(1).sum(1)
    union = (pr_ring + gt_ring - pr_ring * gt_ring).flatten(1).sum(1)
    return (inter + eps) / (union + eps)


def _per_sample_metrics(pred_prob, gt, r=2, eps=1e-7):
    """Returns dict of (B,) tensors: iou, f1, mae, contour_miou."""
    p = (pred_prob > 0.5).float()
    g = (gt > 0.5).float()
    inter = (p * g).flatten(1).sum(1)
    union = (p + g - p * g).flatten(1).sum(1)
    iou = inter / (union + eps)
    tp = inter
    fp = (p * (1 - g)).flatten(1).sum(1)
    fn = ((1 - p) * g).flatten(1).sum(1)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    mae = (pred_prob - gt).abs().flatten(1).mean(1)
    contour = _contour_miou(pred_prob, gt, r=r)
    return {'iou': iou, 'f1': f1, 'mae': mae, 'contour_miou': contour}


# -------------------- image helpers --------------------

def _denorm(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (tensor.float().cpu() * std + mean).clamp(0, 1)


def _tensor_to_png_b64(t, max_side=320):
    """t: (C,H,W) in [0,1] or (H,W) — returns 'data:image/png;base64,...' string."""
    if t.dim() == 2:
        arr = (t.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
    else:
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        arr = (t.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='RGB')
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _error_map(pred_bin, gt_bin):
    """3-channel uint8 tensor: FP=red, FN=blue, TP=white-ish, TN=black."""
    p = (pred_bin > 0.5).float()
    g = (gt_bin > 0.5).float()
    tp = (p * g)
    fp = (p * (1 - g))
    fn = ((1 - p) * g)
    r = tp * 0.6 + fp * 1.0
    gch = tp * 0.6
    b = tp * 0.6 + fn * 1.0
    return torch.cat([r, gch, b], dim=0)


# -------------------- histogram --------------------

def _hist_b64(ious, bins=20):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    ax.hist(ious, bins=bins, range=(0, 1), color='#4a90e2', edgecolor='#1a1a1a')
    ax.set_xlabel('per-sample IoU')
    ax.set_ylabel('count')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


# -------------------- HTML --------------------

_CSS = """
* { box-sizing: border-box; }
body { margin: 0; font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; background: #fafafa; color: #222; }
header { padding: 20px 24px; background: #fff; border-bottom: 1px solid #e3e3e3; }
header h1 { margin: 0 0 4px; font-size: 18px; }
header .meta { color: #666; font-size: 12px; }
section { padding: 18px 24px; }
section h2 { margin: 0 0 10px; font-size: 14px; color: #333; text-transform: uppercase; letter-spacing: 0.04em; }
table { border-collapse: collapse; background: #fff; box-shadow: 0 1px 0 #e3e3e3; }
th, td { padding: 6px 12px; text-align: left; border-bottom: 1px solid #eee; font-variant-numeric: tabular-nums; }
th { background: #f3f3f3; font-weight: 600; color: #444; }
td.num { text-align: right; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }
.tile { background: #fff; border: 1px solid #e3e3e3; border-radius: 4px; padding: 8px; }
.tile .strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
.tile .strip .cell { background: #000; aspect-ratio: 1 / 1; overflow: hidden; border-radius: 2px; position: relative; }
.tile .strip .cell img { width: 100%; height: 100%; object-fit: contain; display: block; }
.tile .strip .cell .lbl { position: absolute; bottom: 2px; left: 4px; font-size: 10px; color: #bbb; background: rgba(0,0,0,.5); padding: 0 4px; border-radius: 2px; }
.tile .meta { margin-top: 6px; font-size: 11px; color: #555; display: flex; justify-content: space-between; }
.hist img { max-width: 720px; }
"""


def _row(cells, num_idx=None):
    num_idx = num_idx or set()
    return '<tr>' + ''.join(
        '<td class="num">{}</td>'.format(c) if i in num_idx else '<td>{}</td>'.format(c)
        for i, c in enumerate(cells)
    ) + '</tr>'


def _fmt(x, d=4):
    return '—' if x is None else '{:.{d}f}'.format(float(x), d=d)


# -------------------- main eval loop --------------------

def evaluate(model, loader, device, autocast_ctx, contour_radius=2):
    """Returns list of per-sample dicts: {iou, f1, mae, contour_miou, dataset, mask_path, image, gt, pred}."""
    results = []
    model.eval()
    n_samples_done = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device, non_blocking=True)
            gts = batch[1]
            label_paths = batch[2]
            with autocast_ctx:
                preds = model(inputs)
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            preds = preds.to(torch.float32).sigmoid()
            preds_cpu = preds.detach().cpu()
            gts_cpu = gts.detach().float().cpu()
            inputs_denorm = _denorm(inputs.detach())
            m = _per_sample_metrics(preds_cpu, gts_cpu, r=contour_radius)
            for i in range(inputs.shape[0]):
                results.append({
                    'iou': float(m['iou'][i]),
                    'f1': float(m['f1'][i]),
                    'mae': float(m['mae'][i]),
                    'contour_miou': float(m['contour_miou'][i]),
                    'mask_path': label_paths[i],
                    'image': inputs_denorm[i].clone(),
                    'gt': gts_cpu[i].clone(),
                    'pred': preds_cpu[i].clone(),
                })
            n_samples_done += inputs.shape[0]
            if n_samples_done % 32 == 0:
                print('  ...{} samples, {:.1f}s'.format(n_samples_done, time.time() - t0))
    print('eval done: {} samples, {:.1f}s'.format(len(results), time.time() - t0))
    return results


def _dataset_of(path, csv_image_root):
    """Pull the dataset subdir name out of the resolved label_path (first dir under csv_image_root)."""
    rel = os.path.relpath(path, csv_image_root) if path.startswith(csv_image_root) else path
    return rel.split(os.sep)[0]


def render_report(results, args, header_extras, out_path):
    n = len(results)
    arr = lambda k: np.array([r[k] for r in results], dtype=np.float64)
    iou_arr, f1_arr, mae_arr, contour_arr = arr('iou'), arr('f1'), arr('mae'), arr('contour_miou')

    # Per-dataset breakdown.
    per_ds = defaultdict(list)
    for r in results:
        per_ds[_dataset_of(r['mask_path'], args.csv_image_root)].append(r)

    # Summary table.
    summary_rows = [
        _row(['n samples', n, '', '', ''], {1}),
        _row(['F1', _fmt(f1_arr.mean()), _fmt(np.median(f1_arr)), _fmt(f1_arr.std()), ''], {1, 2, 3}),
        _row(['IoU', _fmt(iou_arr.mean()), _fmt(np.median(iou_arr)), _fmt(iou_arr.std()), ''], {1, 2, 3}),
        _row(['MAE', _fmt(mae_arr.mean()), _fmt(np.median(mae_arr)), _fmt(mae_arr.std()), ''], {1, 2, 3}),
        _row(['Contour mIoU', _fmt(contour_arr.mean()), _fmt(np.median(contour_arr)), _fmt(contour_arr.std()), ''], {1, 2, 3}),
    ]
    summary_table = (
        '<table><thead><tr><th>metric</th><th>mean</th><th>median</th><th>std</th><th></th></tr></thead><tbody>'
        + ''.join(summary_rows) + '</tbody></table>'
    )

    # Per-dataset table.
    ds_rows = []
    for ds in sorted(per_ds):
        rs = per_ds[ds]
        a = lambda k: np.array([r[k] for r in rs])
        ds_rows.append(_row([
            ds, len(rs),
            _fmt(a('f1').mean()), _fmt(a('iou').mean()), _fmt(a('contour_miou').mean()), _fmt(a('mae').mean()),
        ], {1, 2, 3, 4, 5}))
    per_ds_table = (
        '<table><thead><tr><th>dataset</th><th>n</th><th>F1</th><th>IoU</th><th>Contour mIoU</th><th>MAE</th></tr></thead>'
        '<tbody>' + ''.join(ds_rows) + '</tbody></table>'
    )

    # Histogram.
    hist_uri = _hist_b64(iou_arr)

    # Best / worst tiles.
    order = sorted(range(n), key=lambda i: results[i]['iou'])
    worst_idx = order[:args.top_n]
    best_idx = list(reversed(order[-args.top_n:]))

    def _tiles(idxs, title):
        tiles = []
        for i in idxs:
            r = results[i]
            err = _error_map(r['pred'], r['gt'])
            tiles.append(
                '<div class="tile">'
                '<div class="strip">'
                '<div class="cell"><img src="{img}"><span class="lbl">image</span></div>'
                '<div class="cell"><img src="{gt}"><span class="lbl">gt</span></div>'
                '<div class="cell"><img src="{pred}"><span class="lbl">pred</span></div>'
                '<div class="cell"><img src="{err}"><span class="lbl">err (FP red / FN blue)</span></div>'
                '</div>'
                '<div class="meta"><span>{name}</span>'
                '<span>IoU {iou} • F1 {f1} • Contour {ct}</span></div>'
                '</div>'.format(
                    img=_tensor_to_png_b64(r['image']),
                    gt=_tensor_to_png_b64(r['gt']),
                    pred=_tensor_to_png_b64(r['pred']),
                    err=_tensor_to_png_b64(err),
                    name=os.path.basename(r['mask_path']),
                    iou=_fmt(r['iou']), f1=_fmt(r['f1']), ct=_fmt(r['contour_miou']),
                )
            )
        return '<section><h2>{}</h2><div class="tiles">{}</div></section>'.format(title, ''.join(tiles))

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>BiRefNet metrics report</title>'
        '<style>' + _CSS + '</style></head><body>'
        + '<header><h1>BiRefNet metrics report</h1>'
        + '<div class="meta">'
        + ' • '.join(header_extras)
        + '</div></header>'
        + '<section><h2>summary</h2>' + summary_table + '</section>'
        + '<section><h2>per-dataset</h2>' + per_ds_table + '</section>'
        + '<section class="hist"><h2>IoU distribution</h2><img src="' + hist_uri + '"></section>'
        + _tiles(worst_idx, 'worst {}'.format(len(worst_idx)))
        + _tiles(best_idx, 'best {}'.format(len(best_idx)))
        + '</body></html>'
    )
    with open(out_path, 'w') as f:
        f.write(html)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Precedence: argparse defaults < --config yaml < CLI flags',
    )
    p.add_argument('--config', default=None, type=str, help='YAML config (CLI flags override its values)')
    p.add_argument('--checkpoint', default=None, help='Path to .pth (required unless given via --config)')
    p.add_argument('--csv', default=None, help='CSV index path (defaults to config.csv_data_root/index.csv)')
    p.add_argument('--csv_image_root', default=None, help='Resolves relative paths in the CSV (defaults to config.csv_image_root)')
    p.add_argument('--val_split', default=0.2, type=float, help='Val fraction; must match training')
    p.add_argument('--csv_split_seed', default=42, type=int, help='Split seed; must match training')
    p.add_argument('--val_size', default=512, type=int, help='Square input size for evaluation (W=H, divisible by 32)')
    p.add_argument('--max_samples', default=0, type=int, help='If >0, cap the val set (for fast smoke runs)')
    p.add_argument('--top_n', default=16, type=int, help='Best/worst count per panel')
    p.add_argument('--contour_radius', default=2, type=int)
    p.add_argument('--out', default=None, help='Output dir (default: reports/<ckpt_stem>_<ts>/)')
    return p


def parse_args():
    args = parse_args_with_yaml(build_parser)
    if not args.checkpoint:
        raise SystemExit('--checkpoint is required (either via CLI or YAML config)')
    return args


def main():
    args = parse_args()
    cfg = Config()
    csv_path = args.csv or os.path.join(cfg.csv_data_root, 'index.csv')
    args.csv_image_root = args.csv_image_root or cfg.csv_image_root
    assert args.val_size % 32 == 0, '--val_size must be divisible by 32 (got {})'.format(args.val_size)

    # Build val loader from the same rng split as training.
    val_size_tuple = (args.val_size, args.val_size)
    val_ds = MyData(
        datasets=None, data_size=val_size_tuple, is_train=False,
        csv_path=csv_path, csv_image_root=args.csv_image_root,
        val_split=args.val_split, csv_split_seed=args.csv_split_seed,
        max_samples=args.max_samples,
    )
    print('val set: {} samples ({})'.format(len(val_ds), csv_path))
    loader = torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = BiRefNet(bb_pretrained=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('  loaded ckpt: missing={} unexpected={}'.format(len(missing), len(unexpected)))

    from contextlib import nullcontext
    autocast_ctx = nullcontext()
    if device.type == 'cuda' and cfg.mixed_precision in ('fp16', 'bf16'):
        dtype = torch.float16 if cfg.mixed_precision == 'fp16' else torch.bfloat16
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)

    results = evaluate(model, loader, device, autocast_ctx, contour_radius=args.contour_radius)

    # Output paths.
    ts = time.strftime('%Y%m%d_%H%M%S')
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_dir = args.out or os.path.join('reports', '{}_{}'.format(ckpt_stem, ts))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'report.html')

    n_params = sum(p.numel() for p in model.parameters())
    header_extras = [
        'ckpt: <code>{}</code>'.format(args.checkpoint),
        'csv: <code>{}</code>'.format(csv_path),
        'val_split={} seed={}'.format(args.val_split, args.csv_split_seed),
        'val_size={}'.format(args.val_size),
        'params: {:.1f}M'.format(n_params / 1e6),
        'date: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')),
    ]
    render_report(results, args, header_extras, out_path)
    print('wrote {} ({} bytes)'.format(out_path, os.path.getsize(out_path)))


if __name__ == '__main__':
    main()
