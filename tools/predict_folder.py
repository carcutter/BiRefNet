"""Run a trained BiRefNet checkpoint over a folder of images.

Writes for every input <stem>.<ext>:
  <out>/overlays/<stem>.png   — image with prediction painted in `--color` at `--alpha`
  <out>/masks/<stem>.png      — 0/255 binary mask
  <out>/index.html            — scrollable [image | overlay | mask] grid

Preprocessing mirrors `dataset.MyData` at is_train=False: ImageNet normalization
and a resize to the nearest 32-divisible HxW (or the explicit --size). This is
the same path the training eval / inference uses — do not reimplement it.
"""
import argparse
import html
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.birefnet import BiRefNet
from utils import check_state_dict
from yaml_config import parse_args_with_yaml


_IM_MEAN = (0.485, 0.456, 0.406)
_IM_STD = (0.229, 0.224, 0.225)
_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_IM_MEAN, _IM_STD),
])


def _round32(wh):
    w, h = wh
    return (max(32, int(w) // 32 * 32), max(32, int(h) // 32 * 32))


def _preprocess(img, size=None):
    """img: PIL.Image RGB. size: (W,H) or None ⇒ round-to-32 of source size.
    Returns (tensor[1,3,H,W], (out_w, out_h), (orig_w, orig_h))."""
    orig = img.size  # (W, H)
    target = size if size is not None else _round32(orig)
    if img.size != target:
        img = img.resize(target, Image.BILINEAR)
    t = _TO_TENSOR(img).unsqueeze(0)
    return t, target, orig


def _parse_color(s):
    parts = [int(x.strip()) for x in s.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('--color expects R,G,B (got {!r})'.format(s))
    return tuple(parts)


def _parse_size(s):
    if not s:
        return None
    w, h = (int(x.strip()) for x in s.lower().replace('x', ',').split(','))
    if w % 32 or h % 32:
        raise argparse.ArgumentTypeError('--size dimensions must be divisible by 32 (got {}x{})'.format(w, h))
    return (w, h)


def _paint(image_rgb, mask01, color, alpha):
    """image_rgb: HxWx3 uint8. mask01: HxW float [0,1]. Returns HxWx3 uint8."""
    img_f = image_rgb.astype(np.float32) / 255.0
    c = np.array(color, dtype=np.float32) / 255.0
    m = mask01[:, :, None]
    overlay = img_f * (1 - m * alpha) + c[None, None, :] * (m * alpha)
    return (overlay.clip(0, 1) * 255).astype(np.uint8)


def _index_html(out_dir, rows, header_lines):
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<title>predict_folder — predictions</title>',
        '<style>',
        '* { box-sizing: border-box; }',
        'body { margin: 0; font: 13px/1.5 -apple-system, system-ui, sans-serif; background: #111; color: #ddd; }',
        'header { padding: 14px 20px; background: #1a1a1a; border-bottom: 1px solid #333; }',
        'header h1 { margin: 0 0 4px; font-size: 15px; }',
        'header .meta { color: #888; font-size: 12px; }',
        '.rows { display: flex; flex-direction: column; gap: 8px; padding: 14px; }',
        '.row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px; padding: 6px; }',
        '.row .cell { background: #000; aspect-ratio: 16 / 9; overflow: hidden; position: relative; border-radius: 3px; }',
        '.row .cell img { width: 100%; height: 100%; object-fit: contain; display: block; }',
        '.row .cell .lbl { position: absolute; bottom: 4px; left: 6px; font-size: 10px; color: #ddd; background: rgba(0,0,0,.6); padding: 1px 5px; border-radius: 2px; }',
        '.row .name { grid-column: 1 / -1; font-size: 11px; color: #888; padding: 2px 4px; }',
        '</style></head><body>',
        '<header><h1>predict_folder — predictions</h1><div class="meta">',
        ' • '.join(html.escape(x) for x in header_lines),
        '</div></header>',
        '<div class="rows">',
    ]
    for r in rows:
        parts.append(
            '<div class="row">'
            '<div class="name">{name}</div>'
            '<div class="cell"><img src="{src}"><span class="lbl">image</span></div>'
            '<div class="cell"><img src="{ovl}"><span class="lbl">overlay</span></div>'
            '<div class="cell"><img src="{msk}"><span class="lbl">mask</span></div>'
            '</div>'.format(
                name=html.escape(r['name']),
                src=html.escape(r['src']),
                ovl=html.escape(r['overlay']),
                msk=html.escape(r['mask']),
            )
        )
    parts.append('</div></body></html>')
    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write('\n'.join(parts))


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Precedence: argparse defaults < --config yaml < CLI flags',
    )
    p.add_argument('--config', default=None, type=str, help='YAML config (CLI flags override its values)')
    p.add_argument('--checkpoint', default=None, help='Path to .pth (required)')
    p.add_argument('--input_dir', default=None, help='Folder of images to run on (required)')
    p.add_argument('--out', default=None, help='Output dir (required)')
    p.add_argument('--threshold', default=0.5, type=float)
    p.add_argument('--color', default='255,32,32', type=_parse_color)
    p.add_argument('--alpha', default=0.5, type=float)
    p.add_argument('--ext', default='png,jpg,jpeg,JPG,JPEG,PNG')
    p.add_argument('--size', default=None, type=_parse_size,
                   help='Optional WxH (each divisible by 32). Default: round each source dim down to nearest 32.')
    p.add_argument('--limit', default=0, type=int, help='Cap number of images processed')
    return p


def parse_args():
    args = parse_args_with_yaml(build_parser)
    for req in ('checkpoint', 'input_dir', 'out'):
        if not getattr(args, req):
            raise SystemExit('--{} is required (CLI or YAML)'.format(req))
    return args


def main():
    args = parse_args()
    exts = tuple('.' + e.strip().lstrip('.') for e in args.ext.split(','))
    inputs = sorted(p for p in os.listdir(args.input_dir) if p.endswith(exts))
    if args.limit:
        inputs = inputs[:args.limit]
    assert inputs, 'no images matched --ext={} in {}'.format(args.ext, args.input_dir)
    os.makedirs(os.path.join(args.out, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'masks'), exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device, ' inputs:', len(inputs))
    model = BiRefNet(bb_pretrained=False).to(device).eval()
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('  loaded ckpt: missing={} unexpected={}'.format(len(missing), len(unexpected)))

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for name in inputs:
            src_path = os.path.join(args.input_dir, name)
            stem = os.path.splitext(name)[0]
            with Image.open(src_path) as im:
                im = im.convert('RGB')
                tensor, used_size, orig_size = _preprocess(im, size=args.size)
                tensor = tensor.to(device, non_blocking=True)
                preds = model(tensor)
                if isinstance(preds, (list, tuple)):
                    preds = preds[-1]
                prob = preds.to(torch.float32).sigmoid()[0, 0].detach().cpu().numpy()
                # Resize prob back to original resolution before painting / saving.
                prob_img = Image.fromarray((prob * 255).clip(0, 255).astype(np.uint8))
                if prob_img.size != orig_size:
                    prob_img = prob_img.resize(orig_size, Image.BILINEAR)
                prob_arr = np.asarray(prob_img).astype(np.float32) / 255.0
                mask_arr = (prob_arr > args.threshold).astype(np.uint8) * 255

                # Save mask + overlay.
                mask_out = os.path.join(args.out, 'masks', stem + '.png')
                Image.fromarray(mask_arr, mode='L').save(mask_out)
                overlay_arr = _paint(np.asarray(im), prob_arr * (prob_arr > args.threshold), args.color, args.alpha)
                overlay_out = os.path.join(args.out, 'overlays', stem + '.png')
                Image.fromarray(overlay_arr).save(overlay_out)

                rows.append({
                    'name': '{}   (in {}x{}, eval {}x{})'.format(name, orig_size[0], orig_size[1], used_size[0], used_size[1]),
                    'src': os.path.relpath(src_path, args.out),
                    'overlay': os.path.relpath(overlay_out, args.out),
                    'mask': os.path.relpath(mask_out, args.out),
                })

    header = [
        'ckpt: {}'.format(args.checkpoint),
        'threshold: {}'.format(args.threshold),
        'color: rgb{} alpha: {}'.format(args.color, args.alpha),
        'eval size: {}'.format(args.size if args.size else 'round32 of source'),
        '{} images in {:.1f}s'.format(len(rows), time.time() - t0),
    ]
    _index_html(args.out, rows, header)
    print('wrote {} overlays + {} masks + index.html under {}'.format(len(rows), len(rows), args.out))


if __name__ == '__main__':
    main()
