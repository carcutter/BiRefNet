"""Export a trained UNet mask-refiner checkpoint to ONNX + parity-check it with onnxruntime.

The refiner is a plain `segmentation_models_pytorch.Unet` (efficientnet/resnet encoder),
so unlike BiRefNet it contains no `deform_conv2d` and traces to ONNX cleanly.

Input is the same N-channel stack the model was trained on (see dataset_refiner.py):
  * refiner mode  (use_mask_input=True):  4ch = [crop RGB | degraded mask]
  * crop-only     (use_mask_input=False): 3ch = [crop RGB]

The training-time model emits raw logits. We wrap it in `RefinerOnnxAdapter` so that
`forward(x)` returns the *final refined foreground probability* (B,1,H,W), reproducing
`reconstruct_prob` from train_refiner.py:
  * refiner mode:   refined = clamp(cond_mask + tanh(logits), 0, 1),  cond_mask = x[:, 3:4]
  * crop-only mode: refined = sigmoid(logits)

A sidecar `<out>.json` records the preprocessing assumptions (mean/std/input_size/threshold/
in_channels/use_mask_input) so downstream consumers cannot drift from the training pipeline.

Usage:
    uv run python tools/export_onnx_refiner.py \
        --checkpoint runs/refiner-efficientnet-b4-delta-loss/epoch_200.pth \
        --config configs/refiner-efficientnet-b4.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from yaml_config import parse_args_with_yaml

# ImageNet stats — the refiner's RGB channels are normalized with these (dataset_refiner.py).
_IM_MEAN = [0.485, 0.456, 0.406]
_IM_STD = [0.229, 0.224, 0.225]


def _str2bool(v):
    return v if isinstance(v, bool) else str(v).lower() in ('1', 'true', 'yes', 'y')


class RefinerOnnxAdapter(nn.Module):
    """Wrap the refiner UNet so forward(x) → single (B,1,H,W) refined-probability tensor.

    Mirrors `reconstruct_prob` in train_refiner.py:
      * refiner mode (use_mask_input): refined = clamp(x[:,3:4] + tanh(logits), 0, 1)
      * crop-only mode:                refined = sigmoid(logits)
    Set `raw_logits=True` to emit the bare logits instead (no reconstruction).
    """
    def __init__(self, model, use_mask_input=True, raw_logits=False):
        super().__init__()
        self.model = model
        self.use_mask_input = use_mask_input
        self.raw_logits = raw_logits

    def forward(self, x):
        logits = self.model(x)
        if self.raw_logits:
            return logits
        if self.use_mask_input:
            cond_mask = x[:, 3:4]
            return (cond_mask + torch.tanh(logits)).clamp(0.0, 1.0)
        return torch.sigmoid(logits)


def _parse_size(s):
    parts = [int(p.strip()) for p in str(s).lower().replace('x', ',').split(',')]
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('--input_size must be H,W or a single int (got {!r})'.format(s))
    H, W = parts
    if H % 32 or W % 32:
        raise argparse.ArgumentTypeError('--input_size dims must be divisible by 32 (got {}x{})'.format(H, W))
    return H, W


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Precedence: argparse defaults < --config yaml < CLI flags',
    )
    p.add_argument('--config', default=None, type=str, help='YAML config (CLI flags override its values)')
    p.add_argument('--checkpoint', default=None, help='Path to .pth (required)')
    p.add_argument('--out', default=None, help='Output .onnx (default: <checkpoint>.onnx)')
    p.add_argument('--input_size', default='512', type=_parse_size,
                   help='H,W (or single int) to trace at — each divisible by 32')
    # These mirror train_refiner.py so a run is reproducible from its YAML.
    p.add_argument('--encoder_name', default='efficientnet-b4', type=str)
    p.add_argument('--use_mask_input', default=True, type=_str2bool,
                   help='True ⇒ 4ch input [crop|mask] (refiner). False ⇒ 3ch [crop].')
    p.add_argument('--batch', default='dynamic', help='Static batch size, or "dynamic"')
    p.add_argument('--opset', default=17, type=int)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--raw_logits', action='store_true',
                   help='Emit raw logits instead of the reconstructed refined probability.')
    p.add_argument('--check', action='store_true', default=True,
                   help='Re-run the exported model with onnxruntime and assert parity (default: on).')
    p.add_argument('--no_check', dest='check', action='store_false')
    p.add_argument('--check_tol', default=None, type=float,
                   help='Parity tolerance (defaults: 1e-3 fp32, 1e-2 fp16)')
    p.add_argument('--threshold', default=0.5, type=float,
                   help='Threshold written to the sidecar JSON (for downstream binarization).')
    return p


def main():
    args = parse_args_with_yaml(build_parser)
    if not args.checkpoint:
        raise SystemExit('--checkpoint is required (CLI or YAML)')
    # Convenience: point --checkpoint at a run dir and we export its best.pth (the
    # best-Contour_mIoU checkpoint train_refiner.py saves). Always export the best.
    if os.path.isdir(args.checkpoint):
        best = os.path.join(args.checkpoint, 'best.pth')
        if not os.path.isfile(best):
            raise SystemExit("{} is a directory but has no best.pth — pass an explicit "
                             "epoch_N.pth, or train with --save_best.".format(args.checkpoint))
        print('Resolved run dir → {}'.format(best))
        args.checkpoint = best
    H, W = args.input_size
    out_path = Path(args.out) if args.out else Path(args.checkpoint).with_suffix('.onnx')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    in_channels = 4 if args.use_mask_input else 3
    device = torch.device('cpu')   # Export on CPU for portability — tracing speed is fine.
    print('Loading {} on {}  (encoder={}, in_channels={})'.format(
        args.checkpoint, device, args.encoder_name, in_channels))

    # encoder_weights=None: weights come from the checkpoint, no need to fetch ImageNet pretrained.
    model = smp.Unet(encoder_name=args.encoder_name, encoder_weights=None,
                     in_channels=in_channels, classes=1).to(device).eval()
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('  loaded: missing={} unexpected={}'.format(len(missing), len(unexpected)))
    if missing or unexpected:
        raise SystemExit(
            'state_dict mismatch (missing={} unexpected={}) — check --encoder_name / --use_mask_input '
            'match the trained run.'.format(len(missing), len(unexpected)))

    adapter = RefinerOnnxAdapter(model, use_mask_input=args.use_mask_input,
                                 raw_logits=args.raw_logits).to(device).eval()
    if args.fp16:
        adapter = adapter.half()

    dtype = torch.float16 if args.fp16 else torch.float32
    dummy = torch.randn(1, in_channels, H, W, dtype=dtype, device=device)
    if args.use_mask_input:
        # Channel 3 is a mask in [0,1]; randn would push the clamp to saturate and hide
        # real numerical drift in the parity check. Use a plausible soft mask instead.
        dummy[:, 3:4] = torch.rand(1, 1, H, W, dtype=dtype, device=device)

    dynamic_axes = None
    if str(args.batch) == 'dynamic':
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    else:
        dummy = dummy.repeat(int(args.batch), 1, 1, 1)

    print('Tracing → {}  (input {}x{}x{}x{}, opset={}, fp16={}, batch={}, raw_logits={})'.format(
        out_path, dummy.shape[0], in_channels, H, W, args.opset, args.fp16, args.batch, args.raw_logits))
    torch.onnx.export(
        adapter, dummy, str(out_path),
        input_names=['input'], output_names=['output'],
        opset_version=args.opset, dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
    )
    print('wrote {}  ({:.1f} MB)'.format(out_path, out_path.stat().st_size / 1e6))

    sidecar = {
        'mean': _IM_MEAN,
        'std': _IM_STD,
        'input_size': [H, W],
        'in_channels': in_channels,
        'use_mask_input': bool(args.use_mask_input),
        'mask_channel': 3 if args.use_mask_input else None,
        'output': 'logits' if args.raw_logits else (
            'refined_prob=clamp(mask+tanh(logits),0,1)' if args.use_mask_input else 'sigmoid(logits)'),
        'threshold': args.threshold,
        'opset': args.opset,
        'fp16': bool(args.fp16),
        'encoder_name': args.encoder_name,
        'checkpoint': str(args.checkpoint),
    }
    sidecar_path = out_path.with_suffix('.json')
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=2)
    print('wrote {}'.format(sidecar_path))

    if args.check:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise SystemExit('--check requested but onnxruntime not installed: {}'.format(e))
        sess = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
        with torch.no_grad():
            torch_out = adapter(dummy).detach().cpu().float().numpy()
        onnx_out = sess.run(None, {'input': dummy.cpu().numpy()})[0].astype(np.float32)
        tol = args.check_tol if args.check_tol is not None else (1e-2 if args.fp16 else 1e-3)
        diff = float(np.abs(torch_out - onnx_out).max())
        assert diff < tol, 'ONNX parity failed: max |Δ| = {:.3e} > tol {:.3e}'.format(diff, tol)
        print('parity OK (max |Δ| = {:.3e}, tol = {:.3e})'.format(diff, tol))


if __name__ == '__main__':
    main()
