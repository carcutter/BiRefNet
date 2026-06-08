"""Probe the largest UNet mask-refiner training batch size the current GPU can hold.

This is the refiner counterpart to find_max_batch_size.py (which probes BiRefNet). The
refiner is a plain segmentation_models_pytorch UNet with a 4-channel input
([crop RGB | degraded mask], or 3ch when --use_mask_input is false) and a single-class
output — see train_refiner.py. The training loop runs plain fp32 forward+backward (no
autocast), so the default probe here is fp32 too; pass --amp only if you also wrap the
real train step in autocast.

Strategy (same as the BiRefNet probe):
  1. Doubling phase     — start at --start_batch, double until OOM (or --max_batch).
  2. Binary-search phase — narrow between last-OK and first-OOM to ±1.
  3. Apply --margin     — int(max_ok * margin), default 0.9, to leave headroom
                          for allocator fragmentation across long runs.

Each candidate runs forward + backward + optimizer.step for --warmup_iters real
iterations because a single forward pass under-estimates memory: PyTorch's caching
allocator only commits gradient buffers on the first backward.

Reads the same YAML as train_refiner.py (--config), so encoder_name / input_size /
use_mask_input stay in sync with the run you are sizing. CLI flags override the YAML.
"""
import argparse
import gc
import os
import sys

import torch
import segmentation_models_pytorch as smp

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from yaml_config import parse_args_with_yaml


def _is_oom(e):
    if isinstance(e, torch.cuda.OutOfMemoryError):   # PyTorch >= 2.0
        return True
    return isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _try_batch(model, opt, scaler, B, C, H, W, device, amp, do_backward, iters):
    """Run `iters` real training steps at batch size B. Raises on OOM."""
    for _ in range(iters):
        x = torch.randn(B, C, H, W, device=device)
        gt = torch.rand(B, 1, H, W, device=device)
        opt.zero_grad(set_to_none=True)
        amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if amp else _NullCtx()
        with amp_ctx:
            logits = model(x)
            # Cheap surrogate loss — we only exercise the memory path, not accuracy.
            loss = ((logits.float() - gt) ** 2).mean()
        if not do_backward:
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
    torch.cuda.synchronize()


def _build_model(encoder_name, encoder_weights, in_channels, device):
    enc_weights = None if str(encoder_weights).lower() in ('none', '', 'null') else encoder_weights
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=enc_weights,
                     in_channels=in_channels, classes=1).to(device)
    model.train()
    return model


def find_max_batch(model, C, H, W, device, amp, do_backward,
                   start=2, hard_cap=1024, margin=0.9, warmup=3, pow2=False):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-9)   # AdamW: matches train_refiner's optimizer state
    scaler = torch.amp.GradScaler() if (amp and do_backward) else None

    # Phase 1: doubling. last_ok is always a power of two (start is, and we only ever double).
    last_ok, first_bad = 0, None
    B = start
    while B <= hard_cap:
        try:
            _try_batch(model, opt, scaler, B, C, H, W, device, amp, do_backward, warmup)
            print('  ok    batch={:4d}'.format(B))
            last_ok = B
            if B == hard_cap:
                break
            B = min(B * 2, hard_cap)
        except Exception as e:
            if not _is_oom(e):
                raise
            print('  OOM   batch={:4d}'.format(B))
            first_bad = B
            torch.cuda.empty_cache()
            gc.collect()
            break

    # Power-of-two mode: the largest power of two that fit IS the answer. The fractional margin
    # would break the power-of-two property, so it is skipped; step down one notch (÷2) yourself
    # if a long run later OOMs on allocator fragmentation.
    if pow2:
        return last_ok, last_ok

    if first_bad is None:
        return int(last_ok * margin), last_ok

    # Phase 2: binary search in (last_ok, first_bad).
    lo, hi = last_ok, first_bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            _try_batch(model, opt, scaler, mid, C, H, W, device, amp, do_backward, warmup)
            print('  ok    batch={:4d}  (binsearch)'.format(mid))
            lo = mid
        except Exception as e:
            if not _is_oom(e):
                raise
            print('  OOM   batch={:4d}  (binsearch)'.format(mid))
            hi = mid
            torch.cuda.empty_cache()
            gc.collect()
    return int(lo * margin), lo


def _str2bool(v):
    return v if isinstance(v, bool) else str(v).lower() in ('1', 'true', 'yes', 'y')


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Precedence: argparse defaults < --config yaml < CLI flags',
    )
    p.add_argument('--config', default=None, type=str, help='YAML config (CLI flags override its values)')
    # These three names match train_refiner.py argparse dests so the same --config applies cleanly.
    p.add_argument('--encoder_name', default='resnet34', type=str)
    p.add_argument('--encoder_weights', default='imagenet', type=str)
    p.add_argument('--use_mask_input', default=True, type=_str2bool,
                   help='True => 4ch input [crop|degraded mask]; False => 3ch [crop].')
    p.add_argument('--input_size', default=512, type=int, help='Square input size (divisible by 32)')
    p.add_argument('--start_batch', default=2, type=int,
                   help='Start batch size; default 2 because BatchNorm in train() mode rejects batches of 1.')
    p.add_argument('--max_batch', default=1024, type=int, help='Hard safety cap; default 1024')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action='store_true', help='Wrap forward in bf16 autocast (only if real training does too)')
    p.add_argument('--no_backward', action='store_true', help='Forward-only probe (inference number — NOT for training)')
    p.add_argument('--margin', default=0.9, type=float, help='Multiply the ceiling by this; default 0.9 (ignored with --pow2)')
    p.add_argument('--pow2', action='store_true',
                   help='Report the largest power-of-2 batch size that fits (skips binary search + fractional margin).')
    p.add_argument('--warmup_iters', default=3, type=int)
    p.add_argument('--out', default=None, help='Write the result as YAML to this path')
    return p


def main():
    args = parse_args_with_yaml(build_parser)
    assert torch.cuda.is_available(), '--device requires CUDA'
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    if args.input_size % 32:
        raise SystemExit('--input_size must be divisible by 32 (got {})'.format(args.input_size))
    H = W = args.input_size
    in_channels = 4 if args.use_mask_input else 3
    do_backward = not args.no_backward
    gpu_name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print('Probing  device={}  ({})  total_mem={:.1f} GB'.format(device, gpu_name, total_mem))
    print('  encoder={}  in_channels={}  input_size={}x{}  amp={}  backward={}  warmup_iters={}'.format(
        args.encoder_name, in_channels, H, W, args.amp, do_backward, args.warmup_iters))

    print('Building UNet refiner on {}'.format(device))
    model = _build_model(args.encoder_name, args.encoder_weights, in_channels, device)

    recommended, ceiling = find_max_batch(
        model, in_channels, H, W, device,
        amp=args.amp, do_backward=do_backward,
        start=args.start_batch, hard_cap=args.max_batch,
        margin=args.margin, warmup=args.warmup_iters, pow2=args.pow2,
    )
    print()
    if args.pow2:
        print('recommended={}  (largest power-of-2 that fits)'.format(recommended))
    else:
        print('recommended={}  ceiling={}'.format(recommended, ceiling))
    if not do_backward:
        print('WARNING: --no_backward probe — do NOT use this number as the training batch size.')

    if args.out:
        payload = {
            'recommended_batch_size': int(recommended),
            'hard_ceiling': int(ceiling),
            'encoder_name': args.encoder_name,
            'in_channels': int(in_channels),
            'input_size': [H, W],
            'device': str(device),
            'amp': bool(args.amp),
            'backward': bool(do_backward),
            'margin': float(args.margin),
            'pow2': bool(args.pow2),
            'warmup_iters': int(args.warmup_iters),
            'gpu_name': gpu_name,
            'gpu_mem_gb': round(total_mem, 1),
            'torch': str(torch.__version__),
        }
        try:
            import yaml
            with open(args.out, 'w') as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            print('wrote {}'.format(args.out))
        except ImportError:
            print('pyyaml not installed; skipped writing {}'.format(args.out))


if __name__ == '__main__':
    main()
