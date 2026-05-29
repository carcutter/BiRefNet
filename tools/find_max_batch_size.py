"""Probe the largest BiRefNet training batch size the current GPU can hold.

Strategy:
  1. Doubling phase     — start at --start_batch, double until OOM (or --max_batch).
  2. Binary-search phase — narrow between last-OK and first-OOM to ±1.
  3. Apply --margin     — int(max_ok * margin), default 0.9, to leave headroom
                          for allocator fragmentation across long runs.

Each candidate runs forward + backward + optimizer.step for --warmup_iters real
iterations because a single forward pass under-estimates memory: PyTorch's
caching allocator only commits gradient buffers on the first backward.

Use --no_backward for an inference-only probe. The resulting number is NOT
safe as the training batch size — gradients + optimizer state typically double
or triple memory use.
"""
import argparse
import gc
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.birefnet import BiRefNet
from utils import check_state_dict
from yaml_config import parse_args_with_yaml


def _parse_size(s):
    parts = [int(p.strip()) for p in s.lower().replace('x', ',').split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('--input_size must be H,W (got {!r})'.format(s))
    H, W = parts
    if H % 32 or W % 32:
        raise argparse.ArgumentTypeError('--input_size dims must be divisible by 32 (got {}x{})'.format(H, W))
    return H, W


def _is_oom(e):
    if isinstance(e, torch.cuda.OutOfMemoryError):   # PyTorch ≥ 2.0
        return True
    return isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()


def _try_batch(model, opt, scaler, B, H, W, device, amp, do_backward, iters):
    """Run `iters` real training steps at batch size B. Raises on OOM."""
    for _ in range(iters):
        x = torch.randn(B, 3, H, W, device=device)
        gt = torch.rand(B, 1, H, W, device=device)
        opt.zero_grad(set_to_none=True)
        amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if amp else _NullCtx()
        with amp_ctx:
            out = model(x)
            # Training-mode forward returns [scaled_preds, class_preds_lst]; in eval mode a tensor list.
            # With out_ref, scaled_preds itself is ((gdt_pred, gdt_label), scaled_preds_list).
            if isinstance(out, (list, tuple)) and len(out) == 2 and not torch.is_tensor(out[0]):
                inner = out[0]   # the scaled_preds (possibly wrapped by out_ref)
            else:
                inner = out
            if isinstance(inner, (list, tuple)) and len(inner) == 2 and not torch.is_tensor(inner[0]):
                inner = inner[1]   # out_ref wraps as ((gdt_pred, gdt_label), scaled_preds)
            if isinstance(inner, (list, tuple)):
                preds = inner[-1]
            else:
                preds = inner
            # Cheap surrogate loss — we're not optimizing accuracy, just exercising the memory path.
            loss = ((preds.float() - gt) ** 2).mean()
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


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _build_model(checkpoint, device):
    model = BiRefNet(bb_pretrained=False).to(device)
    if checkpoint:
        sd = torch.load(checkpoint, map_location='cpu', weights_only=True)
        sd = check_state_dict(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print('  loaded {}  (missing={} unexpected={})'.format(checkpoint, len(missing), len(unexpected)))
    model.train()
    return model


def find_max_batch(model, H, W, device, amp, do_backward,
                   start=1, hard_cap=1024, margin=0.9, warmup=3):
    opt = torch.optim.SGD(model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler() if (amp and do_backward) else None

    # Phase 1: doubling.
    last_ok, first_bad = 0, None
    B = start
    while B <= hard_cap:
        try:
            _try_batch(model, opt, scaler, B, H, W, device, amp, do_backward, warmup)
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

    if first_bad is None:
        return int(last_ok * margin), last_ok

    # Phase 2: binary search in (last_ok, first_bad).
    lo, hi = last_ok, first_bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            _try_batch(model, opt, scaler, mid, H, W, device, amp, do_backward, warmup)
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


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Precedence: argparse defaults < --config yaml < CLI flags',
    )
    p.add_argument('--config', default=None, type=str, help='YAML config (CLI flags override its values)')
    p.add_argument('--checkpoint', default=None, help='Optional .pth — load weights if given')
    p.add_argument('--input_size', default='1024,1024', type=_parse_size, help='H,W (each divisible by 32)')
    p.add_argument('--start_batch', default=2, type=int,
                   help='Start batch size; default 2 because BatchNorm in train() mode rejects batches of 1.')
    p.add_argument('--max_batch', default=1024, type=int, help='Hard safety cap; default 1024')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action='store_true', help='Wrap forward in bf16 autocast (matches typical training)')
    p.add_argument('--no_backward', action='store_true', help='Forward-only probe (inference-time number — NOT for training)')
    p.add_argument('--margin', default=0.9, type=float, help='Multiply the ceiling by this; default 0.9')
    p.add_argument('--warmup_iters', default=3, type=int)
    p.add_argument('--out', default=None, help='Write the result as YAML to this path')
    return p


def main():
    args = parse_args_with_yaml(build_parser)
    assert torch.cuda.is_available(), '--device requires CUDA'
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    H, W = args.input_size
    do_backward = not args.no_backward
    gpu_name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print('Probing  device={}  ({})  total_mem={:.1f} GB'.format(device, gpu_name, total_mem))
    print('  input_size = {}x{}   amp={}   backward={}   warmup_iters={}'.format(
        H, W, args.amp, do_backward, args.warmup_iters))

    print('Building BiRefNet on {}'.format(device))
    model = _build_model(args.checkpoint, device)

    recommended, ceiling = find_max_batch(
        model, H, W, device,
        amp=args.amp, do_backward=do_backward,
        start=args.start_batch, hard_cap=args.max_batch,
        margin=args.margin, warmup=args.warmup_iters,
    )
    print()
    print('recommended={}  ceiling={}'.format(recommended, ceiling))
    if not do_backward:
        print('WARNING: --no_backward probe — do NOT use this number as the training batch size.')

    if args.out:
        payload = {
            'recommended_batch_size': int(recommended),
            'hard_ceiling': int(ceiling),
            'input_size': [H, W],
            'device': str(device),
            'amp': bool(args.amp),
            'backward': bool(do_backward),
            'margin': float(args.margin),
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
