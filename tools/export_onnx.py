"""Export a trained BiRefNet checkpoint to ONNX + parity-check it with onnxruntime.

The training-time `BiRefNet.forward(x)` returns a *list of side-map logits*
where `[-1]` is the main prediction. ONNX exporters dislike dict/list returns,
so we wrap the model in `OnnxAdapter`: a thin `nn.Module` whose `forward(x)`
returns a single sigmoided HxW tensor.

A sidecar `<out>.json` is written with the preprocessing assumptions
(`mean`, `std`, `input_size`, `threshold`) so downstream consumers cannot drift
from the training-time pipeline.
"""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.birefnet import BiRefNet
from utils import check_state_dict
from yaml_config import parse_args_with_yaml


@contextlib.contextmanager
def _annotate_deform_conv2d_sizes(scope_to_hw):
    """Wrap the deform_conv2d ONNX symbolic to set static input/offset H/W on the JIT
    tensors before calling the real symbolic. `scope_to_hw` maps a scope-name substring
    to `(H, W)` for that deform_conv2d call. Input channel count is read from the
    weight tensor (known-static); offset channel count from the offset tensor type
    when available, else `2 * Kh * Kw` (no offset groups in this model).
    """
    import deform_conv2d_onnx_exporter as dce
    from torch.onnx.symbolic_helper import parse_args
    from torch.onnx import JitScalarType
    import torch.onnx.symbolic_helper as sh

    # Bridge a PyTorch API change: the exporter looks up `sh.cast_pytorch_to_onnx[JitScalarType.X]`
    # but in current PyTorch the dict is keyed by string names ('Float', 'Half', ...).
    # Add JitScalarType-keyed aliases so the lookup works without monkey-patching the dict shape.
    _added_keys = []
    _name_to_jit = {
        'Byte': JitScalarType.UINT8, 'Char': JitScalarType.INT8,
        'Double': JitScalarType.DOUBLE, 'Float': JitScalarType.FLOAT,
        'Half': JitScalarType.HALF, 'Int': JitScalarType.INT,
        'Long': JitScalarType.INT64, 'Short': JitScalarType.INT16,
        'Bool': JitScalarType.BOOL,
    }
    for name, jit_type in _name_to_jit.items():
        if name in sh.cast_pytorch_to_onnx and jit_type not in sh.cast_pytorch_to_onnx:
            sh.cast_pytorch_to_onnx[jit_type] = sh.cast_pytorch_to_onnx[name]
            _added_keys.append(jit_type)

    # Pre-build the inner symbolic that does the actual decomposition.
    _inner = dce.deform_conv2d_func(use_gathernd=True, enable_openvino_patch=False)

    def _resolve_hw(scope_name):
        for k, hw in scope_to_hw.items():
            if k in scope_name:
                return hw
        return None

    def _set_sizes_if_missing(jit_value, sizes):
        try:
            cur = jit_value.type().sizes()
        except Exception:
            cur = None
        if cur is None or any(s is None for s in (cur or [])):
            try:
                jit_value.setType(jit_value.type().with_sizes(sizes))
                return True
            except Exception:
                return False
        return False

    @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
    def wrapped_symbolic(g, input, weight, offset, mask, bias, stride_h, stride_w,
                         pad_h, pad_w, dilation_h, dilation_w,
                         n_weight_grps, n_offset_grps, use_mask):
        # We need the SCOPE OF THE DEFORM_CONV2D OP itself, not of `input`. The input
        # tensor was produced by an upstream op (the offset_conv) whose scope sits inside
        # the same DeformableConv2d module — so input.node().scopeName() works for the
        # offset-conv case but not always for the deform input. Walk upward through the
        # graph's call sites instead.
        scope = ''
        try:
            # Each JIT value carries its producing node; for the deform_conv2d call site
            # itself we look at the consumer. There's no direct consumer pointer, so we
            # fall back to checking the input's node first, then the offset's.
            for cand in (input.node(), offset.node(), weight.node()):
                s = cand.scopeName()
                if 'dec_att' in s or 'squeeze_module' in s:
                    scope = s; break
            if not scope:
                scope = input.node().scopeName()
        except Exception:
            pass
        hw = _resolve_hw(scope)
        if hw is not None:
            try:
                w_sizes = weight.type().sizes()
                kh, kw = int(w_sizes[2]), int(w_sizes[3])
                in_ch = int(w_sizes[1]) * int(n_weight_grps)
            except Exception:
                kh = kw = None; in_ch = None
            H, W = hw
            if in_ch is not None:
                _set_sizes_if_missing(input, [1, in_ch, H, W])
            if kh is not None and kw is not None:
                offset_ch = 2 * kh * kw * int(n_offset_grps)
                _set_sizes_if_missing(offset, [1, offset_ch, H, W])
                if use_mask and mask is not None:
                    mask_ch = kh * kw * int(n_offset_grps)
                    _set_sizes_if_missing(mask, [1, mask_ch, H, W])
        return _inner(g, input, weight, offset, mask, bias, stride_h, stride_w,
                      pad_h, pad_w, dilation_h, dilation_w,
                      n_weight_grps, n_offset_grps, use_mask)

    dce.register_custom_op_symbolic('torchvision::deform_conv2d', wrapped_symbolic, dce.onnx_opset_version)
    try:
        yield
    finally:
        # Restore the upstream symbolic so subsequent runs aren't poisoned.
        dce.register_deform_conv2d_onnx_op()
        for k in _added_keys:
            sh.cast_pytorch_to_onnx.pop(k, None)


@contextlib.contextmanager
def _static_shapes_for_export(model, dummy):
    """At export time, monkey-patch BiRefNet's `forward_enc` cxt-concat and the
    `Decoder.forward` interpolate calls to use Python-int sizes captured from a
    dry forward, instead of `tensor.shape[2:]`. The JIT tracer otherwise records
    these as `aten::size(...)` SymInt feeds, which leaves the deform_conv2d
    input's spatial dims untyped — and the GatherND decomposer needs static H/W.

    Restores everything on exit.
    """
    import torch.nn.functional as Fn
    from models.birefnet import BiRefNet, Decoder, image2patches
    from kornia.filters import laplacian

    # 1. Dry forward to learn intermediate spatial dims at the export resolution.
    with torch.no_grad():
        x1, x2, x3, x4 = model.bb(dummy)
    sizes = {
        'x1': tuple(int(v) for v in x1.shape[2:]),
        'x2': tuple(int(v) for v in x2.shape[2:]),
        'x3': tuple(int(v) for v in x3.shape[2:]),
        'x4': tuple(int(v) for v in x4.shape[2:]),
        'x': tuple(int(v) for v in dummy.shape[2:]),
    }
    print('  captured static spatial sizes: {}'.format(sizes))

    # 2. Build static replacements for forward_enc and Decoder.forward.
    _orig_forward_enc = BiRefNet.forward_enc
    _orig_decoder_forward = Decoder.forward

    def static_forward_enc(self, x):
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.bb.conv1(x); x2 = self.bb.conv2(x1); x3 = self.bb.conv3(x2); x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)
        if self.config.mul_scl_ipt:
            B, C, H, W = x.shape
            x_pyramid = Fn.interpolate(x, size=(sizes['x'][0]//2, sizes['x'][1]//2), mode='bilinear', align_corners=True)
            if self.config.mul_scl_ipt == 'cat':
                if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
                    x1_ = self.bb.conv1(x_pyramid); x2_ = self.bb.conv2(x1_); x3_ = self.bb.conv3(x2_); x4_ = self.bb.conv4(x3_)
                else:
                    x1_, x2_, x3_, x4_ = self.bb(x_pyramid)
                x1 = torch.cat([x1, Fn.interpolate(x1_, size=sizes['x1'], mode='bilinear', align_corners=True)], dim=1)
                x2 = torch.cat([x2, Fn.interpolate(x2_, size=sizes['x2'], mode='bilinear', align_corners=True)], dim=1)
                x3 = torch.cat([x3, Fn.interpolate(x3_, size=sizes['x3'], mode='bilinear', align_corners=True)], dim=1)
                x4 = torch.cat([x4, Fn.interpolate(x4_, size=sizes['x4'], mode='bilinear', align_corners=True)], dim=1)
            elif self.config.mul_scl_ipt == 'add':
                x1_, x2_, x3_, x4_ = self.bb(x_pyramid)
                x1 = x1 + Fn.interpolate(x1_, size=sizes['x1'], mode='bilinear', align_corners=True)
                x2 = x2 + Fn.interpolate(x2_, size=sizes['x2'], mode='bilinear', align_corners=True)
                x3 = x3 + Fn.interpolate(x3_, size=sizes['x3'], mode='bilinear', align_corners=True)
                x4 = x4 + Fn.interpolate(x4_, size=sizes['x4'], mode='bilinear', align_corners=True)
        class_preds = self.cls_head(self.avgpool(x4).view(x4.shape[0], -1)) if self.training and self.config.auxiliary_classification else None
        if self.config.cxt:
            x4 = torch.cat(
                (
                    *[
                        Fn.interpolate(x1, size=sizes['x4'], mode='bilinear', align_corners=True),
                        Fn.interpolate(x2, size=sizes['x4'], mode='bilinear', align_corners=True),
                        Fn.interpolate(x3, size=sizes['x4'], mode='bilinear', align_corners=True),
                    ][-len(self.config.cxt):],
                    x4
                ),
                dim=1
            )
        return (x1, x2, x3, x4), class_preds

    def static_decoder_forward(self, features):
        # eval mode only — no gdt branches.
        x, x1, x2, x3, x4 = features
        outs = []
        # Apply image-patch branch with static target sizes derived from x4/x3/x2/x1/x.
        # ipt_blk5 → x4, ipt_blk4 → x3, ipt_blk3 → x2, ipt_blk2 → x1, ipt_blk1 → x.
        from models.birefnet import image2patches as _i2p
        if self.config.dec_ipt:
            patches = _i2p(x, patch_ref=x4, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            x4 = torch.cat((x4, self.ipt_blk5(Fn.interpolate(patches, size=sizes['x4'], mode='bilinear', align_corners=True))), 1)
        p4 = self.decoder_block4(x4)
        if self.config.out_ref:
            p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        _p4 = Fn.interpolate(p4, size=sizes['x3'], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)
        if self.config.dec_ipt:
            patches = _i2p(x, patch_ref=_p3, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p3 = torch.cat((_p3, self.ipt_blk4(Fn.interpolate(patches, size=sizes['x3'], mode='bilinear', align_corners=True))), 1)
        p3 = self.decoder_block3(_p3)
        if self.config.out_ref:
            p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        _p3 = Fn.interpolate(p3, size=sizes['x2'], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)
        if self.config.dec_ipt:
            patches = _i2p(x, patch_ref=_p2, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p2 = torch.cat((_p2, self.ipt_blk3(Fn.interpolate(patches, size=sizes['x2'], mode='bilinear', align_corners=True))), 1)
        p2 = self.decoder_block2(_p2)
        if self.config.out_ref:
            p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        _p2 = Fn.interpolate(p2, size=sizes['x1'], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)
        if self.config.dec_ipt:
            patches = _i2p(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk2(Fn.interpolate(patches, size=sizes['x1'], mode='bilinear', align_corners=True))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = Fn.interpolate(_p1, size=sizes['x'], mode='bilinear', align_corners=True)
        if self.config.dec_ipt:
            patches = _i2p(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk1(Fn.interpolate(patches, size=sizes['x'], mode='bilinear', align_corners=True))), 1)
        p1_out = self.conv_out1(_p1)
        outs.append(p1_out)
        return outs

    BiRefNet.forward_enc = static_forward_enc
    Decoder.forward = static_decoder_forward
    print('  applied static-shape monkey-patches: BiRefNet.forward_enc, Decoder.forward')
    try:
        yield sizes
    finally:
        BiRefNet.forward_enc = _orig_forward_enc
        Decoder.forward = _orig_decoder_forward
        print('  reverted static-shape monkey-patches')


@contextlib.contextmanager
def _interpolate_size_as_python_ints():
    """During tracing, BiRefNet's Decoder calls `F.interpolate(..., size=feat.shape[2:])`.
    Under `torch.onnx.export` this records a graph where the `size` arg is a SymInt
    tensor, which then leaves downstream conv inputs without a static H/W in the
    JIT type system — `deform_conv2d_onnx_exporter` then fails inside `create_dcn_params`
    with `NoneType + int`. Coercing `size` to a tuple of Python ints during the trace
    keeps intermediate shapes static throughout the graph."""
    original = F.interpolate

    def _to_python_int(v):
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            return int(v.item())
        if isinstance(v, torch.SymInt):
            return int(v)
        return v

    def patched(input, size=None, scale_factor=None, mode='nearest',
                align_corners=None, recompute_scale_factor=None, antialias=False):
        if size is not None:
            if isinstance(size, torch.Tensor):
                size = tuple(int(s) for s in size.tolist())
            elif isinstance(size, (list, tuple)):
                size = tuple(_to_python_int(s) for s in size)
            else:
                size = _to_python_int(size)
        return original(input, size=size, scale_factor=scale_factor, mode=mode,
                        align_corners=align_corners,
                        recompute_scale_factor=recompute_scale_factor, antialias=antialias)

    F.interpolate = patched
    try:
        yield
    finally:
        F.interpolate = original


def _register_deform_conv2d_symbolic():
    """torchvision::deform_conv2d has no built-in ONNX symbolic. The dedicated
    `deform_conv2d_onnx_exporter` package decomposes the op into ops the legacy
    TorchScript exporter understands (requires opset ≥ 12).

    Patches `get_tensor_dim_size` so that when the JIT type system can't infer the
    input's spatial dims (BiRefNet's first deform_conv2d ingests features whose
    shapes go through `F.interpolate(size=tensor.shape[2:])`, leaving the JIT type
    as the bare `Tensor`), we fall back to the offset tensor — its H/W equals the
    deform_conv2d's *output* H/W, which equals input H/W since `DeformableConv2d`
    in `models/modules/aspp.py` is hard-coded to `stride=1`.
    """
    try:
        import deform_conv2d_onnx_exporter as dce
    except ImportError as e:
        raise SystemExit(
            "Model uses torchvision.ops.deform_conv2d (via ASPPDeformable) but the "
            "`deform_conv2d_onnx_exporter` package is missing. Install it with "
            "`uv add deform-conv2d-onnx-exporter`.\nOriginal error: {}".format(e)
        )

    dce.register_deform_conv2d_onnx_op()


_IM_MEAN = [0.485, 0.456, 0.406]
_IM_STD = [0.229, 0.224, 0.225]


class OnnxAdapter(nn.Module):
    """Wrap BiRefNet so forward(x) → single (B,1,H,W) sigmoided tensor.

    Keeps the training-time model untouched. `apply_sigmoid` defaults to True
    so downstream consumers can threshold the output directly without having
    to know the model produces logits.
    """
    def __init__(self, model, apply_sigmoid=True):
        super().__init__()
        self.model = model
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out


def _parse_size(s):
    parts = [int(p.strip()) for p in s.lower().replace('x', ',').split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('--input_size must be H,W (got {!r})'.format(s))
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
    p.add_argument('--out', default=None, help='Output .onnx (default: <checkpoint_dir>/model.onnx)')
    p.add_argument('--input_size', default='1024,1024', type=_parse_size, help='H,W to trace at (each divisible by 32)')
    p.add_argument('--batch', default='dynamic', help='Static batch size, or "dynamic"')
    p.add_argument('--opset', default=17, type=int)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--no_sigmoid', action='store_true',
                   help='Emit raw logits instead of sigmoided probabilities.')
    p.add_argument('--check', action='store_true', default=True,
                   help='Re-run the exported model with onnxruntime and assert parity (default: on).')
    p.add_argument('--no_check', dest='check', action='store_false')
    p.add_argument('--check_tol', default=None, type=float,
                   help='Parity tolerance (defaults: 1e-3 fp32, 1e-2 fp16)')
    p.add_argument('--threshold', default=0.5, type=float,
                   help='Threshold written to the sidecar JSON (for downstream binarization).')
    return p


_DEFORM_SCOPES = ('atrous_conv', 'dec_att')   # ASPPDeformable deform_conv2d decomposition


def make_deform_batch_dynamic(model):
    """Free the batch dim in the deform_conv2d (ASPPDeformable) decomposition.

    The `deform_conv2d_onnx_exporter` decomposition reads the export-time batch (1) off the
    input tensor and bakes it as a constant leading dim into its per-batch Reshapes
    ([b, group, ch, h, w], [b, group, K, 2, h, w], and the rank-4 K*H*W flatten reshapes).
    That makes the ONNX usable only at batch=1 even though `input` has a dynamic batch axis.

    `b` is only ever a Reshape leading-dim (never arithmetic), so flipping that constant 1 -> -1
    lets ONNX infer the batch at runtime — every other dim in those shapes is a concrete positive
    int, so exactly one inferred dim per reshape. Scoped to the deform decomposition; flipping a
    leading 1 is safe regardless (a genuine batch dim infers N; a batch-independent singleton
    always infers 1). H/W stay static (the GatherND index math needs them). Returns #reshapes fixed.
    """
    g = model.graph
    reshape_shapes = {
        n.input[1] for n in g.node
        if n.op_type == 'Reshape' and len(n.input) >= 2
        and any(s in (n.name or '') for s in _DEFORM_SCOPES)
    }
    n_fixed = 0

    def _flip(arr):
        if arr.ndim == 1 and int(arr[0]) == 1 and not (arr < 0).any():
            out = arr.copy()
            out[0] = -1
            return out
        return None

    for init in g.initializer:
        if init.name in reshape_shapes:
            new = _flip(numpy_helper.to_array(init))
            if new is not None:
                init.CopyFrom(numpy_helper.from_array(new, init.name))
                n_fixed += 1
    for n in g.node:
        if n.op_type == 'Constant' and n.output and n.output[0] in reshape_shapes:
            for a in n.attribute:
                if a.name == 'value':
                    new = _flip(numpy_helper.to_array(a.t))
                    if new is not None:
                        a.t.CopyFrom(numpy_helper.from_array(new))
                        n_fixed += 1
    return n_fixed


def main():
    args = parse_args_with_yaml(build_parser)
    if not args.checkpoint:
        raise SystemExit('--checkpoint is required (CLI or YAML)')
    H, W = args.input_size
    out_path = Path(args.out) if args.out else Path(args.checkpoint).with_suffix('.onnx')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')   # Export on CPU for portability — tracing speed is fine.
    print('Loading {} on {}'.format(args.checkpoint, device))
    model = BiRefNet(bb_pretrained=False).to(device).eval()
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('  loaded: missing={} unexpected={}'.format(len(missing), len(unexpected)))

    adapter = OnnxAdapter(model, apply_sigmoid=not args.no_sigmoid).to(device).eval()
    if args.fp16:
        adapter = adapter.half()

    dummy = torch.randn(1, 3, H, W, dtype=torch.float16 if args.fp16 else torch.float32, device=device)

    dynamic_axes = None
    if str(args.batch) == 'dynamic':
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    else:
        bs = int(args.batch)
        dummy = dummy.repeat(bs, 1, 1, 1)

    # Register the missing torchvision::deform_conv2d ONNX symbolic. Opset 12+
    # is required by the GatherND-based decomposition; the script defaults to 17.
    if args.opset < 12:
        raise SystemExit('--opset must be ≥ 12 for deform_conv2d export (got {})'.format(args.opset))
    _register_deform_conv2d_symbolic()

    print('Tracing → {}  (input {}x3x{}x{}, opset={}, fp16={}, batch={})'.format(
        out_path, dummy.shape[0], H, W, args.opset, args.fp16, args.batch))
    # dynamo=False ⇒ legacy TorchScript-based exporter. The new dynamo exporter (PyTorch ≥ 2.5)
    # fails on Swin v1 attention's view-after-transpose with symbolic batch dimensions:
    #   "Cannot view a tensor with shape [36*s77, 144, 4, 32]..."
    # Vision-model coverage of the legacy path is broader; keep it as the default.
    with _static_shapes_for_export(model, dummy) as sizes:
        # All deform_conv2d in this model sit inside ASPPDeformable, which appears
        # in: (a) squeeze_module — operates at x4 resolution, and (b) decoder_block{1..4} —
        # operate at x1/x2/x3/x4 resolution respectively. Map scope substrings to HW.
        scope_to_hw = {
            'squeeze_module': sizes['x4'],
            'decoder_block4': sizes['x4'],
            'decoder_block3': sizes['x3'],
            'decoder_block2': sizes['x2'],
            'decoder_block1': sizes['x1'],
        }
        with _annotate_deform_conv2d_sizes(scope_to_hw):
            torch.onnx.export(
                adapter, dummy, str(out_path),
                input_names=['input'], output_names=['output'],
                opset_version=args.opset, dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                dynamo=False,
            )
    print('wrote {}  ({:.1f} MB)'.format(out_path, out_path.stat().st_size / 1e6))

    # The deform_conv2d decomposition bakes batch=1 into its Reshapes; free it so the dynamic
    # input batch axis actually works (otherwise the model only runs at batch=1).
    _m = onnx.load(str(out_path))
    _n_fixed = make_deform_batch_dynamic(_m)
    onnx.save(_m, str(out_path))
    print('  made deform-conv batch dynamic: {} reshape dims -> -1'.format(_n_fixed))

    # Sidecar preprocessing manifest so downstream cannot drift.
    sidecar = {
        'mean': _IM_MEAN,
        'std': _IM_STD,
        'input_size': [H, W],
        'threshold': args.threshold,
        'output_sigmoided': not args.no_sigmoid,
        'opset': args.opset,
        'fp16': bool(args.fp16),
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
