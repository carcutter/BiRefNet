"""Export ONLY the Swin-L encoder (backbone) of a BiRefNet checkpoint to ONNX, as an image
EMBEDDING model: input image -> global-average-pooled deepest-stage feature -> [batch, C] vector.
For use as a domain-specific embedding backbone (e.g. image-explorer / FAISS similarity).

No deform-conv here (that lives in the decoder), so the export is a plain Swin trace.
Run with BIREFNET_BB=swin_v1_l so BiRefNet builds the matching backbone.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.birefnet import BiRefNet
from utils import check_state_dict

_IM_MEAN = [0.485, 0.456, 0.406]
_IM_STD = [0.229, 0.224, 0.225]


class EncoderEmbed(nn.Module):
    """Swin backbone -> deepest feature stage -> global average pool -> [B, C] embedding."""
    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, x):
        feats = self.bb(x)                 # Swin/PVT: 4-tuple (x1..x4), deepest last
        deep = feats[-1] if isinstance(feats, (list, tuple)) else feats
        return deep.mean(dim=(2, 3))       # global average pool over H,W -> [B, C]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--input_size', default='1024,1024')
    p.add_argument('--opset', default=17, type=int)
    p.add_argument('--no_check', action='store_true')
    args = p.parse_args()
    H, W = [int(v) for v in args.input_size.replace('x', ',').split(',')]

    device = torch.device('cpu')
    print('Loading {} (bb={})'.format(args.checkpoint, os.environ.get('BIREFNET_BB')))
    model = BiRefNet(bb_pretrained=False).to(device).eval()
    sd = check_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('  loaded: missing={} unexpected={}'.format(len(missing), len(unexpected)))

    enc = EncoderEmbed(model.bb).to(device).eval()
    dummy = torch.randn(1, 3, H, W)
    with torch.no_grad():
        emb = enc(dummy)
    print('  embedding dim = {}  (output shape {})'.format(emb.shape[1], tuple(emb.shape)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    print('Tracing encoder -> {}  (input 1x3x{}x{}, opset={})'.format(args.out, H, W, args.opset))
    torch.onnx.export(
        enc, dummy, args.out,
        input_names=['input'], output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    print('wrote {}  ({:.1f} MB)'.format(args.out, os.path.getsize(args.out) / 1e6))

    sidecar = {'mean': _IM_MEAN, 'std': _IM_STD, 'input_size': [H, W],
               'embedding_dim': int(emb.shape[1]), 'pooling': 'global_avg_pool_deepest_stage',
               'opset': args.opset, 'checkpoint': args.checkpoint,
               'note': 'Swin-L encoder only; RGB [0,1] then ImageNet-normalized; output = L2-unnormalized embedding.'}
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(sidecar, f, indent=2)
    print('wrote {}'.format(os.path.splitext(args.out)[0] + '.json'))

    if not args.no_check:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        onnx_out = sess.run(None, {'input': dummy.numpy()})[0]
        d = float(np.abs(emb.numpy() - onnx_out).max())
        print('parity OK (max |Δ| = {:.2e})'.format(d) if d < 1e-3 else 'PARITY WARN (max |Δ| = {:.2e})'.format(d))


if __name__ == '__main__':
    main()
