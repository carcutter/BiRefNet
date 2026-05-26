"""Smoke test: exercise the full finetuning pipeline (dataset -> model -> loss
-> backward -> validation image logging) at the smallest possible batch and
resolution. Uses bb_pretrained=False so it does not need local backbone weights.

Run: uv run python smoke_test.py
"""
import os
import sys
import shutil
import traceback


# Force a minimal Config BEFORE importing anything that instantiates Config().
from config import Config

_orig_init = Config.__init__


def _patched_init(self):
    _orig_init(self)
    # batch_size must be > 1 so BatchNorm modules are real (they are nn.Identity at bs=1),
    # otherwise the resumed checkpoint reports unexpected BN keys.
    self.batch_size = 2
    self.batch_size_valid = 1
    self.size = (512, 512)
    self.dynamic_size = None
    self.compile = False
    self.mixed_precision = 'bf16'
    self.num_workers = 0
    self.load_all = False
    self.precisionHigh = True
    self.use_csv_data = True
    self.train_csv = 'test_smoke.csv'
    self.val_csv = 'test_smoke.csv'
    self.val_every_n_epochs = 1
    self.val_num_samples = 0
    # Adjust lr for batch_size=1 (the formula in config scales by sqrt(bs/4)).
    import math
    self.lr = 1e-5 * math.sqrt(self.batch_size / 4)
    # Disable lambdas that aren't needed for a 1-iter smoke (keep bce + iou + ssim).
    # Already minimal for 'General' default.


Config.__init__ = _patched_init


# Now safe to import the rest.
import torch
import torch.nn as nn
from dataset import MyData
from models.birefnet import BiRefNet
from loss import PixLoss
from utils import save_tensor_img, check_state_dict
from kornia.filters import laplacian  # noqa: F401  (sanity: birefnet imports this)


CKPT_DIR = 'ckpts/smoke_test'
VIS_DIR = os.path.join(CKPT_DIR, 'val_vis', 'epoch_1')


def cleanup():
    if os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)


def check(cond, msg):
    if cond:
        print('  [OK]  {}'.format(msg))
    else:
        print('  [FAIL] {}'.format(msg))
        raise AssertionError(msg)


def main():
    cleanup()
    os.makedirs(VIS_DIR, exist_ok=True)
    cfg = Config()
    print('==> Config: bs={}, size={}, bb={}, compile={}, mp={}'.format(
        cfg.batch_size, cfg.size, cfg.bb, cfg.compile, cfg.mixed_precision))
    print('==> CUDA available: {}'.format(torch.cuda.is_available()))
    assert torch.cuda.is_available(), 'CUDA required'
    device = torch.device('cuda:0')

    # --- 1. Datasets / loaders ---
    print('\n==> Building train + val loaders from CSV')
    train_csv = os.path.join(cfg.csv_data_root, cfg.train_csv)
    val_csv = os.path.join(cfg.csv_data_root, cfg.val_csv)
    train_ds = MyData(datasets=None, data_size=cfg.size, is_train=True, csv_path=train_csv, csv_image_root=cfg.csv_image_root)
    val_ds = MyData(datasets=None, data_size=cfg.size, is_train=False, csv_path=val_csv, csv_image_root=cfg.csv_image_root)
    check(len(train_ds) > 0, 'train dataset non-empty ({} samples)'.format(len(train_ds)))
    check(len(val_ds) > 0, 'val dataset non-empty ({} samples)'.format(len(val_ds)))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=0, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size_valid, num_workers=0, shuffle=False, drop_last=False)

    sample = next(iter(train_loader))
    img_t, lbl_t, cls_t = sample
    check(tuple(img_t.shape) == (cfg.batch_size, 3, cfg.size[1], cfg.size[0]),
          'train image tensor shape {}'.format(tuple(img_t.shape)))
    check(tuple(lbl_t.shape) == (cfg.batch_size, 1, cfg.size[1], cfg.size[0]),
          'train label tensor shape {}'.format(tuple(lbl_t.shape)))
    check(lbl_t.min() >= 0 and lbl_t.max() <= 1, 'train label in [0,1]')
    # Mask binarization sanity: a non-trivial fraction should be both fg and bg.
    fg_frac = (lbl_t > 0.5).float().mean().item()
    bg_frac = (lbl_t < 0.5).float().mean().item()
    check(fg_frac > 0.001 and bg_frac > 0.001,
          'train label is non-degenerate after color->binary (fg={:.3f}, bg={:.3f})'.format(fg_frac, bg_frac))
    # And the bulk of pixels should be near-binary (after color thresholding + linear resize edge softening).
    near_binary = ((lbl_t < 0.05) | (lbl_t > 0.95)).float().mean().item()
    check(near_binary > 0.95, 'train label is near-binary (>95% pixels at 0 or 1; got {:.3f})'.format(near_binary))

    val_sample = next(iter(val_loader))
    v_img, v_lbl, v_path = val_sample
    check(v_img.shape[1] == 3, 'val image is 3-channel')
    check(v_lbl.shape[1] == 1, 'val label is 1-channel')
    check(isinstance(v_path, (list, tuple)) and len(v_path) == cfg.batch_size_valid,
          'val returns label_path list (got {})'.format(type(v_path).__name__))

    # --- 2. Model + optimizer ---
    # Load the full BiRefNet-DIS checkpoint that ships in weights/. Since we resume from a
    # complete model checkpoint, bb_pretrained=False (skips backbone .pth lookup that fails on
    # this machine), and the state_dict overrides every parameter anyway.
    resume_ckpt = 'weights/BiRefNet-DIS-bb_swin_v1_base-epoch_595.pth'
    check(os.path.isfile(resume_ckpt), 'resume checkpoint exists: {}'.format(resume_ckpt))
    print('\n==> Building BiRefNet and loading {}'.format(resume_ckpt))
    model = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(resume_ckpt, map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Allow small drift but require the core to load.
    check(len(unexpected) < 10, 'unexpected keys in checkpoint kept small (got {}: {})'.format(len(unexpected), unexpected[:3]))
    check(len(missing) < 10, 'missing keys in model kept small (got {}: {})'.format(len(missing), missing[:3]))
    if unexpected:
        print('  WARN unexpected keys: {} (first 3: {})'.format(len(unexpected), unexpected[:3]))
    if missing:
        print('  WARN missing keys: {} (first 3: {})'.format(len(missing), missing[:3]))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('  model params: {:.1f}M  (loaded {} tensors from checkpoint)'.format(n_params / 1e6, len(state_dict)))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=1e-2)
    pix_loss = PixLoss()
    crit_gdt = nn.BCELoss() if cfg.out_ref else None

    mp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(cfg.mixed_precision)
    autocast = torch.amp.autocast(device_type='cuda', dtype=mp_dtype) if mp_dtype else None

    # --- 3. One train iteration ---
    print('\n==> Running 1 training iteration')
    model.train()
    optimizer.zero_grad()
    inputs = img_t.to(device)
    gts = lbl_t.to(device)

    if autocast is not None:
        ctx = autocast
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    # Autocast wraps only the model forward (matches accelerate's behavior);
    # BCELoss is autocast-unsafe so loss computation stays in fp32.
    with ctx:
        scaled_preds, class_preds_lst = model(inputs)
    if cfg.out_ref:
        (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
        loss_gdt = 0.0
        for i, (gp, gl) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
            gp_r = nn.functional.interpolate(gp.float(), size=gl.shape[2:], mode='bilinear', align_corners=True).sigmoid()
            gl_s = gl.float().sigmoid()
            _ld = crit_gdt(gp_r, gl_s)
            loss_gdt = _ld if i == 0 else loss_gdt + _ld
    # PixLoss internally sigmoids predictions and uses BCELoss too, so cast preds to fp32 first.
    scaled_preds_fp32 = [p.float() for p in scaled_preds]
    loss_pix, loss_dict = pix_loss(scaled_preds_fp32, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
    loss = loss_pix
    if cfg.out_ref:
        loss = loss + loss_gdt
    check(torch.isfinite(loss).item(), 'training loss is finite (loss={:.4f})'.format(loss.item()))
    check(None in class_preds_lst, 'class_preds is [None] when aux classification disabled')

    loss.backward()
    # Check at least one gradient is nonzero (training-eligible param updated)
    grad_norms = [p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None]
    check(len(grad_norms) > 0 and sum(grad_norms) > 0, 'backward produced nonzero gradients ({} params with grads)'.format(len(grad_norms)))
    optimizer.step()
    print('  train loss: {:.4f}  pix loss dict keys: {}'.format(loss.item(), list(loss_dict.keys())))

    # --- 4. Validation pass with image logging (mirrors train.py:Trainer.validate) ---
    print('\n==> Running validation pass with image logging')
    model.eval()
    imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    sample_idx = 0
    written = []
    with torch.no_grad():
        for batch in val_loader:
            v_inputs = batch[0].to(device, non_blocking=True)
            v_gts = batch[1]
            v_paths = batch[2]
            with ctx:
                preds = model(v_inputs)
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            preds = preds.sigmoid().to(torch.float32)
            inputs_denorm = (v_inputs.detach().float().cpu() * imnet_std + imnet_mean).clamp(0, 1)
            preds_cpu = preds.detach().cpu()
            gts_cpu = v_gts.detach().float().cpu()
            for i in range(v_inputs.shape[0]):
                stem = os.path.splitext(os.path.basename(v_paths[i]))[0]
                tag = '{:03d}_{}'.format(sample_idx, stem)
                for kind, t in (('input', inputs_denorm[i:i+1]),
                                ('pred', preds_cpu[i:i+1]),
                                ('gt', gts_cpu[i:i+1])):
                    p = os.path.join(VIS_DIR, '{}_{}.png'.format(tag, kind))
                    save_tensor_img(t, p)
                    written.append(p)
                sample_idx += 1

    # --- 5. Assertions on logged outputs ---
    print('\n==> Verifying validation outputs')
    expected_count = len(val_ds) * 3
    check(len(written) == expected_count,
          '{} files written (expected {} = {} samples x 3)'.format(len(written), expected_count, len(val_ds)))
    for p in written:
        check(os.path.isfile(p) and os.path.getsize(p) > 0, 'wrote non-empty file {}'.format(os.path.basename(p)))

    # Open one image of each kind to confirm decodability
    from PIL import Image
    for kind in ('input', 'pred', 'gt'):
        any_path = [p for p in written if p.endswith('_{}.png'.format(kind))][0]
        with Image.open(any_path) as im:
            im.load()
            check(im.size == cfg.size, '{} image size {} matches config.size {}'.format(kind, im.size, cfg.size))

    print('\n==> SMOKE TEST PASSED')
    print('    val_vis dir: {}'.format(VIS_DIR))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
