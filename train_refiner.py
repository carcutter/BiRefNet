"""Train a standalone UNet mask-refiner on the blob crops.

The refiner learns to turn a coarse (degraded) mask into a crisp one. Input is a 7-channel
stack — crop RGB + full source frame + degraded mask — and the target is the clean GT crop
mask (see dataset_refiner.py). It is fully self-contained: it does NOT import BiRefNet's
Config/backbone machinery, but it reuses the repo's metric helpers, experiment logger, and
YAML-config plumbing.

Usage:
    uv run python train_refiner.py --config configs/refiner.yaml --ckpt_dir runs/refiner
    # CPU smoke (no GPU contention):
    CUDA_VISIBLE_DEVICES="" uv run python train_refiner.py --config configs/refiner.yaml --smoke_test 2 --logger none

Precedence: argparse defaults < --config yaml < CLI flags.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import segmentation_models_pytorch as smp

from dataset_refiner import RefinerData, DEFAULT_DEGRADE_CFG
from metrics import _binary_metrics, _ring, _to_3ch, _denormalize
from experiment_logger import ExperimentLogger, parse_backends
from yaml_config import parse_args_with_yaml, dump_resolved

def _str2bool(v):
    return v if isinstance(v, bool) else str(v).lower() in ('1', 'true', 'yes', 'y')


def build_parser():
    p = argparse.ArgumentParser(
        description='UNet mask-refiner training entry point.',
        epilog='Precedence: argparse defaults < --config yaml < CLI flags',
    )
    p.add_argument('--config', default=None, type=str, help='YAML config (CLI flags override its values).')
    # ---- training loop ----
    p.add_argument('--resume', default=None, type=str, help='Checkpoint .pth to resume from.')
    p.add_argument('--epochs', default=50, type=int)
    p.add_argument('--ckpt_dir', default='runs/refiner',
                   help='Run dir for tb events, wandb logs, ckpts, resolved config.')
    p.add_argument('--batch_size', default=8, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--input_size', default=512, type=int, help='Square input size (divisible by 32).')
    p.add_argument('--num_workers', default=8, type=int)
    p.add_argument('--smoke_test', default=0, type=int,
                   help='If > 0, cap train+val to this many batches and force 1 epoch.')
    p.add_argument('--save_best', default='contour_miou', type=str,
                   help="Also save runs/<run>/best.pth whenever this val metric improves. "
                        "One of: loss | mae (lower=better) | f1 | iou | contour_miou (higher=better) | "
                        "none (disable, per-epoch ckpts only). Default contour_miou — the boundary "
                        "metric that catches the blobs-not-edges failure mode.")
    p.add_argument('--save_freq', default=5, type=int,
                   help='Save a per-epoch checkpoint (epoch_N.pth) every N epochs instead of every '
                        'epoch. A new-best epoch is always saved regardless of this interval (it also '
                        'refreshes best.pth).')
    # ---- model ----
    p.add_argument('--use_mask_input', default=True, type=_str2bool,
                   help='True ⇒ 4ch input [crop|degraded mask] (refiner). False ⇒ 3ch [crop] (plain segmentation).')
    p.add_argument('--encoder_name', default='efficientnet-b4', type=str)
    p.add_argument('--encoder_weights', default='imagenet', type=str,
                   help="Pretrained encoder weights; 'none' for random init.")
    # ---- data / split ----
    p.add_argument('--csv_index', default='data/processed/blob_crops.csv', type=str)
    p.add_argument('--val_split', default=0.2, type=float)
    p.add_argument('--csv_split_seed', default=42, type=int)
    p.add_argument('--val_num_samples', default=512, type=int,
                   help='Cap val set size for speed (0 = full val split).')
    # ---- loss weights ----
    p.add_argument('--w_delta', default=1.0, type=float,
                   help='Weight of the L1 on the predicted mask DELTA (gt - degraded input) — the '
                        'residual refinement target. Active only in refiner mode (use_mask_input=True).')
    p.add_argument('--w_bce', default=1.0, type=float,
                   help='Weight of full-image BCE vs GT on the reconstructed refined mask. Active in '
                        'both modes (in refiner mode it is computed on the reconstructed prob, not the '
                        'raw tanh-delta logits).')
    p.add_argument('--w_dice', default=1.0, type=float)
    p.add_argument('--w_boundary', default=1.0, type=float, help='Weight of BCE restricted to GT contour ring.')
    p.add_argument('--w_contour_iou', default=1.0, type=float,
                   help='Weight of (1 - soft IoU) computed on the GT contour ring; pushes crisp edges.')
    p.add_argument('--contour_radius', default=2, type=int, help='Ring half-width for boundary loss + metric.')
    p.add_argument('--w_locality', default=0.0, type=float,
                   help='Weight of the locality penalty: predicted prob outside a dilation of the '
                        'conditioning mask. 0=off. Suppresses far-from-mask hallucinations.')
    p.add_argument('--locality_dilation', default=8, type=int,
                   help='Dilation radius (px) of the conditioning mask defining the allowed region; '
                        'predictions within this band of the mask are NOT penalized.')
    # ---- degradation knobs (override dataset_refiner.DEFAULT_DEGRADE_CFG) ----
    p.add_argument('--degrade_max_radius', default=DEFAULT_DEGRADE_CFG['max_radius'], type=int)
    p.add_argument('--degrade_max_down', default=DEFAULT_DEGRADE_CFG['max_down'], type=int)
    p.add_argument('--degrade_morph_prob', default=DEFAULT_DEGRADE_CFG['morph_prob'], type=float)
    p.add_argument('--degrade_downsample_prob', default=DEFAULT_DEGRADE_CFG['downsample_prob'], type=float)
    p.add_argument('--degrade_affine_prob', default=DEFAULT_DEGRADE_CFG['affine_prob'], type=float)
    p.add_argument('--degrade_noise_prob', default=DEFAULT_DEGRADE_CFG['noise_prob'], type=float)
    p.add_argument('--degrade_spurious_prob', default=DEFAULT_DEGRADE_CFG['spurious_prob'], type=float,
                   help='P(inject spurious far-from-GT blobs into the degraded mask) — false positives the '
                        'refiner must erase (GT target stays clean). 0=off.')
    p.add_argument('--degrade_spurious_max_blobs', default=DEFAULT_DEGRADE_CFG['spurious_max_blobs'], type=int)
    p.add_argument('--degrade_spurious_min_radius', default=DEFAULT_DEGRADE_CFG['spurious_min_radius'], type=int)
    p.add_argument('--degrade_spurious_max_radius', default=DEFAULT_DEGRADE_CFG['spurious_max_radius'], type=int)
    p.add_argument('--degrade_spurious_margin', default=DEFAULT_DEGRADE_CFG['spurious_margin'], type=int,
                   help='Min gap (px) between injected blobs and the true object.')
    # ---- logging ----
    p.add_argument('--logger', default='both', type=str,
                   help="Logging backend(s): tensorboard | wandb | mlflow | both (tb+wandb) | "
                        "all (tb+wandb+mlflow) | none | comma-separated mix e.g. 'wandb,mlflow'.")
    p.add_argument('--wandb_project', default='birefnet-interior', type=str)
    p.add_argument('--wandb_run_name', default=None, type=str)
    p.add_argument('--wandb_entity', default='meero_rd', type=str)
    return p


def make_degrade_cfg(args):
    return dict(DEFAULT_DEGRADE_CFG,
                max_radius=args.degrade_max_radius, max_down=args.degrade_max_down,
                morph_prob=args.degrade_morph_prob, downsample_prob=args.degrade_downsample_prob,
                affine_prob=args.degrade_affine_prob, noise_prob=args.degrade_noise_prob,
                spurious_prob=args.degrade_spurious_prob,
                spurious_max_blobs=args.degrade_spurious_max_blobs,
                spurious_min_radius=args.degrade_spurious_min_radius,
                spurious_max_radius=args.degrade_spurious_max_radius,
                spurious_margin=args.degrade_spurious_margin)


def reconstruct_prob(logits, cond_mask):
    """Refined foreground probability from the model output.

    Residual mode (cond_mask given): the model head predicts a signed delta in [-1, 1] via tanh,
    and the refined mask is the conditioning (degraded) mask plus that delta, clamped to [0, 1]:
        refined = clamp(cond_mask + tanh(logits), 0, 1)
    Direct mode (cond_mask is None, i.e. crop-only): plain sigmoid(logits).
    """
    if cond_mask is None:
        return torch.sigmoid(logits)
    return (cond_mask + torch.tanh(logits)).clamp(0.0, 1.0)


def refiner_loss(logits, gt, args, cond_mask=None):
    """Residual mask-refinement loss. Returns (total, parts, refined_prob).

    In refiner mode (cond_mask = the degraded input mask) the model predicts a signed *delta*, not
    the full mask:
        pred_delta   = tanh(logits)                in [-1, 1]
        target_delta = gt - cond_mask              in [-1, 1]  (0 where the input already matches GT)
        refined prob = clamp(cond_mask + pred_delta, 0, 1)
    The primary term `w_delta` is an L1 between pred_delta and target_delta — so supervision is the
    DIFFERENCE between GT and the perturbed input mask (the missing/spurious pixels), not the full
    GT mask. It is dense over the whole frame: pred_delta is pushed to 0 wherever the input is
    already correct, which keeps correct regions untouched.

    `w_bce` / `w_dice` / `w_boundary` / `w_contour_iou` are computed on the *reconstructed* refined
    mask vs GT (the boundary/contour terms live on the GT contour ring, exactly where the delta
    concentrates) to keep edges crisp. `w_locality` penalizes refined foreground far from a dilation
    of cond_mask.

    Crop-only mode (cond_mask is None) has no coarse baseline to refine: it falls back to a direct
    sigmoid prediction and the w_delta term is inactive. The w_bce / w_dice / w_boundary /
    w_contour_iou terms apply in both modes.
    """
    eps = 1e-6
    ring = _ring(gt, r=args.contour_radius)               # (N,1,H,W) binary, no grad through gt
    prob = reconstruct_prob(logits, cond_mask)

    if cond_mask is not None:
        target_delta = gt - cond_mask                     # supervise on the delta, not the full GT
        delta = (torch.tanh(logits) - target_delta).abs().mean()
    else:
        delta = logits.new_zeros(())

    dice = smp.losses.DiceLoss(mode='binary', from_logits=False)(prob, gt)

    # Full-image BCE on the reconstructed refined probability. In refiner mode the model output is a
    # tanh delta (not a mask logit), so BCE is taken on `prob`, not on `logits`. bce_map is reused
    # for the contour-ring boundary term below.
    prob_c = prob.clamp(eps, 1.0 - eps)
    bce_map = F.binary_cross_entropy(prob_c, gt, reduction='none')
    bce = bce_map.mean()
    boundary = (bce_map * ring).sum() / (ring.sum() + eps)

    # Differentiable (soft) IoU between the refined prediction and GT within the GT contour ring.
    inter = (prob * gt * ring).flatten(1).sum(1)
    union = ((prob + gt - prob * gt) * ring).flatten(1).sum(1)
    contour_iou = (inter + eps) / (union + eps)
    contour_iou_loss = (1.0 - contour_iou).mean()

    # Locality penalty — mean refined prob in the region outside the dilated conditioning mask.
    locality = logits.new_zeros(())
    if args.w_locality > 0 and cond_mask is not None:
        d = int(args.locality_dilation)
        m = (cond_mask > 0.5).float()                     # binarize the (soft) conditioning mask
        allowed = F.max_pool2d(m, kernel_size=2 * d + 1, stride=1, padding=d)  # dilate by d px
        outside = 1.0 - allowed
        locality = (prob * outside).sum() / (outside.sum() + eps)

    total = (args.w_delta * delta + args.w_bce * bce + args.w_dice * dice
             + args.w_boundary * boundary + args.w_contour_iou * contour_iou_loss
             + args.w_locality * locality)
    parts = {'delta': delta.item(), 'bce': bce.item(), 'dice': dice.item(),
             'boundary': boundary.item(), 'contour_iou': contour_iou_loss.item(),
             'locality': locality.detach().item()}
    return total, parts, prob


@torch.no_grad()
def validate(model, loader, device, args, exp_logger, epoch, max_batches=None):
    model.eval()
    agg = {k: 0.0 for k in ('loss', 'iou', 'f1', 'mae', 'contour_miou')}
    base = {k: 0.0 for k in ('iou', 'f1', 'mae', 'contour_miou')}  # passthrough (degraded vs GT)
    n = 0
    panel = None
    for bi, (x, gt) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, gt = x.to(device), gt.to(device)
        logits = model(x)
        cond = x[:, 3:4] if args.use_mask_input else None
        loss, _, prob = refiner_loss(logits, gt, args, cond_mask=cond)
        m = _binary_metrics(prob, gt, contour_radius=args.contour_radius)
        agg['loss'] += loss.item()
        for k in ('iou', 'f1', 'mae', 'contour_miou'):
            agg[k] += m[k]
        if args.use_mask_input:  # passthrough baseline = degraded-input mask vs GT
            bm = _binary_metrics(x[:, 3:4], gt, contour_radius=args.contour_radius)
            for k in base:
                base[k] += bm[k]
        n += 1
        if panel is None:
            ns = min(4, x.shape[0])
            cols = [_denormalize(x[:ns, :3])]
            if args.use_mask_input:
                cols.append(_to_3ch(x[:ns, 3:4]))   # degraded-input column
            cols += [_to_3ch(gt[:ns]), _to_3ch(prob[:ns])]
            panel = torch.cat(cols, dim=3)  # [crop | (degraded) | gt | pred]

    if n == 0:
        return None
    for k in agg:
        agg[k] /= n
    msg = ('Val @ epoch {}: loss={:.4f}  IoU={:.4f}  F1={:.4f}  MAE={:.4f}  Contour_mIoU={:.4f}'
           .format(epoch, agg['loss'], agg['iou'], agg['f1'], agg['mae'], agg['contour_miou']))
    if args.use_mask_input:
        for k in base:
            base[k] /= n
        msg += '  | baseline(degraded): IoU={:.4f} Contour_mIoU={:.4f}'.format(base['iou'], base['contour_miou'])
    print(msg + '  (n_batches={})'.format(n))
    if exp_logger is not None:
        for k in ('loss', 'iou', 'f1', 'mae', 'contour_miou'):
            tag = 'Val/' + {'iou': 'IoU', 'f1': 'F1', 'mae': 'MAE',
                            'contour_miou': 'Contour_mIoU', 'loss': 'loss'}[k]
            exp_logger.add_scalar(tag, agg[k], epoch, axis='epoch')
        if args.use_mask_input:
            # Passthrough baseline — the un-refined coarse mask. Refiner should beat these.
            exp_logger.add_scalar('Baseline/IoU', base['iou'], epoch, axis='epoch')
            exp_logger.add_scalar('Baseline/Contour_mIoU', base['contour_miou'], epoch, axis='epoch')
        if panel is not None:
            exp_logger.add_image('Val/panel', make_grid(panel, nrow=1, padding=4), epoch, axis='epoch')
    return agg


def main():
    args = parse_args_with_yaml(build_parser)
    if args.smoke_test:
        args.epochs = 1
        args.num_workers = 0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    dump_resolved(args, os.path.join(args.ckpt_dir, 'config.resolved.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device={}  input_size={}  batch_size={}  encoder={}'.format(
        device, args.input_size, args.batch_size, args.encoder_name))

    # ---- data ----
    degrade_cfg = make_degrade_cfg(args)
    common = dict(csv_path=args.csv_index, image_size=args.input_size,
                  val_split=args.val_split, csv_split_seed=args.csv_split_seed,
                  degrade_cfg=degrade_cfg, use_mask_input=args.use_mask_input)
    train_ds = RefinerData(is_train=True, **common)
    val_ds = RefinerData(is_train=False, max_samples=args.val_num_samples, **common)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    print('train={} crops, val={} crops'.format(len(train_ds), len(val_ds)))

    # ---- model ----
    in_channels = 4 if args.use_mask_input else 3  # [crop|degraded mask] vs [crop]
    enc_weights = None if str(args.encoder_weights).lower() in ('none', '', 'null') else args.encoder_weights
    model = smp.Unet(encoder_name=args.encoder_name, encoder_weights=enc_weights,
                     in_channels=in_channels, classes=1).to(device)
    print('use_mask_input={} ⇒ in_channels={}'.format(args.use_mask_input, in_channels))

    epoch_st = 1
    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        model.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=True))
        if not args.smoke_test:
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # ---- logging ----
    backends = parse_backends(args.logger)
    exp_logger = None
    if backends:
        run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.ckpt_dir))
        exp_logger = ExperimentLogger(
            backends=backends, run_dir=os.path.join(args.ckpt_dir, 'tb'),
            project=args.wandb_project, run_name=run_name, entity=args.wandb_entity,
            config={'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
                    'encoder_name': args.encoder_name, 'input_size': args.input_size,
                    'use_mask_input': args.use_mask_input,
                    'val_split': args.val_split, 'csv_split_seed': args.csv_split_seed,
                    'w_delta': args.w_delta, 'w_bce': args.w_bce, 'w_dice': args.w_dice,
                    'w_boundary': args.w_boundary,
                    'w_contour_iou': args.w_contour_iou, 'w_locality': args.w_locality,
                    'locality_dilation': args.locality_dilation,
                    'degrade': degrade_cfg, 'smoke_test': args.smoke_test})

    # ---- best-checkpoint tracking ----
    # loss/mae are lower-better; f1/iou/contour_miou are higher-better.
    lower_better = {'loss', 'mae'}
    save_best = None if str(args.save_best).lower() in ('none', '', 'null') else args.save_best.lower()
    if save_best is not None and save_best not in lower_better | {'f1', 'iou', 'contour_miou'}:
        raise SystemExit("--save_best must be one of: loss|mae|f1|iou|contour_miou|none (got {!r})".format(args.save_best))
    best_metric = float('inf') if save_best in lower_better else float('-inf')

    # ---- train ----
    global_step = 0
    max_batches = args.smoke_test or None
    for epoch in range(epoch_st, args.epochs + 1):
        model.train()
        running = 0.0
        for bi, (x, gt) in enumerate(train_loader):
            if max_batches is not None and bi >= max_batches:
                break
            x, gt = x.to(device), gt.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            cond = x[:, 3:4] if args.use_mask_input else None
            loss, parts, _ = refiner_loss(logits, gt, args, cond_mask=cond)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if bi % 50 == 0:
                print('Epoch[{}/{}] Iter[{}/{}]  loss={:.4f}  delta={:.4f} bce={:.4f} dice={:.4f} '
                      'boundary={:.4f} contour_iou={:.4f} locality={:.4f}'.format(
                          epoch, args.epochs, bi, len(train_loader), loss.item(),
                          parts['delta'], parts['bce'], parts['dice'], parts['boundary'],
                          parts['contour_iou'], parts['locality']))
            if exp_logger is not None:
                exp_logger.add_scalar('Train/loss', loss.item(), global_step)
                for k, v in parts.items():
                    exp_logger.add_scalar('Train/{}'.format(k), v, global_step)
            global_step += 1
        scheduler.step()
        if exp_logger is not None:
            exp_logger.add_scalar('Train/loss_epoch_avg', running / max(1, (bi + 1)), epoch, axis='epoch')
            exp_logger.add_scalar('Optim/lr', optimizer.param_groups[0]['lr'], epoch, axis='epoch')

        val_metrics = validate(model, val_loader, device, args, exp_logger, epoch, max_batches=max_batches)

        # Is this epoch a new best? Decided before saving so a best epoch is always checkpointed
        # even when it doesn't land on the save_freq interval.
        is_best = False
        if save_best is not None and val_metrics is not None:
            cur = val_metrics[save_best]
            is_best = cur < best_metric if save_best in lower_better else cur > best_metric

        # Per-epoch checkpoint only every save_freq epochs (default 5), to avoid one .pth per epoch.
        # Exceptions always saved: a new-best epoch (also refreshes best.pth below) and the last epoch.
        if epoch % args.save_freq == 0 or is_best or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch)))

        if is_best:
            best_metric = cur
            best_path = os.path.join(args.ckpt_dir, 'best.pth')
            torch.save(model.state_dict(), best_path)
            print('  ↳ new best {}={:.4f} (epoch {}) → saved {}'.format(save_best, cur, epoch, best_path))
            if exp_logger is not None:
                exp_logger.add_scalar('Val/best_' + save_best, best_metric, epoch, axis='epoch')

    if exp_logger is not None:
        exp_logger.close()


if __name__ == '__main__':
    main()
