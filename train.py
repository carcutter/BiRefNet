import os
import sys
import math
import shutil
import datetime
from contextlib import nullcontext
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def _preset_backbone_env():
    """The backbone (config.bb) drives channel counts that are read when each module builds its
    own Config() — including the model's, inside BiRefNet.__init__. Resolve the requested backbone
    from --bb / the --config YAML and export it as BIREFNET_BB *before* Config is imported below,
    so every Config() derives matching channels. CLI --bb wins over the YAML key. Mirrors the
    'set before import' approach in smoke_test.py."""
    argv = sys.argv[1:]
    def _flag(name):
        for i, a in enumerate(argv):
            if a == name and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(name + '='):
                return a.split('=', 1)[1]
        return None
    bb = _flag('--bb')
    if bb is None:
        cfg_path = _flag('--config')
        if cfg_path and os.path.isfile(cfg_path):
            import yaml
            with open(cfg_path) as f:
                bb = (yaml.safe_load(f) or {}).get('bb')
    if bb:
        os.environ['BIREFNET_BB'] = str(bb)


_preset_backbone_env()

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet
from utils import Logger, AverageMeter, set_seed, check_state_dict, save_tensor_img

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision.utils import make_grid

from experiment_logger import ExperimentLogger, parse_backends
from yaml_config import parse_args_with_yaml, dump_resolved


# Metric + image-panel helpers live in metrics.py so train_refiner.py can reuse them
# without importing this module (whose body parses argv and builds a Config() on import).
from metrics import (
    _IMNET_MEAN, _IMNET_STD, _denormalize, _to_3ch, _overlay,
    _ring, _contour_miou, _binary_metrics,
)


def build_parser():
    p = argparse.ArgumentParser(
        description='BiRefNet training entry point.',
        epilog='Precedence: argparse defaults < --config yaml < CLI flags',
    )
    p.add_argument('--config', default=None, type=str,
                   help='Path to a YAML config. CLI flags override YAML values.')
    p.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    p.add_argument('--epochs', default=120, type=int)
    p.add_argument('--ckpt_dir', default='runs/tmp',
                   help='Run dir for tb events, wandb logs, ckpts, resolved config.')
    p.add_argument('--dist', default=False, type=lambda x: x == 'True')
    p.add_argument('--use_accelerate', action='store_true',
                   help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
    p.add_argument('--logger', default='both', type=str,
                   help='Experiment-logging backend(s): tensorboard | wandb | mlflow | both (tb+wandb) | '
                        "all (tb+wandb+mlflow) | none | comma-separated mix e.g. 'wandb,mlflow'.")
    p.add_argument('--contour_radius', default=2, type=int,
                   help='Ring half-width (px) used for the boundary mIoU metric. Scale up at higher input resolutions.')
    p.add_argument('--wandb_project', default='birefnet-interior', type=str)
    p.add_argument('--wandb_run_name', default=None, type=str,
                   help='W&B run name; defaults to the basename of --ckpt_dir.')
    p.add_argument('--wandb_entity', default='meero_rd', type=str,
                   help='W&B entity (team) the run is logged under.')
    p.add_argument('--smoke_test', default=0, type=int,
                   help='If > 0, cap train + val to this many batches per epoch and force --epochs=1. '
                        'Skips checkpoint pruning and any heavy artifacts.')
    p.add_argument('--csv_index', default=None, type=str,
                   help='CSV filename under config.csv_data_root used as the single source of pairs. '
                        'When set together with --val_split>0, train/val are an rng-based deterministic '
                        'split of this file. Overrides config.csv_index. Pass "" to disable.')
    p.add_argument('--val_split', default=None, type=float,
                   help='Fraction of --csv_index reserved for validation (e.g. 0.2 = 80/20). '
                        'Overrides config.val_split.')
    p.add_argument('--csv_split_seed', default=None, type=int,
                   help='Seed for the rng-based 80/20 split. Same seed + flipped is_train ⇒ disjoint '
                        'subsets. Overrides config.csv_split_seed.')
    p.add_argument('--input_size', default=None, type=str,
                   help='Override config.size. Accepts a single int (square, e.g. 512) or "H,W"/"HxW". '
                        'Each side must be divisible by 32.')
    p.add_argument('--batch_size', default=None, type=int,
                   help='Override config.batch_size. Re-scales lr by sqrt(new/old) to keep the '
                        'config.py batch-lr relationship.')
    p.add_argument('--bb', default=None, type=str,
                   help='Backbone override (e.g. swin_v1_l for the general-resolution checkpoint, '
                        'swin_v1_b for the DIS/interior ones). Must match the resume checkpoint\'s '
                        'architecture. Exported as BIREFNET_BB before Config is built.')
    p.add_argument('--num_workers', default=8, type=int,
                   help='DataLoader worker processes for the train/val loaders (capped at os.cpu_count()). '
                        'Overrides the batch-size-derived default. Smoke runs force 0.')
    p.add_argument('--early_stop_patience', default=0, type=int,
                   help='Early stop if Val/Contour_mIoU does not improve for this many consecutive '
                        'validations (0 = disabled). Requires validation to be enabled. The best '
                        'checkpoint is always saved to <ckpt_dir>/best.pth when the metric improves.')
    p.add_argument('--early_stop_min_delta', default=0.0, type=float,
                   help='Minimum Val/Contour_mIoU increase to count as an improvement for early stopping.')
    p.add_argument('--save_step', default=None, type=int,
                   help='Save a per-epoch checkpoint every N epochs (overrides config.save_step). '
                        'Combine with --save_last to control the cadence over the whole run.')
    p.add_argument('--save_last', default=None, type=int,
                   help='Only save per-epoch checkpoints within the last N epochs (overrides '
                        'config.save_last). Set >= total epochs to checkpoint across the entire run.')
    p.add_argument('--val_vis_num_samples', default=16, type=int,
                   help='How many validation samples to dump as input/pred/gt PNGs per validation '
                        '(the first N in val order). 0 disables. Only the LATEST validated epoch is '
                        'kept on disk (val_vis/epoch_<n>); prior epochs are removed. Metrics still '
                        'run on the full val set.')
    p.add_argument('--window_blob_loss', default=False, type=lambda x: str(x).lower() == 'true',
                   help='Upweight the BCE on FG (red/green) blobs isolated inside window (blue) regions. '
                        'The dataset emits a 0/1 weight map; PixLoss applies a weighted BCE there. '
                        'Detection params live in config.py (window_blob_*).')
    p.add_argument('--window_blob_loss_weight', default=5.0, type=float,
                   help='Multiplier k for the window-blob weighted BCE: flagged pixels count k× '
                        '(weight = 1 + (k-1)*map). Only used when --window_blob_loss is set.')
    return p


args = parse_args_with_yaml(build_parser)

if args.smoke_test:
    args.epochs = 1

config = Config()
# CLI overrides for the CSV split (None ⇒ keep config defaults).
if args.csv_index is not None:
    config.csv_index = args.csv_index
if args.val_split is not None:
    config.val_split = args.val_split
if args.csv_split_seed is not None:
    config.csv_split_seed = args.csv_split_seed
if args.input_size is not None:
    parts = [int(v) for v in str(args.input_size).lower().replace('x', ',').split(',') if v.strip()]
    if len(parts) == 1:
        h = w = parts[0]
    elif len(parts) == 2:
        h, w = parts
    else:
        raise SystemExit('--input_size must be "N" or "H,W", got: {}'.format(args.input_size))
    if h % 32 or w % 32:
        raise SystemExit('--input_size sides must be divisible by 32, got: {}x{}'.format(h, w))
    config.size = (h, w)
if args.batch_size is not None:
    if args.batch_size < 1:
        raise SystemExit('--batch_size must be >= 1, got: {}'.format(args.batch_size))
    # config.lr/num_workers are derived from batch_size in Config.__init__, so refresh both.
    # Scale lr by sqrt(new/old) to preserve config.py's sqrt-batch lr relationship without
    # re-stating the task/base-lr formula here.
    config.lr *= math.sqrt(args.batch_size / config.batch_size)
    config.num_workers = max(4, args.batch_size)
    config.batch_size = args.batch_size
# Explicit --num_workers wins over the batch-size-derived value above (smoke still forces 0 below).
config.num_workers = args.num_workers
# Checkpoint cadence overrides (smoke block below still forces save_last=epochs, save_step=1).
if args.save_step is not None:
    config.save_step = args.save_step
if args.save_last is not None:
    config.save_last = args.save_last
if args.smoke_test:
    # Smoke runs should validate the pipeline, not the slow paths.
    config.compile = False
    config.size = (512, 512)
    config.dynamic_size = None
    config.num_workers = 0
    config.load_all = False
    # Always save a ckpt at the (single) final epoch — defeats the train.sh-driven
    # save_last/save_step gating which would otherwise skip epoch 1.
    config.save_last = args.epochs
    config.save_step = 1

if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = config.mixed_precision
    kwargs_handlers = [
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=False),
            utils.GradScalerKwargs(backoff_factor=0.5),
    ]
    if mixed_precision == 'fp8':
        kwargs_handlers.append(utils.AORecipeKwargs())
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=kwargs_handlers,
    )
    accelerator.print(accelerator.state)
    accelerator.print('backbone:', config.bb, ', freeze_bb:', config.freeze_bb)
    args.dist = False

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    if args.use_accelerate:
        device = accelerator.local_process_index
    else:
        device = config.device

if config.rand_seed:
    set_seed(config.rand_seed + device)

epoch_st = 1
# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Dump the resolved (defaults < yaml < CLI) config so the run is reproducible from artifacts.
dump_resolved(args, os.path.join(args.ckpt_dir, 'config.resolved.yaml'))

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
# if args.use_accelerate and accelerator.mixed_precision != 'no':
#     config.compile = False
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)

from dataset import custom_collate_fn

def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders. Worker count tracks config.num_workers (capped at CPU cores) — NOT the
    # batch size, which would throttle small-batch / big-model runs (e.g. Swin-L @1024, bs=2) and
    # starve the GPU. persistent_workers keeps the pool alive across epochs when workers > 0.
    nw = min(config.num_workers, os.cpu_count() or config.num_workers)
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=nw, pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True, persistent_workers=nw > 0,
            collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=nw, pin_memory=True,
            shuffle=is_train, sampler=None, drop_last=True, persistent_workers=nw > 0,
            collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )


def _resolve_csv(name):
    # Allow a CSV name to be either an already-valid path (absolute, or relative to the repo root,
    # e.g. data/processed/blob_crops.csv) or a bare filename living under config.csv_data_root.
    if name and (os.path.isabs(name) or os.path.isfile(name)):
        return name
    return os.path.join(config.csv_data_root, name)


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    if config.use_csv_data:
        use_index_split = bool(getattr(config, 'csv_index', '')) and getattr(config, 'val_split', 0.0) > 0
        if use_index_split:
            index_csv = _resolve_csv(config.csv_index)
            train_dataset = MyData(
                datasets=None,
                data_size=None if config.dynamic_size else config.size,
                is_train=True,
                csv_path=index_csv,
                csv_image_root=config.csv_image_root,
                val_split=config.val_split,
                csv_split_seed=config.csv_split_seed,
                window_blob_loss=args.window_blob_loss,
            )
            train_source_str = '{} (rng split, seed={}, val={})'.format(
                index_csv, config.csv_split_seed, config.val_split)
        else:
            train_csv = _resolve_csv(config.train_csv)
            train_dataset = MyData(
                datasets=None,
                data_size=None if config.dynamic_size else config.size,
                is_train=True,
                csv_path=train_csv,
                csv_image_root=config.csv_image_root,
                window_blob_loss=args.window_blob_loss,
            )
            train_source_str = train_csv
        train_loader = prepare_dataloader(train_dataset, config.batch_size, to_be_distributed=to_be_distributed, is_train=True)
        print(len(train_loader), "batches of train dataloader from {} have been created.".format(train_source_str))

        val_loader = None
        if config.val_every_n_epochs:
            if use_index_split:
                val_dataset = MyData(
                    datasets=None,
                    data_size=config.size,
                    is_train=False,
                    csv_path=_resolve_csv(config.csv_index),
                    csv_image_root=config.csv_image_root,
                    val_split=config.val_split,
                    csv_split_seed=config.csv_split_seed,
                    max_samples=config.val_num_samples,
                )
                val_source_str = '{} (rng split val slice)'.format(config.csv_index)
            elif config.val_csv:
                val_csv = _resolve_csv(config.val_csv)
                val_dataset = MyData(
                    datasets=None,
                    data_size=config.size,
                    is_train=False,
                    csv_path=val_csv,
                    csv_image_root=config.csv_image_root,
                    max_samples=config.val_num_samples,
                )
                val_source_str = val_csv
            else:
                val_dataset = None
            if val_dataset is not None:
                # Plain (un-distributed) loader: validation runs only on the main process.
                _val_nw = min(config.num_workers, os.cpu_count() or config.num_workers)
                val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset, batch_size=config.batch_size_valid,
                    num_workers=_val_nw, pin_memory=True, shuffle=False, drop_last=False,
                    persistent_workers=_val_nw > 0,
                )
                print(len(val_loader), "batches of val dataloader from {} have been created.".format(val_source_str))
        return train_loader, val_loader

    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, data_size=None if config.dynamic_size else config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    return train_loader, None


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    else:
        print('Undefined model: {}.'.format(config.model))
        return None
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            # Smoke runs want to actually iterate, so ignore the resume file's epoch suffix —
            # otherwise range(epoch_st, args.epochs+1) is empty and the training loop never runs.
            if not args.smoke_test:
                epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader, self.val_loader = data_loaders
        if args.use_accelerate:
            self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()

        # Image-denorm constants for validation visualization (ImageNet mean/std used in dataset.py).
        self._imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Experiment logger (main process only). TB events live under ckpts/<RUN>/tb/.
        self.exp_logger = None
        if self._is_main_process():
            tb_dir = os.path.join(args.ckpt_dir, 'tb')
            backends = parse_backends(args.logger)
            run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.ckpt_dir))
            self.exp_logger = ExperimentLogger(
                backends=backends,
                run_dir=tb_dir,
                project=args.wandb_project,
                entity=args.wandb_entity if hasattr(args, 'wandb_entity') else 'meero_rd',
                run_name=run_name,
                config={
                    'epochs': args.epochs, 'batch_size': config.batch_size, 'lr': config.lr,
                    'bb': config.bb, 'task': config.task, 'size': list(config.size),
                    'mixed_precision': config.mixed_precision, 'compile': config.compile,
                    'val_split': getattr(config, 'val_split', 0.0),
                    'csv_split_seed': getattr(config, 'csv_split_seed', 42),
                    'logger': args.logger,
                    'smoke_test': args.smoke_test,
                },
            )
            if backends:
                logger.info('experiment logging backends={}  tb_dir={}'.format(sorted(backends), tb_dir))
            else:
                logger.info('experiment logging disabled (--logger none)')
        self.global_step = 0

        # Others
        self.loss_log = AverageMeter()

    def _is_main_process(self):
        if args.use_accelerate:
            return accelerator.is_main_process
        if to_be_distributed:
            return int(os.environ.get("LOCAL_RANK", "0")) == 0
        return True

    def _unwrap_model(self):
        if args.use_accelerate:
            return accelerator.unwrap_model(self.model)
        if to_be_distributed:
            return self.model.module
        return self.model

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None or not config.val_every_n_epochs:
            return
        if epoch % config.val_every_n_epochs != 0:
            return
        if not self._is_main_process():
            return

        model = self._unwrap_model()
        was_training = model.training
        model.eval()

        # Keep only the LATEST validated epoch's visualizations: wipe the whole val_vis tree
        # (train.py owns it — it holds nothing but these epoch dirs) before writing this epoch's.
        n_vis = max(0, int(args.val_vis_num_samples))
        vis_root = os.path.join(args.ckpt_dir, 'val_vis')
        if os.path.isdir(vis_root):
            shutil.rmtree(vis_root, ignore_errors=True)
        vis_dir = os.path.join(vis_root, 'epoch_{}'.format(epoch))
        os.makedirs(vis_dir, exist_ok=True)
        logger.info('Validation @ epoch {}: writing up to {} sample viz to {} (latest epoch only)'.format(
            epoch, n_vis, vis_dir))

        mixed_precision = config.mixed_precision
        if mixed_precision == 'fp16':
            val_autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        elif mixed_precision == 'bf16':
            val_autocast = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            val_autocast = nullcontext()

        val_device = next(model.parameters()).device
        sample_idx = 0
        agg = {'iou': 0.0, 'f1': 0.0, 'mae': 0.0, 'contour_miou': 0.0, 'loss': 0.0, 'count': 0}
        first_batch_tb = None    # cache the first val batch's tensors for TB image panel
        for batch_idx, batch in enumerate(self.val_loader):
            if args.smoke_test and batch_idx >= args.smoke_test:
                break
            inputs = batch[0].to(val_device, non_blocking=True)
            gts = batch[1]
            label_paths = batch[2]
            with val_autocast:
                preds = model(inputs)
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            preds_logits = preds.to(torch.float32)
            preds = preds_logits.sigmoid()

            # Main-loss-only val loss: PixLoss on the final prediction only (no gdt, no cls,
            # no multi-scale supervision). Comparable to Train/loss without aux terms.
            gts_dev = gts.to(val_device, non_blocking=True).float().clamp(0, 1)
            main_loss, _ = self.pix_loss([preds_logits], gts_dev, pix_loss_lambda=1.0)
            agg['loss'] += float(main_loss.item()) * inputs.shape[0]

            inputs_denorm = (inputs.detach().float().cpu() * self._imnet_std + self._imnet_mean).clamp(0, 1)
            preds_cpu = preds.detach().cpu()
            gts_cpu = gts.detach().float().cpu()

            # Metrics on this batch (binary mask vs probability).
            m = _binary_metrics(preds_cpu, gts_cpu, contour_radius=args.contour_radius)
            bs = inputs.shape[0]
            agg['iou'] += m['iou'] * bs
            agg['f1'] += m['f1'] * bs
            agg['mae'] += m['mae'] * bs
            agg['contour_miou'] += m['contour_miou'] * bs
            agg['count'] += bs

            if first_batch_tb is None:
                first_batch_tb = (inputs_denorm.clone(), preds_cpu.clone(), gts_cpu.clone())

            # Only dump the first n_vis samples (val order is deterministic — shuffle=False — so the
            # saved subset is the same images every epoch, good for side-by-side visual comparison).
            for i in range(inputs.shape[0]):
                if sample_idx >= n_vis:
                    break
                stem = os.path.splitext(os.path.basename(label_paths[i]))[0]
                tag = '{:03d}_{}'.format(sample_idx, stem)
                save_tensor_img(inputs_denorm[i:i+1], os.path.join(vis_dir, '{}_input.png'.format(tag)))
                save_tensor_img(preds_cpu[i:i+1], os.path.join(vis_dir, '{}_pred.png'.format(tag)))
                save_tensor_img(gts_cpu[i:i+1], os.path.join(vis_dir, '{}_gt.png'.format(tag)))
                sample_idx += 1

        n = agg['count']
        contour_miou = (agg['contour_miou'] / n) if n > 0 else None

        # Always surface the val metrics in the log (needed for early stopping even with --logger none).
        if n > 0:
            logger.info('Val @ epoch {}: loss={:.4f}  IoU={:.4f}  F1={:.4f}  MAE={:.4f}  Contour_mIoU={:.4f}  (n={})'.format(
                epoch, agg['loss'] / n, agg['iou'] / n, agg['f1'] / n, agg['mae'] / n, contour_miou, n))

        # TensorBoard: epoch-level val scalars + image panel.
        if self.exp_logger is not None and n > 0:
            self.exp_logger.add_scalar('Val/loss', agg['loss'] / n, epoch, axis='epoch')
            self.exp_logger.add_scalar('Val/IoU', agg['iou'] / n, epoch, axis='epoch')
            self.exp_logger.add_scalar('Val/F1', agg['f1'] / n, epoch, axis='epoch')
            self.exp_logger.add_scalar('Val/MAE', agg['mae'] / n, epoch, axis='epoch')
            self.exp_logger.add_scalar('Val/Contour_mIoU', contour_miou, epoch, axis='epoch')
            if first_batch_tb is not None:
                img, pred, gt = first_batch_tb
                n_show = min(4, img.shape[0])
                pred3 = _to_3ch(pred[:n_show])
                gt3 = _to_3ch(gt[:n_show])
                panel = torch.cat([img[:n_show], gt3, pred3], dim=3)    # [img | gt | pred]
                grid = make_grid(panel, nrow=1, padding=4)
                self.exp_logger.add_image('Val/predictions', grid, epoch, axis='epoch')

        if was_training:
            model.train()
        return contour_miou    # higher = better; used for early stopping in main()

    def _train_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
            weight_map = batch[3] if len(batch) > 3 else None   # window-blob loss weight map
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
            weight_map = batch[3].to(device) if len(batch) > 3 else None
        self.optimizer.zero_grad()
        scaled_preds, class_preds_lst = self.model(inputs)
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            # self.loss_dict['loss_gdt'] = loss_gdt.item()
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels)
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix, loss_dict_pix = self.pix_loss(
            scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0,
            weight_map=weight_map, weight_k=args.window_blob_loss_weight)
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        self.loss_log.update(loss.item(), inputs.size(0))
        # TensorBoard: per-step scalars (main process only).
        if self.exp_logger is not None:
            self.exp_logger.add_scalar('Train/loss', loss.item(), self.global_step)
            for k, v in self.loss_dict.items():
                self.exp_logger.add_scalar('Train/{}'.format(k), v, self.global_step)
            if config.out_ref:
                self.exp_logger.add_scalar('Train/loss_gdt', loss_gdt.item(), self.global_step)
        self.global_step += 1
        if args.use_accelerate:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}
        if epoch > args.epochs + config.finetune_last_epochs:
            if config.task == 'Matting':
                self.pix_loss.lambdas_pix_last['mae'] *= 1
                self.pix_loss.lambdas_pix_last['mse'] *= 0.9
                self.pix_loss.lambdas_pix_last['ssim'] *= 0.9
            else:
                self.pix_loss.lambdas_pix_last['bce'] *= 0
                self.pix_loss.lambdas_pix_last['ssim'] *= 1
                self.pix_loss.lambdas_pix_last['iou'] *= 0.5
                self.pix_loss.lambdas_pix_last['mae'] *= 0.9

        self.loss_log.reset()
        for batch_idx, batch in enumerate(self.train_loader):
            if args.smoke_test and batch_idx >= args.smoke_test:
                break
            # with nullcontext if not args.use_accelerate or accelerator.gradient_accumulation_steps <= 1 else accelerator.accumulate(self.model):
            # TensorBoard: log the (post-augmentation) input batch once per epoch.
            if batch_idx == 0 and self.exp_logger is not None:
                self._log_augmented_batch_to_tb(batch, epoch)
            self._train_batch(batch)
            # Logger
            if (epoch < 2 and batch_idx < 100 and batch_idx % 20 == 0) or batch_idx % max(100, len(self.train_loader) / 100 // 100 * 100) == 0:
                info_progress = f'Epoch[{epoch}/{args.epochs}] Iter[{batch_idx}/{len(self.train_loader)}].'
                info_loss = 'Training Losses:'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += f' {loss_name}: {loss_value:.5g} |'
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = f'@==Final== Epoch[{epoch}/{args.epochs}]  Training Loss: {self.loss_log.avg:.5g}  '
        logger.info(info_loss)

        # TensorBoard: per-epoch summary scalars.
        if self.exp_logger is not None:
            self.exp_logger.add_scalar('Train/loss_epoch_avg', self.loss_log.avg, epoch, axis='epoch')
            for pg in self.optimizer.param_groups:
                self.exp_logger.add_scalar('Optim/lr', pg['lr'], epoch, axis='epoch')
                break

        self.lr_scheduler.step()
        return self.loss_log.avg

    def _log_augmented_batch_to_tb(self, batch, epoch, n=4):
        """Log a denormalized [img | gt | overlay] panel to TensorBoard."""
        try:
            inputs = batch[0][:n].detach()
            gts = batch[1][:n].detach()
        except Exception:
            return
        img = _denormalize(inputs)
        gt3 = _to_3ch(gts)
        ovl = _overlay(img, gts.float().cpu())
        # Each sample becomes a horizontal triple [img | gt | overlay].
        panel = torch.cat([img, gt3, ovl], dim=3)
        grid = make_grid(panel, nrow=1, padding=4)
        self.exp_logger.add_image('Inputs/augmented', grid, epoch, axis='epoch')


def main():

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )

    def _state_dict():
        if args.use_accelerate:
            return trainer.model.state_dict()
        return trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()

    best_miou = float('-inf')
    best_epoch = None
    epochs_since_improve = 0
    for epoch in range(epoch_st, args.epochs+1):
        train_loss = trainer.train_epoch(epoch)
        val_miou = trainer.validate(epoch)
        # Save checkpoint
        if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
            torch.save(_state_dict(), os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch)))

        # Early stopping on Val/Contour_mIoU (higher = better). Only on the main process, and only
        # on epochs where validation actually ran (val_miou is None otherwise). Always keep best.pth.
        if val_miou is not None and trainer._is_main_process():
            if val_miou > best_miou + args.early_stop_min_delta:
                best_miou, best_epoch, epochs_since_improve = val_miou, epoch, 0
                torch.save(_state_dict(), os.path.join(args.ckpt_dir, 'best.pth'))
                logger.info('New best Val/Contour_mIoU={:.4f} @ epoch {} -> saved best.pth'.format(best_miou, epoch))
            else:
                epochs_since_improve += 1
                if args.early_stop_patience and epochs_since_improve >= args.early_stop_patience:
                    logger.info('Early stopping at epoch {}: Val/Contour_mIoU has not improved for {} '
                                'validations (best={:.4f} @ epoch {}).'.format(
                                    epoch, epochs_since_improve, best_miou, best_epoch))
                    break
    if best_epoch is not None:
        logger.info('Training done. Best Val/Contour_mIoU={:.4f} @ epoch {} (best.pth).'.format(best_miou, best_epoch))
    if trainer.exp_logger is not None:
        trainer.exp_logger.close()
    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()
