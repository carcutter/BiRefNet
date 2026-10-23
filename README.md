# Interior Segmentation — BiRefNet (coarse full-frame)

Fork of [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) for **car interior/exterior full-frame
binary segmentation**. This README documents the current production training run
(`coarse-swin-l-add-old-details`). The upstream project README is preserved as
[`README_ORIGINAL.md`](README_ORIGINAL.md).

## Training details

| Aspect | Value |
|---|---|
| **Model** | BiRefNet — Swin-L (`swin_v1_l`) encoder, `ASPPDeformable` decoder, multi-scale input (`cat`), bilateral-reference gradient supervision (`out_ref`) |
| **Task / mode** | Coarse full-frame binary segmentation (`General`), **full fine-tune** (encoder + decoder trained; `freeze_bb=false`) |
| **Pretraining weights** | `BiRefNet-DIS-epoch_590.pth` — official public DIS-pretrained Swin-L checkpoint |
| **Data** | `data_link/train/index.csv` — **15,279** image/mask pairs |
| **Split** | 90 / 10 reproducible RNG split (`--val_split 0.1`, `--csv_split_seed 42`) → **13,751 train / 1,528 val** |
| **Mask binarization** | `auto` — coloured masks ⇒ red+green = FG; black-and-white masks ⇒ white = FG |
| **Input size** | 1024 × 1024 (each side ÷ 32) |
| **Augmentation** | flip 0.5 · half-size crop 0.3 · rotate ±15° 0.2 · letterbox 0.3 · colour-enhance (always) · Gaussian blur 0.15 |
| **Optimizer** | AdamW, `weight_decay = 1e-2` |
| **Learning rate** | **≈ 7.07e-6** (base 1e-5 for `General` × √(batch/4), batch = 2) |
| **LR scheduler** | `MultiStepLR` (γ = 0.5), milestone at epoch 100,000 ⇒ **effectively constant** for this run; last 20 epochs taper the pixel-loss weights (`finetune_last_epochs = -20`) |
| **Epochs** | Absolute target **690** (100 epochs from DIS ep590); **early stopping** on `Val/Contour_mIoU`, patience 10 |
| **Batch size** | 2 (max-batch probe: safe max at 1024² fp16; hard ceiling 3) |
| **Mixed precision** | **fp16 AMP** — autocast the model forward; losses computed in fp32 (BCE is unsafe under fp16 autocast); `GradScaler` for fp16 |
| **Validation metrics** | loss · IoU · F1 · MAE · **Contour_mIoU** (boundary mean-IoU, radius 4) — early stopping monitors Contour_mIoU |
| **Throughput opts** | `num_workers=8`, `pin_memory` + `non_blocking` H2D, `prefetch_factor=4`, `persistent_workers`, `cudnn.benchmark`, TF32 |
| **Experiment logging** | TensorBoard + MLflow (local SQLite at `runs/<run>/tb/mlflow.db`) |
| **Hardware** | 1× NVIDIA L40S (44.4 GB); ~78 min/epoch at fp16 |

## Notes on each choice

- **Data.** The index mixes interior full-frames (`gold_dataset/*`: kw2552/2604/2610/2614/2616/2622
  and the kw2624 refinement set) with **detail crops** (`details/rims`, `center_console`, `rooftop`,
  `cropped_exterior_images`) and a legacy `old_dataset_details` set. Because these buckets use two
  mask conventions, binarization is **`auto`**: masks containing colour use the red+green rule;
  purely black-and-white masks use white-as-foreground. Missing image/mask pairs are dropped
  automatically, so a partially-synced index will not crash training.

- **Augmentation** (train split only, applied per sample in order flip → crop → rotate → letterbox →
  enhance → blur). Geometric ops (flip, half-size crop, rotate, letterbox) transform image **and**
  mask jointly. The **half-size crop** (p=0.3) takes a random 0.5× window and resizes it back (2×
  zoom). **Letterbox** (p=0.3) shrinks content vertically and pads top/bottom — **white** bars on the
  image, **black** on the mask. Colour-enhance and blur touch the image only.

- **Optimizer / LR / scheduler.** AdamW with the BiRefNet `General`-task learning rate, scaled by
  √(batch/4). The scheduler's only milestone sits far beyond the run length, so the LR is constant;
  the final 20 epochs instead taper the pixel-loss weights (BCE → 0, IoU × 0.5, MAE × 0.9) for a
  fine-tuning polish. Input preprocessing: ImageNet normalization (mean `[0.485,0.456,0.406]`, std
  `[0.229,0.224,0.225]`).

- **Mixed precision.** fp16 autocast on the model forward (tensor-core speedup + ~35% lower activation
  memory: ~34 GB → ~22 GB). Losses run in fp32 because `binary_cross_entropy`/`BCELoss` are unsafe
  under fp16 autocast; a `GradScaler` handles fp16 gradient underflow. Master weights, gradients and
  optimizer state stay fp32 (torch autocast keeps no fp16 weight copy).

- **Validation & early stopping.** Every epoch reports loss, IoU, F1, MAE and **Contour_mIoU**. The
  last is a boundary IoU (a ±4 px ring around each contour) and is the early-stopping signal: it
  catches the failure mode where interior IoU keeps rising while edges degrade. `best.pth` is saved
  whenever Contour_mIoU improves.

## Loss variants

The pixel loss (`loss.py:PixLoss`) is a weighted sum of criteria; each is only constructed when its
weight in `config.lambdas_pix_last` is nonzero, so toggling a loss is just editing its weight. Weights
below are for the active `General` task.

| Key | Criterion | What it optimizes | Weight (General) |
|---|---|---|---|
| `bce` | `BCELoss` | per-pixel foreground/background classification | **30** |
| `iou` | `IoULoss` (soft IoU) | region overlap (1 − I/U) | **0.5** |
| `ssim` | `SSIMLoss` | structural / local-edge similarity | **10** |
| `mae` | `L1Loss` | pixel absolute error | **100** |
| `mse` | `MSELoss` | pixel squared error | 0 (off) |
| `iou_patch` | `PatchIoULoss` | IoU on local patches (fine structure) | 0 |
| `cnt` | `ContourLoss` | boundary/contour emphasis | 0 |
| `structure` | `StructureLoss` | boundary-weighted BCE+IoU (hard edge pixels) | 0 |
| `reg`, `triplet` | regularizer / triplet | auxiliary | 0 |

So the effective objective here is **30·BCE + 0.5·IoU + 100·MAE + 10·SSIM**. Two extra loss terms sit
outside `PixLoss`:

- **Bilateral-reference gradient loss** (`out_ref`, `criterion_gdt = BCELoss`): at three decoder
  stages the model predicts a gradient-attention map supervised against the input's Laplacian
  gradient label — pushes sharper boundaries.
- **Auxiliary classification** (`ClsLoss`, cross-entropy): off unless `config.auxiliary_classification`.

Two BCE **variants** exist for special cases (both off in this run):

- **Window-blob weighted BCE** — upweights (×k, default 5) the BCE on foreground (red/green) blobs
  that sit *isolated inside window (blue) regions*, so small objects seen through windows aren't
  dropped. Enable with `--window_blob_loss`.
- **Fine-tuning taper** — during the last `finetune_last_epochs` (20 for `General`) the weights shift:
  `bce → 0`, `iou × 0.5`, `ssim × 1`, `mae × 0.9`, trading pixel-classification for boundary/regression
  polish.

## Model variants

### Two-stage pipeline (coarse → refiner) — the "two-step decoder" setup

1. **Stage 1 — coarse full-frame model** (this README's model): BiRefNet Swin-L predicts the whole
   mask at 1024². Fast, globally consistent, but boundaries are only as sharp as a 1024² decode.
2. **Stage 2 — mask refiner** (`train_refiner.py`, `dataset_refiner.py`): a lightweight
   `segmentation-models-pytorch` **UNet** that runs on **512² crops** around the foreground and
   sharpens the coarse mask's edges. It is trained with **mask degradation** (the GT's edge sharpness
   is degraded to imitate a coarse-mask input), so it learns to *fix* an approximate mask rather than
   segment from scratch.

Refiner sub-variants (see `configs/refiner-*.yaml`):

| Variant | Input | Loss | Notes |
|---|---|---|---|
| Residual refiner (default) | 4ch: crop RGB **+ degraded mask** | **delta loss** — predict signed Δ; `refined = clamp(degraded + tanh(logits))`; L1 on `gt − degraded` | supervises only the missing/spurious pixels |
| Full-mask refiner | 4ch: crop RGB + degraded mask | L1/BCE on the full GT mask | predicts the whole mask, not a residual |
| Crop-only (control) | 3ch: crop RGB only | full-mask | measures how much the coarse-mask input + delta loss actually buy |
| Encoder options | — | — | `resnet34` / `resnet50` / `efficientnet-b4` |

### Decoder-only / shared-encoder variants (`ft-decoder`)

`--freeze_bb` freezes the Swin-L **encoder** (195 M params) and trains **only the decoder** (25 M).
This lets one shared, fixed encoder host several **domain-specialized decoders** — we trained separate
decoders for `interior`, `360 cubemap`, and `details` off the same frozen encoder
(`configs/ft-decoder-*.yaml`). Cheaper to train and keeps a common feature space across heads.

### Backbone variants

`config.bb` selects the encoder: `swin_v1_l` (this run) / `swin_v1_b` (smaller/faster) / PVT /
DINOv3 (auto-frozen) / vgg / resnet. The backbone drives channel counts and freezing rules.

## Multi-head decoder

**Strategy:** deploy **one shared, frozen Swin-L encoder** with **several specialized decoders**, one
per data regime, instead of one monolithic model that must handle everything at once. The encoder is
trained once (on a broad dataset) and then **frozen** (`freeze_bb=true`); each decoder is fine-tuned
decoder-only on its target regime. At inference the encoder runs once and the appropriate decoder head
is selected for the input.

**Why:**
- **Specialization without interference** — a decoder tuned for close-ups doesn't have to compromise
  with full-frame views (and vice-versa), so each head can push quality on its own distribution.
- **Cheap to add heads** — only the 25 M-param decoder trains (the 195 M encoder is fixed), so a new
  head is fast and small, and all heads share one set of encoder weights/activations at deploy.
- **Stable feature space** — every head reads the same frozen features, so heads stay mutually
  compatible and comparable.

**Heads so far:**

| Head | Data regime | Init | Config |
|---|---|---|---|
| General interior | full-frame views (enter driver/codriver, rear seats, …) | — (existing) | — |
| **Close-up (detail)** | interior + exterior close-ups: `details/*` + `old_dataset_details` | ft-rim `epoch_685` | `configs/coarse-swin-l-decoder-closeup.yaml` |

**Anti-forgetting replay.** A decoder trained *only* on its niche can lose competence on everything
else. To keep the close-up head usable on general interiors too, its training set mixes in a small
**replay buffer** — ~**10 % general-interior gold images** (700 of 6,892) sampled from the full-frame
`gold_dataset/kw*` sets — alongside the close-up data. The head specializes on close-ups while
retaining baseline capacity on normal views; the 90/10 val split spans both, so retention is
monitored during training.

**Add a new head:**
1. Build a CSV for the head's data regime (+ a small general-data replay slice to prevent forgetting).
2. Point a config at it with `freeze_bb: true` and `resume:` the shared-encoder checkpoint.
3. Train — only the decoder updates. `Val/Contour_mIoU` on the mixed val set tracks both specialization
   and retention.

## Reproduce

```bash
BIREFNET_BB=swin_v1_l uv run python train.py --config configs/coarse-swin-l-add-old-details.yaml
```

The resolved (defaults < YAML < CLI) config is dumped to
`runs/coarse-swin-l-add-old-details/config.resolved.yaml` at train start.
