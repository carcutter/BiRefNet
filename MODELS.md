# MODELS.md — trained model registry (v8 weights)

Per-model record of **data, architecture, losses, and augmentations** for every model version we
train. Exports live in the sibling weights tree:
`../interior-segmentation/models/v8/weights/<model-name>/` (no-sigmoid ONNX, fp32 + fp16).

> **Upkeep rule:** whenever a new model is trained (or an existing one is retrained/continued),
> add or update its entry below in the same turn — before/at export time. Record anything that
> differs from **Shared defaults**; if it matches, say "defaults". Keep the **Augmentation
> timeline** accurate so it's always clear which augs a run saw.

---

## Shared defaults — coarse BiRefNet (`task='General'`)

Unless an entry says otherwise, every **coarse** model uses these. (Refiners are a separate
architecture — see that section.)

- **Architecture:** BiRefNet, `bb=swin_v1_l` (Swin-L) encoder + `ASPPDeformable` decoder.
  - Multi-scale input `mul_scl_ipt='cat'` (backbone run at full + half res, concatenated).
  - Bilateral reference / gradient supervision `out_ref=True`; multi-scale supervision on.
  - Context concat `cxt_num=3`; squeeze `BasicDecBlk_x1`; `dec_blk=BasicDecBlk`.
  - Input **1024×1024**. Params: **encoder 195.2M / decoder 25.0M**.
- **Losses** (`lambdas_pix_last`, General): `bce×30`, `iou×0.5`, `mae×100`, `ssim×10` (others 0),
  plus `out_ref` gradient BCE (`criterion_gdt`). `finetune_last_epochs=-20` tapers pixel lambdas
  over the final 20 epochs (bce→0, iou×0.5, mae×0.9) — shows as `bce: 0` late in a run.
  - *window-blob* variants add an extra window-blob loss (weight 5.0) on a blob loss-weight map.
- **Optimizer / LR:** AdamW; `lr = 1e-5 × sqrt(batch/4)`, **constant** (milestone `1e5` never fires).
- **Mask binarization:** `mask_color_to_binary='auto'` (colored → red/green FG rule; B&W → white=FG).
- **Validation metric:** `Val/Contour_mIoU` (boundary IoU, ±4 px) selects `best.pth`; also IoU/F1/MAE.
- **Precision:** fp16 mixed precision on runs **2026-09-11+**; earlier runs fp32.
- **Logging:** MLflow + TensorBoard → `runs/<run>/tb/`.

## Augmentation timeline (global `config.preproc_methods`, training only)

- **Before 2026-09-07:** `flip`, `enhance`, `rotate` (±15°), `blur`, `crop` (pre-existing).
  **No letterbox, no half-size zoom crop.**
- **2026-09-07 (`f7a445e`) onward:** added **half-size crop** (0.5× zoom, p=0.3) and **letterbox**
  (WHITE bars on image / BLACK bars on mask, top+bottom, 5–30% of H, p=0.3). Current set:
  `[flip, enhance, rotate, blur, crop(half), letterbox]`.
- **Rule of thumb:** a run trained **on/after 2026-09-07 → has letterbox + half-crop**; before → not.
- **2026-09-25 onward (opt-in knobs, per-run via CLI/YAML):**
  - `letterbox_fg=true` → **foreground-bar** letterbox variant: bars on **top/bottom OR left/right**
    (random), **black OR white** (random), and the mask is **foreground on the bars** (teaches
    matte bars = FG). `letterbox_prob` sets its probability. Default (flag off) = the background-bar
    letterbox above.
  - `crop_frac_area=true` → the runtime wheel/random crop draws its size as a fraction of image
    **AREA** (e.g. 40–70% area) instead of per-side.
  - `preproc_methods` is now overridable per-run (drop/keep specific augs).

---

## Coarse full-model lineage (encoder + decoder trained)

Architecture & losses = **Shared defaults** unless noted. "Augs" per the timeline above.

| Export name | Trained | Base / resume | Data (CSV) | Split | bs · prec | Epochs | Augs | Result (Contour_mIoU) |
|---|---|---|---|---|---|---|---|---|
| `coarse-swin-l` | 2026-06 | BiRefNet-DIS-590 → self | `index.csv` (interior) | 80/20 | 2 · fp32 | →640 | ❌ | not logged |
| `coarse-swin-b` | 2026-06 | — (Swin-B variant, `bb=swin_v1_b`) | interior | — | — | — | ❌ | details not captured |
| `coarse-swin-l-4x3` | 2026-06 | BiRefNet-DIS-590 | `index.csv` | 80/20 | 2 · fp32 | →640 | ❌ | not logged · **1024×768 (4:3)** |
| `coarse-swin-l-4x3-1MP` | 2026-06 | self (4x3) 625 | `index.csv` | 80/20 | 2 · fp32 | →640 | ❌ | not logged · **1152×864 (~1MP 4:3)** |
| `coarse-swin-l-window-blob-from-dis` | 2026-07 | BiRefNet-DIS-590 | `train_interior_rims_refinement.csv` | 80/20 | 2 · fp32 | →690 | ❌ | not logged · **+window-blob loss** |
| `coarse-swin-l-window-blob` | 2026-07 | self 640 | `train_interior_rims_refinement.csv` | 80/20 | 2 · fp32 | →690 | ❌ | not logged · **+window-blob loss; `epoch_685` = shared encoder base** |
| `coarse-swin-l-window-blob-ft-rim` | 2026-08-18 | window-blob | rim + detail data | — | 2 · fp32 | — | ❌ | rim/detail fine-tune. **Its `.pth` == `runs/coarse-swin-l-window-blob/epoch_685.pth`** (per the ft-rim ONNX sidecar `checkpoint` field); run dir cleaned |
| `coarse-swin-l-window-blob-ft-refinement` | 2026-07-30 | window-blob | rim **+ details** | — | 2 · fp32 | — | ❌ | broader refine of ft-rim (run dir cleaned) |
| `coarse-swin-l-det-roof-cons` | 2026-07-29 | window-blob-from-dis 659 | `train_interior_rims_refinement.csv` | 80/20 | 2 · fp32 | →690 | ❌ | **window-blob loss REMOVED (noblob)**; run dir = `coarse-swin-l-from-dis-noblob`; base for the ft-decoder family |
| `coarse-swin-l-add-old-details` | 2026-09-08 | from DIS 590 (resume 641) | `data_link/train/index.csv` | 90/10 | 2 · **fp16** | →690 | ✅ | **0.7917 @ 656** (early-stopped) |
| `extended_dataset` | 2026-09-10 | window-blob `epoch_685` | `data_link/train/index.csv` | 90/10 | 2 · **fp16** | →785 | ✅ | run dir = `coarse-swin-l-extended-from-ftrim` |

## Decoder-only specializations (encoder FROZEN, `freeze_bb=true`)

Shared frozen Swin-L encoder; only the 25.0M decoder trains. Architecture & losses = defaults.

| Export name | Trained | Encoder base | Data (CSV) | Split | bs · prec | Epochs | Augs | Result (Contour_mIoU) |
|---|---|---|---|---|---|---|---|---|
| `ft-decoder/interior` | 2026-08-25 | from-dis-noblob 690 | `ft_decoder_interior.csv` (interior_train + kw2622 + kw2624) | 80/20 | 2 · fp32 | →730 | ❌ | **0.8680 @ 693** |
| `ft-decoder/cubemap` | 2026-08-25 | from-dis-noblob 690 | `ft_decoder_cubemap.csv` (notbg cube imgs + interior_360_cube masks) | 80/20 | 2 · fp32 | →730 | ❌ | **0.8406 @ 729** |
| `ft-decoder/details` | 2026-08-26 | from-dis-noblob 690 | `ft_decoder_details.csv` (detail_crops rims/wheels) | 80/20 | 2 · fp32 | →730 | ❌ | **0.8366 @ 730** |
| `coarse-swin-l-decoder-closeup` | 2026-09-10 | window-blob 685 | `details_closeup.csv` | 90/10 | 2 · fp16 | →735 | ✅ | **0.7711 @ 732** (old data) |
| `coarse-swin-l-decoder-closeup-newdata` | 2026-09-14→16 | window-blob 685 | `details_closeup_newdata.csv` — ALL details (incl. full kw2628) + 20% gold interior, **8,944 pairs** | 90/10 (tile) | 8 · fp16 | 686→735, then **+30 → 762** | ✅ | **0.8156 @ 757** (`best.pth`) |
| `coarse-swin-l-decoder-cubemap` | 2026-09-22→23 | window-blob 685 | `interior_360_cube` cube faces (masks projected from `interior_360/masks`), **1,068 faces / 178 panos** | **80/20 pano-level** (`train_csv`/`val_csv`, zero pano overlap) | 8 · fp16 | 686→735 | ✅ | **0.7852 @ 726** (pano-disjoint) |
| `coarse-swin-l-decoder-ftrim-gold-ext` | 2026-09-25→27 | **ft-rim** (window-blob 685) | `ftrim_gold_plus_exterior.csv` — ALL interior gold (8 subdirs, 14,890) + **30%** of `details/exterior_originals` train (337, seed 42), **15,227 pairs** | 90/10 | 8 · fp16 | 686→735 (50 ep) | ✅✱ | **0.8314 @ 734** |
| `coarse-swin-l-decoder-strip` | 2026-10-14 | window-blob 685 | `interior_360_notbg_strip` — equirect sliced into **4 square 1:1 tiles** (make_360_tiles strip, aspect 1.0, overlap 0.5, cyclic), **712 tiles / 178 panos** | **80/20 pano-level** (`train_csv`/`val_csv`, **same panos as cubemap**) | 8 · fp16 | 686→**696** (early-stop) | ✅ | **0.8055 @ 686** |

- `runs/coarse-swin-l-decoder-only` (not separately exported): frozen-encoder decoder on the combined
  `train_interior_rims_refinement.csv` (80/20, from-dis-noblob 690 base) — the precursor experiment
  before splitting into the interior/cubemap/details buckets.

### Notes on the newest two (flagship detail)
- **decoder-closeup-newdata:** masks mixed conventions (white detail masks + red/green gold) → `'auto'`
  binarization. The +30 continuation retrained on the *updated* details (kw2628 moved in, full-size);
  0.8156 is on the harder post-update val slice (not comparable to the stage-1 0.8379 on old data).
- **decoder-cubemap:** cube masks made with `interior_segmentation.cubemap.equirect_to_faces`
  (gnomonic, face_size 1024, nearest for masks); regenerated images were pixel-identical to the
  existing cube images (verified |Δ|=0). Pano-level split added via new `--train_csv/--val_csv`.
- **decoder-strip:** same experiment as decoder-cubemap but tiling the equirect into **4 overlapping
  square 1:1 strips** instead of 6 cube faces (`make_360_tiles.py --method strip --aspect 1.0
  --overlap 0.5`; for a 2:1 image, `n=round(2/(1−overlap))=4`, tiles at x=0,H/2,H,3H/2, cyclic wrap).
  Square 2976² tiles, masks keep source RGB polarity (auto-binarized). Uses the **same 142/36 pano
  split** as the cubemap run → directly comparable (strips vs cube faces, same base/recipe/augs).
  - **Done:** best **Val/Contour_mIoU = 0.8055 @ epoch 686** — the *first* validation; 50 epochs of
    strip-tile finetuning never beat the ft-rim base's zero-shot score on strips, so early-stopping
    fired at epoch 696. Exported no-sigmoid fp32+fp16 (parity OK). NB: absolute metric is **not**
    directly comparable to cubemap's 0.7852 (different tile geometry/val tiles — cube faces vs square
    strips of the same panos); the finding is that decoder finetuning didn't help on strips.
  - **Head-to-head (same 36 test panos, each via its OWN pipeline, reassembled to full equirect,
    scored @2048×1024):** cubemap IoU **0.9832** / F1 0.9909 / **Bound-IoU 0.7411**; strip 0.9765 /
    0.9871 / 0.6879; whole-image ft-rim (ref) 0.9785 / 0.9880 / 0.6621. **Cubemap wins on every
    metric, decisively on boundaries;** strip is even slightly below the whole-image baseline on
    region IoU. → Use the **cubemap** approach for 360. (script: scratchpad `compare_360.py`.)
- **decoder-ftrim-gold-ext** (✱ special augs): decoder-only continuation of **ft-rim**.
  - Augs = `flip, enhance, rotate, blur` + **foreground letterbox** (`letterbox_fg=true`,
    `letterbox_prob=0.5`: random top/bottom OR left/right bars, black OR white, **GT foreground on
    the bars**). The default half-size `crop` is dropped.
  - Exterior rows (`details/exterior_originals`, red=FG masks) carry `wheel_boxes` → runtime crop at
    load: **50% centred on a random wheel, 50% random**, size **40–70% of image AREA**
    (`crop_frac_area=true`). Gold rows have no wheel_boxes → never cropped.
  - Enabled by new per-run flags in `train.py`: `--preproc_methods --letterbox_fg --letterbox_prob
    --crop_frac_area` (+ `--train_csv/--val_csv`).
  - **Done:** best **Val/Contour_mIoU = 0.8314 @ epoch 734** (final ep735: IoU 0.9954, F1 0.9977,
    MAE 0.0037; n=1523). Exported no-sigmoid fp32+fp16 to the weights tree (parity OK).

## Refiner stage (separate architecture — NOT BiRefNet coarse)

Second-stage mask refiners over **cropped windows** around predicted blobs. SMP-style
encoder–UNet decoder; ImageNet-initialized encoder. Input **512×512**, fp32/bf16, data
`data/processed/blob_crops.csv` (80/20, seed 42). Mask-refiner variants take a **4-ch input**
(crop RGB + degraded mask, `use_mask_input=true`); crop-only variants are 3-ch RGB.

| Export name | Trained | Encoder | Input | bs | Epochs | Notes |
|---|---|---|---|---|---|---|
| `refiner-efficientnet-b4` | 2026-06-13 | efficientnet-b4 | 512 · 4ch | 16 | →200 | RGB + degraded mask |
| `refiner-efficientnet-b4-crop-only` | 2026-06 | efficientnet-b4 | 512 · 3ch | 16 | →200 | RGB only (no mask input) |
| `refiner-efficientnet-b4-delta-loss` | 2026-06-08 | efficientnet-b4 | 512 · 4ch | 16 | →200 | delta-loss variant (supervise mask *change*) |
| `refiner-resnet50` (run) | 2026-06-05 | resnet50 | 512 · 4ch | 32 | →200 | probe experiment |
| `refiner-birefnet-resume` | 2026-06 | BiRefNet-init refiner | 512 | — | — | refine-crops from a BiRefNet ckpt (`ckpts/interior-ft/epoch_645`) |
| `refiner-efficientnet-b4-crop-only` (weights) | 2026-06 | efficientnet-b4 | 512 · 3ch | 16 | →200 | exported crop-only refiner |

> Refiner max-batch (L40S, resnet34-UNet @512 fp32): ceiling 96 / recommended 86 (see memory).

## External / third-party

- **`PDFNet`** (2026-06-25): third-party salient-object/segmentation network kept in the weights
  tree for comparison — **not trained in this repo**; architecture/data/losses per its own upstream.

---

*Metrics marked "not logged" are older runs whose training logs were pruned; values recovered from
`config.resolved.yaml`, checkpoint dates, and git history. Fill in if ever retrained under this log.*
