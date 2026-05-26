# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BiRefNet — official implementation of "Bilateral Reference for High-Resolution Dichotomous Image Segmentation" (CAAI AIR 2024). Despite the name, it is used as a general binary image segmentation framework: DIS, camouflaged-object detection (COD), high-res salient-object detection (HRSOD), background removal, and matting.

## Environment

Managed with [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` (mirrors `requirements.txt`, kept around for the shell scripts).

```bash
uv sync                                # creates .venv and installs from pyproject.toml
source .venv/bin/activate
```

Prefix one-off commands with `uv run` to execute inside the project environment without activating it, e.g. `uv run python config.py --print_task`.

## Common commands

Most workflows are driven by `config.py` plus a handful of shell scripts. There is no test suite — "test" here means model evaluation on a benchmark.

```bash
# Full pipeline: train + inference + evaluation
./train_test.sh RUN_NAME TRAIN_GPUS TEST_GPU
# e.g. ./train_test.sh tmp-proj 0,1,2,3,4,5,6,7 0

# Train only (multi-GPU auto-detected from the comma-separated list)
./train.sh RUN_NAME 0,1,2,3

# Inference + evaluation on the latest ckpts/* folder
./test.sh GPU_ID [pred_root] [resolutions]
# resolutions arg accepts "config.size", "None" (original size), or "WxH" tokens, e.g. "1024x1024 None"

# Inference only (selects the latest ckpts/* by default)
python inference.py --ckpt_folder ckpts/RUN_NAME --pred_root e_preds --resolution config.size

# Evaluate predictions against the configured testsets
python eval_existingOnes.py --pred_root e_preds --data_lst DIS-VD --metrics all

# After multiple eval runs, pick the best checkpoint per metric (Sm / wFm / HCE)
python gen_best_ep.py

# SLURM submission wrapper (cluster-specific module loads inside)
./sub.sh RUN_NAME 0,1,2,3 0

# Clean up checkpoints, predictions, eval logs, pycache
./rm_cache.sh
```

`train.sh` reads task-specific epoch counts (`epochs / val_last / step`) from a `case` block keyed off `python3 config.py --print_task`. `config.py` reads `train.sh` back to populate `self.save_last` and `self.save_step` — these two files are coupled, edit them together.

## Configuration is centralized

**`config.py` is the single source of truth for all hyperparameters, paths, model variants, and task selection.** There are no CLI flags for most of this; you change behavior by editing the indexed-list defaults at the top of `Config.__init__`, e.g.:

```python
self.task = ['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'][0]
self.bb   = [..., 'swin_v1_l', 'swin_v1_b', ..., 'dino_v3_l', ...][3]
self.mixed_precision = ['no', 'fp16', 'bf16', 'fp8'][2]
```

Switching tasks/backbones means changing the trailing index. Key knobs:
- `task` drives `testsets`, `training_set`, default `size`, `lr`, loss weights (`lambdas_pix_last`), and per-task field ordering in `eval_existingOnes.py`.
- `bb` drives `lateral_channels_in_collection`, freezing rules (`freeze_bb` is auto-true for DINOv3), and the pretrained-weight filename map in `self.weights`.
- `compile`, `mixed_precision`, `SDPA_enabled`, `dynamic_size`, `load_all` are the main memory/throughput dials. `compile=True` requires PyTorch ≥ 2.5; `dynamic_size` can break `torch.compile`.
- `size` defaults to `(1024,1024)` except for `General-2K` (`(2560,1440)`). All input H/W must be divisible by 32 (the dataset rounds down at inference time).

`python config.py --print_task` / `--print_testsets` is how the shell scripts read the active settings.

## Paths and data layout

`config.py` assumes a fixed directory layout under `sys_home_dir` (default `/workspace`, second entry in the list):

```
${sys_home_dir}/codes/dis/BiRefNet           # this repo
${sys_home_dir}/datasets/dis/<TASK>/<SET>/im # images
${sys_home_dir}/datasets/dis/<TASK>/<SET>/gt # masks (matched by basename)
${sys_home_dir}/weights/cv/...               # backbone pretrained weights
```

The dataset loader pairs each `*/im/x.ext` with `*/gt/x.{png,jpg,...}` and raises if counts disagree. DINOv3 weights live in a `DINOv3-timm/` subfolder; Swin/PVT live directly under `weights/cv/`.

Output conventions (all gitignored): `ckpts/RUN_NAME/epoch_N.pth`, `e_preds/...` (raw predicted masks), `e_logs/eval_<testset>.out`, `e_results/<testset>_eval.txt`.

## Architecture

Entry point: `models/birefnet.py`. The class is `nn.Module` + `huggingface_hub.PyTorchModelHubMixin`, so it can be loaded from HF Hub with `from_pretrained`.

Encoder–decoder with optional bells:
- **Encoder (`forward_enc`)**: `build_backbone(config.bb)` returns one of vgg/resnet (4-stage `conv1..conv4` sequential), Swin/PVT (4-tuple), or DINOv3 ViT (4-tuple via `vit_model_to_out_indices`). For ViT-style backbones, all four stages share the same channel count — the `Decoder` detects this via `bbs_without_pyramid = ['vit', 'dino']` and inserts a `pyramid_neck_x{1..4}` to remap them onto a Swin-L-style channel pyramid before the decoder runs.
- **Multi-scale input** (`mul_scl_ipt='cat'|'add'`): the backbone is run twice — once on `x`, once on a half-resolution copy — and the two feature pyramids are fused. With `'cat'`, lateral channels double (handled in `config.py`).
- **Context concat** (`cxt_num`, default 3): low-level features are downsampled and concatenated onto `x4` before the squeeze block.
- **Decoder**: four `BasicDecBlk`/`ResBlk` stages with lateral skips, each optionally fed an extra "image-patches" branch (`dec_ipt` + `ipt_blk{1..5}`) where the input image is split into patches matching the current feature-map grid (`image2patches`).
- **Multi-scale supervision** (`ms_supervision`): the three coarse decoder stages output side maps `m4/m3/m2` for training only.
- **Bilateral reference / gradient output** (`out_ref`): at each of three decoder stages, predict a gradient-attention map and a gradient label (from the Laplacian of the input); these become extra losses (`criterion_gdt`, BCE on sigmoids) on top of the pixel loss.
- **Forward return shape changes by mode**: `training=True` returns `[scaled_preds, class_preds_lst]` where `scaled_preds` may itself be `([outs_gdt_pred, outs_gdt_label], outs)` when `out_ref`; `training=False` returns just the list of side-map tensors and `inference.py` takes `[-1]` as the final prediction.

Losses (`loss.py`): `PixLoss` builds a dict of criteria from `config.lambdas_pix_last` (keys: `bce iou iou_patch ssim mae mse reg cnt structure`). Loss criteria are only constructed when their lambda is nonzero, so toggling losses is just editing the weights. `train.py` separately decays/zeros several lambdas during the last `config.finetune_last_epochs` for fine-tuning.

## Training internals worth knowing

- Two execution paths in `train.py`: `--use_accelerate` (default in `train.sh`, supports FP16/BF16/FP8 via `accelerate`) and the manual DDP/`init_process_group` path. `accelerator.prepare` is called on the loader/model/optimizer; do not also `.to(device)` in that path.
- `torch.compile` is applied after model construction when `config.compile`. Compile + `mixed_precision='fp8'` does not accelerate. Compile + `dynamic_size` may fail.
- `--resume PATH.pth` parses the starting epoch from the filename (`...epoch_N.pth` → starts at `N+1`). Backbone pretrained weights are skipped automatically when resuming.
- Checkpoints are saved only in the last `save_last` epochs every `save_step` epochs (both read from `train.sh`).
- `check_state_dict` strips `module.` and `_orig_mod.` prefixes — apply it to any externally-loaded checkpoint that may have been saved under DDP or `torch.compile`.

## Inference and evaluation flow

`inference.py` walks `ckpts/RUN_NAME/*.pth` (or a single `--ckpt`), produces masks under `e_preds/<method>/<testset>/`, and uses `<method>` of the form `RUN_NAME--epoch_N-reso_WxH`. `eval_existingOnes.py` then walks `e_preds/*` in epoch order, produces one PrettyTable per testset under `e_results/<testset>_eval.txt`, and `gen_best_ep.py` parses those tables to pick the best checkpoint per metric. The per-task column ordering in the eval table is hardcoded in both `eval_existingOnes.py` (field names) and `gen_best_ep.py` (`targe_idx`) — keep them in sync if you add metrics.

## Fine-tuning on custom data

1. Place data at `${data_root_dir}/<TASK>/<DATASET>/im` and `.../gt`.
2. Either reuse one of the existing task names or rename a task (e.g. replace every `'General'` in the project with your task name — there are references in `config.py`, `train.sh`, `eval_existingOnes.py`, `gen_best_ep.py`).
3. Adjust `testsets`, `training_set`, `lambdas_pix_last` in `config.py`.
4. Resume from a pretrained `.pth` via the `resume_weights_path` variable inside `train.sh` — `--epochs` is the **absolute** target epoch, not "epochs to add" (because `epoch_st` is parsed from the resume filename).
