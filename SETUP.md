# Setup

End-to-end setup for a fresh instance: environment, backbone weights, dataset (via DVC), and a first training run. Assumes a CUDA-capable GPU and an NVIDIA driver that supports CUDA 12.8 (the wheel index pinned in `pyproject.toml`). Bump the `pytorch-cuXXX` index in `pyproject.toml` if your driver is older.

## 1. Clone

```bash
git clone <repo-url> BiRefNet
cd BiRefNet
```

## 2. Python environment (uv)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it's not already on the box:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the venv and install dependencies from `pyproject.toml` (resolves against `uv.lock`):

```bash
uv sync
source .venv/bin/activate
```

Sanity-check the install:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
uv run python config.py --print_task
```

## 3. Backbone pretrained weights

`config.py` expects backbone checkpoints under `${sys_home_dir}/weights/cv/` (default `sys_home_dir=/workspace`). The default backbone is `swin_v1_l` ([config.py:91-102](config.py#L91-L102)). Download only the file(s) for the backbone you'll use:

| Backbone | File | Subdir under `weights/cv/` |
|---|---|---|
| `swin_v1_l` (default) | `swin_large_patch4_window12_384_22kto1k.pth` | `./` |
| `swin_v1_b` | `swin_base_patch4_window12_384_22kto1k.pth` | `./` |
| `pvt_v2_b5` | `pvt_v2_b5.pth` | `./` |
| `dino_v3_l` | `vit_large_patch16_dinov3.lvd1689m.pth` | `DINOv3-timm/` |
| (others) | see [config.py:181-188](config.py#L181-L188) | as above |

Example (Swin-L default):

```bash
sudo mkdir -p /workspace/weights/cv
sudo chown -R "$USER" /workspace
# Pull from the project's weights store (HF Hub / S3 / wherever your team keeps them):
# e.g. huggingface-cli download <repo> swin_large_patch4_window12_384_22kto1k.pth \
#        --local-dir /workspace/weights/cv
```

If `/workspace` is read-only on your instance, change `sys_home_dir` in [config.py:16](config.py#L16) to a writable path.

## 4. Dataset (DVC)

The training set lives in `data/interior_segmentation/` and is tracked by [data/interior_segmentation.dvc](data/interior_segmentation.dvc) (~3.6 GB, 21,914 files). `dvc` and `dvc-s3` are already in `pyproject.toml`.

Configure AWS credentials for the S3 remote (one of):

```bash
aws configure                        # writes ~/.aws/credentials
# or export them for the current shell:
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...
```

If the repo doesn't yet have a `.dvc/config` with a remote set, configure it once:

```bash
uv run dvc remote add -d storage s3://<bucket>/<prefix>
uv run dvc remote modify storage region <region>     # if needed
```

Pull the data:

```bash
uv run dvc pull
```

After this you should have:

```
data/interior_segmentation/
├── train.csv
├── test.csv
├── test_smoke.csv
├── index.csv
└── kw*_car_interior_segmentation/...
```

The CSVs reference image/mask paths relative to their own directory ([dataset.py:58-72](dataset.py#L58-L72)). `config.py` defaults to reading `train.csv` / `test_smoke.csv` from `csv_data_root` ([config.py:23-28](config.py#L23-L28)) — adjust those fields if your CSV locations differ.

## 5. Train

Single-GPU smoke run:

```bash
./train.sh my-run 0
```

Multi-GPU (auto-detected from the comma-separated GPU list):

```bash
./train.sh my-run 0,1,2,3
```

Resume from a pretrained `.pth` by editing `resume_weights_path` in [train.sh:21](train.sh#L21). Epoch counts (`epochs`, `val_last`, `step`) come from the `case` block in [train.sh:5-12](train.sh#L5-L12), keyed off `config.task`.

Train + inference + evaluation in one shot:

```bash
./train_test.sh my-run 0,1,2,3 0    # train_gpus, eval_gpu
```

Outputs (all gitignored):
- `ckpts/my-run/epoch_N.pth` — checkpoints (only the last `save_last` epochs, every `save_step`)
- `e_preds/` — predicted masks
- `e_results/<testset>_eval.txt` — eval tables
- `tb_logs/` — TensorBoard scalars and image samples

## 6. Monitor

```bash
uv run tensorboard --logdir tb_logs --port 6006
```

## Troubleshooting

- **`torch.cuda.is_available() == False`**: driver too old for the cu128 wheels. Either upgrade the driver or change the index in [pyproject.toml:31-34](pyproject.toml#L31-L34) to `cu124`/`cu126` and re-run `uv sync`.
- **`FileNotFoundError` on a backbone `.pth`**: the backbone selected by `config.bb` isn't present under `weights/cv/`. Re-check the table in step 3 — DINOv3 weights go under `DINOv3-timm/`, the others go directly under `weights/cv/`.
- **`dvc pull` fails on auth**: confirm `aws sts get-caller-identity` works, then retry. `dvc-s3` reads the standard AWS credential chain.
- **OOM at default size**: lower `config.size` (must stay divisible by 32), drop `config.mixed_precision` to `'fp16'`, or disable `config.compile`.
