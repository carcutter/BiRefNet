#!/bin/bash
# Sequential decoder-only fine-tunes (single GPU, ~34GB each ⇒ one at a time).
# Shared FROZEN Swin-L encoder (from-dis-noblob epoch 690); each run trains its own decoder on one
# data bucket. Order follows the user's request: interior, cubemap, details. A failure in one run is
# logged and does NOT abort the others.
cd /home/renato/projects/interior-segmentation/BiRefNet || exit 1
for t in interior cubemap details; do
  mkdir -p "runs/ft-decoder/$t"
  echo "[$(date '+%F %T')] === starting ft-decoder-$t ==="
  BIREFNET_BB=swin_v1_l uv run python train.py --config "configs/ft-decoder-$t.yaml" \
    > "runs/ft-decoder/$t/train.out" 2>&1
  echo "[$(date '+%F %T')] === finished ft-decoder-$t (exit $?) ==="
done
echo "[$(date '+%F %T')] === all three ft-decoder runs done ==="
