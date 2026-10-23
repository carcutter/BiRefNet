#!/bin/bash
# Poll GPU memory every 30 min; when total used < 10 GB, launch the data/train full-FT run.
# A 20s re-check debounces a transient dip before committing to the (~27h) run.
cd /home/renato/projects/interior-segmentation/BiRefNet || exit 1
THRESH_MIB=10240   # 10 GB
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
while true; do
  u=$(used)
  echo "[$(date '+%F %T')] GPU used=${u}MiB (threshold ${THRESH_MIB})"
  if [ "${u:-999999}" -lt "$THRESH_MIB" ]; then
    sleep 20; u2=$(used)                                   # debounce
    if [ "${u2:-999999}" -lt "$THRESH_MIB" ]; then
      echo "[$(date '+%F %T')] GPU free (${u2}MiB) -> launching training"
      mkdir -p runs/coarse-swin-l-data-train
      BIREFNET_BB=swin_v1_l uv run python train.py --config configs/coarse-swin-l-data-train.yaml \
        > runs/coarse-swin-l-data-train/train.out 2>&1
      echo "[$(date '+%F %T')] training exited ($?)"
      break
    fi
    echo "[$(date '+%F %T')] dip was transient (${u2}MiB), keep waiting"
  fi
  sleep 1800   # 30 min
done
