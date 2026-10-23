#!/bin/bash
# Wait for coarse-swin-l-decoder-closeup to finish, then export best.pth to no-sigmoid ONNX
# (fp32 + fp16, dynamic batch, 1024^2) and copy into the weights tree. Export is CPU-only.
cd /home/renato/projects/interior-segmentation/BiRefNet || exit 1
C=runs/coarse-swin-l-decoder-closeup/train.out
R=runs/coarse-swin-l-decoder-closeup
BASE=coarse-swin-l-decoder-closeup
DST=/home/renato/projects/interior-segmentation/interior-segmentation/models/v8/weights/$BASE

echo "[$(date '+%F %T')] waiting for training to finish..."
until grep -qE "Training done|Early stopping" "$C" 2>/dev/null; do
  if ! pgrep -f "[t]rain.py.*coarse-swin-l-decoder-closeup" >/dev/null; then
    grep -qE "Training done|Early stopping" "$C" 2>/dev/null || { echo "[$(date '+%F %T')] process gone with no terminal log line — aborting"; exit 1; }
    break
  fi
  sleep 60
done
echo "[$(date '+%F %T')] training finished:"; grep -inE "Training done|Early stopping" "$C" | tail -1

# fp32 (no sigmoid)
BIREFNET_BB=swin_v1_l uv run python tools/export_onnx.py \
  --checkpoint "$R/best.pth" --out "$R/$BASE.onnx" \
  --input_size 1024,1024 --batch dynamic --no_sigmoid --check || { echo "FP32 EXPORT FAILED"; exit 1; }
# fp16 (no sigmoid)
BIREFNET_BB=swin_v1_l uv run python tools/export_onnx.py \
  --checkpoint "$R/best.pth" --out "$R/${BASE}_fp16.onnx" \
  --input_size 1024,1024 --batch dynamic --no_sigmoid --fp16 --check || { echo "FP16 EXPORT FAILED"; exit 1; }

mkdir -p "$DST"
cp "$R/$BASE.onnx" "$R/$BASE.json" "$R/${BASE}_fp16.onnx" "$R/${BASE}_fp16.json" "$DST/"
echo "[$(date '+%F %T')] EXPORTED (no-sigmoid) -> $DST"
ls -la --time-style=+%H:%M "$DST"