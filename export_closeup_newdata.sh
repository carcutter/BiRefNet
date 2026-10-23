#!/bin/bash
# Export decoder-closeup-newdata best.pth (epoch 732) to no-sigmoid ONNX (fp32 + fp16, dynamic batch,
# 1024^2) and copy into the weights tree. CPU-only.
cd /home/renato/projects/interior-segmentation/BiRefNet || exit 1
R=runs/coarse-swin-l-decoder-closeup-newdata
BASE=coarse-swin-l-decoder-closeup-newdata
DST=/home/renato/projects/interior-segmentation/interior-segmentation/models/v8/weights/$BASE

BIREFNET_BB=swin_v1_l uv run python tools/export_onnx.py \
  --checkpoint "$R/best.pth" --out "$R/$BASE.onnx" \
  --input_size 1024,1024 --batch dynamic --no_sigmoid --check || { echo "FP32 FAILED"; exit 1; }
BIREFNET_BB=swin_v1_l uv run python tools/export_onnx.py \
  --checkpoint "$R/best.pth" --out "$R/${BASE}_fp16.onnx" \
  --input_size 1024,1024 --batch dynamic --no_sigmoid --fp16 --check || { echo "FP16 FAILED"; exit 1; }

mkdir -p "$DST"
cp "$R/$BASE.onnx" "$R/$BASE.json" "$R/${BASE}_fp16.onnx" "$R/${BASE}_fp16.json" "$DST/"
echo "[$(date '+%F %T')] EXPORTED (no-sigmoid) -> $DST"
ls -la --time-style=+%H:%M "$DST"
