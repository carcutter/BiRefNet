#!/bin/bash
# Export the three decoder-only models (best.pth each) to fp32 + fp16 ONNX, then copy into the
# weights tree under ft-decoder/<type>/. CPU export; runs sequentially.
cd /home/renato/projects/interior-segmentation/BiRefNet || exit 1
WEIGHTS=/home/renato/projects/interior-segmentation/interior-segmentation/models/v8/weights/ft-decoder
for t in interior cubemap details; do
  R="runs/ft-decoder/$t"
  base="ft-decoder-$t"
  echo "=========== exporting $t ==========="
  BIREFNET_BB=swin_v1_l uv run python tools/export_onnx.py \
    --checkpoint "$R/best.pth" --out "$R/$base.onnx" \
    --input_size 1024,1024 --batch dynamic --no_sigmoid --check || { echo "FP32 FAILED: $t"; continue; }
  BIREFNET_BB=swin_v1_l uv run python tools/export_onnx.py \
    --checkpoint "$R/best.pth" --out "$R/${base}_fp16.onnx" \
    --input_size 1024,1024 --batch dynamic --no_sigmoid --fp16 --check || { echo "FP16 FAILED: $t"; continue; }
  mkdir -p "$WEIGHTS/$t"
  cp "$R/$base.onnx" "$R/$base.json" "$R/${base}_fp16.onnx" "$R/${base}_fp16.json" "$WEIGHTS/$t/"
  echo "COPIED $t -> $WEIGHTS/$t/"
done
echo "=========== all exports done ==========="
