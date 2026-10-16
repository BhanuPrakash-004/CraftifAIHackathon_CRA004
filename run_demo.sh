#!/usr/bin/env bash
# run_demo.sh — full ADAS pipeline, front + rear side by side, for screen recording.
#
#   ./run_demo.sh [front_video] [rear_video]
#
# Runs PipeGen's FP16 TensorRT engines (YOLOv8 + MiDaS) on both videos inside
# the PipeGen container, fuses depth+detection into FCW (front) and BSD (rear)
# warnings with adas_fusion, stitches the two annotated streams side by side,
# and plays the result fullscreen. Everything is FP16 end to end.
set -euo pipefail
cd "$(dirname "$0")"
A="$(pwd)"
FRONT="${1:-$HOME/personal/pipegen/front.mov}"
REAR="${2:-$HOME/personal/pipegen/rear.mov}"
CT=craftifai-amd64-v3
YE=pipelines/engines_pipegen/yolov8m.fp16.engine
ME=pipelines/engines_pipegen/midas_small.fp16.engine
mkdir -p pipelines/out_pipegen demos

echo "[1/4] perception (YOLOv8 + MiDaS, FP16 TensorRT) — front"
docker exec "$CT" bash -c "cd $A/pipelines && python3 perception.py '$FRONT' \
  out_pipegen/live_front.mp4 out_pipegen/live_front.json $YE $ME" 2>/dev/null
echo "[2/4] perception — rear"
docker exec "$CT" bash -c "cd $A/pipelines && python3 perception.py '$REAR' \
  out_pipegen/live_rear.mp4 out_pipegen/live_rear.json $YE $ME" 2>/dev/null

