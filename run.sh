#!/bin/bash
# Run the Qwen3-Omni + GR00T N1.6 system on SO-101.
# Usage:
#   bash run.sh            (no recording)
#   bash run.sh --record   (records to ~/sessions/<timestamp>.mp4)

cd "$(dirname "$0")"

EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--record" ]]; then
    OUTFILE="$HOME/sessions/$(date +%Y%m%d_%H%M%S).mp4"
    mkdir -p "$HOME/sessions"
    EXTRA_ARGS+=(--record "$OUTFILE" --record-fps 10)
    echo "Recording to: $OUTFILE"
  else
    EXTRA_ARGS+=("$arg")
  fi
done

python run_system_groot.py \
  --vllm-url http://192.168.2.25:8000/v1 \
  --tasks tasks.yaml \
  --robot-port /dev/tty.usbmodem5AE70452961 \
  --camera-index 0 \
  --robot-camera-index 0 \
  --policy-host 192.168.2.25 \
  --policy-port 5555 \
  "${EXTRA_ARGS[@]}"
