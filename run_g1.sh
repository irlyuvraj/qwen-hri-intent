#!/usr/bin/env bash
# Launch the G1 HRI brain on the MacBook.
#
# Prereqs (must already be running):
#   - Qwen vLLM on s99           : http://100.118.24.127:8000/v1
#   - GR00T policy server (G1 PC) : port 5555
#   - eval_g1_isaac_gr00t.py --hri_enable (G1 PC), pressed 's'
#   - Tailscale up on all three (warm the link: see the ping below)
#
# Usage:
#   bash run_g1.sh                # full: --no-vad + view + record + metrics
#   bash run_g1.sh --vad          # use the energy-VAD speech path instead
#   bash run_g1.sh --no-record    # skip the MP4 (still records metrics)
#   extra flags are passed through, e.g.:  bash run_g1.sh --no-completion-check

set -euo pipefail

# ── Config (edit these if IPs change) ────────────────────────────────────
VLLM_URL="http://100.118.24.127:8000/v1"   # s99 (Qwen)
G1_HOST="100.99.67.4"                       # gqu6x (G1 PC), Tailscale IP
TASKS="tasks_g1.yaml"
SESS_DIR="$HOME/sessions"
TS_CLI="/Applications/Tailscale.app/Contents/MacOS/Tailscale"

# ── Parse a couple of convenience flags; pass everything else through ─────
SPEECH="--no-vad"          # anticipatory multimodal path (your contribution)
RECORD=1
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --vad)        SPEECH="" ;;            # revert to energy-VAD transcribe path
    --no-record)  RECORD=0 ;;
    *)            EXTRA+=("$arg") ;;
  esac
done

mkdir -p "$SESS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ── Warm the Tailscale direct paths so the first seconds aren't relayed ───
if [ -x "$TS_CLI" ]; then
  echo "Warming Tailscale direct paths..."
  "$TS_CLI" ping -c 2 "$G1_HOST" >/dev/null 2>&1 || true
  "$TS_CLI" ping -c 2 100.118.24.127 >/dev/null 2>&1 || true
fi

# ── Build the command ────────────────────────────────────────────────────
CMD=(python run_system_g1.py
     --vllm-url "$VLLM_URL"
     --g1-host  "$G1_HOST"
     --tasks    "$TASKS"
     --view
     --metrics  "$SESS_DIR/g1_${STAMP}.jsonl")

[ -n "$SPEECH" ] && CMD+=("$SPEECH")
[ "$RECORD" -eq 1 ] && CMD+=(--record "$SESS_DIR/g1_${STAMP}.mp4")
CMD+=("${EXTRA[@]}")

echo "Launching: ${CMD[*]}"
echo "  (Ctrl-C to stop cleanly so the MP4 is finalized + audio muxed)"
exec "${CMD[@]}"
