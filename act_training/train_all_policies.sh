#!/bin/bash
# ═══════════════════════════════════════════════════════════
# ACT Policy Training — 3 policies for SO101
# ═══════════════════════════════════════════════════════════
#
# Prerequisites:
#   1. Stop vLLM on s99 first (frees ~176GB VRAM)
#   2. pip install lerobot
#
# Usage:
#   bash train_all_policies.sh          # train all 3
#   bash train_all_policies.sh pink     # train only pink
#   bash train_all_policies.sh yellow   # train only yellow
#   bash train_all_policies.sh interrupt # train only interrupt
# ═══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}/outputs"
GPU=0  # Use GPU 0 (single GPU is enough for ACT)

# Training hyperparams
EPOCHS=2000
BATCH_SIZE=8
LR=1e-5
CHUNK_SIZE=100

mkdir -p "${OUTPUT_BASE}"

echo "═══════════════════════════════════════════════════"
echo "  ACT Training for SO101 — 3 Policies"
echo "═══════════════════════════════════════════════════"
echo ""

# ─── Policy 1: Pick Pink Cotton Ball ───
train_pink() {
    echo "━━━ [1/3] Training: pick_pink_ball ━━━"
    echo "  Dataset: u539285g/so101-pink-cotton-ball-v1"
    echo "  Output:  ${OUTPUT_BASE}/pick_pink_ball/"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPU} python -m lerobot.scripts.train \
        --policy.type=act \
        --dataset.repo_id=u539285g/so101-pink-cotton-ball-v1 \
        --output_dir="${OUTPUT_BASE}/pick_pink_ball" \
        --training.num_epochs=${EPOCHS} \
        --training.batch_size=${BATCH_SIZE} \
        --training.lr=${LR} \
        --policy.chunk_size=${CHUNK_SIZE}

    echo "✅ pick_pink_ball done"
    echo ""
}

# ─── Policy 2: Pick Yellow Cotton Ball ───
train_yellow() {
    echo "━━━ [2/3] Training: pick_yellow_ball ━━━"
    echo "  Dataset: u539285g/so101-yellow-cotton-ball-v1"
    echo "  Output:  ${OUTPUT_BASE}/pick_yellow_ball/"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPU} python -m lerobot.scripts.train \
        --policy.type=act \
        --dataset.repo_id=u539285g/so101-yellow-cotton-ball-v1 \
        --output_dir="${OUTPUT_BASE}/pick_yellow_ball" \
        --training.num_epochs=${EPOCHS} \
        --training.batch_size=${BATCH_SIZE} \
        --training.lr=${LR} \
        --policy.chunk_size=${CHUNK_SIZE}

    echo "✅ pick_yellow_ball done"
    echo ""
}

# ─── Policy 3: Pick Up with Interruption ───
train_interrupt() {
    echo "━━━ [3/3] Training: pick_and_correct ━━━"
    echo "  Dataset: u539285g/so101-pick-up-interruption-v1"
    echo "  Output:  ${OUTPUT_BASE}/pick_and_correct/"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPU} python -m lerobot.scripts.train \
        --policy.type=act \
        --dataset.repo_id=u539285g/so101-pick-up-interruption-v1 \
        --output_dir="${OUTPUT_BASE}/pick_and_correct" \
        --training.num_epochs=${EPOCHS} \
        --training.batch_size=${BATCH_SIZE} \
        --training.lr=${LR} \
        --policy.chunk_size=${CHUNK_SIZE}

    echo "✅ pick_and_correct done"
    echo ""
}

# ─── Run ───
POLICY="${1:-all}"

case "$POLICY" in
    pink)      train_pink ;;
    yellow)    train_yellow ;;
    interrupt) train_interrupt ;;
    all)
        train_pink
        train_yellow
        train_interrupt
        echo "═══════════════════════════════════════════════════"
        echo "  ALL 3 POLICIES TRAINED"
        echo "═══════════════════════════════════════════════════"
        echo "  Outputs:"
        echo "    ${OUTPUT_BASE}/pick_pink_ball/"
        echo "    ${OUTPUT_BASE}/pick_yellow_ball/"
        echo "    ${OUTPUT_BASE}/pick_and_correct/"
        echo ""
        echo "  Next: restart vLLM, then run policy_router.py"
        ;;
    *)
        echo "Usage: $0 [pink|yellow|interrupt|all]"
        exit 1
        ;;
esac
