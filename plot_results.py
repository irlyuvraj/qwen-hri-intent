#!/usr/bin/env python3
"""Plot predicted intents vs ground truth tasks for interrupt test."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load predictions
predictions = []
with open("results/interrupt_test_v2.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d["type"] == "prediction":
            predictions.append(d)

# Ground truth task segments (from injected interrupts)
gt_segments = [
    (0.0, 10.0, "blue bottle", "#4C9AFF"),
    (10.0, 18.5, "headphone", "#6554C0"),
    (18.5, 21.0, "blue bottle", "#4C9AFF"),
    (21.0, 31.0, "other bottle", "#FF9F43"),
]

# Intent colors
intent_colors = {
    "approach": "#4C9AFF",
    "gesture": "#36B37E",
    "withdraw": "#FF7452",
    "unknown": "#97A0AF",
    "continue": "#B3BAC5",
    "change_target": "#6554C0",
}

# Object colors
def obj_color(obj):
    if "headphone" in obj: return "#6554C0"
    if "bottle" in obj: return "#4C9AFF"
    return "#97A0AF"

times = [p["elapsed_s"] for p in predictions]
intents = [p["predicted_intent"] for p in predictions]
objects = [p["target_object"] for p in predictions]
confs = [p["confidence"] for p in predictions]
lats = [p["latency_ms"] for p in predictions]

fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 1.5, 1]})
fig.suptitle("Qwen3-Omni Interrupt Detection — Prediction vs Ground Truth",
             fontsize=14, fontweight="bold", y=0.97)

# ── Row 1: Ground Truth Tasks ──
ax = axes[0]
for start, end, label, color in gt_segments:
    ax.barh(0, end - start, left=start, height=0.6, color=color, alpha=0.7, edgecolor="white")
    mid = (start + end) / 2
    ax.text(mid, 0, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
ax.set_yticks([0])
ax.set_yticklabels(["Ground\nTruth"])
ax.set_ylim(-0.5, 0.5)
ax.set_title("Active Task (injected commands)", fontsize=10, loc="left", pad=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Task switch lines on all axes
for ts in [10.0, 18.5, 21.0]:
    for a in axes:
        a.axvline(ts, color="#172B4D", linewidth=1, linestyle="--", alpha=0.5)

# ── Row 2: Predicted Object ──
ax = axes[1]
for i, p in enumerate(predictions):
    t = p["elapsed_s"]
    dt = predictions[i+1]["elapsed_s"] - t if i < len(predictions)-1 else 0.5
    color = obj_color(p["target_object"])
    ax.barh(0, dt, left=t, height=0.6, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
ax.set_yticks([0])
ax.set_yticklabels(["Predicted\nObject"])
ax.set_ylim(-0.5, 0.5)
ax.set_title("Predicted Target Object", fontsize=10, loc="left", pad=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Row 3: Intent over time ──
ax = axes[2]
intent_order = ["approach", "gesture", "withdraw", "unknown"]
intent_to_y = {v: i for i, v in enumerate(intent_order)}

for i, p in enumerate(predictions):
    t = p["elapsed_s"]
    y = intent_to_y.get(p["predicted_intent"], 3)
    color = intent_colors.get(p["predicted_intent"], "#97A0AF")
    size = 30 + p["confidence"] * 60
    ax.scatter(t, y, c=color, s=size, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)
    # Connect to next
    if i < len(predictions) - 1:
        nt = predictions[i+1]["elapsed_s"]
        ny = intent_to_y.get(predictions[i+1]["predicted_intent"], 3)
        ax.plot([t, nt], [y, ny], color=color, alpha=0.2, linewidth=1, zorder=1)

# Interrupt marker
ax.axvline(21.0, color="#FF5630", linewidth=2.5, alpha=0.8)
ax.annotate("VERBAL\nINTERRUPT", xy=(21.0, 3.3), fontsize=8, color="#FF5630",
            fontweight="bold", ha="center", va="bottom")

ax.set_yticks(range(len(intent_order)))
ax.set_yticklabels(intent_order)
ax.set_ylim(-0.5, len(intent_order) - 0.5)
ax.set_title("Predicted Intent (dot size = confidence)", fontsize=10, loc="left", pad=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.15)

# ── Row 4: Latency ──
ax = axes[3]
ax.fill_between(times, lats, alpha=0.15, color="#FF7452")
ax.plot(times, lats, color="#FF7452", linewidth=1.2, alpha=0.8)
ax.axhline(360, color="#FF7452", linewidth=0.8, linestyle="--", alpha=0.5)
ax.text(0.3, 370, "avg 360ms", fontsize=8, color="#FF7452", alpha=0.7)
ax.axhline(1400, color="#36B37E", linewidth=0.8, linestyle="--", alpha=0.4)
ax.text(0.3, 1420, "1.4s advance window", fontsize=8, color="#36B37E", alpha=0.6)
ax.set_ylabel("ms", fontsize=9)
ax.set_ylim(0, 1600)
ax.set_title("Inference Latency", fontsize=10, loc="left", pad=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("Time (s)", fontsize=10)
ax.set_xlim(0, 31.5)

# Legend
legend_patches = [
    mpatches.Patch(color="#4C9AFF", label="approach / blue bottle"),
    mpatches.Patch(color="#36B37E", label="gesture"),
    mpatches.Patch(color="#FF7452", label="withdraw"),
    mpatches.Patch(color="#6554C0", label="headphones"),
    mpatches.Patch(color="#FF9F43", label="other bottle (GT)"),
    mpatches.Patch(color="#97A0AF", label="unknown"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("results/interrupt_results_v3.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig("results/interrupt_results_v3.pdf", bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: results/interrupt_results_v3.png")
print("Saved: results/interrupt_results_v3.pdf")
plt.show()
