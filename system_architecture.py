#!/usr/bin/env python3
"""Simple System Architecture — Multimodal Intent Prediction for HRI"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors
BLUE   = ("#DBEAFE", "#1D4ED8")   # sensor
GREEN  = ("#D1FAE5", "#047857")   # model
PURPLE = ("#EDE9FE", "#6D28D9")   # decision
RED    = ("#FEE2E2", "#B91C1C")   # robot
GRAY   = ("#F3F4F6", "#4B5563")   # info

def box(x, y, w, h, title, subtitle, colors, fontsize=11):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                       facecolor=colors[0], edgecolor=colors[1], linewidth=2, zorder=2)
    ax.add_patch(b)
    if subtitle:
        ax.text(x+w/2, y+h/2+0.13, title, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=colors[1], zorder=3)
        ax.text(x+w/2, y+h/2-0.17, subtitle, ha='center', va='center',
                fontsize=8, color='#6B7280', zorder=3)
    else:
        ax.text(x+w/2, y+h/2, title, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=colors[1], zorder=3)

def arrow(x1, y1, x2, y2, label=None, color='#374151', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.15, label, ha='center', va='bottom', fontsize=7.5,
                color='#6B7280', style='italic', zorder=3,
                bbox=dict(fc='white', ec='none', pad=1, alpha=0.85))

# ── Title ──
ax.text(7, 8.15, "Real-Time Multimodal Intent Prediction for HRI",
        ha='center', fontsize=14, fontweight='bold', color='#111827')
ax.text(7, 7.8, "Single Qwen3-Omni model replaces VAD + ASR + Vision + Fusion",
        ha='center', fontsize=9, color='#6B7280')

# ══════════ Row 1: Inputs (y=6.6) ══════════
box(0.5, 6.4, 3.0, 0.85, "Camera", "30fps RGB", BLUE)
box(4.0, 6.4, 3.0, 0.85, "Microphone", "16kHz mono", BLUE)
box(8.5, 6.4, 3.0, 0.85, "Robot Task", "active object + goal", BLUE)

# ══════════ Row 2: Qwen (y=4.5) — big central box ══════════
qwen = FancyBboxPatch((0.5, 4.2), 11.0, 1.5, boxstyle="round,pad=0.2",
                      facecolor=GREEN[0], edgecolor=GREEN[1], linewidth=2.5, zorder=2)
ax.add_patch(qwen)
ax.text(6.0, 5.2, "Qwen3-Omni-30B", ha='center', fontsize=14,
        fontweight='bold', color=GREEN[1], zorder=3)
ax.text(6.0, 4.75, "audio + video + task context  →  intent + object + confidence",
        ha='center', fontsize=9, color='#374151', zorder=3)

# Latency box
lat = FancyBboxPatch((12.0, 4.3), 1.8, 1.3, boxstyle="round,pad=0.1",
                     facecolor=GRAY[0], edgecolor=GRAY[1], linewidth=1.2, zorder=2)
ax.add_patch(lat)
ax.text(12.9, 5.25, "~360ms", ha='center', fontsize=11, fontweight='bold', color=GRAY[1], zorder=3)
ax.text(12.9, 4.92, "avg latency", ha='center', fontsize=7.5, color='#9CA3AF', zorder=3)
ax.text(12.9, 4.65, "~1.4s advance", ha='center', fontsize=7.5, color='#9CA3AF', zorder=3)

# Arrows: inputs → Qwen
arrow(2.0, 6.4, 3.5, 5.7, "frames")
arrow(5.5, 6.4, 5.5, 5.7, "audio")
arrow(10.0, 6.4, 8.5, 5.7, "task context")

# ══════════ Row 3: Decision (y=2.6) ══════════
box(0.5, 2.4, 3.2, 0.85, "State Machine", "filters transitions", PURPLE)
box(4.2, 2.4, 3.2, 0.85, "Mismatch Detector", "object ≠ task → alert", PURPLE)
box(7.9, 2.4, 3.6, 0.85, "Interrupt Detector", "verbal STOP + visual", PURPLE)

# Arrows: Qwen → Decision
arrow(3.0, 4.2, 2.1, 3.25, "intent")
arrow(6.0, 4.2, 5.8, 3.25, "predicted object")
arrow(8.5, 4.2, 9.7, 3.25, "interrupt signal")
# Direct audio to interrupt detector
arrow(5.5, 6.4, 9.7, 3.25, "raw audio", color='#7C3AED', lw=1.2)

# ══════════ Row 4: Robot (y=0.6) ══════════
box(0.5, 0.5, 3.0, 0.85, "Policy Router", "selects ACT policy", RED)
box(4.5, 0.5, 2.0, 0.85, "pick_pink", None, ("#FCE7F3", "#BE185D"), fontsize=9)
box(6.7, 0.5, 2.0, 0.85, "pick_yellow", None, ("#FEF3C7", "#B45309"), fontsize=9)
box(8.9, 0.5, 2.2, 0.85, "pick_correct", None, ("#DBEAFE", "#1D4ED8"), fontsize=9)
box(11.5, 0.5, 2.3, 0.85, "SO101 Robot", "6-DOF arm", RED)

# Arrows: Decision → Robot
arrow(2.1, 2.4, 2.0, 1.35, "task switch")
arrow(5.8, 2.4, 2.0, 1.35, "mismatch")
arrow(9.7, 2.4, 2.0, 1.35, "STOP")

# Policy Router → policies
arrow(3.5, 0.92, 4.5, 0.92, color=RED[1], lw=1.2)
arrow(3.5, 0.92, 6.7, 0.92, color=RED[1], lw=1.2)
arrow(3.5, 0.92, 8.9, 0.92, color=RED[1], lw=1.2)

# Policy → Robot
arrow(11.1, 0.92, 11.5, 0.92, "actions", color=RED[1])

# ── Layer labels on left ──
for y, label in [(6.82, "INPUT"), (4.95, "PERCEPTION"), (2.82, "DECISION"), (0.92, "ACTION")]:
    ax.text(0.15, y, label, fontsize=7, fontweight='bold', color='#9CA3AF',
            rotation=90, va='center', ha='center')

plt.tight_layout(pad=0.3)
plt.savefig("results/system_architecture.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig("results/system_architecture.pdf", bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: results/system_architecture.png")
print("Saved: results/system_architecture.pdf")
plt.show()
