"""
Live telemetry dashboard for the Qwen-HRI + GR00T + SO-101 system.

Subscribes (via ZMQ) to a publisher running inside run_system_groot.py and
plots, in real time:

    ┌────────────────────────────┬────────────────────────────┐
    │  Qwen confidence           │  Audio RMS (post-HPF)      │
    │  + 0.85 streak threshold   │  + 0.012 speech gate       │
    └────────────────────────────┴────────────────────────────┘
    Status: intent · phase · target · policy · robot_state · events

Two panels + status bar. The two things you actually look at while the
robot is running: "did Qwen become confident enough to act?" and "did
Qwen hear my voice?". Latency, Hz, and pre-HPF audio were removed —
those are post-hoc analysis metrics, better read from the JSONL/CSV
than watched live.

Vertical event markers are drawn on every plot when a routing decision
fires: STOP (red), SWITCH (orange), RESUME (green), COMPLETE (blue),
COLD-START (cyan).

────────────────────────────────────────────────────────────────────
USAGE

    # Standalone with fake data — to check the look:
    python telemetry_dashboard.py --demo

    # Real telemetry (publisher must be running):
    python telemetry_dashboard.py
        # defaults to tcp://127.0.0.1:5601

    # Custom address / window length:
    python telemetry_dashboard.py --address tcp://127.0.0.1:5601 --window-s 30

    # Save CSV on close:
    python telemetry_dashboard.py --csv-dir telemetry/

────────────────────────────────────────────────────────────────────
WIRE PROTOCOL (when you hook into run_system_groot.py later)

Two message types on a single ZMQ PUB socket:

(1) "prediction" — one per Qwen prediction (~2 Hz):
    {
      "type": "prediction",
      "t": 1718...,
      "seq": 42,
      "intent": "approach",
      "phase": "approaching",
      "confidence": 0.91,
      "target": "pink cotton ball",
      "latency_ms": 412.0,
      "audio_rms_pre": 0.07,
      "audio_rms_post": 0.04,
      "qwen_hz": 1.9,
      "groot_hz": 1.7,
      "robot_state": "running",
      "active_policy": "pick_pink_ball"
    }

(2) "event" — one per routing decision (sparse):
    {
      "type": "event",
      "t": 1718...,
      "kind": "STOP" | "SWITCH" | "RESUME" | "COMPLETE" | "COLD-START",
      "policy": "pick_pink_ball",
      "reason": "Multimodal interrupt streak 2/2"
    }
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import random
import threading
import time
from collections import deque
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt

try:
    import zmq
except ImportError:
    zmq = None  # demo mode does not need zmq


# ─── tuning ──────────────────────────────────────────────────────────
DEFAULT_ADDRESS = "tcp://127.0.0.1:5601"
DEFAULT_WINDOW_S = 30.0           # how many seconds of history to show
PREDICTION_RATE_HZ = 2.0          # roughly — used only for deque sizing
EVENT_HISTORY = 64                # how many event markers to keep
REFRESH_MS = 200                  # plot update interval

CONFIDENCE_THRESHOLD = 0.85       # streak threshold used by PolicyRouter
SPEECH_GATE_RMS = 0.012           # audio-callback speech-onset gate

EVENT_COLORS = {
    "STOP":       "#e53935",
    "SWITCH":     "#fb8c00",
    "RESUME":     "#43a047",
    "COMPLETE":   "#1e88e5",
    "COLD-START": "#00acc1",
}


# ─── data buffers ────────────────────────────────────────────────────
class TelemetryBuffers:
    """Bounded deques for one rolling time window."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        cap = max(64, int(window_s * PREDICTION_RATE_HZ * 4))

        self.t        = deque(maxlen=cap)
        self.conf     = deque(maxlen=cap)
        self.rms_post = deque(maxlen=cap)

        # last-seen values for the status bar
        self.intent: str = "—"
        self.phase: str = "—"
        self.target: str = "—"
        self.policy: str = "—"
        self.robot_state: str = "—"

        # event markers: list of (t_rel, kind, label)
        self.events: deque = deque(maxlen=EVENT_HISTORY)

        # full history copy for CSV
        self._all_rows: list = []
        self._all_events: list = []

        self._lock = threading.Lock()
        self._t0 = time.time()

    def add_prediction(self, msg: dict):
        with self._lock:
            t_rel = msg.get("t", time.time()) - self._t0
            self.t.append(t_rel)
            self.conf.append(float(msg.get("confidence", 0.0)))
            self.rms_post.append(float(msg.get("audio_rms_post", 0.0)))

            self.intent = str(msg.get("intent", "—"))
            self.phase = str(msg.get("phase", "—"))
            self.target = str(msg.get("target", "—"))
            self.policy = str(msg.get("active_policy", "—") or "—")
            self.robot_state = str(msg.get("robot_state", "—"))

            # CSV keeps the FULL schema (latency, pre-HPF, Hz) for post-hoc
            # analysis even though the dashboard no longer plots them.
            self._all_rows.append({
                "t_rel": t_rel,
                "seq": msg.get("seq"),
                "intent": self.intent,
                "phase": self.phase,
                "confidence": self.conf[-1],
                "target": self.target,
                "latency_ms": float(msg.get("latency_ms", 0.0)),
                "rms_pre": float(msg.get("audio_rms_pre", 0.0)),
                "rms_post": self.rms_post[-1],
                "qwen_hz": float(msg.get("qwen_hz", 0.0)),
                "groot_hz": float(msg.get("groot_hz", 0.0)),
                "robot_state": self.robot_state,
                "active_policy": self.policy,
            })

    def add_event(self, msg: dict):
        with self._lock:
            t_rel = msg.get("t", time.time()) - self._t0
            kind = str(msg.get("kind", "?")).upper()
            label = msg.get("policy") or msg.get("reason") or ""
            self.events.append((t_rel, kind, label))
            self._all_events.append({
                "t_rel": t_rel, "kind": kind, "label": label,
            })

    def snapshot(self):
        """Copy current buffers for plotting (held under lock briefly)."""
        with self._lock:
            return (
                list(self.t),
                list(self.conf),
                list(self.rms_post),
                list(self.events),
                (self.intent, self.phase, self.target,
                 self.policy, self.robot_state),
            )


# ─── ZMQ subscriber thread ───────────────────────────────────────────
def _subscriber_loop(address: str, buffers: TelemetryBuffers,
                     stop_event: threading.Event):
    if zmq is None:
        print("[telemetry] pyzmq not installed — install with `pip install pyzmq`")
        return
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(address)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    print(f"[telemetry] subscribed to {address}")
    while not stop_event.is_set():
        try:
            events = dict(poller.poll(timeout=200))
            if sock in events:
                raw = sock.recv_string(flags=zmq.NOBLOCK)
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "prediction":
                    buffers.add_prediction(msg)
                elif kind == "event":
                    buffers.add_event(msg)
        except zmq.ZMQError:
            break
    sock.close(linger=0)


# ─── demo data thread (so you can run --demo with no publisher) ──────
def _demo_publisher_loop(buffers: TelemetryBuffers,
                         stop_event: threading.Event):
    """Synthesizes prediction + event traffic so the layout is visible."""
    intents = ["continue", "approach", "gesture", "approach",
               "withdraw", "continue", "interrupt", "change_target"]
    phases = ["idle", "approaching", "grasping", "transporting",
              "placing", "retracting", "idle"]
    policy = "pick_pink_ball"
    seq = 0
    last_event_t = time.time()
    next_event_in = 6.0
    robot_state = "idle"

    while not stop_event.is_set():
        seq += 1
        now = time.time()
        # simulate a noisy confidence with occasional dips
        conf = 0.75 + 0.20 * math.sin(now * 0.7) + random.uniform(-0.05, 0.05)
        conf = max(0.30, min(0.99, conf))
        latency = 320 + 120 * math.sin(now * 0.4) + random.uniform(-40, 60)
        rms_pre = 0.04 + 0.02 * math.sin(now * 1.1) + random.uniform(-0.005, 0.02)
        rms_post = rms_pre * 0.55 + random.uniform(-0.003, 0.003)
        # occasional speech spike
        if random.random() < 0.03:
            rms_pre += 0.08
            rms_post += 0.05
        qwen_hz = 1.9 + random.uniform(-0.2, 0.2)
        groot_hz = 1.7 + random.uniform(-0.15, 0.15) if robot_state == "running" else 0.0

        intent = random.choice(intents)
        phase = random.choice(phases)

        buffers.add_prediction({
            "t": now, "seq": seq,
            "intent": intent, "phase": phase,
            "confidence": conf, "target": "pink cotton ball",
            "latency_ms": latency,
            "audio_rms_pre": max(0, rms_pre),
            "audio_rms_post": max(0, rms_post),
            "qwen_hz": qwen_hz, "groot_hz": groot_hz,
            "robot_state": robot_state, "active_policy":
                policy if robot_state == "running" else None,
        })

        # fire fake events on a slow random cadence
        if now - last_event_t > next_event_in:
            kind = random.choice(["COLD-START", "STOP", "SWITCH",
                                  "RESUME", "COMPLETE"])
            if kind == "COLD-START":
                robot_state = "running"
                policy = random.choice(["pick_pink_ball", "pick_yellow_ball"])
            elif kind == "STOP":
                robot_state = "idle"
            elif kind == "RESUME":
                robot_state = "running"
            elif kind == "SWITCH":
                policy = ("pick_yellow_ball" if policy == "pick_pink_ball"
                          else "pick_pink_ball")
            elif kind == "COMPLETE":
                robot_state = "idle"
            buffers.add_event({"t": now, "kind": kind,
                               "policy": policy, "reason": "demo"})
            last_event_t = now
            next_event_in = random.uniform(4.0, 9.0)

        time.sleep(1.0 / PREDICTION_RATE_HZ)


# ─── plotting ────────────────────────────────────────────────────────
def _draw_event_markers(ax, events, t_min: float, t_max: float):
    """Vertical lines for routing events within the visible window."""
    for t_rel, kind, _label in events:
        if not (t_min <= t_rel <= t_max):
            continue
        ax.axvline(t_rel, color=EVENT_COLORS.get(kind, "gray"),
                   alpha=0.35, linewidth=1.2, linestyle="--")


def run_dashboard(buffers: TelemetryBuffers, window_s: float,
                  on_close=None):
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor":   "#252525",
        "axes.edgecolor":   "#666",
        "axes.labelcolor":  "#ddd",
        "axes.titlecolor":  "#eee",
        "xtick.color":      "#bbb",
        "ytick.color":      "#bbb",
        "grid.color":       "#3a3a3a",
        "legend.facecolor": "#2c2c2c",
        "legend.edgecolor": "#555",
        "text.color":       "#ddd",
    })

    fig = plt.figure(figsize=(12, 5))
    fig.canvas.manager.set_window_title("Qwen-HRI telemetry")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.2], hspace=0.45,
                          wspace=0.22, left=0.07, right=0.97,
                          top=0.92, bottom=0.12)

    ax_conf = fig.add_subplot(gs[0, 0])
    ax_rms  = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    # static decorations
    ax_conf.set_title("Qwen prediction confidence")
    ax_conf.set_ylabel("conf")
    ax_conf.set_ylim(0, 1.05)
    ax_conf.axhline(CONFIDENCE_THRESHOLD, color="#ef5350", linestyle="--",
                    linewidth=1, label=f"streak gate {CONFIDENCE_THRESHOLD}")
    ax_conf.grid(True, alpha=0.3)

    ax_rms.set_title("Audio RMS  (post-HPF)")
    ax_rms.set_ylabel("RMS")
    ax_rms.axhline(SPEECH_GATE_RMS, color="#ffb74d", linestyle="--",
                   linewidth=1, label=f"speech gate {SPEECH_GATE_RMS}")
    ax_rms.grid(True, alpha=0.3)

    for ax in (ax_conf, ax_rms):
        ax.set_xlabel("time (s)")

    (line_conf,)     = ax_conf.plot([], [], color="#66bb6a", linewidth=1.3,
                                    label="confidence")
    (line_rms_post,) = ax_rms.plot([], [], color="#42a5f5", linewidth=1.3,
                                   label="post-HPF")

    ax_conf.legend(loc="upper right", fontsize=8)
    ax_rms.legend(loc="upper right", fontsize=8)

    status_text = ax_text.text(
        0.01, 0.5, "", transform=ax_text.transAxes,
        family="monospace", fontsize=11, va="center",
    )

    # legend strip for event markers
    legend_strip = "events:   " + "   ".join(
        f"{k}" for k in EVENT_COLORS
    )
    fig.text(0.5, 0.005, legend_strip, ha="center", fontsize=8,
             color="#888", family="monospace")

    def update(_frame):
        ts, conf, rms_post, events, status = buffers.snapshot()

        if not ts:
            return ()

        t_max = ts[-1]
        t_min = max(0.0, t_max - window_s)

        # slice to visible window
        def _slice(arr):
            return [v for t, v in zip(ts, arr) if t >= t_min]
        x = [t for t in ts if t >= t_min]

        line_conf.set_data(x, _slice(conf))
        line_rms_post.set_data(x, _slice(rms_post))

        for ax in (ax_conf, ax_rms):
            ax.set_xlim(t_min, t_min + window_s)
            # clear previously-drawn event marker lines (matplotlib stores
            # axvline as a line, so we filter by linestyle '--' AND color)
            for ln in [ln for ln in ax.lines if ln.get_linestyle() == "--"
                       and ln.get_color() in EVENT_COLORS.values()]:
                ln.remove()
            _draw_event_markers(ax, events, t_min, t_min + window_s)

        # autoscale RMS (conf has fixed 0-1 range)
        recent_post = _slice(rms_post)
        if recent_post:
            ymax = max(recent_post + [SPEECH_GATE_RMS * 1.5])
            ax_rms.set_ylim(0, ymax * 1.15)

        intent, phase, target, policy, robot_state = status
        status_text.set_text(
            f"  intent: {intent:<14s}  phase: {phase:<14s}  "
            f"target: {target:<22s}\n"
            f"  policy: {policy:<22s}  robot: {robot_state:<10s}  "
            f"events: {len(events):d}"
        )
        return ()

    ani = animation.FuncAnimation(fig, update, interval=REFRESH_MS,
                                  blit=False, cache_frame_data=False)

    if on_close is not None:
        fig.canvas.mpl_connect("close_event", lambda _e: on_close())

    plt.show()
    return ani


def save_csv(buffers: TelemetryBuffers, out_dir: str):
    if not buffers._all_rows and not buffers._all_events:
        print("[telemetry] no data — nothing saved")
        return
    os.makedirs(out_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(out_dir, f"predictions_{ts}.csv")
    evt_path  = os.path.join(out_dir, f"events_{ts}.csv")

    if buffers._all_rows:
        fields = list(buffers._all_rows[0].keys())
        with open(pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in buffers._all_rows:
                w.writerow(row)
        print(f"[telemetry] saved {len(buffers._all_rows)} predictions → "
              f"{pred_path}")

    if buffers._all_events:
        with open(evt_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_rel", "kind", "label"])
            w.writeheader()
            for row in buffers._all_events:
                w.writerow(row)
        print(f"[telemetry] saved {len(buffers._all_events)} events → "
              f"{evt_path}")


# ─── main ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Qwen-HRI telemetry dashboard")
    p.add_argument("--address", default=DEFAULT_ADDRESS,
                   help=f"ZMQ SUB address (default {DEFAULT_ADDRESS})")
    p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S,
                   help="Seconds of history shown (default %(default)s)")
    p.add_argument("--demo", action="store_true",
                   help="Synthesize fake data in-process (no publisher needed)")
    p.add_argument("--csv-dir", default=None,
                   help="If set, save full session CSVs here on window close")
    args = p.parse_args()

    buffers = TelemetryBuffers(window_s=args.window_s)
    stop_event = threading.Event()

    if args.demo:
        print("[telemetry] demo mode — generating fake data")
        worker = threading.Thread(
            target=_demo_publisher_loop,
            args=(buffers, stop_event), daemon=True)
    else:
        if zmq is None:
            raise SystemExit(
                "pyzmq is required for live mode. "
                "Install with: pip install pyzmq  "
                "(or pass --demo to preview the layout)")
        worker = threading.Thread(
            target=_subscriber_loop,
            args=(args.address, buffers, stop_event), daemon=True)

    worker.start()

    def _on_close():
        stop_event.set()
        if args.csv_dir:
            save_csv(buffers, args.csv_dir)

    try:
        run_dashboard(buffers, args.window_s, on_close=_on_close)
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
