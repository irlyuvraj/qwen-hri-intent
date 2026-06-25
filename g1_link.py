"""
G1Link — brain-side ZMQ transport between the Qwen HRI brain (this repo,
running natively) and the Unitree G1 control loop (eval_g1_isaac_gr00t.py,
running inside docker on the G1 host).

Topology (the G1 host binds both ports; the brain is a pure client, exactly
like ImageClient connecting to image_host):

    G1 host (eval loop)                         Brain (this process)
    ───────────────────                         ────────────────────
    PUB  bind tcp://*:STATE_PORT   ── state ──►  SUB  connect  (latest state + cam_head frame)
    SUB  bind tcp://*:CMD_PORT     ◄─ command ── PUB  connect  (run / hold / switch / home)

Wire format
-----------
State channel (eval → brain), multipart [topic, payload]:
    [b"state", json]   {seq, t, state, active_task, hz}      ~30 Hz
    [b"frame", jpeg]   BGR JPEG bytes of cam_head             ~10 Hz
Command channel (brain → eval), single-part json:
    {seq, t, command, task}   command ∈ {run, hold, switch, home}

The brain treats the eval loop as a dumb executor: the brain owns the
idle/running/active-task state machine (in G1ControlBridge) and just tells
the loop what to do each tick. `state` coming back from the loop is advisory
(loop liveness / hz), not the source of truth for the policy state.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("g1_link")

try:
    import zmq
except ImportError as e:  # hard requirement on the G1 brain
    raise ImportError(
        "pyzmq is required for the G1 brain (g1_link). Install with "
        "`pip install pyzmq` in the qwen-hri-intent venv."
    ) from e

try:
    import cv2
except ImportError as e:
    raise ImportError("opencv-python is required for g1_link (JPEG decode).") from e


class G1Link:
    """Brain-side ZMQ client. Background thread keeps the latest state + frame;
    send_command() pushes a routing decision to the eval loop."""

    def __init__(
        self,
        g1_host: str,
        state_port: int = 5701,
        cmd_port: int = 5702,
        recv_timeout_ms: int = 200,
    ):
        self._ctx = zmq.Context.instance()

        # SUB to eval's state+frame PUB.
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{g1_host}:{state_port}")
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")  # all topics
        self._sub.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        # Keep queues short — we only ever care about the newest message.
        self._sub.setsockopt(zmq.RCVHWM, 4)
        self._sub.setsockopt(zmq.CONFLATE, 0)  # multipart: can't conflate; drain instead

        # PUB to eval's command SUB.
        self._cmd = self._ctx.socket(zmq.PUB)
        self._cmd.setsockopt(zmq.SNDHWM, 4)
        self._cmd.connect(f"tcp://{g1_host}:{cmd_port}")

        self._lock = threading.Lock()
        self._latest_state: dict = {"state": "unknown", "active_task": "", "hz": 0.0}
        self._latest_frame: Optional[np.ndarray] = None  # BGR np.ndarray
        self._last_frame_t: float = 0.0
        self._last_state_t: float = 0.0
        self._cmd_seq = 0

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="g1-link-recv")
        self._thread.start()
        log.info("G1Link started — subscribing to state/frame, publishing commands")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        with _suppress():
            self._sub.close(0)
        with _suppress():
            self._cmd.close(0)
        log.info("G1Link stopped")

    # ── receive (background) ─────────────────────────────────────────────
    def _recv_loop(self):
        while self._running:
            try:
                parts = self._sub.recv_multipart()
            except zmq.Again:
                continue
            except Exception as e:
                log.debug("recv_multipart failed: %s", e)
                continue
            if len(parts) != 2:
                continue
            topic, payload = parts
            now = time.time()
            if topic == b"state":
                try:
                    st = json.loads(payload.decode("utf-8"))
                    with self._lock:
                        self._latest_state = st
                        self._last_state_t = now
                except Exception:
                    log.debug("bad state payload")
            elif topic == b"frame":
                buf = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # decodes to BGR
                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._last_frame_t = now

    # ── accessors ────────────────────────────────────────────────────────
    def latest_frame(self) -> Optional[np.ndarray]:
        """Most recent cam_head frame as a BGR np.ndarray (cv2 convention,
        matching what SO-101's cap.read() feeds the predictor), or None."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def loop_state(self) -> dict:
        with self._lock:
            return dict(self._latest_state)

    def loop_hz(self) -> float:
        with self._lock:
            return float(self._latest_state.get("hz", 0.0))

    def frame_age(self) -> float:
        with self._lock:
            return time.time() - self._last_frame_t if self._last_frame_t else float("inf")

    def is_loop_alive(self, max_age_s: float = 2.0) -> bool:
        with self._lock:
            return (time.time() - self._last_state_t) < max_age_s if self._last_state_t else False

    # ── send ─────────────────────────────────────────────────────────────
    def send_command(self, command: str, task: str = ""):
        """Publish a routing decision to the eval loop.
        command ∈ {run, hold, switch, home}. `task` is the GR00T lang string
        for run/switch (ignored for hold/home)."""
        self._cmd_seq += 1
        msg = {
            "seq": self._cmd_seq,
            "t": time.time(),
            "command": command,
            "task": task,
        }
        try:
            self._cmd.send_string(json.dumps(msg), flags=zmq.NOBLOCK)
        except Exception as e:
            log.debug("send_command dropped (%s): %s", command, e)


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
