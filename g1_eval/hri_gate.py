"""
HRIGate — eval-loop side of the Qwen HRI bridge (self-contained; lives in the
Unitree repo so it can be imported inside the gr00t docker container).

>>> THIS IS A COPY <<<  The authoritative location when running is inside the
Unitree repo at:
    unitree_lerobot/unitree_lerobot/eval_robot/utils/hri_gate.py
Copy this file there (see g1_eval/APPLY_PATCH.md). It is kept here so the G1
integration is fully reproducible from the qwen-hri-intent repo alone.

It is the counterpart of g1_link.G1Link in the qwen-hri-intent repo. The G1
host (the eval process) BINDS both ZMQ ports; the brain is a pure client that
connects to them.

    eval loop (this file)                          brain (qwen-hri-intent)
    ─────────────────────                          ───────────────────────
    PUB  bind tcp://*:state_port  ── state/frame ►  SUB connect   (cam_head + state)
    SUB  bind tcp://*:cmd_port    ◄── command ────  PUB connect   (run/hold/switch/home)

Wire format (must match g1_link.py):
    state channel, multipart:
        [b"state", json]  {seq, t, state, active_task, hz}
        [b"frame", jpeg]  BGR JPEG bytes of cam_head
    command channel, single-part json:
        {seq, t, command, task}    command ∈ {run, hold, switch, home}

Only deps: pyzmq, numpy, opencv (cv2). All present in the gr00t container
except possibly pyzmq — `uv pip install pyzmq` if missing.
"""

import json
import time

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("HRIGate needs opencv (cv2).") from e

try:
    import zmq
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "HRIGate needs pyzmq. Inside the gr00t container run: "
        "`uv pip install pyzmq` (or add it to the image)."
    ) from e


class HRIGate:
    def __init__(
        self,
        state_port: int = 5701,
        cmd_port: int = 5702,
        frame_hz: float = 10.0,
        jpeg_quality: int = 70,
        bind_host: str = "*",
    ):
        self._ctx = zmq.Context.instance()

        # PUB state + frames (bind — the brain connects).
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 4)
        self._pub.bind(f"tcp://{bind_host}:{state_port}")

        # SUB commands (bind + conflate — only the latest command matters).
        self._cmd = self._ctx.socket(zmq.SUB)
        self._cmd.setsockopt(zmq.SUBSCRIBE, b"")
        self._cmd.setsockopt(zmq.CONFLATE, 1)
        self._cmd.setsockopt(zmq.RCVHWM, 1)
        self._cmd.bind(f"tcp://{bind_host}:{cmd_port}")
        self._poller = zmq.Poller()
        self._poller.register(self._cmd, zmq.POLLIN)

        self._frame_min_dt = 1.0 / frame_hz if frame_hz > 0 else 0.0
        self._last_frame_t = 0.0
        self._jpeg_q = int(jpeg_quality)
        self._state_seq = 0

        # Give SUBs on the brain side a moment to connect before first send
        # (PUB drops messages with no subscriber — "slow joiner").
        time.sleep(0.2)

    # ── publish ──────────────────────────────────────────────────────────
    def publish_state(self, state: str, active_task: str, hz: float = 0.0):
        self._state_seq += 1
        msg = {
            "seq": self._state_seq, "t": time.time(),
            "state": state, "active_task": active_task, "hz": float(hz),
        }
        try:
            self._pub.send_multipart([b"state", json.dumps(msg).encode("utf-8")],
                                     flags=zmq.NOBLOCK)
        except Exception:
            pass

    def publish_frame(self, bgr: np.ndarray):
        """Publish a cam_head frame (BGR np.ndarray) as JPEG, throttled to
        frame_hz. Pass the BGR image so the brain receives exactly what a cv2
        webcam would deliver on SO-101."""
        if bgr is None:
            return
        now = time.time()
        if self._frame_min_dt and (now - self._last_frame_t) < self._frame_min_dt:
            return
        ok, buf = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_q])
        if not ok:
            return
        try:
            self._pub.send_multipart([b"frame", buf.tobytes()], flags=zmq.NOBLOCK)
            self._last_frame_t = now
        except Exception:
            pass

    # ── receive ──────────────────────────────────────────────────────────
    def poll_command(self):
        """Return the latest command dict {command, task, seq, t} if one has
        arrived since the last poll, else None. Non-blocking."""
        latest = None
        socks = dict(self._poller.poll(timeout=0))
        if self._cmd in socks:
            try:
                latest = json.loads(self._cmd.recv_string(flags=zmq.NOBLOCK))
            except Exception:
                latest = None
        return latest

    def close(self):
        for sock in (self._pub, self._cmd):
            try:
                sock.close(0)
            except Exception:
                pass
