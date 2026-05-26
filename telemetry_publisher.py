"""
ZMQ telemetry publisher — wire format matches telemetry_dashboard.py.

Usage:
    import telemetry_publisher as tel

    # Once, at startup:
    tel.init(port=5601)   # safe no-op if pyzmq is missing

    # Each Qwen prediction:
    tel.publish_prediction(
        pred=pred,                 # PredictionOutput
        qwen_hz=_qwen_hz_ema,
        groot_hz=robot.current_hz,
        robot_state=_qwen_state,
        active_policy=robot.active_policy,
        audio_rms_pre=None,        # optional
        audio_rms_post=None,
    )

    # Each routing decision:
    tel.publish_event("STOP", policy="pick_pink_ball",
                      reason="Multimodal interrupt streak 2/2")

Design notes
------------
- Module-level singleton: keeps call sites tiny (no plumbing a handle through
  PolicyRouter / GrootRobotController).
- Safe before `init()` is called: every publish_*() is a no-op so existing
  code can call into it unconditionally.
- Non-blocking sends: a tight lock + NOBLOCK; dropped messages are logged
  at debug level. The control loop must never stall on the dashboard.
- pyzmq is an optional dependency. If missing, `init()` warns once and the
  publisher stays disabled.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

log = logging.getLogger("telemetry")

try:
    import zmq
except ImportError:
    zmq = None

_socket = None         # zmq.Socket or None
_lock = threading.Lock()
_enabled = False
_dropped = 0


def init(port: int = 5601, bind_host: str = "127.0.0.1") -> bool:
    """Start the PUB socket. Returns True if active, False if disabled.

    Idempotent: calling twice is a no-op.
    """
    global _socket, _enabled
    if _enabled:
        return True
    if zmq is None:
        log.warning("pyzmq not installed — telemetry disabled "
                    "(pip install pyzmq to enable)")
        return False
    try:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        # SNDHWM controls how many outbound messages we'll buffer before
        # dropping; keep it small so the dashboard sees fresh data.
        sock.setsockopt(zmq.SNDHWM, 1000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://{bind_host}:{port}")
        _socket = sock
        _enabled = True
        log.info("telemetry PUB on tcp://%s:%d", bind_host, port)
        # ZMQ slow-joiner: subscribers attaching right after bind miss the
        # first message. A short sleep + warm-up message keeps demos clean.
        time.sleep(0.1)
        return True
    except Exception as e:
        log.warning("telemetry init failed: %s", e)
        return False


def close() -> None:
    global _socket, _enabled
    with _lock:
        if _socket is not None:
            try:
                _socket.close(linger=0)
            except Exception:
                pass
        _socket = None
        _enabled = False


def _send(payload: dict) -> None:
    """Best-effort non-blocking send. Drops on backpressure."""
    global _dropped
    if not _enabled or _socket is None:
        return
    try:
        msg = json.dumps(payload, default=str)
    except (TypeError, ValueError) as e:
        log.debug("telemetry serialize failed: %s", e)
        return
    with _lock:
        try:
            _socket.send_string(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            _dropped += 1
            if _dropped % 100 == 1:
                log.debug("telemetry: %d messages dropped (backpressure)",
                          _dropped)
        except Exception as e:
            log.debug("telemetry send error: %s", e)


def publish_prediction(
    pred: Any,
    qwen_hz: float = 0.0,
    groot_hz: float = 0.0,
    robot_state: str = "",
    active_policy: Optional[str] = None,
    audio_rms_pre: Optional[float] = None,
    audio_rms_post: Optional[float] = None,
) -> None:
    """Emit one prediction message. ``pred`` is duck-typed (PredictionOutput)."""
    if not _enabled:
        return
    payload = {
        "type":           "prediction",
        "t":              float(getattr(pred, "timestamp", time.time())),
        "seq":            int(getattr(pred, "sequence_id", 0)),
        "intent":         getattr(pred, "predicted_intent", "unknown"),
        "phase":          getattr(pred, "predicted_phase", "unknown"),
        "confidence":     float(getattr(pred, "confidence", 0.0)),
        "target":         getattr(pred, "target_object", "") or "",
        "task_complete":  bool(getattr(pred, "task_complete", False)),
        "latency_ms":     float(getattr(pred, "latency_ms", 0.0)),
        "audio_rms_pre":  audio_rms_pre if audio_rms_pre is not None else 0.0,
        "audio_rms_post": audio_rms_post if audio_rms_post is not None else 0.0,
        "qwen_hz":        float(qwen_hz),
        "groot_hz":       float(groot_hz),
        "robot_state":    robot_state,
        "active_policy":  active_policy,
    }
    _send(payload)


def publish_event(
    kind: str,
    policy: Optional[str] = None,
    reason: str = "",
) -> None:
    """Emit a routing-event marker (STOP / SWITCH / RESUME / COMPLETE / COLD-START)."""
    if not _enabled:
        return
    _send({
        "type":   "event",
        "t":      time.time(),
        "kind":   kind.upper(),
        "policy": policy or "",
        "reason": reason,
    })


def is_enabled() -> bool:
    return _enabled
