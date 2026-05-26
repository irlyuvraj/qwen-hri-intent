"""
Structured per-prediction metrics logger.

One JSONL line per Qwen prediction. Designed to power post-hoc evaluation:
precision/recall by intent class, latency distributions, parse-failure
breakdown, threshold sweeps, etc. Does NOT replace the existing log file
(predictions.jsonl) — that one is older and lighter; this is the canonical
research-grade record.

Usage:
    metrics = MetricsLogger("logs/metrics_2026-04-30.jsonl")
    metrics.log(
        prediction=pred,           # streaming_intent_predictor.PredictionOutput
        robot_state="running",
        active_policy="pick_pink_ball",
        audio_rms_pre_hpf=0.07,
        audio_rms_post_hpf=0.04,
        parse_failed=False,        # set from raw_response inspection
    )

Each line schema:
    {
      "ts":               float,    # unix time, prediction emit
      "seq":              int,      # sequence_id
      "intent":           str,
      "confidence":       float,
      "target_object":    str,
      "task_complete":    bool,
      "spoken_command":   str,
      "reason":           str,      # truncated
      "latency_ms":       float,
      "robot_state":      str,
      "active_policy":    str | null,
      "audio_rms_pre":    float | null,
      "audio_rms_post":   float | null,
      "parse_failed":     bool,
      "raw_response_len": int,
    }
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional


class MetricsLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1)  # line-buffered
        # Header line: a single comment-style record so future tooling can
        # detect the file format without parsing every line.
        self._fh.write(json.dumps({
            "_meta": "qwen-hri-intent metrics v1",
            "_started": time.time(),
        }) + "\n")

    def log(
        self,
        prediction: Any,                       # PredictionOutput (duck-typed)
        robot_state: str = "",
        active_policy: Optional[str] = None,
        audio_rms_pre_hpf: Optional[float] = None,
        audio_rms_post_hpf: Optional[float] = None,
        parse_failed: bool = False,
    ) -> None:
        raw = getattr(prediction, "raw_response", "") or ""
        reason = getattr(prediction, "reason", "") or ""
        record = {
            "ts":               getattr(prediction, "timestamp", time.time()),
            "seq":              int(getattr(prediction, "sequence_id", 0)),
            "intent":           getattr(prediction, "predicted_intent", "unknown"),
            "phase":            getattr(prediction, "predicted_phase", "unknown"),
            "confidence":       float(getattr(prediction, "confidence", 0.0)),
            "target_object":    getattr(prediction, "target_object", "") or "",
            "task_complete":    bool(getattr(prediction, "task_complete", False)),
            "spoken_command":   getattr(prediction, "spoken_command", "") or "",
            "reason":           reason[:140],
            "latency_ms":       float(getattr(prediction, "latency_ms", 0.0)),
            "robot_state":      robot_state,
            "active_policy":    active_policy,
            "audio_rms_pre":    audio_rms_pre_hpf,
            "audio_rms_post":   audio_rms_post_hpf,
            "parse_failed":     bool(parse_failed),
            "raw_response_len": len(raw),
        }
        with self._lock:
            self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
