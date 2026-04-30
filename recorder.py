"""
recorder.py — Async Session Recorder for Qwen3-Omni + GR00T HRI System
========================================================================
Records a composite 1280×720 MP4 with audio:
  LEFT  : Live camera feed with intent prediction overlay
  RIGHT : Scrolling terminal log (top) + confidence timeline graph (bottom)

Audio is captured in parallel and muxed in at stop() via ffmpeg (-c copy
for video, AAC for audio). If ffmpeg is absent the video-only file is kept.

Non-blocking: push_frame / push_audio / push_log never block the control loop.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("recorder")

# Local stand-in — recorder only reads 5 fields via duck typing so it works
# with any PredictionOutput version without a hard import dependency.
@dataclass
class PredictionOutput:
    predicted_intent: str = "unknown"
    confidence: float = 0.0
    target_object: str = ""
    task_complete: bool = False
    reason: Optional[str] = None
    spoken_command: str = ""

# ── optional: matplotlib for the confidence graph ──────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":          (15,  15,  20),
    "panel_bg":    (25,  28,  35),
    "accent":      (0,  210, 140),
    "warn":        (0,  160, 255),
    "danger":      (50,  50, 230),
    "text_hi":     (240, 240, 240),
    "text_lo":     (130, 130, 145),
}
INTENT_COLORS = {
    "approach":  (0,  200, 120),
    "gesture":   (0,  200, 255),
    "withdraw":  (50, 100, 230),
    "continue":  (160, 160, 160),
    "unknown":   (80,  80,  80),
}


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg finder
# ─────────────────────────────────────────────────────────────────────────────
def _find_ffmpeg() -> Optional[str]:
    candidates = [
        os.environ.get("FFMPEG_BIN"),
        "/opt/miniconda3/envs/lerobot/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    for c in candidates:
        if not c:
            continue
        try:
            subprocess.run([c, "-version"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return c
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), PALETTE["bg"], dtype=np.uint8)
    canvas[(h - nh) // 2:(h - nh) // 2 + nh,
           (w - nw) // 2:(w - nw) // 2 + nw] = resized
    return canvas


def _put(img, text, x, y, color=None, scale=0.55, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color or PALETTE["text_hi"], thickness, cv2.LINE_AA)


def _rect(img, x1, y1, x2, y2, color, alpha=1.0):
    if alpha < 1.0:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Panel renderers
# ─────────────────────────────────────────────────────────────────────────────
def _draw_camera_panel(frame_bgr: np.ndarray, pred: PredictionOutput,
                       panel_w: int, panel_h: int, elapsed: float,
                       qwen_hz: float = 0.0, groot_hz: float = 0.0,
                       pred_seq: int = 0) -> np.ndarray:
    cam = _fit(frame_bgr, panel_w, panel_h)

    intent = pred.predicted_intent or "unknown"
    conf   = max(0.0, min(1.0, pred.confidence))
    icolor = INTENT_COLORS.get(intent.lower(), INTENT_COLORS["unknown"])
    target = (pred.target_object or "").strip()

    # ── Top bar: timestamp ────────────────────────────────────────────────
    _rect(cam, 0, 0, panel_w, 32, (10, 10, 10), alpha=0.70)
    _put(cam, f"t = {elapsed:06.1f}s", 10, 21,
         color=PALETTE["text_lo"], scale=0.50)
    _put(cam, "SO-101  ·  Qwen3-Omni Monitor", panel_w - 230, 21,
         color=PALETTE["text_lo"], scale=0.45)

    # ── Bottom HUD card (bottom-left, not covering the robot centre) ──────
    card_x, card_y = 10, panel_h - 142
    card_w, card_h = 310, 130
    _rect(cam, card_x, card_y, card_x + card_w, card_y + card_h,
          (10, 10, 10), alpha=0.72)
    # coloured left accent stripe
    _rect(cam, card_x, card_y, card_x + 4, card_y + card_h, icolor)

    # label row + prediction sequence counter
    _put(cam, "FUTURE INTENT PREDICTION", card_x + 10, card_y + 17,
         color=PALETTE["text_lo"], scale=0.40)
    _put(cam, f"#{pred_seq}", card_x + card_w - 38, card_y + 17,
         color=PALETTE["text_lo"], scale=0.40)

    # intent value — large coloured text
    cv2.putText(cam, intent.upper(), (card_x + 10, card_y + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, icolor, 2, cv2.LINE_AA)

    # confidence pill next to intent
    conf_str = f"{conf:.0%}"
    (cw, _), _ = cv2.getTextSize(intent.upper(),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
    pill_x = card_x + 14 + cw
    _rect(cam, pill_x, card_y + 28, pill_x + 42, card_y + 48,
          icolor, alpha=0.30)
    _put(cam, conf_str, pill_x + 4, card_y + 44,
         color=icolor, scale=0.45)

    # target object row
    obj_label = f"Target:  {target}" if target else "Target:  —"
    _put(cam, obj_label, card_x + 10, card_y + 68,
         color=PALETTE["text_lo"], scale=0.44)

    # confidence bar (thin)
    bx, by_ = card_x + 10, card_y + 82
    blen = card_w - 20
    _rect(cam, bx, by_, bx + blen, by_ + 6, (40, 42, 48))
    _rect(cam, bx, by_, bx + int(blen * conf), by_ + 6, icolor)

    # reason — WHY Qwen predicted this intent
    reason_raw = pred.reason or ""
    reason_txt = (reason_raw[:55] + "…") if len(reason_raw) > 55 else reason_raw
    _put(cam, f"Reason: {reason_txt}", card_x + 10, card_y + 97,
         color=PALETTE["text_lo"], scale=0.36)

    # speed stats row
    qwen_str  = f"Qwen {qwen_hz:.1f}Hz"  if qwen_hz  > 0.05 else "Qwen ---"
    groot_str = f"GR00T {groot_hz:.1f}Hz" if groot_hz > 0.05 else "GR00T ---"
    _put(cam, f"{qwen_str}  ·  {groot_str}", card_x + 10, card_y + 113,
         color=PALETTE["text_lo"], scale=0.35)

    # ── TASK COMPLETE badge (top-right) ───────────────────────────────────
    if pred.task_complete:
        _rect(cam, panel_w - 144, 40, panel_w - 8, 64, PALETTE["accent"])
        _put(cam, "[DONE] TASK COMPLETE", panel_w - 140, 57,
             color=(10, 10, 10), scale=0.44, thickness=1)

    return cam


def _draw_log_panel(log_lines: deque, panel_w: int, panel_h: int) -> np.ndarray:
    img = np.full((panel_h, panel_w, 3), PALETTE["panel_bg"], dtype=np.uint8)
    _rect(img, 0, 0, panel_w, 28, PALETTE["bg"])
    _put(img, "TERMINAL LOG", 10, 19, color=PALETTE["accent"], scale=0.52)

    line_h    = 18
    max_lines = (panel_h - 36) // line_h
    for i, line in enumerate(list(log_lines)[-max_lines:]):
        y = 36 + i * line_h + 13
        ll = line.lower()
        if any(k in ll for k in ["interrupt", "stop", "error", "warn"]):
            color = PALETTE["danger"]
        elif any(k in ll for k in ["complete", "success", "start", "heard"]):
            color = PALETTE["accent"]
        elif any(k in ll for k in ["approach", "gesture", "withdraw"]):
            color = PALETTE["warn"]
        else:
            color = PALETTE["text_hi"]
        _put(img, line[:70], 8, y, color=color, scale=0.40)
    return img


def _draw_graph_panel_mpl(history: deque, panel_w: int, panel_h: int) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(panel_w / 100, panel_h / 100), dpi=100)
    fig.patch.set_facecolor("#191c23")
    ax.set_facecolor("#191c23")

    times   = [h[0] for h in history]
    confs   = [h[1] for h in history]
    intents = [h[2] for h in history]

    ax.plot(times, confs, color="#00d48c", linewidth=1.5)
    ax.fill_between(times, confs, alpha=0.15, color="#00d48c")

    intent_map = {"approach": 0.9, "gesture": 0.75, "withdraw": 0.55,
                  "continue": 0.4, "unknown": 0.1}
    iy = [intent_map.get(i.lower(), 0.1) for i in intents]
    scatter_colors = [
        "#00c878" if i == "approach" else
        "#00c8ff" if i == "gesture" else
        "#e63232" if i == "withdraw" else "#aaaaaa"
        for i in intents
    ]
    ax.scatter(times, iy, c=scatter_colors, s=12, zorder=5)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(max(0, times[-1] - 30) if times else 0, (times[-1] + 1) if times else 30)
    ax.set_xlabel("time (s)", color="#808090", fontsize=7)
    ax.set_ylabel("conf / intent", color="#808090", fontsize=7)
    ax.tick_params(colors="#606070", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#303040")
    ax.set_title("Intent + Confidence Timeline", color="#e0e0e0", fontsize=8, pad=4)
    ax.legend(handles=[
        mpatches.Patch(color="#00c878", label="approach"),
        mpatches.Patch(color="#00c8ff", label="gesture"),
        mpatches.Patch(color="#e63232", label="withdraw"),
        mpatches.Patch(color="#aaaaaa", label="continue"),
    ], fontsize=5, loc="upper left",
       facecolor="#191c23", edgecolor="#404050", labelcolor="#c0c0c0")

    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close("all")
    return cv2.cvtColor(cv2.resize(buf, (panel_w, panel_h)), cv2.COLOR_RGB2BGR)


def _draw_graph_panel_cv2(history: deque, panel_w: int, panel_h: int) -> np.ndarray:
    img = np.full((panel_h, panel_w, 3), PALETTE["panel_bg"], dtype=np.uint8)
    _rect(img, 0, 0, panel_w, 28, PALETTE["bg"])
    _put(img, "CONFIDENCE TIMELINE", 10, 19, color=PALETTE["accent"], scale=0.52)
    if not history:
        return img
    plot_top, plot_bot = 36, panel_h - 20
    plot_h = plot_bot - plot_top
    plot_w = panel_w - 20
    n      = len(history)
    bar_w  = max(2, plot_w // max(n, 1))
    x0     = 10
    for i, (t, conf, intent) in enumerate(history):
        bh    = int(plot_h * conf)
        bx    = x0 + i * bar_w
        color = INTENT_COLORS.get(intent.lower(), INTENT_COLORS["unknown"])
        _rect(img, bx, plot_bot - bh, bx + max(bar_w - 1, 1), plot_bot, color)
    cv2.line(img, (x0, plot_bot), (x0 + plot_w, plot_bot), PALETTE["text_lo"], 1)
    _, last_conf, last_intent = history[-1]
    _put(img, f"{last_intent}  {last_conf:.0%}", x0 + 5, plot_top - 5,
         color=PALETTE["text_hi"], scale=0.45)
    return img


def _draw_graph_panel(history, panel_w, panel_h):
    return _draw_graph_panel_cv2(history, panel_w, panel_h)


def compose_frame(camera_bgr: np.ndarray, pred: PredictionOutput,
                  log_lines: deque, history: deque, elapsed: float,
                  out_w: int = 1280, out_h: int = 720,
                  qwen_hz: float = 0.0, groot_hz: float = 0.0,
                  pred_seq: int = 0) -> np.ndarray:
    left_w  = int(out_w * 0.60)
    right_w = out_w - left_w
    log_h   = int(out_h * 0.55)
    graph_h = out_h - log_h
    left  = _draw_camera_panel(camera_bgr, pred, left_w, out_h, elapsed,
                               qwen_hz, groot_hz, pred_seq)
    right = np.vstack([
        _draw_log_panel(log_lines, right_w, log_h),
        _draw_graph_panel(history, right_w, graph_h),
    ])
    return np.hstack([left, right])


# ─────────────────────────────────────────────────────────────────────────────
# SystemRecorder
# ─────────────────────────────────────────────────────────────────────────────
class SystemRecorder:
    """
    Non-blocking session recorder.

    push_frame / push_audio / push_log enqueue work; background threads do
    all encoding and writing. The control loop is never slowed down.
    """

    def __init__(self,
                 output_path: str = "session.mp4",
                 fps: int = 10,
                 out_w: int = 1280,
                 out_h: int = 720,
                 audio_sample_rate: int = 16000,
                 max_log_lines: int = 60,
                 max_history: int = 300,
                 queue_maxsize: int = 120):

        self.output_path = str(output_path)
        self.fps         = fps
        self.out_w       = out_w
        self.out_h       = out_h
        self._audio_sr   = audio_sample_rate

        # Side-car paths — merged by ffmpeg at stop()
        # MJPG → .avi avoids the macOS mp4v grayscale bug in OpenCV.
        base = self.output_path.rsplit(".", 1)[0]
        self._video_tmp_path = base + "_video_raw.avi"
        self._audio_path     = base + "_audio.wav"

        self._ffmpeg = _find_ffmpeg()
        if self._ffmpeg is None:
            log.warning("ffmpeg not found — audio will not be muxed into the final mp4")

        self._log_lines : deque[str]                      = deque(maxlen=max_log_lines)
        self._history   : deque[tuple[float, float, str]] = deque(maxlen=max_history)
        self._video_q   : queue.Queue                     = queue.Queue(maxsize=queue_maxsize)
        self._audio_q   : queue.Queue                     = queue.Queue(maxsize=256)

        self._writer      : Optional[cv2.VideoWriter]  = None
        self._wav         : Optional[wave.Wave_write]  = None
        self._encode_thread : Optional[threading.Thread] = None
        self._audio_thread  : Optional[threading.Thread] = None
        self._stop_evt    : threading.Event            = threading.Event()
        self._start_time  : float                      = 0.0
        self._lock        : threading.Lock             = threading.Lock()

        self._last_pred   : PredictionOutput = PredictionOutput()
        self._last_frame  : Optional[np.ndarray] = None
        self._pred_seq    : int   = 0
        self._qwen_hz     : float = 0.0
        self._groot_hz    : float = 0.0

    # ── public API ──────────────────────────────────────────────────────────

    def start(self):
        # MJPG always writes full colour on macOS; mp4v silently goes greyscale.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(
            self._video_tmp_path, fourcc, self.fps, (self.out_w, self.out_h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {self._video_tmp_path}")

        self._wav = wave.open(self._audio_path, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)   # int16
        self._wav.setframerate(self._audio_sr)

        self._start_time = time.time()
        self._stop_evt.clear()

        self._encode_thread = threading.Thread(
            target=self._encode_loop, name="rec-video", daemon=True)
        self._audio_thread = threading.Thread(
            target=self._audio_writer, name="rec-audio", daemon=True)
        self._encode_thread.start()
        self._audio_thread.start()

        log.info("Recorder started → %s  (%dx%d @ %dfps + audio)",
                 self.output_path, self.out_w, self.out_h, self.fps)

    def push_frame(self,
                   frame_bgr: np.ndarray,
                   prediction: Optional[PredictionOutput] = None,
                   log_line: Optional[str] = None):
        """Non-blocking. Drops frame if queue is full."""
        if prediction is not None:
            self._last_pred = prediction
        if log_line is not None:
            with self._lock:
                self._log_lines.append(f"{_ts()}  {log_line}")
        try:
            self._video_q.put_nowait(
                (frame_bgr.copy(), self._last_pred, list(self._log_lines)))
        except queue.Full:
            pass

    def push_audio(self, audio: np.ndarray):
        """Enqueue an audio chunk (float32 [-1,1] or int16). Non-blocking."""
        if self._stop_evt.is_set():
            return
        if audio.dtype != np.int16:
            audio = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        try:
            self._audio_q.put_nowait(audio.tobytes())
        except queue.Full:
            pass

    def push_log(self, line: str):
        """Append a log line without a new camera frame."""
        with self._lock:
            self._log_lines.append(f"{_ts()}  {line}")

    def push_stats(self, qwen_hz: float, groot_hz: float, pred_seq: int = 0):
        """Update Qwen/GR00T speed stats + prediction counter for HUD overlay. Non-blocking."""
        with self._lock:
            self._qwen_hz  = qwen_hz
            self._groot_hz = groot_hz
            self._pred_seq = pred_seq

    def stop(self, timeout: float = 15.0) -> str:
        """Drain queues, finalise files, mux audio, return output path."""
        self._stop_evt.set()
        if self._encode_thread:
            self._encode_thread.join(timeout=timeout)
        if self._audio_thread:
            self._audio_thread.join(timeout=timeout)
        if self._writer:
            self._writer.release()
        if self._wav:
            try:
                self._wav.close()
            except Exception:
                pass
        self._mux_final()
        return self.output_path

    # ── internal ────────────────────────────────────────────────────────────

    def _encode_loop(self):
        frame_interval = 1.0 / self.fps
        next_tick      = time.time()

        while not self._stop_evt.is_set():
            now = time.time()
            if now < next_tick:
                time.sleep(max(0, next_tick - now))
            next_tick += frame_interval

            cam_bgr = pred = log_snap = None
            try:
                while True:
                    cam_bgr, pred, log_snap = self._video_q.get_nowait()
            except queue.Empty:
                pass

            if cam_bgr is None:
                if self._last_frame is not None:
                    self._writer.write(self._last_frame)
                continue

            elapsed = time.time() - self._start_time
            with self._lock:
                _qhz, _ghz, _seq = self._qwen_hz, self._groot_hz, self._pred_seq
            self._history.append(
                (elapsed, pred.confidence, pred.predicted_intent or "unknown"))
            composite = compose_frame(
                cam_bgr, pred,
                deque(log_snap, maxlen=60), self._history,
                elapsed, self.out_w, self.out_h,
                _qhz, _ghz, _seq)
            self._last_frame = composite
            self._writer.write(composite)

        # flush remaining queue after stop
        try:
            while True:
                cam_bgr, pred, log_snap = self._video_q.get_nowait()
                elapsed = time.time() - self._start_time
                with self._lock:
                    _qhz, _ghz, _seq = self._qwen_hz, self._groot_hz, self._pred_seq
                self._history.append(
                    (elapsed, pred.confidence, pred.predicted_intent or "unknown"))
                composite = compose_frame(
                    cam_bgr, pred,
                    deque(log_snap, maxlen=60), self._history,
                    elapsed, self.out_w, self.out_h,
                    _qhz, _ghz, _seq)
                self._writer.write(composite)
        except queue.Empty:
            pass

    def _audio_writer(self):
        while not self._stop_evt.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.1)
                self._wav.writeframesraw(chunk)
            except queue.Empty:
                continue
            except Exception:
                log.exception("Audio writer error")
                break
        # flush remaining
        try:
            while True:
                chunk = self._audio_q.get_nowait()
                self._wav.writeframesraw(chunk)
        except queue.Empty:
            pass

    def _mux_final(self):
        have_video = (os.path.exists(self._video_tmp_path)
                      and os.path.getsize(self._video_tmp_path) > 0)
        have_audio = (os.path.exists(self._audio_path)
                      and os.path.getsize(self._audio_path) > 44)

        if not have_video:
            log.error("No video data — output not written")
            return

        if self._ffmpeg is None or not have_audio:
            # No ffmpeg or no audio — just rename raw video
            if not have_audio:
                log.warning("No audio — keeping video-only output")
            try:
                os.replace(self._video_tmp_path, self.output_path)
            except Exception:
                log.exception("Failed to move video file")
            return

        # Re-encode to H.264 (videotoolbox on Mac) + mux AAC audio
        cmd = [
            self._ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", self._video_tmp_path,
            "-i", self._audio_path,
            "-c:v", "h264_videotoolbox", "-b:v", "3M",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            self.output_path,
        ]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for p in (self._video_tmp_path, self._audio_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            log.info("Recording saved → %s", self.output_path)
        except subprocess.CalledProcessError:
            # videotoolbox not available — fall back to libx264
            cmd[cmd.index("h264_videotoolbox")] = "libx264"
            try:
                subprocess.run(cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for p in (self._video_tmp_path, self._audio_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
                log.info("Recording saved (libx264) → %s", self.output_path)
            except subprocess.CalledProcessError:
                log.error("Mux failed — raw files kept: %s  %s",
                          self._video_tmp_path, self._audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test  (python recorder.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math
    print("Running 5-second smoke test…")
    rec = SystemRecorder(output_path="/tmp/recorder_test.mp4", fps=10)
    rec.start()
    intents = ["continue", "approach", "gesture", "withdraw"]
    for i in range(50):
        t    = i / 10.0
        pred = PredictionOutput(
            predicted_intent=intents[i % len(intents)],
            confidence=0.5 + 0.45 * math.sin(t),
            target_object="pink cotton ball" if i < 25 else "yellow cotton ball",
            task_complete=(i == 45),
            reason="object visible in bowl" if i == 45 else None,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 1] = np.linspace(20, 80, 640, dtype=np.uint8)
        rec.push_frame(frame, pred, f"[intent] {pred.predicted_intent}  conf={pred.confidence:.2f}")
        if i % 5 == 0:
            rec.push_log(f"[system] tick {i}")
        time.sleep(0.1)
    path = rec.stop()
    print(f"Smoke test done → {path}")
