"""
Qwen3-Omni HRI brain for the Unitree G1 (GR00T N1.7 monster-tray pick-place).

This is the G1 counterpart to run_system_groot.py. It reuses the SAME
robot-agnostic brain — FastQwenInferenceEngine, StreamingIntentPredictor,
InterruptDetectionSystem, PolicyRouter, TaskRegistry — and swaps only the
"body":

    SO-101 (run_system_groot.py):  GrootRobotController drives the arm directly
                                    + local webcam (cv2.VideoCapture)
    G1     (this file):            G1ControlBridge forwards stop/switch/run/home
                                    to Unitree's eval_g1_isaac_gr00t.py over ZMQ
                                    + cam_head frames arrive over the same ZMQ link

Because the brain only touches the robot through the 9-method contract that
G1ControlBridge implements, NOTHING in run_system_groot.py changes and the
SO-101 system keeps working untouched.

IMPORTANT: run this in the SAME venv as run_system_groot.py — it imports
PolicyRouter from that module, which pulls in lerobot/gr00t at import time
(class defs only; no robot connects). Those deps already exist on the SO-101
host. The brain itself never calls lerobot/gr00t.

Run:
    python run_system_g1.py \
        --vllm-url http://192.168.2.25:8000/v1 \
        --tasks tasks_g1.yaml \
        --g1-host 192.168.123.200 \
        --state-port 5701 --cmd-port 5702

The Unitree eval loop must be launched with the matching --hri_enable flags
(see README_G1.md).
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from typing import Optional

import numpy as np
import cv2

from qwen_inference_engine import FastQwenInferenceEngine
from streaming_intent_predictor import StreamingIntentPredictor, StreamConfig
from interrupt_detection_system import InterruptDetectionSystem, connect_to_predictor
from task_registry import TaskRegistry
from metrics_logger import MetricsLogger
import telemetry_publisher as telemetry

# Reuse the brain unchanged from the SO-101 entry point. Import-time this pulls
# in lerobot/gr00t (class defs only) — fine on the SO-101 host venv.
from run_system_groot import PolicyRouter, STOP_WORDS, RESUME_WORDS, RELATIVE_WORDS

from g1_link import G1Link
from g1_bridge import G1ControlBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_g1")


def _show_avatar_view(frame, state, policy, hud, age):
    """Remote 'see-through-the-robot' window: the G1 head-cam frame the brain
    already receives over ZMQ, with a small status bar. Lets the operator
    perceive remotely while issuing voice intent (avatar loop)."""
    disp = frame.copy()
    h, w = disp.shape[:2]
    bar = disp.copy()
    cv2.rectangle(bar, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.5, disp, 0.5, 0, disp)
    cv2.putText(disp, f"{state}:{policy or '-'}  |  {hud}", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if age > 1.0:  # frames stopped arriving — warn the operator
        cv2.putText(disp, "STALE FEED", (w - 135, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("G1 avatar view (head cam)", disp)
    cv2.waitKey(1)


def run(args):
    tasks = TaskRegistry.from_yaml(args.tasks)
    log.info("Loaded %d task(s) from %s: %s",
             len(tasks), args.tasks, ", ".join(tasks.names()))

    # ── Body: ZMQ link + bridge (replaces GrootRobotController + webcam) ──
    link = G1Link(
        g1_host=args.g1_host,
        state_port=args.state_port,
        cmd_port=args.cmd_port,
    )
    robot = G1ControlBridge(tasks=tasks, link=link,
                            home_on_complete=not args.no_home)
    robot.connect()  # starts the link; holds the eval loop (idle)

    # ── Brain: identical wiring to run_system_groot.run() ────────────────
    engine = FastQwenInferenceEngine(
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        scene_objects=tasks.objects(),
        cold_start_choices=[(f"command_{t.name}", t.object) for t in tasks],
    )

    config = StreamConfig(
        inference_interval=0.25,
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        motion_threshold=0,
    )
    predictor = StreamingIntentPredictor(config)
    predictor.scheduler.set_inference_engine(engine)

    interrupt_system = InterruptDetectionSystem(
        consecutive_required=3,
        grace_period=4.0,
    )
    connect_to_predictor(interrupt_system, predictor)

    def _command_validator(text: str) -> bool:
        t = text.lower().strip()
        if any(w in t for w in STOP_WORDS) or "no" in t.split():
            return True
        if tasks.resolve(t) is not None:
            return True
        if any(w in t for w in RESUME_WORDS):
            return True
        return any(w in t for w in RELATIVE_WORDS)

    interrupt_system.command_validator = _command_validator

    router = PolicyRouter(
        robot, interrupt_system, engine,
        predictor=predictor,
        command_validator=_command_validator,
        speech_via_qwen=args.no_vad,
        ground_actions=not args.no_grounding,
        verify_completion=not args.no_completion_check,
    )
    log.info("Grounding %s | completion-check %s | home %s | VAD %s",
             "OFF" if args.no_grounding else "ON",
             "OFF" if args.no_completion_check else "ON",
             "OFF" if args.no_home else "ON",
             "OFF (Qwen owns speech)" if args.no_vad else "ON")

    def on_interrupt(event):
        log.info("INTERRUPT: %s (object=%s, cmd='%s')",
                 event.reason, event.predicted_object, event.raw_command)
        result = router.handle_interrupt(event)
        log.info("Router action: %s", result)

    interrupt_system.on_interrupt(on_interrupt)
    interrupt_system.command_interface.on_new_task(
        lambda task: router.handle_voice_command(task)
    )

    metrics: Optional[MetricsLogger] = None
    if args.metrics:
        try:
            metrics = MetricsLogger(args.metrics)
            log.info("Metrics → %s", args.metrics)
        except Exception as e:
            log.warning("MetricsLogger failed to start: %s", e)

    if args.telemetry_port:
        telemetry.init(port=args.telemetry_port)

    # ── Audio (local mic on the brain host) — faithful port of the SO-101
    #    audio_callback: energy gate + speech-burst accumulator + fast lane. ─
    audio_stream = None
    if not args.no_audio:
        try:
            import sounddevice as sd

            _SPEECH_RMS_GATE = 0.012
            _SPEECH_ONSET_BLOCKS = 5
            _BURST_END_SILENCE_BLOCKS = 4
            _hpf_enabled = args.hpf

            state = {
                "speech_consec": 0, "silence_consec": 0,
                "onset_armed": True, "burst": [],
                "passed": 0, "dropped": 0, "fast": 0, "log_t": time.time(),
            }

            def audio_callback(indata, frames, time_info, status):
                audio = indata[:, 0].copy()
                if robot.state == "idle":
                    predictor.add_audio(audio)
                else:
                    filtered = (engine._highpass_filter(audio)
                                if _hpf_enabled else audio)
                    rms = float(np.sqrt(np.mean(filtered ** 2)))
                    if rms >= _SPEECH_RMS_GATE:
                        state["passed"] += 1
                        state["speech_consec"] += 1
                        state["silence_consec"] = 0
                        state["burst"].append(filtered)
                        predictor.add_audio(filtered)
                        if (state["speech_consec"] == _SPEECH_ONSET_BLOCKS
                                and state["onset_armed"]):
                            try:
                                predictor.audio_buffer.clear()
                                predictor.add_audio(np.concatenate(state["burst"]))
                            except Exception:
                                pass
                            predictor.request_immediate_inference()
                            state["onset_armed"] = False
                            state["fast"] += 1
                    else:
                        state["dropped"] += 1
                        state["speech_consec"] = 0
                        state["silence_consec"] += 1
                        predictor.add_audio(filtered)
                        if state["silence_consec"] == _BURST_END_SILENCE_BLOCKS:
                            state["burst"].clear()
                            try:
                                predictor.audio_buffer.clear()
                            except Exception:
                                pass
                            state["onset_armed"] = True
                    now = time.time()
                    if now - state["log_t"] >= 10.0:
                        total = state["passed"] + state["dropped"]
                        pct = 100.0 * state["passed"] / total if total else 0.0
                        log.info("Audio gate (10s): passed=%d dropped=%d (%.0f%% "
                                 "speech) fast-lane=%d", state["passed"],
                                 state["dropped"], pct, state["fast"])
                        state["passed"] = state["dropped"] = state["fast"] = 0
                        state["log_t"] = now
                if not args.no_vad:
                    interrupt_system.on_audio(audio)

            audio_stream = sd.InputStream(
                samplerate=16000, channels=1, blocksize=1600,
                callback=audio_callback, device=args.mic_index,
            )
            audio_stream.start()
            log.info("Microphone started (device=%s)", args.mic_index)
        except Exception as e:
            log.warning("Microphone failed: %s (video-only mode)", e)

    predictor.start(num_workers=1)
    log.info("Prediction engine started (interval=%.2fs, 1 worker)",
             config.inference_interval)

    log.info("=" * 55)
    log.info("  G1 HRI BRAIN READY — speak a command to start")
    for task in tasks:
        kw = task.keywords[0] if task.keywords else task.name
        log.info("  '%s'  (keyword: %s)", task.lang, kw)
    log.info("  Say 'stop' to halt the robot")
    log.info("=" * 55)

    last_frame_time = 0.0
    frame_interval = 0.1
    _warned_no_frames = False
    _hud = "waiting"
    _qwen_seq = 0
    _qwen_last_t = 0.0
    _qwen_hz = 0.0

    stop_flag = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_flag.set())

    try:
        while not stop_flag.is_set():
            now = time.time()

            if not link.is_loop_alive() and not _warned_no_frames:
                log.warning("No state from G1 eval loop yet — is it running with "
                            "--hri_enable and reachable at %s:%d?",
                            args.g1_host, args.state_port)
                _warned_no_frames = True

            _qwen_state = "waiting" if robot.state == "idle" else "executing"
            predictor.set_robot_state(
                f"state={_qwen_state}, policy={robot.active_policy or 'none'}"
            )

            if now - last_frame_time >= frame_interval:
                frame = link.latest_frame()  # BGR np, cam_head (no crop — Qwen-only)
                if frame is not None:
                    predictor.add_frame(frame)
                    if args.view:
                        _show_avatar_view(frame, robot.state,
                                          robot.active_policy, _hud, link.frame_age())
                last_frame_time = now

            for pred in predictor.get_all_predictions():
                _qwen_seq += 1
                _np = time.time()
                if _qwen_last_t > 0:
                    dt = _np - _qwen_last_t
                    if dt > 0:
                        _qwen_hz = 0.8 * _qwen_hz + 0.2 / dt
                _qwen_last_t = _np
                _spoken = (getattr(pred, "spoken_command", "") or "").strip()
                _phase = (getattr(pred, "predicted_phase", "") or "unknown").strip()
                _hud = f"{pred.predicted_intent}/{_phase} {pred.confidence:.0%}"
                log.info("Qwen #%d: %s/%s(%s) conf=%.2f task_complete=%s spoken='%s' why=%s",
                         _qwen_seq, pred.predicted_intent, _phase,
                         pred.target_object or "-", pred.confidence,
                         pred.task_complete, _spoken, (pred.reason or "")[:60])
                router.handle_prediction({
                    "predicted_intent": pred.predicted_intent,
                    "predicted_phase": pred.predicted_phase,
                    "confidence": pred.confidence,
                    "target_object": pred.target_object,
                    "task_complete": pred.task_complete,
                    "reason": pred.reason,
                    "spoken_command": _spoken,
                })
                if metrics is not None:
                    raw = (pred.raw_response or "").strip()
                    metrics.log(
                        prediction=pred,
                        robot_state=_qwen_state,
                        active_policy=robot.active_policy,
                        parse_failed=(pred.predicted_intent == "unknown" and bool(raw)),
                    )
                telemetry.publish_prediction(
                    pred=pred, qwen_hz=_qwen_hz, groot_hz=robot.current_hz,
                    robot_state=_qwen_state, active_policy=robot.active_policy,
                )

            time.sleep(0.05)
    finally:
        log.info("Shutting down ...")
        if args.view:
            cv2.destroyAllWindows()
        predictor.stop()
        if audio_stream:
            audio_stream.stop()
        if metrics is not None:
            metrics.close()
        telemetry.close()
        robot.shutdown()
        log.info("G1 HRI brain stopped")


def main():
    p = argparse.ArgumentParser(description="Qwen3-Omni HRI brain for Unitree G1")
    p.add_argument("--tasks", default="tasks_g1.yaml")
    p.add_argument("--vllm-url", default="http://192.168.2.25:8000/v1")
    # G1 host binds both ZMQ ports (state PUB + command SUB); brain connects.
    p.add_argument("--g1-host", default="192.168.123.200",
                   help="Host of the G1 eval loop (binds the ZMQ ports). Often "
                        "the same as image_host.")
    p.add_argument("--state-port", type=int, default=5701,
                   help="Eval-loop PUB port for state + cam_head frames")
    p.add_argument("--cmd-port", type=int, default=5702,
                   help="Eval-loop SUB port for run/hold/switch/home commands")
    p.add_argument("--mic-index", type=int, default=None)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable energy VAD; Qwen WAITING prompt handles cold-start.")
    p.add_argument("--no-grounding", action="store_true")
    p.add_argument("--no-completion-check", action="store_true")
    p.add_argument("--no-home", action="store_true",
                   help="Disable return-to-home on completion (informational; the "
                        "bridge still issues 'home' unless this is set).")
    p.add_argument("--metrics", default=None)
    p.add_argument("--telemetry-port", type=int, default=None)
    p.add_argument("--hpf", action="store_true")
    p.add_argument("--view", action="store_true",
                   help="Show the G1 head-cam feed in a window (remote avatar "
                        "view): see through the robot while you command it. "
                        "Needs a GUI opencv (opencv-python, not -headless).")
    run(p.parse_args())


if __name__ == "__main__":
    main()
