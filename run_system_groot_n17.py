#!/usr/bin/env python3
"""
Qwen3-Omni + GR00T N1.7 on SO101 — Unified System (dual-camera variant)

Differences from run_system_groot.py (N1.6):
  - Dual camera observation: `front` (index 0) + `wrist` (index 2). The N1.7
    checkpoint was trained with both and expects both in every observation.
  - Default tasks file is tasks_n17.yaml (matches the new training lang strings).
  - Everything else — hot-swap, withdraw-stop, PolicyRouter, interrupt handling —
    is identical to the N1.6 version.

Server side (run first on s99):
  cd ~/Isaac-GR00T-N17 && \
  CUDA_VISIBLE_DEVICES=0 uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /home/yuvraj/groot_finetune/multitask-n17/checkpoint-20000 \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda:0 --host 0.0.0.0 --port 5555

Client side (this script):
    python run_system_groot_n17.py \
        --vllm-url http://192.168.2.25:8000/v1 \
        --robot-port /dev/tty.usbmodem5AE70452961 \
        --camera-index 0 \
        --robot-camera-index 0 \
        --wrist-camera-index 2 \
        --policy-host 192.168.2.25 \
        --policy-port 5555
"""

import argparse
import logging
import signal
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from qwen_inference_engine import FastQwenInferenceEngine
from streaming_intent_predictor import StreamingIntentPredictor, StreamConfig
from interrupt_detection_system import (
    InterruptDetectionSystem, connect_to_predictor
)
from task_registry import TaskRegistry

from gr00t.policy.server_client import PolicyClient

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import Robot, make_robot_from_config  # noqa: F401
from lerobot.robots import so_follower  # noqa: F401 — registers so101_follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("run_system_groot_n17")


# Task registry is loaded from YAML at startup (see --tasks). It provides
# the lang instruction, target object, and voice keywords for every policy.
# Default location: ./tasks_n17.yaml next to this file.

DEFAULT_TASKS_FILE = "tasks_n17.yaml"


# ═══════════════════════════════════════════════════════════════
# SO101 Adapter (obs → GR00T VLA input; action chunk → motor commands)
# ═══════════════════════════════════════════════════════════════

_ROBOT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
_CAMERA_KEYS = ["front", "wrist"]  # N1.7 dual-camera setup


def _add_batch_time_dims(obs: Dict) -> Dict:
    """Wrap arrays with (B=1, T=1) dims as GR00T expects."""
    def _once(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = v[np.newaxis, ...]
            elif isinstance(v, dict):
                out[k] = _once(v)
            else:
                out[k] = [v]
        return out
    return _once(_once(obs))


def _obs_to_policy_inputs(obs: Dict[str, Any], lang: str) -> Dict:
    state = np.array([obs[k] for k in _ROBOT_STATE_KEYS], dtype=np.float32)
    model_obs = {
        "video":    {k: obs[k] for k in _CAMERA_KEYS},
        "state":    {"single_arm": state[:5], "gripper": state[5:6]},
        "language": {"annotation.human.task_description": lang},
    }
    return _add_batch_time_dims(model_obs)


def _decode_chunk(chunk: Dict, t: int) -> Dict[str, float]:
    single_arm = chunk["single_arm"][0][t]  # (5,)
    gripper = chunk["gripper"][0][t]        # (1,)
    full = np.concatenate([single_arm, gripper], axis=0)
    return {name: float(full[i]) for i, name in enumerate(_ROBOT_STATE_KEYS)}


# ═══════════════════════════════════════════════════════════════
# GR00T Robot Controller (in-process thread loop)
# ═══════════════════════════════════════════════════════════════

class GrootRobotController:
    """
    Owns the local SO-101 arm and the GR00T PolicyClient.

    start_policy(name)   — begin the control loop with task string for `name`
    switch_policy(name)  — hot-swap the lang string (no thread restart)
    stop()               — halt the loop (keeps arm connected)
    shutdown()           — stop + disconnect arm
    """

    def __init__(
        self,
        tasks: TaskRegistry,
        robot_port: str,
        robot_camera_index: int,
        wrist_camera_index: int,
        robot_id: str,
        policy_host: str,
        policy_port: int,
        action_horizon: int = 8,
        control_hz: float = 30.0,
    ):
        self.tasks = tasks
        front_cfg = OpenCVCameraConfig(
            index_or_path=robot_camera_index, width=640, height=480, fps=30
        )
        wrist_cfg = OpenCVCameraConfig(
            index_or_path=wrist_camera_index, width=640, height=480, fps=30
        )
        self._robot_cfg = SOFollowerRobotConfig(
            port=robot_port,
            id=robot_id,
            cameras={"front": front_cfg, "wrist": wrist_cfg},
        )
        self.robot: Robot = make_robot_from_config(self._robot_cfg)
        self.policy = PolicyClient(host=policy_host, port=policy_port)

        self.action_horizon = action_horizon
        self._dt = 1.0 / control_hz

        self._lang: Optional[str] = None
        self._active_policy: Optional[str] = None
        self._state = "idle"  # idle | running | stopping
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._connected = False

    # ── lifecycle ─────────────────────────────────────────────

    def connect(self):
        if self._connected:
            return
        log.info("Connecting to SO101 arm ...")
        self.robot.connect()
        self._connected = True
        log.info("Arm connected. Pinging GR00T policy server ...")
        if not self.policy.ping():
            log.warning("GR00T policy server did not respond to ping")
        else:
            log.info("GR00T policy server reachable")

    def shutdown(self):
        self.stop()
        if self._connected:
            try:
                self.robot.disconnect()
            except Exception as e:
                log.warning("robot.disconnect() failed: %s", e)
            self._connected = False

    # ── public API used by PolicyRouter ───────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def active_policy(self) -> Optional[str]:
        return self._active_policy

    def start_policy(self, policy_name: str) -> dict:
        with self._lock:
            if policy_name not in self.tasks:
                return {"ok": False, "error": f"Unknown policy: {policy_name}",
                        "available": self.tasks.names()}
            if self._state == "running":
                return {"ok": False, "error": "Already running. Call stop() or switch() first."}

            self._lang = self.tasks.get(policy_name).lang
            self._active_policy = policy_name
            self._stop_event.clear()
            self._state = "running"
            self._thread = threading.Thread(
                target=self._control_loop, name="groot-control", daemon=True
            )
            self._thread.start()

        log.info("Policy started: %s", policy_name)
        return {"ok": True, "policy": policy_name, "state": "running"}

    def switch_policy(self, policy_name: str) -> dict:
        """Hot-swap the lang string — GR00T picks it up on the next obs."""
        if policy_name not in self.tasks:
            return {"ok": False, "error": f"Unknown policy: {policy_name}"}

        with self._lock:
            if self._state != "running":
                # Not running — fall through to a normal start
                pass
            else:
                self._lang = self.tasks.get(policy_name).lang
                prev = self._active_policy
                self._active_policy = policy_name
                log.info("Hot-swapped policy: %s -> %s", prev, policy_name)
                return {"ok": True, "stopped": prev, "started": policy_name,
                        "state": "running", "hot_swap": True}

        return self.start_policy(policy_name)

    def stop(self) -> dict:
        with self._lock:
            if self._state != "running":
                return {"ok": True, "state": self._state, "msg": "Nothing to stop"}
            self._state = "stopping"
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        with self._lock:
            prev = self._active_policy
            self._active_policy = None
            self._lang = None
            self._state = "idle"
        log.info("Policy stopped: %s", prev)
        return {"ok": True, "stopped": prev, "state": "idle"}

    # ── internal ──────────────────────────────────────────────

    def _control_loop(self):
        log.info("GR00T control loop started (horizon=%d, %.0fHz)",
                 self.action_horizon, 1.0 / self._dt)
        tick = 0
        try:
            while not self._stop_event.is_set():
                t_cycle = time.time()
                obs = self.robot.get_observation()
                t_after_obs = time.time()

                with self._lock:
                    lang = self._lang
                if lang is None:
                    break

                obs_in = _obs_to_policy_inputs(obs, lang)
                t_before_inf = time.time()
                action_chunk, _info = self.policy.get_action(obs_in)
                t_after_inf = time.time()

                any_key = next(iter(action_chunk.keys()))
                horizon = min(self.action_horizon, action_chunk[any_key].shape[1])

                for t in range(horizon):
                    if self._stop_event.is_set():
                        break
                    tic = time.time()
                    action = _decode_chunk(action_chunk, t)
                    self.robot.send_action(action)
                    dt = time.time() - tic
                    if dt < self._dt:
                        time.sleep(self._dt - dt)

                t_end = time.time()
                tick += 1
                if tick % 5 == 0:
                    total = t_end - t_cycle
                    log.info(
                        "tick %d: obs=%dms inf=%dms motion=%dms total=%dms rate=%.1fHz",
                        tick,
                        int((t_after_obs - t_cycle) * 1000),
                        int((t_after_inf - t_before_inf) * 1000),
                        int((t_end - t_after_inf) * 1000),
                        int(total * 1000),
                        (1.0 / total) if total > 0 else 0.0,
                    )
        except Exception:
            log.exception("GR00T control loop crashed")
        finally:
            log.info("GR00T control loop exited")


# ═══════════════════════════════════════════════════════════════
# Policy Router (same behavior as run_system.py — swapped RobotController)
# ═══════════════════════════════════════════════════════════════

class PolicyRouter:
    def __init__(
        self,
        robot: GrootRobotController,
        interrupt_system: InterruptDetectionSystem,
        withdraw_stop_count: int = 2,
        withdraw_min_runtime_s: float = 15.0,
    ):
        self.robot = robot
        self.interrupt_system = interrupt_system
        self._lock = threading.Lock()
        self._last_switch_time = 0.0
        self._switch_cooldown = 1.5  # tighter than π0 — GR00T hot-swap is instant

        # Task-completion detection: stop the robot after N consecutive `withdraw`
        # predictions. GR00T has no internal "done" signal, so without this the
        # policy keeps trying to pick up an object that is no longer there.
        # Two safety gates keep the counter from tripping during startup:
        #   - `withdraw_min_runtime_s`: a task must run at least this long before
        #     auto-complete can fire. Covers the robot's initial reach/approach,
        #     where Qwen sometimes mislabels an arm pause as "withdraw".
        #   - Strict consecutive requirement: any non-withdraw prediction resets
        #     the streak (not just approach/gesture). Only a true sustained
        #     withdraw run will count.
        self._withdraw_stop_count = withdraw_stop_count
        self._withdraw_min_runtime_s = withdraw_min_runtime_s
        self._withdraw_streak = 0
        self._policy_start_time = 0.0

    @property
    def state(self):
        return self.robot.state

    @property
    def active_policy(self):
        return self.robot.active_policy

    def handle_voice_command(self, command: str) -> dict:
        log.info("Voice command: '%s'", command)
        policy = self.robot.tasks.resolve(command)
        if not policy:
            log.warning("No policy match for: '%s'", command)
            return {"action": "none", "reason": f"No policy match for: {command}"}
        return self._execute_policy_action(policy, command)

    def handle_interrupt(self, event) -> dict:
        log.info("Interrupt: reason=%s, object=%s, command='%s'",
                 event.reason, event.predicted_object, event.raw_command)

        if event.raw_command:
            policy = self.robot.tasks.resolve(event.raw_command)
            if policy:
                return self._execute_policy_action(policy, event.raw_command)

        if event.predicted_object:
            policy = self.robot.tasks.resolve(event.predicted_object)
            if policy:
                return self._execute_policy_action(
                    policy, f"object_mismatch: {event.predicted_object}"
                )

        log.info("Stopping robot (no clear new policy)")
        return {"action": "stop", "result": self.robot.stop()}

    def handle_prediction(self, prediction: dict):
        if self.state != "running":
            self._withdraw_streak = 0
            return

        intent = prediction.get("predicted_intent", "")
        conf = prediction.get("confidence", 0.0)

        if intent == "withdraw" and conf >= 0.6:
            # Minimum-runtime gate: Qwen often labels the robot's opening
            # approach pose as "withdraw". Don't let the counter start until
            # the task has had time to actually execute.
            runtime = time.time() - self._policy_start_time
            if runtime < self._withdraw_min_runtime_s:
                log.info("Withdraw ignored — task running only %.1fs (<%ds)",
                         runtime, self._withdraw_min_runtime_s)
                return
            self._withdraw_streak += 1
            log.info("Withdraw streak %d/%d (conf=%.2f, runtime=%.1fs) — task may be completing",
                     self._withdraw_streak, self._withdraw_stop_count, conf, runtime)
            if self._withdraw_streak >= self._withdraw_stop_count:
                log.info("Task complete — stopping policy %s", self.active_policy)
                self.robot.stop()
                self._withdraw_streak = 0
        else:
            # Strict consecutive-only: ANY non-withdraw intent breaks the streak.
            # Without this, isolated withdraw frames accumulate over minutes and
            # eventually false-trigger the auto-stop.
            if self._withdraw_streak > 0:
                log.debug("Withdraw streak reset by intent=%s", intent)
            self._withdraw_streak = 0

    def _execute_policy_action(self, policy: str, reason: str) -> dict:
        # No-op: re-issuing the active policy just churns the grace period and
        # resets streaks. Skip cleanly so a duplicate "pick up the X" command
        # doesn't disturb a healthy execution.
        if self.state == "running" and policy == self.active_policy:
            log.info("Skipping no-op hot-swap: %s already active", policy)
            return {"action": "noop", "policy": policy, "state": "running"}

        with self._lock:
            now = time.time()
            if now - self._last_switch_time < self._switch_cooldown:
                remaining = self._switch_cooldown - (now - self._last_switch_time)
                log.info("Switch cooldown (%.1fs remaining)", remaining)
                return {"action": "cooldown", "remaining_s": remaining}

        log.info("Policy action: %s -> %s (reason: %s)",
                 self.active_policy, policy, reason)

        if self.state == "running" and self.active_policy:
            result = self.robot.switch_policy(policy)
        else:
            result = self.robot.start_policy(policy)

        if result.get("ok"):
            with self._lock:
                self._last_switch_time = time.time()
            # Reset withdraw streak so the completion counter from the previous
            # task doesn't falsely kill the new one. Without this, a leftover
            # streak=1 from "pink" will fire streak=2 a few ticks into "yellow"
            # and auto-stop yellow before it has a chance to run.
            self._withdraw_streak = 0
            self._policy_start_time = time.time()
            log.info("Now executing: %s", policy)

            task_obj = self.robot.tasks.get(policy).object
            self.interrupt_system.task_monitor.set_task(
                command=reason,
                intent="approach",
                target_object=task_obj,
            )
        else:
            log.error("Policy action failed: %s", result.get("error"))

        return {"action": "switch" if self.active_policy else "start",
                "policy": policy, "result": result}


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def _check_policy_server_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run(args):
    if not _check_policy_server_reachable(args.policy_host, args.policy_port):
        log.error("GR00T policy server not reachable at %s:%d",
                  args.policy_host, args.policy_port)
        log.error("Launch it on s99 first, e.g.:")
        log.error("  ssh yuvraj@192.168.2.25 'cd ~/Isaac-GR00T && \\")
        log.error("    .venv/bin/python gr00t/eval/run_gr00t_server.py \\")
        log.error("      --embodiment_tag NEW_EMBODIMENT \\")
        log.error("      --model_path <checkpoint> --device cuda:0 \\")
        log.error("      --host 0.0.0.0 --port 5555 --strict'")
        sys.exit(1)
    log.info("GR00T policy server reachable: %s:%d", args.policy_host, args.policy_port)

    tasks = TaskRegistry.from_yaml(args.tasks)
    log.info("Loaded %d task(s) from %s: %s",
             len(tasks), args.tasks, ", ".join(tasks.names()))

    robot = GrootRobotController(
        tasks=tasks,
        robot_port=args.robot_port,
        robot_camera_index=args.robot_camera_index,
        wrist_camera_index=args.wrist_camera_index,
        robot_id=args.robot_id,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        action_horizon=args.action_horizon,
        control_hz=args.control_hz,
    )
    robot.connect()

    engine = FastQwenInferenceEngine(
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        scene_objects=tasks.objects(),
    )

    config = StreamConfig(
        inference_interval=0.5,
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        motion_threshold=1.5,
    )
    predictor = StreamingIntentPredictor(config)
    predictor.scheduler.set_inference_engine(engine)

    interrupt_system = InterruptDetectionSystem(
        consecutive_required=3,
        grace_period=4.0,
    )
    connect_to_predictor(interrupt_system, predictor)

    # Drop transcripts that aren't an actual command. Without this, garbage
    # like "Thank you." or Qwen's "I'm sorry, but I can't provide..." propagates
    # into task_monitor.set_task and corrupts the active-task state.
    _stop_words = {"stop", "no", "wait", "halt", "cancel", "abort"}

    def _command_validator(text: str) -> bool:
        t = text.lower().strip()
        if any(w in t for w in _stop_words):
            return True
        return tasks.resolve(t) is not None

    interrupt_system.command_validator = _command_validator

    router = PolicyRouter(robot, interrupt_system)

    def on_interrupt(event):
        log.info("INTERRUPT: %s (object=%s, cmd='%s')",
                 event.reason, event.predicted_object, event.raw_command)
        result = router.handle_interrupt(event)
        log.info("Router action: %s", result)

    interrupt_system.on_interrupt(on_interrupt)
    interrupt_system.command_interface.on_new_task(
        lambda task: router.handle_voice_command(task)
    )

    # The robot opens its own camera via LeRobot. If same index as Qwen's,
    # we pause Qwen's capture while the robot loop is running.
    use_separate_cameras = args.camera_index != args.robot_camera_index
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        log.error("Failed to open Qwen camera %d", args.camera_index)
        sys.exit(1)
    log.info("Qwen camera opened (index=%d)", args.camera_index)
    if use_separate_cameras:
        log.info("Robot uses separate camera %d", args.robot_camera_index)
    else:
        log.info("Shared camera — Qwen capture will pause during robot execution")

    audio_stream = None
    if not args.no_audio:
        try:
            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                audio = indata[:, 0].copy()
                predictor.add_audio(audio)
                interrupt_system.on_audio(audio)

            audio_stream = sd.InputStream(
                samplerate=16000, channels=1, blocksize=1600,
                callback=audio_callback, device=args.mic_index,
            )
            audio_stream.start()
            log.info("Microphone started (device=%s)", args.mic_index)
        except Exception as e:
            log.warning("Microphone failed: %s (video-only mode)", e)

    predictor.start(num_workers=2)
    log.info("Prediction engine started (interval=0.5s)")

    log.info("=" * 55)
    log.info("  SYSTEM READY — speak a command to start")
    for task in tasks:
        sample_kw = task.keywords[0] if task.keywords else task.name
        log.info("  '%s'  (matches keyword: %s)", task.lang, sample_kw)
    log.info("  Say 'stop' to halt the robot")
    log.info("=" * 55)

    last_frame_time = 0.0
    frame_interval = 0.1
    camera_paused = False

    # Graceful Ctrl-C across threads
    stop_flag = threading.Event()
    def _sigint(_sig, _frm):
        stop_flag.set()
    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop_flag.is_set():
            now = time.time()

            predictor.set_robot_state(
                f"state={robot.state}, policy={robot.active_policy or 'none'}"
            )

            if not use_separate_cameras:
                if robot.state == "running" and not camera_paused:
                    cap.release()
                    camera_paused = True
                    log.info("Qwen camera paused (robot using shared camera)")
                elif robot.state != "running" and camera_paused:
                    cap = cv2.VideoCapture(args.camera_index)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera_paused = False
                    log.info("Qwen camera resumed")

            if not camera_paused and now - last_frame_time >= frame_interval:
                ret, frame = cap.read()
                if ret:
                    predictor.add_frame(frame)
                last_frame_time = now

            for pred in predictor.get_all_predictions():
                router.handle_prediction({
                    "predicted_intent": pred.predicted_intent,
                    "confidence": pred.confidence,
                    "target_object": pred.target_object,
                    "reason": pred.reason,
                })

            time.sleep(0.05)

    finally:
        log.info("Shutting down ...")
        predictor.stop()
        if not camera_paused:
            cap.release()
        if audio_stream:
            audio_stream.stop()
        robot.shutdown()
        log.info("System stopped")


def main():
    p = argparse.ArgumentParser(description="Qwen3-Omni + GR00T N1.7 (dual-cam) on SO101")
    p.add_argument("--tasks", default=DEFAULT_TASKS_FILE,
                   help="Path to task registry YAML (default: ./tasks.yaml)")
    p.add_argument("--vllm-url", default="http://192.168.2.25:8000/v1")
    p.add_argument("--robot-port", default="/dev/tty.usbmodem5AE70452961")
    p.add_argument("--robot-id", default="my_awesome_follower_arm",
                   help="Loads calibration from ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json")
    p.add_argument("--camera-index", type=int, default=0, help="Qwen observation camera")
    p.add_argument("--robot-camera-index", type=int, default=0, help="GR00T front camera (robot POV)")
    p.add_argument("--wrist-camera-index", type=int, default=2, help="GR00T wrist camera (N1.7 dual-cam)")
    p.add_argument("--mic-index", type=int, default=None)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--policy-host", default="192.168.2.25", help="GR00T policy server host (s99)")
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--action-horizon", type=int, default=8)
    p.add_argument("--control-hz", type=float, default=30.0)

    run(p.parse_args())


if __name__ == "__main__":
    main()
