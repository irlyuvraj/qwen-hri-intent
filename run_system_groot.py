#!/usr/bin/env python3
"""
Qwen3-Omni + GR00T N1.6 on SO101 — Unified System (2-PC Setup)

Runs on the robot laptop. vLLM (Qwen3-Omni) and GR00T policy server both run on s99.
Single entry point: camera + mic + Qwen prediction + GR00T control.

Usage:
    python run_system_groot.py \
        --vllm-url http://192.168.2.25:8000/v1 \
        --robot-port /dev/tty.usbmodem5AE70452961 \
        --camera-index 0 \
        --robot-camera-index 0 \
        --policy-host 192.168.2.25 \
        --policy-port 5555

Architecture vs. run_system.py (π0/ACT):
    - GR00T is language-conditioned. Swapping policies = swapping the lang string
      per observation. No subprocess restart, no checkpoint reload. Instant switch.
    - Robot control loop runs in-process in a background thread (not via
      lerobot.async_inference.robot_client subprocess).
    - Single GR00T checkpoint on s99 serves all tasks; task strings come from
      POLICY_TASKS below.
"""

import argparse
import concurrent.futures
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
from recorder import SystemRecorder

from gr00t.policy.server_client import PolicyClient

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import Robot, make_robot_from_config  # noqa: F401
from lerobot.robots import so_follower  # noqa: F401 — registers so101_follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("run_system_groot")


# Task registry is loaded from YAML at startup (see --tasks). It provides
# the lang instruction, target object, and voice keywords for every policy.
# Default location: ./tasks.yaml next to this file.

DEFAULT_TASKS_FILE = "tasks.yaml"


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
_CAMERA_KEYS = ["front"]  # SO-101 single front camera


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
    # Camera delivers 640x480 (the supported mode); GR00T was trained on
    # 640x360. Center-crop here so the policy sees the same field of view
    # it learned. Zero-copy numpy slice — no latency cost.
    model_obs = {
        "video":    {k: _crop_to_training(obs[k]) for k in _CAMERA_KEYS},
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
        robot_id: str,
        policy_host: str,
        policy_port: int,
        action_horizon: int = 8,
        control_hz: float = 30.0,
    ):
        self.tasks = tasks
        # Camera doesn't support 640x360 natively (rejected at connect-time
        # validation). 640x480 is a standard MJPG mode the camera accepts —
        # we then center-crop to the GR00T training resolution (640x360)
        # inside _obs_to_policy_inputs before sending the frame to the policy.
        cam_cfg = OpenCVCameraConfig(
            index_or_path=robot_camera_index, width=640, height=480, fps=30
        )
        self._robot_cfg = SOFollowerRobotConfig(
            port=robot_port,
            id=robot_id,
            cameras={"front": cam_cfg},
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
        # Dedicated thread for get_action calls so we can impose a timeout
        # without blocking the stop_event check or the action execution loop.
        self._infer_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="groot-infer"
        )
        self._current_hz: float = 0.0

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
        self._infer_executor.shutdown(wait=False)
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

    @property
    def current_hz(self) -> float:
        """Smoothed control-loop Hz (EMA over recent cycles). Thread-safe read."""
        return self._current_hz

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
                fut = self._infer_executor.submit(self.policy.get_action, obs_in)
                try:
                    action_chunk, _info = fut.result(timeout=5.0)
                except concurrent.futures.TimeoutError:
                    log.warning("GR00T get_action timed out (>5s) — skipping tick")
                    continue
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
                _cycle_s = t_end - t_cycle
                if _cycle_s > 0:
                    self._current_hz = 0.8 * self._current_hz + 0.2 / _cycle_s
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
        engine: "FastQwenInferenceEngine",
        withdraw_stop_count: int = 2,
        withdraw_min_runtime_s: float = 15.0,
        task_complete_count: int = 1,
        max_task_runtime_s: float = 40.0,
    ):
        self.robot = robot
        self.interrupt_system = interrupt_system
        self._engine = engine
        self._lock = threading.Lock()
        self._last_switch_time = 0.0
        self._switch_cooldown = 1.5  # tighter than π0 — GR00T hot-swap is instant

        # Primary stop: Qwen visual scene-completion signal.
        # Single confident task_complete=true prediction → stop (arm never withdraws
        # while GR00T loops, so we can't require arm-withdrawn as a condition).
        self._task_complete_count = task_complete_count
        self._task_complete_streak = 0

        # Secondary stop: motion-based withdrawal heuristic.
        # Stop after N consecutive `withdraw` predictions past the min runtime gate.
        self._withdraw_stop_count = withdraw_stop_count
        self._withdraw_min_runtime_s = withdraw_min_runtime_s
        self._withdraw_streak = 0

        # Safety stop: cap task runtime so the arm doesn't loop forever if
        # neither task_complete nor withdraw fires (e.g. ball placed but bowl
        # not visible, or GR00T loops back and re-picks the ball).
        self._max_task_runtime_s = max_task_runtime_s

        # Multimodal voice-command path: PolicyRouter reacts to Qwen's
        # continuous prediction reporting predicted_intent="new_command".
        # Streak filter prevents single-tick noise from starting a task.
        # Set to 1 to mirror VAD latency (act on first hit). Bump to 2 if
        # too many false triggers from idle chatter are observed.
        self._new_command_count  = 1
        self._new_command_streak = 0

        # Multimodal COLD-START via predicted_intent="command_<task>".
        # Replaces VAD entirely — Qwen's WAITING prompt classifies verbal
        # task commands directly into intent enum values (e.g.
        # "command_pick_pink_ball"). Streak ≥ 2 protects against single-frame
        # false positives from visible objects alone.
        self._cold_start_count  = 2
        self._cold_start_streak = 0
        self._cold_start_last_intent: Optional[str] = None

        # Multimodal STOP via predicted_intent="interrupt".
        # Qwen reliably emits this when it hears verbal "stop"/"wait"/"halt"
        # while watching a moving arm — the audio + visual evidence fuses
        # into the interrupt class. Streak ≥ 2 protects against a single
        # spurious frame (Qwen sometimes emits interrupt for fast motion alone).
        self._interrupt_count  = 1  # single clean detection sufficient with HPF audio
        self._interrupt_streak = 0

        # Multimodal SWITCH via predicted_intent="change_target".
        # Qwen emits this when the user redirects the arm trajectory OR
        # verbally commands a different object. Same streak protection.
        self._change_target_count  = 2
        self._change_target_streak = 0

        # After a multimodal STOP fires, ignore InterruptDetectionSystem's
        # follow-up callbacks for a few seconds. Otherwise the old visual_interrupt
        # / object_mismatch path immediately restarts the same task because
        # Qwen keeps emitting "interrupt" for a couple more frames.
        self._post_stop_quiet_until = 0.0
        self._post_stop_quiet_seconds = 3.0

        self._policy_start_time = 0.0
        # Tracks the last policy that ran so _resolve_relative works correctly
        # even when the robot is idle (active_policy is None after a stop).
        self._last_active_policy: Optional[str] = None

    _RELATIVE_WORDS = ("other", "not this", "different", "instead", "another",
                       "switch", "change", "else", "instead")

    @property
    def state(self):
        return self.robot.state

    @property
    def active_policy(self):
        return self.robot.active_policy

    def _resolve_relative(self, command: str) -> Optional[str]:
        """If the command contains a relative reference ('other', 'not this', etc.)
        return the task that is NOT currently active. Falls back to last-active policy
        when the robot is idle so 'other ball' after a stop still resolves correctly."""
        text = command.lower()
        if not any(w in text for w in self._RELATIVE_WORDS):
            return None
        # Use last known active policy when idle so we flip away from what just ran.
        current = self.active_policy or self._last_active_policy
        for name in self.robot.tasks.names():
            if name != current:
                log.info("Relative reference '%s' → opposite task: %s", command, name)
                return name
        return None

    _RESUME_WORDS = ("continue", "resume", "keep going", "go on", "carry on", "proceed")

    def handle_voice_command(self, command: str) -> dict:
        log.info("Voice command: '%s'", command)
        text = command.lower()
        # Resume: restart the last paused policy. Only meaningful when idle
        # AND we have a remembered policy to resume.
        if any(w in text for w in self._RESUME_WORDS):
            if self.state == "idle" and self._last_active_policy:
                log.info("Resume: restarting %s", self._last_active_policy)
                return self._execute_policy_action(
                    self._last_active_policy, f"resume: {command}"
                )
            log.info("Resume command ignored (state=%s, last=%s)",
                     self.state, self._last_active_policy)
            return {"action": "none", "reason": "no paused task to resume"}
        policy = self.robot.tasks.resolve(command) or self._resolve_relative(command)
        if not policy:
            log.warning("No policy match for: '%s'", command)
            return {"action": "none", "reason": f"No policy match for: {command}"}
        return self._execute_policy_action(policy, command)

    def handle_interrupt(self, event) -> dict:
        # Drop interrupt callbacks that arrive within the post-stop quiet
        # window — they're Qwen's lingering "interrupt" predictions from the
        # frames around when our multimodal stop already fired.
        if time.time() < self._post_stop_quiet_until:
            log.info("Ignoring interrupt within post-stop quiet window: %s", event.reason)
            return {"action": "quiet", "reason": "post_stop_cooldown"}

        log.info("Interrupt: reason=%s, object=%s, command='%s'",
                 event.reason, event.predicted_object, event.raw_command)

        if event.raw_command:
            policy = (self.robot.tasks.resolve(event.raw_command)
                      or self._resolve_relative(event.raw_command))
            if policy:
                # Pre-assert the correct target immediately so mismatch detection
                # uses the right object during the policy switch, not the stale
                # change_target string the interrupt system stored internally.
                task = self.robot.tasks.get(policy)
                self.interrupt_system.task_monitor.set_task(
                    command=event.raw_command,
                    intent="approach",
                    target_object=task.object,
                )
                return self._execute_policy_action(policy, event.raw_command)

        if event.predicted_object:
            policy = (self.robot.tasks.resolve(event.predicted_object)
                      or self._resolve_relative(event.predicted_object))
            if policy:
                task = self.robot.tasks.get(policy)
                self.interrupt_system.task_monitor.set_task(
                    command=f"object_mismatch: {event.predicted_object}",
                    intent="approach",
                    target_object=task.object,
                )
                return self._execute_policy_action(
                    policy, f"object_mismatch: {event.predicted_object}"
                )

        log.info("Stopping robot (no clear new policy)")
        return {"action": "stop", "result": self.robot.stop()}

    def handle_prediction(self, prediction: dict):
        intent  = prediction.get("predicted_intent", "")
        target  = (prediction.get("target_object") or "").strip()
        conf    = prediction.get("confidence", 0.0)
        spoken  = (prediction.get("spoken_command") or "").strip()

        # ── Multimodal voice-command path (Qwen continuous audio+video) ────
        # Replaces the energy-VAD + transcribe_audio detour. Qwen fills the
        # spoken_command field independently of predicted_intent — vision
        # and audio are now decoupled in the JSON schema, so the model no
        # longer has to choose between describing motion and reporting a
        # heard command.
        if spoken:
            spoken_l = spoken.lower()
            # Verbal stop → halt the running policy.
            if any(w in spoken_l for w in ("stop", "halt", "wait", "cancel", "abort")):
                if self.state == "running":
                    log.info("Multimodal verbal stop (Qwen): '%s' — stopping %s",
                             spoken, self.active_policy)
                    self._new_command_streak = 0
                    self.robot.stop()
                    self._post_stop_quiet_until = time.time() + self._post_stop_quiet_seconds
                    return
                # Idle — nothing to stop, fall through.
            # Verbal RESUME → restart the last paused policy when idle. Without
            # this branch, "continue" goes nowhere in --no-vad mode (the
            # _RESUME_WORDS check in handle_voice_command is only reachable
            # via the VAD callback, which is disabled).
            elif any(w in spoken_l for w in self._RESUME_WORDS):
                if self.state == "idle" and self._last_active_policy:
                    log.info("Multimodal RESUME (Qwen): '%s' → restarting %s",
                             spoken, self._last_active_policy)
                    self._new_command_streak = 0
                    self._execute_policy_action(
                        self._last_active_policy, f"resume(qwen): {spoken}"
                    )
                    return
                # Running or no paused task — ignore.
            else:
                policy = self.robot.tasks.resolve(spoken) or self._resolve_relative(spoken)
                if policy:
                    self._new_command_streak += 1
                    if self._new_command_streak >= self._new_command_count:
                        log.info("Multimodal voice command (Qwen): '%s' → policy=%s",
                                 spoken, policy)
                        self._new_command_streak = 0
                        self._execute_policy_action(policy, f"voice(qwen): {spoken}")
                        return
                else:
                    log.debug("spoken_command '%s' did not resolve to any policy", spoken)
                    self._new_command_streak = 0
        else:
            self._new_command_streak = 0

        # ── Multimodal COLD-START via predicted_intent="command_<task>" ─────
        # Qwen's WAITING prompt emits command_<task_name> when it hears a
        # verbal task command. Replaces VAD for cold-start initiation.
        if (intent.startswith("command_") and conf >= 0.85
                and self.state == "idle"
                and time.time() >= self._post_stop_quiet_until):
            policy = intent[len("command_"):]
            if policy in self.robot.tasks:
                if intent == self._cold_start_last_intent:
                    self._cold_start_streak += 1
                else:
                    self._cold_start_streak = 1
                    self._cold_start_last_intent = intent
                log.info("Multimodal cold-start streak %d/%d (conf=%.2f) — %s",
                         self._cold_start_streak, self._cold_start_count, conf, policy)
                if self._cold_start_streak >= self._cold_start_count:
                    log.info("Multimodal COLD-START (Qwen command_*) → %s", policy)
                    self._cold_start_streak = 0
                    self._cold_start_last_intent = None
                    self._execute_policy_action(policy, f"qwen-command: {intent}")
                    return
            else:
                log.warning("command_* intent '%s' did not match any task", intent)
                self._cold_start_streak = 0
                self._cold_start_last_intent = None
        elif not intent.startswith("command_"):
            self._cold_start_streak = 0
            self._cold_start_last_intent = None

        # ── Multimodal STOP via predicted_intent="interrupt" ────────────────
        # Qwen emits "interrupt" when it hears a verbal stop AND/OR sees the
        # arm freeze mid-motion. Use this as the actual stop trigger so the
        # system can stop without VAD when Qwen does multimodal fusion well.
        if intent == "interrupt" and conf >= 0.85 and self.state == "running":
            self._interrupt_streak += 1
            log.info("Multimodal interrupt streak %d/%d (conf=%.2f) — '%s'",
                     self._interrupt_streak, self._interrupt_count,
                     conf, self.active_policy)
            if self._interrupt_streak >= self._interrupt_count:
                log.info("Multimodal STOP (Qwen interrupt) — stopping %s", self.active_policy)
                self.robot.stop()
                self._interrupt_streak = 0
                self._change_target_streak = 0
                # Quiet the InterruptDetectionSystem for a few seconds so its
                # follow-up visual_interrupt / object_mismatch callbacks
                # (caused by Qwen still emitting "interrupt" intent for the
                # next 1-2 frames) don't immediately restart the same task.
                self._post_stop_quiet_until = time.time() + self._post_stop_quiet_seconds
                return
        elif intent != "interrupt":
            self._interrupt_streak = 0

        # ── Multimodal SWITCH via predicted_intent="change_target" ───────────
        # Qwen emits "change_target" when the human redirects (visually OR
        # verbally). Hot-swap the policy directly.
        if intent == "change_target" and target and conf >= 0.85 and self.state == "running":
            policy = self.robot.tasks.resolve(target) or self._resolve_relative(target)
            if policy and policy != self.active_policy:
                self._change_target_streak += 1
                log.info("Multimodal change_target streak %d/%d (conf=%.2f) — %s → %s",
                         self._change_target_streak, self._change_target_count,
                         conf, self.active_policy, policy)
                if self._change_target_streak >= self._change_target_count:
                    log.info("Multimodal SWITCH (Qwen change_target) → %s", policy)
                    self._change_target_streak = 0
                    self._interrupt_streak = 0
                    self._execute_policy_action(policy, f"qwen-change_target: {target}")
                    return
            else:
                self._change_target_streak = 0
        elif intent != "change_target":
            self._change_target_streak = 0

        if self.state != "running":
            self._withdraw_streak = 0
            self._task_complete_streak = 0
            return

        runtime = time.time() - self._policy_start_time

        # ── Safety: max task runtime ──────────────────────────────────────
        # GR00T has no internal "done" signal and will loop indefinitely.
        # If neither task_complete nor withdraw fires within the time limit,
        # stop anyway so the arm doesn't pick-and-place forever.
        if runtime >= self._max_task_runtime_s:
            log.info("Max task runtime %.0fs reached — stopping policy %s",
                     self._max_task_runtime_s, self.active_policy)
            self.robot.stop()
            self._withdraw_streak = 0
            self._task_complete_streak = 0
            return

        # ── Primary: Qwen visual scene-completion ────────────────────────
        if prediction.get("task_complete", False) and runtime >= self._withdraw_min_runtime_s:
            self._task_complete_streak += 1
            log.info("Qwen task_complete %d/%d (runtime=%.1fs) — '%s'",
                     self._task_complete_streak, self._task_complete_count,
                     runtime, self.active_policy)
            if self._task_complete_streak >= self._task_complete_count:
                log.info("Task complete (Qwen visual) — stopping policy %s", self.active_policy)
                self.robot.stop()
                self._task_complete_streak = 0
                self._withdraw_streak = 0
                return
        else:
            self._task_complete_streak = 0

        # ── Fallback: motion-based withdrawal heuristic ───────────────────
        if intent == "withdraw" and conf >= 0.6:
            if runtime < self._withdraw_min_runtime_s:
                log.info("Withdraw ignored — task running only %.1fs (<%ds)",
                         runtime, self._withdraw_min_runtime_s)
                return
            self._withdraw_streak += 1
            log.info("Withdraw streak %d/%d (conf=%.2f, runtime=%.1fs) — task may be completing",
                     self._withdraw_streak, self._withdraw_stop_count, conf, runtime)
            if self._withdraw_streak >= self._withdraw_stop_count:
                log.info("Task complete (withdraw heuristic) — stopping policy %s", self.active_policy)
                self.robot.stop()
                self._withdraw_streak = 0
        else:
            # Strict consecutive-only: ANY non-withdraw intent breaks the streak,
            # so isolated withdraw frames don't accumulate over minutes and
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
            self._withdraw_streak = 0
            self._task_complete_streak = 0
            self._policy_start_time = time.time()
            self._last_active_policy = policy
            log.info("Now executing: %s", policy)

            task = self.robot.tasks.get(policy)
            self._engine.active_task_lang = task.lang
            self.interrupt_system.task_monitor.set_task(
                command=reason,
                intent="approach",
                target_object=task.object,
            )
        else:
            log.error("Policy action failed: %s", result.get("error"))

        return {"action": "switch" if self.active_policy else "start",
                "policy": policy, "result": result}


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

# Resolution the robot was trained on. Frames that already match are returned
# as-is (zero-copy numpy slice). Larger frames are center-cropped, not scaled,
# so pixel density is preserved and GR00T sees the same field of view as training.
_TRAIN_W, _TRAIN_H = 640, 360


def _crop_to_training(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == _TRAIN_H and w == _TRAIN_W:
        return frame
    if h < _TRAIN_H or w < _TRAIN_W:
        log.warning("Frame %dx%d smaller than training size %dx%d — skipping crop",
                    w, h, _TRAIN_W, _TRAIN_H)
        return frame
    y = (h - _TRAIN_H) // 2
    x = (w - _TRAIN_W) // 2
    return frame[y : y + _TRAIN_H, x : x + _TRAIN_W]


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
        cold_start_choices=[(f"command_{t.name}", t.object) for t in tasks],
    )

    config = StreamConfig(
        inference_interval=0.5,
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        # motion_threshold=0 disables the optical-flow gate so Qwen polls
        # every inference_interval regardless of scene motion. Was 1.5 but
        # under-fired during execution — arm motion through the cropped
        # 640×360 frame downsampled to 160×120 was below threshold.
        # Continuous prediction is needed for visual intent + task_complete
        # + opportunistic multimodal interrupt during execution AND for
        # responsive cold-start in --no-vad demo mode.
        motion_threshold=0,
    )
    predictor = StreamingIntentPredictor(config)
    predictor.scheduler.set_inference_engine(engine)

    interrupt_system = InterruptDetectionSystem(
        consecutive_required=3,
        grace_period=4.0,
    )
    connect_to_predictor(interrupt_system, predictor)

    # Drop transcripts that aren't an actual command. Without this, garbage
    # like "Thank you." or "Shh." propagates into task_monitor.set_task and
    # corrupts the active-task state.
    _stop_words = {"stop", "no", "wait", "halt", "cancel", "abort"}

    _relative_words = ("other", "not this", "different", "instead", "another",
                       "switch", "change", "else")
    _resume_words = ("continue", "resume", "keep going", "go on", "carry on", "proceed")

    def _command_validator(text: str) -> bool:
        t = text.lower().strip()
        if any(w in t for w in _stop_words):
            return True
        if tasks.resolve(t) is not None:
            return True
        if any(w in t for w in _resume_words):
            return True
        # Allow relative-reference phrases through so "other ball", "not this one",
        # etc. reach handle_voice_command → _resolve_relative.
        return any(w in t for w in _relative_words)

    interrupt_system.command_validator = _command_validator

    router = PolicyRouter(robot, interrupt_system, engine)

    def on_interrupt(event):
        log.info("INTERRUPT: %s (object=%s, cmd='%s')",
                 event.reason, event.predicted_object, event.raw_command)
        result = router.handle_interrupt(event)
        log.info("Router action: %s", result)

    interrupt_system.on_interrupt(on_interrupt)
    interrupt_system.command_interface.on_new_task(
        lambda task: router.handle_voice_command(task)
    )

    # ── Recorder (optional) ───────────────────────────────────
    recorder: Optional[SystemRecorder] = None
    if args.record:
        try:
            recorder = SystemRecorder(
                output_path=args.record,
                fps=args.record_fps,
            )
            recorder.start()

            interrupt_system.on_interrupt(
                lambda ev: recorder.push_log(
                    f"[STOP] {ev.reason.value} conf={ev.confidence:.2f}"
                )
            )
            interrupt_system.command_interface.on_new_task(
                lambda task: recorder.push_log(f"[HEARD] {task}")
            )
        except Exception as e:
            log.error("Failed to start recorder: %s — continuing without recording", e)
            recorder = None

    # Same physical camera as the GR00T OpenCVCamera — both must agree on
    # device resolution or LeRobot's read_loop crashes when this one calls
    # cap.set() and reconfigures the underlying AVFoundation device.
    # 640x480 matches OpenCVCameraConfig; _crop_to_training() then center-
    # crops Qwen frames down to 640x360 (training resolution).
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        log.error("Failed to open camera %d", args.camera_index)
        sys.exit(1)
    log.info("Camera opened (index=%d)", args.camera_index)

    audio_stream = None
    if not args.no_audio:
        try:
            import sounddevice as sd

            _vad_enabled = not args.no_vad
            if not _vad_enabled:
                log.warning("VAD DISABLED — cold-start must come from Qwen "
                            "predicted_intent='command_*'")
            else:
                log.info("Inverse hybrid: VAD active during execution only; "
                         "cold-start handled by Qwen predicted_intent='command_*'")

            def audio_callback(indata, frames, time_info, status):
                audio = indata[:, 0].copy()
                # Send HPF-filtered audio to Qwen during execution to suppress
                # motor-noise rumble below 200Hz while keeping speech intact.
                # This lets Qwen detect stop/switch via interrupt/change_target
                # without encoder degeneration from raw motor noise.
                # Idle state gets clean unfiltered audio (no motor noise).
                if robot.state == "idle":
                    predictor.add_audio(audio)
                else:
                    predictor.add_audio(engine._highpass_filter(audio))
                # VAD always gets raw audio — it detects energy, not frequency.
                if _vad_enabled:
                    interrupt_system.on_audio(audio)
                if recorder is not None:
                    recorder.push_audio(audio)

            audio_stream = sd.InputStream(
                samplerate=16000, channels=1, blocksize=1600,
                callback=audio_callback, device=args.mic_index,
            )
            audio_stream.start()
            log.info("Microphone started (device=%s)", args.mic_index)
        except Exception as e:
            log.warning("Microphone failed: %s (video-only mode)", e)

    predictor.start(num_workers=1)
    log.info("Prediction engine started (interval=0.5s, 1 worker)")

    log.info("=" * 55)
    log.info("  SYSTEM READY — speak a command to start")
    for task in tasks:
        sample_kw = task.keywords[0] if task.keywords else task.name
        log.info("  '%s'  (matches keyword: %s)", task.lang, sample_kw)
    log.info("  Say 'stop' to halt the robot")
    log.info("=" * 55)

    last_frame_time = 0.0
    frame_interval = 0.1
    _last_pred_obj = None   # latest PredictionOutput for recorder overlay
    _last_robot_state = None
    _qwen_pred_seq = 0
    _qwen_last_pred_time = 0.0
    _qwen_hz_ema = 0.0

    # Graceful Ctrl-C across threads
    stop_flag = threading.Event()
    def _sigint(_sig, _frm):
        stop_flag.set()
    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop_flag.is_set():
            now = time.time()

            # State→prompt mapping:
            # - --no-vad mode: idle → WAITING prompt so Qwen can fire
            #   command_<task> for cold-start (research-demo path).
            # - default mode (VAD on): always use EXECUTING prompt — Qwen
            #   continuously classifies VISUAL intent (continue/approach/
            #   gesture/withdraw) and the reason field describes the scene.
            #   VAD handles cold-start, so the WAITING prompt's audio-only
            #   classification isn't needed.
            if args.no_vad and robot.state == "idle":
                _qwen_state = "waiting"
            else:
                _qwen_state = "executing"
            predictor.set_robot_state(
                f"state={_qwen_state}, policy={robot.active_policy or 'none'}"
            )

            if now - last_frame_time >= frame_interval:
                ret, frame = cap.read()
                if ret:
                    frame = _crop_to_training(frame)
                    predictor.add_frame(frame)
                    if recorder is not None:
                        recorder.push_frame(frame, _last_pred_obj)
                last_frame_time = now

            if recorder is not None:
                rs_state = f"{robot.state}:{robot.active_policy or 'idle'}"
                if rs_state != _last_robot_state:
                    recorder.push_log(
                        f"[robot] {robot.state} | policy={robot.active_policy or 'none'}"
                    )
                    _last_robot_state = rs_state

            for pred in predictor.get_all_predictions():
                _qwen_pred_seq += 1
                _now_pred = time.time()
                if _qwen_last_pred_time > 0:
                    _dt = _now_pred - _qwen_last_pred_time
                    if _dt > 0:
                        _qwen_hz_ema = 0.8 * _qwen_hz_ema + 0.2 / _dt
                _qwen_last_pred_time = _now_pred

                # Only update HUD with successful predictions. Unknowns from
                # parse failures (Qwen3-Omni audio encoder degeneration during
                # execution) shouldn't overwrite the last good intent — the
                # underlying model is still firing useful intents at a similar
                # cadence to before, but ~80% of execution-time inferences
                # return garbage. Keeping the last good display gives the
                # user (and the recording) continuous meaningful context.
                if pred.predicted_intent and pred.predicted_intent != "unknown":
                    _last_pred_obj = pred
                _spoken = (getattr(pred, "spoken_command", "") or "").strip()
                log.info("Qwen #%d: %s(%s) conf=%.2f task_complete=%s spoken='%s' why=%s",
                         _qwen_pred_seq,
                         pred.predicted_intent,
                         pred.target_object or "-",
                         pred.confidence,
                         pred.task_complete,
                         _spoken,
                         (pred.reason or "")[:60])
                router.handle_prediction({
                    "predicted_intent": pred.predicted_intent,
                    "confidence": pred.confidence,
                    "target_object": pred.target_object,
                    "task_complete": pred.task_complete,
                    "reason": pred.reason,
                    "spoken_command": _spoken,
                })
                if recorder is not None:
                    recorder.push_stats(_qwen_hz_ema, robot.current_hz, _qwen_pred_seq)

            time.sleep(0.05)

    finally:
        log.info("Shutting down ...")
        predictor.stop()
        cap.release()
        if audio_stream:
            audio_stream.stop()
        if recorder is not None:
            recorder.stop()
        robot.shutdown()
        log.info("System stopped")


def main():
    p = argparse.ArgumentParser(description="Qwen3-Omni + GR00T on SO101")
    p.add_argument("--tasks", default=DEFAULT_TASKS_FILE,
                   help="Path to task registry YAML (default: ./tasks.yaml)")
    p.add_argument("--vllm-url", default="http://192.168.2.25:8000/v1")
    p.add_argument("--robot-port", default="/dev/tty.usbmodem5AE70452961")
    p.add_argument("--robot-id", default="my_awesome_follower_arm",
                   help="Loads calibration from ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json")
    p.add_argument("--camera-index", type=int, default=0, help="Qwen observation camera")
    p.add_argument("--robot-camera-index", type=int, default=0, help="GR00T robot camera")
    p.add_argument("--mic-index", type=int, default=None)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable energy-VAD path. Cold-start must come from "
                        "Qwen's predicted_intent='command_*' (test mode).")
    p.add_argument("--policy-host", default="192.168.2.25", help="GR00T policy server host (s99)")
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--action-horizon", type=int, default=8)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--record", default=None,
                   help="Path to .mp4 — record session video (Qwen camera + mic + overlays)")
    p.add_argument("--record-fps", type=int, default=10,
                   help="Recording frame rate (default 10, matches Qwen capture cadence)")

    run(p.parse_args())


if __name__ == "__main__":
    main()
