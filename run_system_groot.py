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
from metrics_logger import MetricsLogger
import telemetry_publisher as telemetry

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

# Voice command vocabulary used by both PolicyRouter (handle_voice_command,
# handle_prediction) and the run() command_validator. Keep in one place.
RELATIVE_WORDS = ("other", "not this", "different", "instead", "another",
                  "switch", "change", "else")
RESUME_WORDS = ("continue", "resume", "keep going", "go on", "carry on", "proceed")
STOP_WORDS = ("stop", "halt", "wait", "cancel", "abort")


# ═══════════════════════════════════════════════════════════════
# RobotProfile — robot-specific adapter configuration
#
# Everything above and below this block (Qwen perception, PolicyRouter,
# TaskRegistry, MetricsLogger) is robot-agnostic. Only this dataclass and
# the GrootRobotController.__init__ touch robot-specific drivers.
#
# To port to a new robot (e.g. Unitree G1):
#   1. Define G1_PROFILE = RobotProfile(state_keys=(...), ...)
#   2. Write a LeRobot-compatible driver (or use the existing Unitree one)
#   3. Replace SOFollowerRobotConfig in GrootRobotController.__init__
# ═══════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dataclass, field as _field


@_dataclass
class RobotProfile:
    """Robot-specific kinematic + sensor configuration.

    Decouples the perception/decision stack from hardware. The only places
    that read this object are _obs_to_policy_inputs, _decode_chunk,
    _crop_to_training, and GrootRobotController — everything else is generic.

    To port to a new robot, define a new profile (e.g. G1_PROFILE) with the
    correct state_keys, camera_keys, arm_dof, resolutions, and a
    release_overrides dict that describes what "let go of held object" means
    for that embodiment.
    """
    state_keys: tuple        # ordered joint-position keys in robot obs dict
    camera_keys: tuple       # camera names in robot obs dict
    arm_dof: int             # non-gripper joints (state[:arm_dof] → single_arm)
    native_w: int            # camera capture width (must match driver config)
    native_h: int            # camera capture height (must match driver config)
    train_w: int             # GR00T training frame width (crop target)
    train_h: int             # GR00T training frame height (crop target)
    # Joint overrides applied during hard-switch release. Keys are state_keys
    # entries; values are target positions. The hard-switch action is built by
    # copying current obs values for all state_keys, then applying these
    # overrides. SO-101 → open the single gripper. G1 → open both hands /
    # extend specific finger joints. Empty dict → "freeze in place" (no
    # release happens, just stop+restart).
    release_overrides: dict = _field(default_factory=dict)


# SO-101 follower arm — 5-DOF arm + 1 gripper, single front camera.
# Camera delivers 640×480 natively (640×360 is unsupported by AVFoundation);
# frames are center-cropped to the GR00T training resolution inside
# _obs_to_policy_inputs / _crop_to_training.
SO101_PROFILE = RobotProfile(
    state_keys=(
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ),
    camera_keys=("front",),
    arm_dof=5,
    native_w=640,
    native_h=480,
    train_w=640,
    train_h=360,
    release_overrides={"gripper.pos": 50.0},  # open gripper, calibration-specific
)


# Example shape for a future G1 profile (commented — write the driver first):
#
# G1_PROFILE = RobotProfile(
#     state_keys=("left_shoulder.pos", ..., "left_hand_thumb.pos", ...),
#     camera_keys=("head_front",),
#     arm_dof=14,                     # both arms, no fingers
#     native_w=..., native_h=...,
#     train_w=..., train_h=...,
#     release_overrides={
#         "left_hand_thumb.pos": 0.0,  # open left hand
#         "left_hand_index.pos": 0.0,
#         "right_hand_thumb.pos": 0.0,  # open right hand
#         "right_hand_index.pos": 0.0,
#     },
# )


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


def _obs_to_policy_inputs(obs: Dict[str, Any], lang: str, profile: RobotProfile) -> Dict:
    state = np.array([obs[k] for k in profile.state_keys], dtype=np.float32)
    model_obs = {
        "video":    {k: _crop_to_training(obs[k], profile) for k in profile.camera_keys},
        "state":    {"single_arm": state[:profile.arm_dof], "gripper": state[profile.arm_dof:]},
        "language": {"annotation.human.task_description": lang},
    }
    return _add_batch_time_dims(model_obs)


def _decode_chunk(chunk: Dict, t: int, profile: RobotProfile) -> Dict[str, float]:
    single_arm = chunk["single_arm"][0][t]  # (arm_dof,)
    gripper = chunk["gripper"][0][t]        # (1,)
    full = np.concatenate([single_arm, gripper], axis=0)
    return {name: float(full[i]) for i, name in enumerate(profile.state_keys)}


# ═══════════════════════════════════════════════════════════════
# GR00T Robot Controller (in-process thread loop)
# ═══════════════════════════════════════════════════════════════

class GrootRobotController:
    """
    Owns the local robot arm and the GR00T PolicyClient.

    The robot-specific parts are confined to __init__ (driver config) and
    the profile passed to _obs_to_policy_inputs / _decode_chunk. Everything
    else is generic across robot embodiments.

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
        profile: RobotProfile = SO101_PROFILE,
        action_horizon: int = 8,
        control_hz: float = 30.0,
        hard_switch: bool = True,
        home_on_complete: bool = True,
    ):
        self.tasks = tasks
        self.profile = profile
        # Return-to-home: after a task-completion stop, smoothly drive the arm
        # back to the pose captured at connect() (the "initial position"). The
        # pose is auto-captured, so this is robot-agnostic — it works unchanged
        # on any embodiment without per-robot tuning. Disable with --no-home.
        self._home_on_complete = home_on_complete
        self._home_pose: Optional[Dict[str, float]] = None
        # Hard switch: on policy switch, stop the loop, apply profile.release_overrides
        # (e.g. open gripper for SO-101) to drop any held object, then start
        # the new policy from a clean state. Avoids GR00T's "finish current task
        # before switching" behavior caused by datasets that only contain
        # complete trajectories. The release behavior itself is robot-agnostic
        # — what counts as "release" lives in the RobotProfile, not here.
        # Set hard_switch=False to revert to legacy hot-swap (lang change only).
        self._hard_switch = hard_switch
        # Native camera resolution comes from the profile. The camera must be
        # opened at profile.native_w × profile.native_h; frames are then
        # center-cropped to profile.train_w × profile.train_h inside
        # _obs_to_policy_inputs before the policy sees them.
        cam_cfg = OpenCVCameraConfig(
            index_or_path=robot_camera_index,
            width=profile.native_w, height=profile.native_h, fps=30
        )
        self._robot_cfg = SOFollowerRobotConfig(
            port=robot_port,
            id=robot_id,
            cameras={profile.camera_keys[0]: cam_cfg},
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
        log.info("Connecting to robot arm (%s) ...", type(self._robot_cfg).__name__)
        self.robot.connect()
        self._connected = True
        # Capture the startup pose as "home" for return-to-home on completion.
        # Whatever joints the arm rests in at connect become the target — no
        # hardcoded angles, automatically correct for any embodiment.
        if self._home_on_complete:
            try:
                obs = self.robot.get_observation()
                self._home_pose = {k: float(obs[k]) for k in self.profile.state_keys}
                log.info("Captured home pose (%d joints) for return-on-complete",
                         len(self._home_pose))
            except Exception as e:
                log.warning("Could not capture home pose: %s — return-to-home off", e)
                self._home_pose = None
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
        """Switch to a new policy.

        Default behavior (self._hard_switch=True): stop current loop, open
        gripper to release any held object, then start the new policy from a
        clean state. This avoids GR00T's "finish current task before
        switching" behavior — a dataset limitation, not a code bug. The
        policy was trained on full task trajectories and has not seen
        mid-task abort or state-conditional starts, so a clean restart from
        a known pose gives it the best chance.

        Legacy behavior (self._hard_switch=False): hot-swap the lang string
        only — GR00T picks it up on the next obs but typically completes the
        original action first.
        """
        if policy_name not in self.tasks:
            return {"ok": False, "error": f"Unknown policy: {policy_name}"}

        # Legacy hot-swap path (revert toggle)
        if not self._hard_switch:
            with self._lock:
                if self._state == "running":
                    self._lang = self.tasks.get(policy_name).lang
                    prev = self._active_policy
                    self._active_policy = policy_name
                    log.info("Hot-swapped policy: %s -> %s", prev, policy_name)
                    return {"ok": True, "stopped": prev, "started": policy_name,
                            "state": "running", "hot_swap": True}
            return self.start_policy(policy_name)

        # Hard-switch path
        if self._state != "running":
            # Not running — fall through to a clean start
            return self.start_policy(policy_name)

        prev = self._active_policy
        log.info("Hard switch: %s -> %s (stop, release, restart)",
                 prev, policy_name)

        # 1. Stop the current control loop
        self.stop()

        # 2. Release any held object — apply profile.release_overrides to the
        # current pose. SO-101: opens the single gripper. G1: would open both
        # hands. Empty overrides → freeze in place (no release).
        if self.profile.release_overrides:
            try:
                obs = self.robot.get_observation()
                release_action = {k: float(obs[k]) for k in self.profile.state_keys}
                release_action.update(self.profile.release_overrides)
                for _ in range(8):
                    self.robot.send_action(release_action)
                    time.sleep(0.04)
            except Exception as e:
                log.warning("Release-action failed during hard switch: %s", e)

        # 3. Start the new policy from clean state
        result = self.start_policy(policy_name)
        result["hot_swap"] = False
        result["hard_switch"] = True
        result["dropped"] = prev
        return result

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

    def go_home(self, duration_s: float = 1.2, steps: int = 30) -> None:
        """Smoothly interpolate the arm back to the pose captured at connect().

        Called after a task-completion stop so the arm returns to its ready
        position instead of freezing over the bowl. Linear joint interpolation
        over ~1.2s. No-op if return-to-home is disabled, no home pose was
        captured, or the control loop is still running (must stop() first).
        Robot-agnostic: it just drives every state_key from its current value
        to the captured home value, so it works on any embodiment.
        """
        if not self._home_on_complete or self._home_pose is None:
            return
        if self._state == "running":
            log.debug("go_home skipped — control loop still running")
            return
        try:
            obs = self.robot.get_observation()
            start = {k: float(obs[k]) for k in self.profile.state_keys}
        except Exception as e:
            log.warning("go_home: could not read current pose: %s", e)
            return
        log.info("Returning to home pose ...")
        dt = duration_s / max(steps, 1)
        for i in range(1, steps + 1):
            a = i / steps
            action = {k: start[k] + (self._home_pose[k] - start[k]) * a
                      for k in self.profile.state_keys}
            try:
                self.robot.send_action(action)
            except Exception as e:
                log.warning("go_home: send_action failed: %s", e)
                return
            time.sleep(dt)
        log.info("Home pose reached")

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

                obs_in = _obs_to_policy_inputs(obs, lang, self.profile)
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
                    action = _decode_chunk(action_chunk, t, self.profile)
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
        predictor: Optional["StreamingIntentPredictor"] = None,
        command_validator: Optional[Any] = None,
        withdraw_stop_count: int = 2,
        withdraw_min_runtime_s: float = 15.0,
        task_complete_count: int = 1,
        max_task_runtime_s: float = 40.0,
        placing_min_runtime_s: float = 12.0,
        speech_via_qwen: bool = True,
        ground_actions: bool = True,
        verify_completion: bool = True,
    ):
        self.robot = robot
        self.interrupt_system = interrupt_system
        self._engine = engine
        # Who owns the speech channel:
        #   True  (--no-vad): Qwen handles ALL spoken commands via its
        #         predicted_intent (command_pick_*/interrupt/change_target/
        #         command_resume) and spoken_command field.
        #   False (default, VAD on): the energy VAD + transcribe_audio path
        #         owns all speech. Qwen's audio-derived intents are ignored
        #         here so the two paths don't double-fire; Qwen still drives
        #         VISUAL completion (task_complete / placing / withdraw).
        self._speech_via_qwen = speech_via_qwen
        # Visual grounding gate: when True, a pick command is verified against
        # the live frame (object actually visible?) before the robot starts or
        # switches. Disable with --no-grounding for A/B comparison or if the
        # perception veto misfires during a demo.
        self._ground_actions = ground_actions
        self._predictor = predictor
        self._command_validator = command_validator
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

        # Tertiary stop: placing-phase sliding-window fallback (May 20).
        # Qwen labels predicted_phase="placing" during a put-down but the label
        # is sparse and oscillates (placing→transporting→approaching), so the
        # original strict-consecutive count rarely tripped — especially for
        # yellow, which the model labels "placing" at roughly half the rate of
        # pink and almost never twice in a row (see eval_metrics.py baseline:
        # max consecutive yellow placing = 2, usually 0-1). Instead, count
        # placing hits in a sliding window of the last N gated predictions:
        # ≥hits within the window → complete. Robust to the oscillation.
        self._placing_window_size = 4
        self._placing_window_hits = 2
        self._placing_window: list = []
        # Placing has its own runtime gate (default 12s), separate from the
        # withdraw gate. It's a "don't auto-complete before N seconds" floor,
        # not an object- or task-specific value: nothing here references a
        # particular task. The 12s default suits short tabletop pick-place
        # (eval_metrics.py shows those finish ~7-13s and never emit "placing"
        # before ~10s); a longer-horizon task can raise it at construction
        # without touching this logic.
        self._placing_min_runtime_s = placing_min_runtime_s

        # Primary completion: dedicated visual verifier (May 21).
        # GR00T loops forever; the inline task_complete field and the placing
        # fallback both miss most real completions (esp. yellow), so the robot
        # kept picking at the empty table until the max-runtime cap. This tier
        # runs a focused vision-only Qwen call ("object in bowl AND gripper
        # empty?") — reliable because it asks one question and judges the image,
        # not the phase label. Throttled + phase-gated so it adds at most ~one
        # extra Qwen call every 1.5s late in a task. Confirmed twice → stop.
        self._verify_completion = verify_completion
        self._complete_min_runtime_s = 8.0
        self._complete_check_interval = 2.0
        self._complete_confirm_count = 2
        self._complete_min_conf = 0.7
        self._complete_streak = 0
        self._last_complete_check = 0.0

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

    _RELATIVE_WORDS = RELATIVE_WORDS

    @property
    def state(self):
        return self.robot.state

    @property
    def active_policy(self):
        return self.robot.active_policy

    def _clear_audio_buffer(self):
        """Wipe the predictor's 2s rolling buffer after a verbal stop fires.
        Otherwise the word "stop" lingers in the buffer for up to 2 seconds
        and Qwen can re-detect it on subsequent ticks."""
        if self._predictor is not None:
            try:
                self._predictor.audio_buffer.clear()
            except Exception:
                log.debug("audio_buffer.clear() failed (predictor may be stopped)")

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

    _RESUME_WORDS = RESUME_WORDS

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
        phase   = (prediction.get("predicted_phase") or "").strip().lower()

        # When VAD owns the speech channel (default mode), ignore Qwen's
        # audio-derived intents so the two command paths don't double-fire.
        # Blank the spoken command and rewrite audio-driven intents to a
        # neutral "continue" — Qwen still drives VISUAL completion below
        # (task_complete / placing-phase / withdraw all key off phase or
        # the task_complete flag, not these intents).
        if not self._speech_via_qwen:
            spoken = ""
            if (intent in ("interrupt", "change_target", "command_resume")
                    or intent.startswith("command_pick_")
                    or intent.startswith("command_")):
                intent = "continue"

        # ── Multimodal voice-command path (Qwen continuous audio+video) ────
        # Replaces the energy-VAD + transcribe_audio detour. Qwen fills the
        # spoken_command field independently of predicted_intent — vision
        # and audio are now decoupled in the JSON schema, so the model no
        # longer has to choose between describing motion and reporting a
        # heard command.
        if spoken:
            spoken_l = spoken.lower()
            # Switch-takes-priority-over-stop: if the spoken command names a
            # task target ("stop, pick up the yellow") OR a relative reference
            # ("not this one, the other"), treat it as a SWITCH and route
            # through _execute_policy_action — which in turn calls the
            # hard-switch path in switch_policy(). A switch implies a stop +
            # restart with cleanup, so this subsumes the verbal-stop branch
            # whenever a target is also present.
            _switch_target = (self.robot.tasks.resolve(spoken)
                              or self._resolve_relative(spoken))
            if _switch_target and self.state == "running" and _switch_target != self.active_policy:
                # Validator gate (same as below) keeps garbage from triggering a switch
                if not self._command_validator or self._command_validator(spoken):
                    log.info("Multimodal SWITCH (Qwen spoken+target): '%s' → %s",
                             spoken, _switch_target)
                    self._new_command_streak = 0
                    self._post_stop_quiet_until = time.time() + self._post_stop_quiet_seconds
                    self._clear_audio_buffer()
                    self._execute_policy_action(_switch_target, f"voice-switch(qwen): {spoken}")
                    return
            # Verbal stop → halt the running policy (only if no switch target).
            if any(w in spoken_l for w in STOP_WORDS):
                if self.state == "running":
                    log.info("Multimodal verbal stop (Qwen): '%s' — stopping %s",
                             spoken, self.active_policy)
                    self._new_command_streak = 0
                    self.robot.stop()
                    self._post_stop_quiet_until = time.time() + self._post_stop_quiet_seconds
                    self._clear_audio_buffer()
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
                # Gate through the same validator the VAD path uses — keeps
                # garbage Qwen transcriptions ("Thank you", "Shh.") from
                # spuriously starting tasks.
                if self._command_validator and not self._command_validator(spoken):
                    log.debug("spoken_command '%s' rejected by validator", spoken)
                    self._new_command_streak = 0
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

        # ── Multimodal COLD-START / RESUME via predicted_intent="command_*" ──
        # Qwen's WAITING prompt emits:
        #   command_<task_name>   — heard a verbal task command (start a new task)
        #   command_resume        — heard "continue" / "resume" / "keep going"
        #                           (restart the LAST paused task without
        #                            having to name it again)
        # Both share the same streak filter (≥2 consecutive at conf≥0.85).
        if (intent.startswith("command_") and conf >= 0.85
                and self.state == "idle"
                and time.time() >= self._post_stop_quiet_until):
            # Resolve the intent to a concrete policy name.
            if intent == "command_resume":
                policy = self._last_active_policy
                resolved_from = "resume"
            else:
                policy = intent[len("command_"):]
                resolved_from = "task-name"

            if policy and policy in self.robot.tasks:
                if intent == self._cold_start_last_intent:
                    self._cold_start_streak += 1
                else:
                    self._cold_start_streak = 1
                    self._cold_start_last_intent = intent
                log.info("Multimodal %s streak %d/%d (conf=%.2f) — %s",
                         "RESUME" if resolved_from == "resume" else "cold-start",
                         self._cold_start_streak, self._cold_start_count, conf, policy)
                if self._cold_start_streak >= self._cold_start_count:
                    _kind = "RESUME" if resolved_from == "resume" else "COLD-START"
                    log.info("Multimodal %s (Qwen %s) → %s", _kind, intent, policy)
                    telemetry.publish_event(_kind, policy=policy,
                                            reason=f"qwen-command: {intent}")
                    self._cold_start_streak = 0
                    self._cold_start_last_intent = None
                    self._execute_policy_action(policy, f"qwen-command: {intent}")
                    return
            elif intent == "command_resume":
                # User asked to resume but we have nothing to resume to.
                log.info("command_resume heard but no last policy to resume")
                self._cold_start_streak = 0
                self._cold_start_last_intent = None
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
                telemetry.publish_event("STOP", policy=self.active_policy,
                                        reason="multimodal interrupt streak")
                self.robot.stop()
                self._interrupt_streak = 0
                self._change_target_streak = 0
                # Quiet the InterruptDetectionSystem for a few seconds so its
                # follow-up visual_interrupt / object_mismatch callbacks
                # (caused by Qwen still emitting "interrupt" intent for the
                # next 1-2 frames) don't immediately restart the same task.
                self._post_stop_quiet_until = time.time() + self._post_stop_quiet_seconds
                self._clear_audio_buffer()
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
                    telemetry.publish_event("SWITCH", policy=policy,
                                            reason="multimodal change_target streak")
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
            self._placing_window.clear()
            return

        runtime = time.time() - self._policy_start_time

        # ── Safety: max task runtime (per-task override, falls back to default) ──
        # GR00T has no internal "done" signal and will loop indefinitely.
        # If neither task_complete nor withdraw fires within the time limit,
        # stop anyway so the arm doesn't pick-and-place forever.
        active = self.robot.tasks.get(self.active_policy) if self.active_policy else None
        max_rt = (active.max_runtime_s
                  if active and active.max_runtime_s is not None
                  else self._max_task_runtime_s)
        if runtime >= max_rt:
            log.info("Max task runtime %.0fs reached — stopping policy %s",
                     max_rt, self.active_policy)
            telemetry.publish_event("COMPLETE", policy=self.active_policy,
                                    reason=f"max runtime {max_rt:.0f}s")
            self.robot.stop()
            self._withdraw_streak = 0
            self._task_complete_streak = 0
            self._placing_window.clear()
            return

        # ── Primary completion: dedicated visual verifier ─────────────────
        # Throttled focused Qwen call: "is a ball now resting inside the bowl?".
        # Judges the bowl contents, not the phase label or the gripper state —
        # because after placing, GR00T loops back and Qwen mislabels the loop as
        # "approaching/grasping yellow ball" with a closed (empty) gripper. So we
        # do NOT gate on phase (the loop looks like real work) and we do NOT key
        # off gripper state (the empty closed gripper looks "holding"). We just
        # ask whether the ball made it into the bowl. Two consecutive
        # confirmations → stop. Every check is logged so the perception can be
        # audited. Fails safe (max-runtime backstop).
        now = time.time()
        if (self._verify_completion and self.state == "running"
                and self._predictor is not None
                and runtime >= self._complete_min_runtime_s
                and now - self._last_complete_check >= self._complete_check_interval):
            self._last_complete_check = now
            _ct = self.robot.tasks.get(self.active_policy) if self.active_policy else None
            _frame, _ = self._predictor.frame_buffer.get_latest()
            if _ct is not None and _frame is not None:
                c = self._engine.verify_task_complete(
                    _ct.object, _frame, completion_check=_ct.completion_check)
                if c.get("grounded"):
                    log.info("Completion check: complete=%s conf=%.2f why=%s",
                             c["complete"], c["confidence"], c.get("reason", ""))
                if (c.get("grounded") and c["complete"]
                        and c["confidence"] >= self._complete_min_conf):
                    self._complete_streak += 1
                    if self._complete_streak >= self._complete_confirm_count:
                        log.info("Task complete (visual verifier) — stopping %s",
                                 self.active_policy)
                        telemetry.publish_event("COMPLETE", policy=self.active_policy,
                                                reason="visual completion verifier")
                        self.robot.stop()
                        self.robot.go_home()
                        self._complete_streak = 0
                        self._task_complete_streak = 0
                        self._withdraw_streak = 0
                        self._placing_window.clear()
                        return
                else:
                    self._complete_streak = 0

        # ── Secondary: inline Qwen task_complete field ────────────────────
        if prediction.get("task_complete", False) and runtime >= self._withdraw_min_runtime_s:
            self._task_complete_streak += 1
            log.info("Qwen task_complete %d/%d (runtime=%.1fs) — '%s'",
                     self._task_complete_streak, self._task_complete_count,
                     runtime, self.active_policy)
            if self._task_complete_streak >= self._task_complete_count:
                log.info("Task complete (Qwen visual) — stopping policy %s", self.active_policy)
                telemetry.publish_event("COMPLETE", policy=self.active_policy,
                                        reason="qwen task_complete=true")
                self.robot.stop()
                self.robot.go_home()
                self._task_complete_streak = 0
                self._withdraw_streak = 0
                self._placing_window.clear()
                return
        else:
            self._task_complete_streak = 0

        # ── Placing-phase sliding-window fallback (May 20) ────────────────
        # Qwen reports predicted_phase="placing" when the arm is putting an
        # object down, but the label is sparse/oscillating and it is
        # conservative about task_complete=true. Count placing hits in a
        # sliding window of the last N gated predictions: ≥hits → complete.
        # Restricted to non-command intents (gesture/continue/approach) so we
        # don't auto-stop if Qwen also emits an interrupt or change_target —
        # those take precedence via their own branches above (which return
        # before reaching here).
        _placing_intents = {"gesture", "continue", "approach"}
        if runtime >= self._placing_min_runtime_s:
            is_placing = (phase == "placing"
                          and intent in _placing_intents
                          and conf >= 0.85)
            self._placing_window.append(is_placing)
            if len(self._placing_window) > self._placing_window_size:
                self._placing_window.pop(0)
            hits = sum(self._placing_window)
            if is_placing:
                log.info("Placing window %d/%d in last %d (conf=%.2f, runtime=%.1fs) — '%s'",
                         hits, self._placing_window_hits, len(self._placing_window),
                         conf, runtime, self.active_policy)
            if hits >= self._placing_window_hits:
                log.info("Task complete (placing-phase fallback) — stopping policy %s",
                         self.active_policy)
                telemetry.publish_event("COMPLETE", policy=self.active_policy,
                                        reason="placing-phase fallback")
                self.robot.stop()
                self.robot.go_home()
                self._placing_window.clear()
                self._withdraw_streak = 0
                self._task_complete_streak = 0
                return

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
                telemetry.publish_event("COMPLETE", policy=self.active_policy,
                                        reason="withdraw heuristic")
                self.robot.stop()
                self._withdraw_streak = 0
                self._placing_window.clear()
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

        # ── Visual grounding gate (SayCan-style world-grounding) ───────────
        # Before STARTING a pick from idle, verify the workspace has a graspable
        # object — don't reach into an empty table. Cold-start only: we skip the
        # gate on mid-task switches (self.state == "running") because the
        # switch-time frame is cluttered (arm holding the current object) which
        # makes the pickability check misfire, and a ball is obviously present
        # mid-task anyway. Done outside the lock — a ~300ms Qwen call that only
        # fires at a start decision. Fails open: a grounding outage degrades to
        # the previous ungated behavior, never a frozen robot.
        if self._ground_actions and self.state != "running" and self._predictor is not None:
            task = self.robot.tasks.get(policy)
            frame, _ = self._predictor.frame_buffer.get_latest()
            if task is not None and frame is not None:
                g = self._engine.verify_object_present(task.object, frame)
                if g.get("grounded") and not g["present"]:
                    # Lenient refusal. Qwen sometimes answers "not pickable"
                    # while STILL listing a graspable object in `seen` (e.g.
                    # "seen: purple yarn ball" — perception noise that otherwise
                    # blocks every resume/re-pick and looks like a hang). If the
                    # seen list names anything ball/yarn/object-like, there IS
                    # something to pick, so allow it. Refuse only when the scene
                    # truly shows nothing graspable — the real empty-table case.
                    _graspable = ("ball", "yarn", "pom", "cotton", "cube",
                                  "block", "object", "toy", "sphere")
                    seen_lc = " ".join(g.get("seen", [])).lower()
                    if any(w in seen_lc for w in _graspable):
                        log.info("Grounding override for %s — refused but seen "
                                 "list has a graspable object (%s); allowing",
                                 policy, seen_lc or "?")
                    else:
                        seen = ", ".join(g["seen"]) or "nothing recognizable"
                        log.warning("Grounding REFUSED %s — no graspable object in "
                                    "workspace (seen: %s)", policy, seen)
                        telemetry.publish_event("REFUSED", policy=policy,
                                                reason=f"empty workspace (seen: {seen})")
                        return {"action": "refused", "policy": policy,
                                "reason": "no_pickable_object", "seen": g["seen"]}

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
            self._placing_window.clear()
            self._complete_streak = 0
            self._last_complete_check = time.time()
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

def _crop_to_training(frame: np.ndarray, profile: RobotProfile) -> np.ndarray:
    """Center-crop frame to profile.train_w × profile.train_h (zero-copy when already correct size)."""
    th, tw = profile.train_h, profile.train_w
    h, w = frame.shape[:2]
    if h == th and w == tw:
        return frame
    if h < th or w < tw:
        log.warning("Frame %dx%d smaller than training size %dx%d — skipping crop",
                    w, h, tw, th)
        return frame
    y = (h - th) // 2
    x = (w - tw) // 2
    return frame[y : y + th, x : x + tw]


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

    # CLI override for the SO-101 gripper release pose. For other robots,
    # tune profile.release_overrides directly on the profile object.
    if args.gripper_open_pos is not None:
        SO101_PROFILE.release_overrides["gripper.pos"] = args.gripper_open_pos

    robot = GrootRobotController(
        tasks=tasks,
        robot_port=args.robot_port,
        robot_camera_index=args.robot_camera_index,
        robot_id=args.robot_id,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        profile=SO101_PROFILE,
        action_horizon=args.action_horizon,
        control_hz=args.control_hz,
        hard_switch=not args.no_hard_switch,
        home_on_complete=not args.no_home,
    )
    robot.connect()

    engine = FastQwenInferenceEngine(
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        scene_objects=tasks.objects(),
        cold_start_choices=[(f"command_{t.name}", t.object) for t in tasks],
    )

    config = StreamConfig(
        # 0.25s gives ~4 Hz target rate without overwhelming vLLM.
        # Continuous (0) caused vLLM queue saturation — video-only retries
        # started returning empty in ~40ms (queue full), doubling the failure
        # rate vs the old 0.5s. 0.25s halves worst-case reaction time (250ms
        # gap vs 500ms) while giving vLLM enough breathing room between calls.
        inference_interval=0.25,
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        # motion_threshold=0 disables the optical-flow gate so Qwen fires
        # on every cycle regardless of scene motion.
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
    # corrupts the active-task state. Uses module-level word constants.
    def _command_validator(text: str) -> bool:
        t = text.lower().strip()
        if any(w in t for w in STOP_WORDS) or "no" in t.split():
            return True
        if tasks.resolve(t) is not None:
            return True
        if any(w in t for w in RESUME_WORDS):
            return True
        # Allow relative-reference phrases through so "other ball", "not this one",
        # etc. reach handle_voice_command → _resolve_relative.
        return any(w in t for w in RELATIVE_WORDS)

    interrupt_system.command_validator = _command_validator

    router = PolicyRouter(
        robot, interrupt_system, engine,
        predictor=predictor,
        command_validator=_command_validator,
        # --no-vad → Qwen owns speech; default → VAD owns speech and Qwen's
        # audio-derived intents are ignored by the router.
        speech_via_qwen=args.no_vad,
        ground_actions=not args.no_grounding,
        verify_completion=not args.no_completion_check,
    )
    if args.no_grounding:
        log.warning("GROUNDING DISABLED (--no-grounding) — picks are not "
                    "verified against the live frame")
    else:
        log.info("Visual grounding ON (cold-start only) — a pick from idle is "
                 "vetoed if the workspace is empty (disable with --no-grounding)")
    if args.no_completion_check:
        log.warning("COMPLETION CHECK DISABLED (--no-completion-check) — robot "
                    "stops only via placing fallback / max-runtime cap")
    else:
        log.info("Visual completion ON — robot auto-stops when Qwen confirms "
                 "the object is placed and the gripper is empty "
                 "(disable with --no-completion-check)")
    if args.no_home:
        log.warning("RETURN-TO-HOME DISABLED (--no-home) — arm freezes in place "
                    "on completion")
    else:
        log.info("Return-to-home ON — arm returns to its startup pose after a "
                 "confirmed completion (disable with --no-home)")

    def on_interrupt(event):
        log.info("INTERRUPT: %s (object=%s, cmd='%s')",
                 event.reason, event.predicted_object, event.raw_command)
        result = router.handle_interrupt(event)
        log.info("Router action: %s", result)

    interrupt_system.on_interrupt(on_interrupt)
    interrupt_system.command_interface.on_new_task(
        lambda task: router.handle_voice_command(task)
    )

    # ── Metrics logger (always-on; cheap, JSONL append) ───────
    metrics: Optional[MetricsLogger] = None
    if args.metrics:
        try:
            metrics = MetricsLogger(args.metrics)
            log.info("Metrics → %s", args.metrics)
        except Exception as e:
            log.warning("MetricsLogger failed to start: %s", e)

    # ── Telemetry publisher (optional; for live dashboard) ────
    if args.telemetry_port:
        telemetry.init(port=args.telemetry_port)

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
    # device resolution (profile.native_w × profile.native_h) or LeRobot's
    # read_loop crashes when this call to cap.set() reconfigures the
    # AVFoundation device. Frames are then center-cropped to the GR00T
    # training resolution (profile.train_w × profile.train_h) in _crop_to_training.
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SO101_PROFILE.native_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SO101_PROFILE.native_h)
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

            # Minimum RMS a 1600-sample block must have (post-HPF) to be
            # added to Qwen's rolling buffer during execution. Motor noise
            # after HPF sits well below 0.01; speech typically lands 0.02-0.15.
            _SPEECH_RMS_GATE = 0.012

            # Speech-onset → fast-lane Qwen inference. We wait until the
            # user has accumulated enough speech in the rolling buffer that
            # the model has a recognizable utterance to classify. Too few
            # blocks → only a fragment ("s...") → classifier returns "none".
            # Too many → we lose the latency benefit. ~500 ms is the sweet
            # spot: long enough for "stop", "yellow", "pink" to be fully in
            # the buffer; short enough to be much faster than the worst-case
            # 250 ms polling wait.
            #
            # IMPORTANT: do NOT clear the buffer on onset — the audio gate
            # has already been filtering motor noise for seconds, so the
            # buffer already contains mostly clean speech. Clearing it
            # destroys the very audio the classifier needs.
            _SPEECH_ONSET_BLOCKS = 5  # ~500 ms of voiced energy → fire

            # Gate diagnostic counters — logged every 10s so you can verify
            # the threshold without reading raw audio.
            _gate_passed = 0
            _gate_dropped = 0
            _gate_log_time = time.time()
            _speech_consecutive = 0
            _silence_consecutive = 0
            _onset_armed = True   # one fast-lane fire per speech burst
            _fast_lane_count = 0
            # Speech-burst accumulator — holds ONLY the contiguous voiced
            # blocks of the current utterance. Mirrors what the VAD path
            # gives transcribe_audio (a tight clean segment), which is far
            # more reliable for Qwen3-Omni's audio encoder than the 2 s
            # rolling buffer's mixed content. Cleared on sustained silence.
            _speech_burst: list = []
            _BURST_END_SILENCE_BLOCKS = 4  # ~400 ms of silence → burst ends

            # Whisper-community consensus (and Qwen3-Omni's AuT encoder is
            # architecturally similar): denoising audio before a Whisper-style
            # encoder tends to HURT, not help — these encoders are trained on
            # noisy real-world audio and the HPF strips spectral cues they
            # rely on. Default OFF as of May 19 — A/B testing showed
            # significantly better stop detection without the HPF (multimodal
            # path catches stop in ~8s vs HPF-on needing fast-lane fallback
            # after ~28s). Pass --hpf to re-enable for comparison.
            _hpf_enabled = args.hpf
            if _hpf_enabled:
                log.warning("HPF ENABLED (--hpf) — pre-filtering audio with "
                            "200Hz high-pass before gate. Default is OFF; "
                            "use this only for A/B comparison.")

            def audio_callback(indata, frames, time_info, status):
                nonlocal _gate_passed, _gate_dropped, _gate_log_time
                nonlocal _speech_consecutive, _silence_consecutive
                nonlocal _onset_armed, _fast_lane_count
                audio = indata[:, 0].copy()
                if robot.state == "idle":
                    # No motor noise when idle — feed raw audio directly.
                    predictor.add_audio(audio)
                else:
                    # During execution: optionally HPF to strip servo rumble,
                    # then gate on energy. Default is no HPF (--hpf to enable).
                    filtered = (engine._highpass_filter(audio)
                                if _hpf_enabled else audio)
                    rms = float(np.sqrt(np.mean(filtered ** 2)))
                    if rms >= _SPEECH_RMS_GATE:
                        _gate_passed += 1
                        _speech_consecutive += 1
                        _silence_consecutive = 0
                        # Accumulate this block into the current speech burst.
                        _speech_burst.append(filtered)
                        # Also add to the predictor's rolling buffer so the
                        # main 250 ms polling path keeps working normally.
                        predictor.add_audio(filtered)
                        # When the user has clearly started speaking, replace
                        # the predictor's audio buffer with JUST the burst
                        # (mimicking what VAD's transcribe_audio sends) and
                        # fire an immediate inference. This is the critical
                        # fix: Qwen sees only the tight utterance, not 2 s
                        # of mixed history.
                        if _speech_consecutive == _SPEECH_ONSET_BLOCKS and _onset_armed:
                            try:
                                predictor.audio_buffer.clear()
                                predictor.add_audio(np.concatenate(_speech_burst))
                            except Exception:
                                pass
                            predictor.request_immediate_inference()
                            _onset_armed = False
                            _fast_lane_count += 1
                    else:
                        _gate_dropped += 1
                        _speech_consecutive = 0
                        _silence_consecutive += 1
                        # CRITICAL (May 19 fix): also feed silence into the
                        # rolling buffer. Previously this branch did nothing,
                        # so during sustained silence the buffer kept stale
                        # content from earlier (e.g. a 358 ms audio sample
                        # captured during the brief idle window of a hard
                        # switch). Every Qwen call then read the SAME stale
                        # audio for many seconds — visible in session logs
                        # as 20+ consecutive parse failures all reporting
                        # the identical RMS value (e.g. 0.0092). Qwen's
                        # audio encoder receives the same low-energy
                        # not-quite-silent fragment over and over and
                        # returns empty streams / garbage tokens.
                        # Feeding silence here keeps the buffer rolling
                        # forward so:
                        #   (a) the engine's silence gate (audio_energy <
                        #       1e-6) trips correctly during real silence
                        #       and forces video-only mode;
                        #   (b) speech bursts are surrounded by natural
                        #       silence padding, which matches how Qwen3
                        #       Omni's AuT encoder was trained.
                        predictor.add_audio(filtered)
                        # Clear BOTH the speech-burst accumulator AND the
                        # predictor's rolling audio buffer once we've heard
                        # enough sustained silence. The buffer clear is
                        # still useful: it kills any residual speech-burst
                        # content from the previous utterance so the next
                        # inference sees a clean silence-only buffer.
                        if _silence_consecutive == _BURST_END_SILENCE_BLOCKS:
                            if _speech_burst:
                                _speech_burst.clear()
                            try:
                                predictor.audio_buffer.clear()
                            except Exception:
                                pass
                            _onset_armed = True
                    # Periodic diagnostic log — helps tune _SPEECH_RMS_GATE
                    now = time.time()
                    if now - _gate_log_time >= 10.0:
                        total = _gate_passed + _gate_dropped
                        pct = 100.0 * _gate_passed / total if total else 0.0
                        log.info(
                            "Audio gate (last 10s): passed=%d dropped=%d "
                            "(%.0f%% speech) fast-lane fires=%d threshold=%.3f",
                            _gate_passed, _gate_dropped, pct,
                            _fast_lane_count, _SPEECH_RMS_GATE,
                        )
                        _gate_passed = 0
                        _gate_dropped = 0
                        _fast_lane_count = 0
                        _gate_log_time = now
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
    log.info("Prediction engine started (interval=%.2fs, 1 worker)",
             config.inference_interval)

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

            # State→prompt mapping (idle → WAITING in BOTH modes):
            # - idle → WAITING prompt. In --no-vad this lets Qwen fire
            #   command_<task> for cold-start. In VAD mode the command output
            #   is ignored by the router, but the WAITING prompt stops Qwen
            #   from hallucinating arm motion ("approaching pink ball") when
            #   nothing is moving — important for clean recordings/demos.
            # - running → EXECUTING prompt for visual intent + completion.
            if robot.state == "idle":
                _qwen_state = "waiting"
            else:
                _qwen_state = "executing"
            predictor.set_robot_state(
                f"state={_qwen_state}, policy={robot.active_policy or 'none'}"
            )

            if now - last_frame_time >= frame_interval:
                ret, frame = cap.read()
                if ret:
                    frame = _crop_to_training(frame, SO101_PROFILE)
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
                _phase = (getattr(pred, "predicted_phase", "") or "unknown").strip()
                log.info("Qwen #%d: %s/%s(%s) conf=%.2f task_complete=%s spoken='%s' why=%s",
                         _qwen_pred_seq,
                         pred.predicted_intent,
                         _phase,
                         pred.target_object or "-",
                         pred.confidence,
                         pred.task_complete,
                         _spoken,
                         (pred.reason or "")[:60])
                router.handle_prediction({
                    "predicted_intent": pred.predicted_intent,
                    "predicted_phase": pred.predicted_phase,
                    "confidence": pred.confidence,
                    "target_object": pred.target_object,
                    "task_complete": pred.task_complete,
                    "reason": pred.reason,
                    "spoken_command": _spoken,
                })
                if recorder is not None:
                    recorder.push_stats(_qwen_hz_ema, robot.current_hz, _qwen_pred_seq)
                if metrics is not None:
                    raw = (pred.raw_response or "").strip()
                    metrics.log(
                        prediction=pred,
                        robot_state=_qwen_state,
                        active_policy=robot.active_policy,
                        parse_failed=(pred.predicted_intent == "unknown" and bool(raw)),
                    )
                telemetry.publish_prediction(
                    pred=pred,
                    qwen_hz=_qwen_hz_ema,
                    groot_hz=robot.current_hz,
                    robot_state=_qwen_state,
                    active_policy=robot.active_policy,
                )

            time.sleep(0.05)

    finally:
        log.info("Shutting down ...")
        predictor.stop()
        cap.release()
        if audio_stream:
            audio_stream.stop()
        if recorder is not None:
            recorder.stop()
        if metrics is not None:
            metrics.close()
        telemetry.close()
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
    p.add_argument("--no-grounding", action="store_true",
                   help="Disable the visual-grounding gate. By default a pick "
                        "command from idle is verified against the live frame "
                        "(is the workspace non-empty?) before the robot starts; "
                        "this flag reverts to ungated behavior.")
    p.add_argument("--no-completion-check", action="store_true",
                   help="Disable the dedicated visual completion verifier. By "
                        "default, once a task has run a few seconds the system "
                        "asks Qwen 'is the object placed and the gripper empty?' "
                        "and auto-stops when confirmed; this flag reverts to the "
                        "placing fallback + max-runtime cap only.")
    p.add_argument("--policy-host", default="192.168.2.25", help="GR00T policy server host (s99)")
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--action-horizon", type=int, default=8)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--record", default=None,
                   help="Path to .mp4 — record session video (Qwen camera + mic + overlays)")
    p.add_argument("--record-fps", type=int, default=10,
                   help="Recording frame rate (default 10, matches Qwen capture cadence)")
    p.add_argument("--metrics", default=None,
                   help="Path to .jsonl — write structured per-prediction metrics "
                        "(intent, conf, latency, parse_failed, etc.) for evaluation.")
    p.add_argument("--telemetry-port", type=int, default=None,
                   help="If set, publish live telemetry on tcp://127.0.0.1:<port> "
                        "for telemetry_dashboard.py to subscribe to (e.g. 5601).")
    p.add_argument("--no-hard-switch", action="store_true",
                   help="Revert to legacy hot-swap on policy switch (just change "
                        "the lang string). Default: hard switch — stop loop, "
                        "release gripper, restart cleanly. Use this flag if the "
                        "hard switch causes regressions.")
    p.add_argument("--no-home", action="store_true",
                   help="Disable return-to-home. By default, after a task is "
                        "confirmed complete the arm smoothly returns to the pose "
                        "it was in at startup; this flag leaves it frozen in place.")
    p.add_argument("--gripper-open-pos", type=float, default=None,
                   help="SO-101 only: override the gripper joint value (deg) "
                        "applied during hard-switch release. Defaults to the "
                        "value baked into SO101_PROFILE.release_overrides "
                        "(50°). For other robots, edit the profile's "
                        "release_overrides dict directly.")
    p.add_argument("--hpf", action="store_true",
                   help="Enable the 200Hz high-pass filter on audio during "
                        "robot execution. DEFAULT IS OFF as of May 19 — "
                        "A/B testing showed the HPF stripped speech "
                        "harmonics the AuT encoder needs, hurting stop "
                        "detection and increasing parse failures. Whisper "
                        "community consensus is that denoising hurts ASR "
                        "for encoders trained on noisy real-world audio. "
                        "Pass --hpf if you suspect the raw-audio path is "
                        "letting too much motor noise into Qwen.")

    run(p.parse_args())


if __name__ == "__main__":
    main()
