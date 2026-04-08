#!/usr/bin/env python3
"""
Policy Router — runs on PN62 (prediction machine)
Connects Qwen3-Omni intent prediction to SO101 robot via robot_server.

This is the main entry point for the full system:
  Camera + Mic → Qwen3-Omni → Intent Prediction → Policy Router → SO101 Robot

Usage:
    python policy_router.py \
        --robot-url http://<gqu6x-ip>:5555 \
        --vllm-url http://192.168.2.25:8000/v1 \
        --camera 0 \
        --mic-index 0

Flow:
    1. User speaks: "pick up the yellow ball"
    2. Qwen hears command → Router maps to ACT policy → Sends /start to robot
    3. Qwen monitors continuously (every 0.5s)
    4. User interrupts: "no, the pink one"
    5. Qwen detects interrupt → Router sends /switch to robot
    6. Robot switches ACT policy mid-execution
"""

import argparse
import json
import logging
import re
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("policy_router")


# ── Object-to-policy mapping ──
OBJECT_POLICY_MAP = {
    "pink":    "pick_pink_ball",
    "pink ball": "pick_pink_ball",
    "pink cotton": "pick_pink_ball",
    "yellow":  "pick_yellow_ball",
    "yellow ball": "pick_yellow_ball",
    "yellow cotton": "pick_yellow_ball",
    "other":   "pick_and_correct",
    "correct": "pick_and_correct",
    "interrupted": "pick_and_correct",
}

# Keywords that map to policies
POLICY_KEYWORDS = {
    "pick_pink_ball":   ["pink", "pink ball", "pink cotton", "pink one"],
    "pick_yellow_ball": ["yellow", "yellow ball", "yellow cotton", "yellow one"],
    "pick_and_correct": ["other", "correct", "different", "not this"],
}


class RobotClient:
    """HTTP client for robot_server.py running on gqu6x."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def start(self, policy: str) -> dict:
        return self._post("/start", {"policy": policy})

    def stop(self) -> dict:
        return self._post("/stop", {})

    def switch(self, policy: str) -> dict:
        return self._post("/switch", {"policy": policy})

    def status(self) -> dict:
        return self._get("/status")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest camera frame from robot for Qwen to analyze."""
        try:
            r = self._session.get(f"{self.base_url}/frame", timeout=self.timeout)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except requests.RequestException:
            pass
        return None

    def _post(self, path: str, data: dict) -> dict:
        try:
            r = self._session.post(
                f"{self.base_url}{path}",
                json=data, timeout=self.timeout
            )
            return r.json()
        except requests.RequestException as e:
            return {"ok": False, "error": str(e)}

    def _get(self, path: str) -> dict:
        try:
            r = self._session.get(f"{self.base_url}{path}", timeout=self.timeout)
            return r.json()
        except requests.RequestException as e:
            return {"error": str(e)}


def resolve_policy(text: str) -> Optional[str]:
    """Map a voice command or object name to an ACT policy name."""
    text_lower = text.lower().strip()

    # Direct match
    for policy, keywords in POLICY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return policy

    # Fuzzy: check object-to-policy map
    for obj_key, policy in OBJECT_POLICY_MAP.items():
        if obj_key in text_lower:
            return policy

    return None


class PolicyRouter:
    """
    Connects Qwen predictions to robot policy execution.

    Modes:
    - WAITING: No task active, listening for voice command
    - EXECUTING: Policy running on robot, monitoring for interrupts
    - SWITCHING: In the process of switching policies
    """

    def __init__(self, robot_client: RobotClient, interrupt_system=None):
        self.robot = robot_client
        self.interrupt_system = interrupt_system
        self._active_policy = None
        self._state = "waiting"  # waiting | executing | switching
        self._lock = threading.Lock()
        self._last_switch_time = 0.0
        self._switch_cooldown = 3.0  # min seconds between switches

    @property
    def state(self):
        return self._state

    @property
    def active_policy(self):
        return self._active_policy

    def handle_voice_command(self, command: str) -> dict:
        """Process a voice command from Qwen's audio detection."""
        log.info("Voice command: '%s'", command)

        policy = resolve_policy(command)
        if not policy:
            log.warning("Could not map command to policy: '%s'", command)
            return {"action": "none", "reason": f"No policy match for: {command}"}

        return self._execute_policy_action(policy, command)

    def handle_interrupt(self, event) -> dict:
        """Process an InterruptEvent from the interrupt detection system."""
        from interrupt_detection_system import InterruptReason

        log.info("Interrupt: reason=%s, object=%s, command='%s'",
                 event.reason, event.predicted_object, event.raw_command)

        # Verbal interrupt with new command
        if event.raw_command:
            policy = resolve_policy(event.raw_command)
            if policy:
                return self._execute_policy_action(policy, event.raw_command)

        # Object mismatch — try to find policy from predicted object
        if event.predicted_object:
            policy = resolve_policy(event.predicted_object)
            if policy:
                return self._execute_policy_action(policy, f"object_mismatch: {event.predicted_object}")

        # Just stop if we can't determine new policy
        log.info("Stopping robot (no clear new policy)")
        result = self.robot.stop()
        with self._lock:
            self._state = "waiting"
            self._active_policy = None
        return {"action": "stop", "result": result}

    def handle_prediction(self, prediction: dict):
        """
        Process a regular Qwen prediction (called every 0.5s).
        In executing state, check if prediction suggests task completion.
        """
        if self._state != "executing":
            return

        intent = prediction.get("predicted_intent", "")

        # If we see sustained "continue" with high confidence, task might be done
        # (Robot arm idle after completing pick-and-place)
        # This is informational only — don't auto-stop
        if intent == "withdraw":
            log.info("Withdraw detected — task may be completing")

    def _execute_policy_action(self, policy: str, reason: str) -> dict:
        """Start or switch to a policy."""
        with self._lock:
            # Cooldown to prevent rapid switching
            now = time.time()
            if now - self._last_switch_time < self._switch_cooldown:
                remaining = self._switch_cooldown - (now - self._last_switch_time)
                log.info("Switch cooldown active (%.1fs remaining)", remaining)
                return {"action": "cooldown", "remaining_s": remaining}

            if self._state == "switching":
                return {"action": "busy", "reason": "Already switching"}

            prev_state = self._state
            prev_policy = self._active_policy
            self._state = "switching"

        log.info("Policy action: %s → %s (reason: %s)", prev_policy, policy, reason)

        if prev_state == "executing" and prev_policy:
            # Switch: stop current + start new
            result = self.robot.switch(policy)
        else:
            # Fresh start
            result = self.robot.start(policy)

        with self._lock:
            if result.get("ok"):
                self._active_policy = policy
                self._state = "executing"
                self._last_switch_time = time.time()
                log.info("Now executing: %s", policy)

                # Update interrupt system's active task
                if self.interrupt_system:
                    task_obj = self._policy_to_object(policy)
                    self.interrupt_system.task_monitor.set_task(
                        command=reason,
                        intent="approach",
                        target_object=task_obj
                    )
            else:
                self._state = "waiting" if prev_state == "waiting" else "executing"
                self._active_policy = prev_policy
                log.error("Policy action failed: %s", result.get("error"))

        return {"action": "switch" if prev_policy else "start",
                "policy": policy, "result": result}

    @staticmethod
    def _policy_to_object(policy: str) -> str:
        """Map policy name back to object description for Qwen prompts."""
        mapping = {
            "pick_pink_ball": "pink cotton ball",
            "pick_yellow_ball": "yellow cotton ball",
            "pick_and_correct": "cotton ball",
        }
        return mapping.get(policy, "unknown")


def run_full_system(args):
    """
    Main loop: Qwen watches camera + mic, routes commands to robot.
    """
    from qwen_inference_engine import FastQwenInferenceEngine
    from streaming_intent_predictor import StreamingIntentPredictor, StreamConfig
    from interrupt_detection_system import (
        InterruptDetectionSystem, InterruptReason, connect_to_predictor
    )

    # ── Setup robot client ──
    robot = RobotClient(args.robot_url)
    status = robot.status()
    log.info("Robot server status: %s", status)

    # ── Setup Qwen inference engine ──
    engine = FastQwenInferenceEngine(
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        scene_objects=["pink cotton ball", "yellow cotton ball"],
    )

    # ── Setup streaming predictor ──
    config = StreamConfig(
        inference_interval=0.5,
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        motion_threshold=1.5,
    )
    predictor = StreamingIntentPredictor(config)
    predictor.scheduler.set_inference_engine(engine)

    # ── Setup interrupt detection ──
    interrupt_system = InterruptDetectionSystem(
        consecutive_required=3,
        grace_period=4.0,
    )
    connect_to_predictor(interrupt_system, predictor)

    # ── Setup policy router ──
    router = PolicyRouter(robot, interrupt_system)

    # Register interrupt callback
    def on_interrupt(event):
        log.info("INTERRUPT DETECTED: %s", event.reason)
        result = router.handle_interrupt(event)
        log.info("Router result: %s", result)

    interrupt_system.on_interrupt(on_interrupt)

    # Register command callbacks
    interrupt_system.command_interface.on_stop(
        lambda: log.info("STOP signal sent to robot")
    )
    interrupt_system.command_interface.on_new_task(
        lambda task: router.handle_voice_command(task)
    )

    # ── Camera source ──
    use_robot_camera = args.camera == -1
    cap = None
    if not use_robot_camera:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        log.info("Using local camera %d", args.camera)
    else:
        log.info("Using robot camera via HTTP")

    # ── Audio source ──
    audio_thread = None
    if not args.no_audio:
        try:
            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                predictor.add_audio(indata[:, 0].copy())
                interrupt_system.on_audio(indata[:, 0].copy())

            audio_stream = sd.InputStream(
                samplerate=16000, channels=1, blocksize=1600,  # 100ms blocks
                callback=audio_callback, device=args.mic_index
            )
            audio_stream.start()
            log.info("Microphone started (device=%s)", args.mic_index)
        except Exception as e:
            log.warning("Could not start microphone: %s (running video-only)", e)

    # ── Start prediction engine ──
    predictor.start(num_workers=2)
    log.info("Prediction engine started")

    # ── Main loop ──
    log.info("="*50)
    log.info("SYSTEM READY — Speak a command to start")
    log.info("  'pick up the pink ball'")
    log.info("  'pick up the yellow ball'")
    log.info("  Interrupt: 'no, pick up the other one'")
    log.info("="*50)

    last_frame_time = 0.0
    frame_interval = 0.1  # 10fps — plenty for 2Hz Qwen inference

    try:
        while True:
            now = time.time()

            # Update Qwen with current router state
            predictor.set_robot_state(
                f"state={router.state}, policy={router.active_policy or 'none'}"
            )

            # Get frame at reduced rate (10fps, not 30fps)
            if now - last_frame_time >= frame_interval:
                if use_robot_camera:
                    frame = robot.get_frame()
                else:
                    ret, frame = cap.read()
                    frame = frame if ret else None

                if frame is not None:
                    predictor.add_frame(frame)
                last_frame_time = now

            # Process predictions
            predictions = predictor.get_all_predictions()
            for pred in predictions:
                pred_dict = {
                    "predicted_intent": pred.predicted_intent,
                    "confidence": pred.confidence,
                    "target_object": pred.target_object,
                    "reason": pred.reason,
                }

                # If we're waiting for initial command, check if Qwen heard one
                if router.state == "waiting" and pred.predicted_intent in (
                    "new_command", "approach", "gesture"
                ):
                    if pred.target_object and pred.target_object != "none":
                        policy = resolve_policy(pred.target_object)
                        if policy:
                            log.info("Voice command detected: %s -> %s",
                                     pred.target_object, policy)
                            router.handle_voice_command(
                                f"pick up the {pred.target_object}"
                            )

                router.handle_prediction(pred_dict)

            time.sleep(0.05)  # 20Hz main loop

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        predictor.stop()
        if cap:
            cap.release()
        robot.stop()
        log.info("System stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Policy Router — connects Qwen predictions to SO101 robot"
    )
    parser.add_argument("--robot-url", required=True,
                        help="Robot server URL (e.g., http://192.168.x.x:5555)")
    parser.add_argument("--vllm-url", default="http://192.168.2.25:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--camera", type=int, default=0,
                        help="Local camera index (-1 = use robot camera via HTTP)")
    parser.add_argument("--mic-index", type=int, default=None,
                        help="Microphone device index")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio (video-only mode)")

    # Manual command mode (for testing without mic)
    sub = parser.add_subparsers(dest="command")
    start_cmd = sub.add_parser("start", help="Start a policy manually")
    start_cmd.add_argument("policy", help="Policy name")
    stop_cmd = sub.add_parser("stop", help="Stop robot")
    switch_cmd = sub.add_parser("switch", help="Switch policy")
    switch_cmd.add_argument("policy", help="Policy name")
    status_cmd = sub.add_parser("status", help="Get robot status")

    args = parser.parse_args()

    # Manual command mode
    if args.command:
        robot = RobotClient(args.robot_url)
        if args.command == "start":
            print(json.dumps(robot.start(args.policy), indent=2))
        elif args.command == "stop":
            print(json.dumps(robot.stop(), indent=2))
        elif args.command == "switch":
            print(json.dumps(robot.switch(args.policy), indent=2))
        elif args.command == "status":
            print(json.dumps(robot.status(), indent=2))
        return

    # Full system mode
    run_full_system(args)


if __name__ == "__main__":
    main()
