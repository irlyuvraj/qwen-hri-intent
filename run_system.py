#!/usr/bin/env python3
"""
Qwen3-Omni + SO101 Robot — Unified System (2-PC Setup)

Runs on gqu6x (robot laptop). vLLM runs on s99 (GPU server).
Single entry point: camera + mic + Qwen prediction + policy routing + robot control.

Usage:
    python run_system.py \
        --vllm-url http://192.168.2.25:8000/v1 \
        --robot-port /dev/ttyACM1 \
        --camera-index 0 \
        --robot-camera-index 8

Flow:
    1. System starts, Qwen listens for voice command
    2. You say "pick up the yellow ball"
    3. Qwen detects command → starts pick_yellow_ball ACT policy on robot
    4. Qwen monitors continuously (camera + mic every 0.5s)
    5. You say "no, the pink one"
    6. Qwen detects interrupt → stops robot → switches to pick_pink_ball
"""

import argparse
import logging
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

from qwen_inference_engine import FastQwenInferenceEngine
from streaming_intent_predictor import StreamingIntentPredictor, StreamConfig
from interrupt_detection_system import (
    InterruptDetectionSystem, InterruptReason, connect_to_predictor
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
)
log = logging.getLogger("run_system")

# ═══════════════════════════════════════════════════════════════
# Policy Registry
# ═══════════════════════════════════════════════════════════════

POLICY_REGISTRY = {
    "pick_pink_ball":   "tysyuvraj/so101-act-pick-pink-ball",
    "pick_yellow_ball": "tysyuvraj/so101-act-pick-yellow-ball",
    "pick_and_correct": "tysyuvraj/so101-act-pick-and-correct",
}

POLICY_KEYWORDS = {
    "pick_pink_ball":   ["pink", "pink ball", "pink cotton", "pink one"],
    "pick_yellow_ball": ["yellow", "yellow ball", "yellow cotton", "yellow one"],
    "pick_and_correct": ["other", "correct", "different", "not this"],
}

POLICY_TO_OBJECT = {
    "pick_pink_ball": "pink cotton ball",
    "pick_yellow_ball": "yellow cotton ball",
    "pick_and_correct": "cotton ball",
}


def resolve_policy(text: str) -> Optional[str]:
    """Map a voice command or object name to an ACT policy name."""
    text_lower = text.lower().strip()
    for policy, keywords in POLICY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return policy
    return None


# ═══════════════════════════════════════════════════════════════
# Robot Controller (subprocess management, no HTTP)
# ═══════════════════════════════════════════════════════════════

class RobotController:
    """Manages LeRobot policy_server + robot_client as subprocesses."""

    def __init__(self, robot_port: str, robot_camera_index: int):
        self.robot_port = robot_port
        self.robot_camera_index = robot_camera_index
        self._server_proc = None
        self._client_proc = None
        self._active_policy = None
        self._state = "idle"  # idle | starting | running | stopping
        self._lock = threading.Lock()

    @property
    def state(self):
        return self._state

    @property
    def active_policy(self):
        return self._active_policy

    def start_policy(self, policy_name: str) -> dict:
        """Start executing a policy on the robot."""
        with self._lock:
            if policy_name not in POLICY_REGISTRY:
                return {"ok": False, "error": f"Unknown policy: {policy_name}",
                        "available": list(POLICY_REGISTRY.keys())}

            if self._state == "running":
                return {"ok": False, "error": "Already running. Call stop() or switch() first."}

            self._state = "starting"
            policy_path = POLICY_REGISTRY[policy_name]

            # Start LeRobot policy server
            server_cmd = [
                sys.executable, "-m", "lerobot.async_inference.policy_server",
                f"--policy_type=act",
                f"--pretrained_name_or_path={policy_path}",
                f"--policy_device=cuda",
                f"--port=8080",
            ]
            log.info("Starting policy server: %s", policy_name)
            self._server_proc = subprocess.Popen(
                server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait for server to be ready (poll up to 15s)
            ready = False
            for _ in range(30):
                if self._server_proc.poll() is not None:
                    break
                time.sleep(0.5)
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(0.3)
                    s.connect(("127.0.0.1", 8080))
                    s.close()
                    ready = True
                    break
                except (ConnectionRefusedError, OSError):
                    continue

            if not ready and self._server_proc.poll() is not None:
                stderr = self._server_proc.stderr.read().decode()
                self._state = "idle"
                return {"ok": False, "error": f"Server failed to start: {stderr[:500]}"}

            # Start robot client
            client_cmd = [
                sys.executable, "-m", "lerobot.async_inference.robot_client",
                f"--server_address=127.0.0.1:8080",
                f"--robot.type=so101_follower",
                f"--robot.port={self.robot_port}",
                f"--robot.id=my_awesome_follower_arm",
                f'--robot.cameras={{ front: {{type: opencv, index_or_path: '
                f'{self.robot_camera_index}, width: 640, height: 360, fps: 60}}}}',
                f"--policy_type=act",
                f"--pretrained_name_or_path={policy_path}",
                f"--policy_device=cuda",
                f"--actions_per_chunk=50",
                f"--chunk_size_threshold=0.5",
                f"--aggregate_fn_name=weighted_average",
            ]
            log.info("Starting robot client")
            self._client_proc = subprocess.Popen(
                client_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            time.sleep(2)
            if self._client_proc.poll() is not None:
                stderr = self._client_proc.stderr.read().decode()
                self._kill_server()
                self._state = "idle"
                return {"ok": False, "error": f"Client failed to start: {stderr[:500]}"}

            self._active_policy = policy_name
            self._state = "running"
            log.info("Policy running: %s", policy_name)
            return {"ok": True, "policy": policy_name, "state": "running"}

    def stop(self) -> dict:
        """Stop current policy execution."""
        with self._lock:
            if self._state != "running":
                return {"ok": True, "state": self._state, "msg": "Nothing to stop"}

            self._state = "stopping"
            self._kill_client()
            self._kill_server()
            prev = self._active_policy
            self._active_policy = None
            self._state = "idle"
            log.info("Stopped policy: %s", prev)
            return {"ok": True, "stopped": prev, "state": "idle"}

    def switch_policy(self, policy_name: str) -> dict:
        """Stop current policy and start a new one."""
        if policy_name not in POLICY_REGISTRY:
            return {"ok": False, "error": f"Unknown policy: {policy_name}"}

        stop_result = self.stop()
        time.sleep(0.5)
        start_result = self.start_policy(policy_name)
        return {
            "ok": start_result["ok"],
            "stopped": stop_result.get("stopped"),
            "started": policy_name if start_result["ok"] else None,
            "state": start_result.get("state", "error"),
            "error": start_result.get("error"),
        }

    def _kill_client(self):
        if self._client_proc and self._client_proc.poll() is None:
            self._client_proc.send_signal(signal.SIGINT)
            try:
                self._client_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._client_proc.kill()
            self._client_proc = None

    def _kill_server(self):
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.send_signal(signal.SIGINT)
            try:
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    def shutdown(self):
        self.stop()


# ═══════════════════════════════════════════════════════════════
# Policy Router (direct calls, no HTTP)
# ═══════════════════════════════════════════════════════════════

class PolicyRouter:
    """Routes Qwen predictions to robot policy execution."""

    def __init__(self, robot: RobotController, interrupt_system: InterruptDetectionSystem):
        self.robot = robot
        self.interrupt_system = interrupt_system
        self._lock = threading.Lock()
        self._last_switch_time = 0.0
        self._switch_cooldown = 3.0

    @property
    def state(self):
        return self.robot.state

    @property
    def active_policy(self):
        return self.robot.active_policy

    def handle_voice_command(self, command: str) -> dict:
        """Process a voice command — map to policy and execute."""
        log.info("Voice command: '%s'", command)
        policy = resolve_policy(command)
        if not policy:
            log.warning("No policy match for: '%s'", command)
            return {"action": "none", "reason": f"No policy match for: {command}"}
        return self._execute_policy_action(policy, command)

    def handle_interrupt(self, event) -> dict:
        """Process an InterruptEvent."""
        log.info("Interrupt: reason=%s, object=%s, command='%s'",
                 event.reason, event.predicted_object, event.raw_command)

        # Verbal interrupt with new command
        if event.raw_command:
            policy = resolve_policy(event.raw_command)
            if policy:
                return self._execute_policy_action(policy, event.raw_command)

        # Object mismatch
        if event.predicted_object:
            policy = resolve_policy(event.predicted_object)
            if policy:
                return self._execute_policy_action(
                    policy, f"object_mismatch: {event.predicted_object}"
                )

        # Just stop
        log.info("Stopping robot (no clear new policy)")
        result = self.robot.stop()
        return {"action": "stop", "result": result}

    def handle_prediction(self, prediction: dict):
        """Process a regular Qwen prediction (every 0.5s)."""
        if self.state != "running":
            return
        intent = prediction.get("predicted_intent", "")
        if intent == "withdraw":
            log.info("Withdraw detected — task may be completing")

    def _execute_policy_action(self, policy: str, reason: str) -> dict:
        """Start or switch to a policy."""
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
            log.info("Now executing: %s", policy)

            # Update interrupt system's active task
            task_obj = POLICY_TO_OBJECT.get(policy, "unknown")
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
# Main System
# ═══════════════════════════════════════════════════════════════

def run(args):
    """Main loop: camera + mic + Qwen + policy routing + robot control."""

    # ── Robot controller ──
    robot = RobotController(args.robot_port, args.robot_camera_index)
    log.info("Robot controller ready (port=%s, camera=%d)",
             args.robot_port, args.robot_camera_index)

    # ── Qwen inference engine ──
    engine = FastQwenInferenceEngine(
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        scene_objects=["pink cotton ball", "yellow cotton ball"],
    )

    # ── Streaming predictor ──
    config = StreamConfig(
        inference_interval=0.5,
        vllm_url=args.vllm_url,
        model_name="qwen3-30b-a3b",
        motion_threshold=1.5,
    )
    predictor = StreamingIntentPredictor(config)
    predictor.scheduler.set_inference_engine(engine)

    # ── Interrupt detection ──
    interrupt_system = InterruptDetectionSystem(
        consecutive_required=3,
        grace_period=4.0,
    )
    connect_to_predictor(interrupt_system, predictor)

    # ── Policy router ──
    router = PolicyRouter(robot, interrupt_system)

    # Wire interrupt callback
    def on_interrupt(event):
        log.info("INTERRUPT: %s (object=%s, cmd='%s')",
                 event.reason, event.predicted_object, event.raw_command)
        result = router.handle_interrupt(event)
        log.info("Router action: %s", result)

    interrupt_system.on_interrupt(on_interrupt)

    interrupt_system.command_interface.on_new_task(
        lambda task: router.handle_voice_command(task)
    )

    # ── Camera (local) ──
    use_separate_cameras = args.camera_index != args.robot_camera_index
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        log.error("Failed to open camera %d", args.camera_index)
        sys.exit(1)
    log.info("Camera %d opened (Qwen observation)", args.camera_index)
    if use_separate_cameras:
        log.info("Robot uses separate camera %d (no conflict)", args.robot_camera_index)
    else:
        log.info("Same camera for Qwen and robot — will pause Qwen during execution")

    # ── Microphone (local) ──
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

    # ── Start predictor ──
    predictor.start(num_workers=2)
    log.info("Prediction engine started (interval=0.5s)")

    # ── Ready ──
    log.info("=" * 55)
    log.info("  SYSTEM READY — Speak a command to start")
    log.info("  'pick up the pink ball'")
    log.info("  'pick up the yellow ball'")
    log.info("  Interrupt: 'no, pick up the other one'")
    log.info("  Say 'stop' to halt the robot")
    log.info("=" * 55)

    last_frame_time = 0.0
    frame_interval = 0.1  # 10fps for Qwen (inference at 2Hz)
    camera_paused = False

    try:
        while True:
            now = time.time()

            # Update Qwen with current state
            predictor.set_robot_state(
                f"state={robot.state}, policy={robot.active_policy or 'none'}"
            )

            # Camera management: pause if same camera and robot is running
            if not use_separate_cameras:
                if robot.state == "running" and not camera_paused:
                    cap.release()
                    camera_paused = True
                    log.info("Qwen camera paused (robot using camera)")
                elif robot.state != "running" and camera_paused:
                    cap = cv2.VideoCapture(args.camera_index)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera_paused = False
                    log.info("Qwen camera resumed")

            # Grab frames at 10fps
            if not camera_paused and now - last_frame_time >= frame_interval:
                ret, frame = cap.read()
                if ret:
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

                # Waiting for initial voice command
                if robot.state in ("idle", "stopping") and pred.predicted_intent in (
                    "new_command", "approach", "gesture"
                ):
                    if pred.target_object and pred.target_object != "none":
                        policy = resolve_policy(pred.target_object)
                        if policy:
                            log.info("Voice command: '%s' -> %s",
                                     pred.target_object, policy)
                            router.handle_voice_command(
                                f"pick up the {pred.target_object}"
                            )

                router.handle_prediction(pred_dict)

            time.sleep(0.05)

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        predictor.stop()
        if not camera_paused:
            cap.release()
        if audio_stream:
            audio_stream.stop()
        robot.shutdown()
        log.info("System stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni + SO101 Robot — Unified System"
    )
    parser.add_argument("--vllm-url", default="http://192.168.2.25:8000/v1",
                        help="vLLM server URL on s99")
    parser.add_argument("--robot-port", default="/dev/ttyACM1",
                        help="Robot serial port")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Camera index for Qwen observation")
    parser.add_argument("--robot-camera-index", type=int, default=8,
                        help="Camera index for robot policy (LeRobot)")
    parser.add_argument("--mic-index", type=int, default=None,
                        help="Microphone device index")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable microphone (video-only)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
