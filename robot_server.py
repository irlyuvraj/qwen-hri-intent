#!/usr/bin/env python3
"""
Robot Control Server — runs on SO101 machine (gqu6x)
Receives commands from Qwen prediction system (PN62) and controls LeRobot policy execution.

Usage:
    python robot_server.py --port 5555 --robot-port /dev/ttyACM1 --camera-index 8

The server exposes a simple HTTP API:
    POST /start    {"policy": "pick_pink_ball"}     → load policy + start execution
    POST /stop     {}                                → stop current execution
    POST /switch   {"policy": "pick_yellow_ball"}   → stop + load new policy + start
    GET  /status                                     → current state + active policy
    GET  /frame                                      → latest camera frame (JPEG)
    GET  /health                                     → connectivity check
"""

import argparse
import json
import logging
import subprocess
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("robot_server")

# ── Policy registry ──
# Maps friendly names to local checkpoint paths or HuggingFace repo IDs
POLICY_REGISTRY = {
    "pick_pink_ball":   "tysyuvraj/so101-act-pick-pink-ball",
    "pick_yellow_ball": "tysyuvraj/so101-act-pick-yellow-ball",
    "pick_and_correct": "tysyuvraj/so101-act-pick-and-correct",
}


class RobotController:
    """Manages LeRobot policy server + robot client processes."""

    def __init__(self, robot_port: str, camera_index: int, fps: int = 30):
        self.robot_port = robot_port
        self.camera_index = camera_index
        self.fps = fps
        self._server_proc = None
        self._client_proc = None
        self._active_policy = None
        self._state = "idle"  # idle | starting | running | stopping
        self._lock = threading.Lock()
        self._camera = None
        self._latest_frame = None
        self._camera_lock = threading.Lock()
        self._camera_thread = None
        self._camera_running = False

    def start_camera(self):
        """Start background camera capture for streaming to Qwen."""
        if self._camera_running:
            return
        self._camera_running = True
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        log.info("Camera capture started (index=%d)", self.camera_index)

    def stop_camera(self):
        """Stop background camera to avoid conflict with robot_client."""
        self._camera_running = False
        if self._camera_thread:
            self._camera_thread.join(timeout=3)
            self._camera_thread = None
        with self._camera_lock:
            self._latest_frame = None
        log.info("Camera capture stopped")

    def _camera_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
        while self._camera_running:
            ret, frame = cap.read()
            if ret:
                with self._camera_lock:
                    self._latest_frame = frame
            time.sleep(0.03)  # ~30fps
        cap.release()

    def get_frame(self) -> bytes | None:
        """Get latest camera frame as JPEG bytes."""
        with self._camera_lock:
            if self._latest_frame is None:
                return None
            _, buf = cv2.imencode(".jpg", self._latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes()

    def start_policy(self, policy_name: str) -> dict:
        """Start executing a policy on the robot."""
        with self._lock:
            if policy_name not in POLICY_REGISTRY:
                return {"ok": False, "error": f"Unknown policy: {policy_name}",
                        "available": list(POLICY_REGISTRY.keys())}

            if self._state == "running":
                return {"ok": False, "error": "Already running. Call /stop or /switch first."}

            self._state = "starting"
            policy_path = POLICY_REGISTRY[policy_name]

            # Stop camera to avoid conflict with robot_client
            self.stop_camera()

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
                    import socket
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
                self.start_camera()
                return {"ok": False, "error": f"Server failed to start: {stderr[:500]}"}

            # Start robot client
            client_cmd = [
                sys.executable, "-m", "lerobot.async_inference.robot_client",
                f"--server_address=127.0.0.1:8080",
                f"--robot.type=so101_follower",
                f"--robot.port={self.robot_port}",
                f"--robot.id=my_awesome_follower_arm",
                f'--robot.cameras={{ front: {{type: opencv, index_or_path: {self.camera_index}, '
                f'width: 640, height: 360, fps: 60}}}}',
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
                self.start_camera()
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
            # Restart camera for Qwen frame streaming
            self.start_camera()
            return {"ok": True, "stopped": prev, "state": "idle"}

    def switch_policy(self, policy_name: str) -> dict:
        """Stop current policy and start a new one."""
        with self._lock:
            if policy_name not in POLICY_REGISTRY:
                return {"ok": False, "error": f"Unknown policy: {policy_name}",
                        "available": list(POLICY_REGISTRY.keys())}

        # Stop current (releases lock)
        stop_result = self.stop()
        time.sleep(0.5)
        # Start new
        start_result = self.start_policy(policy_name)
        return {
            "ok": start_result["ok"],
            "stopped": stop_result.get("stopped"),
            "started": policy_name if start_result["ok"] else None,
            "state": start_result.get("state", "error"),
            "error": start_result.get("error"),
        }

    def status(self) -> dict:
        return {
            "state": self._state,
            "active_policy": self._active_policy,
            "available_policies": list(POLICY_REGISTRY.keys()),
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
        self._camera_running = False
        self.stop()


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for robot commands."""

    controller: RobotController = None  # set by server setup

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        if self.path == "/start":
            policy = data.get("policy", "")
            result = self.controller.start_policy(policy)
            self._respond(200 if result["ok"] else 400, result)

        elif self.path == "/stop":
            result = self.controller.stop()
            self._respond(200, result)

        elif self.path == "/switch":
            policy = data.get("policy", "")
            result = self.controller.switch_policy(policy)
            self._respond(200 if result["ok"] else 400, result)

        else:
            self._respond(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"ok": True})

        elif self.path == "/status":
            self._respond(200, self.controller.status())

        elif self.path == "/frame":
            frame_bytes = self.controller.get_frame()
            if frame_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(frame_bytes))
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self._respond(503, {"error": "No frame available"})

        else:
            self._respond(404, {"error": "Not found"})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        log.debug(format, *args)


def main():
    parser = argparse.ArgumentParser(description="SO101 Robot Control Server")
    parser.add_argument("--port", type=int, default=5555, help="HTTP port")
    parser.add_argument("--robot-port", default="/dev/ttyACM1", help="Robot serial port")
    parser.add_argument("--camera-index", type=int, default=8, help="Camera index")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    controller = RobotController(args.robot_port, args.camera_index, args.fps)
    controller.start_camera()

    RequestHandler.controller = controller

    server = HTTPServer(("0.0.0.0", args.port), RequestHandler)
    log.info("Robot server listening on 0.0.0.0:%d", args.port)
    log.info("Available policies: %s", list(POLICY_REGISTRY.keys()))

    def shutdown_handler(sig, frame):
        log.info("Shutting down...")
        controller.shutdown()
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        server.serve_forever()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
