"""
G1ControlBridge — implements the exact robot interface PolicyRouter depends on
(the 9-method contract), but instead of driving an arm directly (as
GrootRobotController does for SO-101) it forwards routing decisions to the
Unitree G1 control loop over ZMQ (see g1_link.G1Link).

The brain owns the policy state machine (idle / running / active_task) here,
locally — identical semantics to GrootRobotController — and the eval loop is a
dumb executor that does whatever the latest command says.

Router contract reproduced (verified against PolicyRouter usage):
    state            (property)  → "idle" | "running"
    active_policy    (property)  → policy name | None
    current_hz       (property)  → control-loop Hz (advisory, from eval)
    tasks                        → TaskRegistry
    start_policy(name) -> dict   → cold-start            (sends "run")
    switch_policy(name) -> dict  → mid-task switch       (sends "switch")
    stop() -> dict               → halt                  (sends "hold")
    go_home(...)                 → return to start pose   (sends "home")
    place_object(...)            → put-back on interrupt  (v1: no-op, see note)
    clear_grasp_pose()           → reset put-back snapshot (v1: no-op)

v1 scope note (matches the agreed "full audio+visual, gate the loop" milestone):
The dexterous put-back / grasp-pose-snapshot behaviors (place_object,
clear_grasp_pose) are SO-101-specific arm interpolation and are NOT yet ported
to the G1's 16-DoF dual-arm + dex grippers. They are safe no-ops here: on a
mid-task interrupt the bridge still issues "hold" (the arm stops), it just
doesn't trace the object back to where it was grasped. go_home IS supported —
the eval loop returns both arms to the captured init pose.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from g1_link import G1Link

log = logging.getLogger("g1_bridge")


class G1ControlBridge:
    def __init__(self, tasks, link: G1Link, home_on_complete: bool = True):
        self.tasks = tasks                 # TaskRegistry (robot-agnostic)
        self._link = link
        self._home_on_complete = home_on_complete
        self._lock = threading.Lock()
        self._state = "idle"               # "idle" | "running"
        self._active: Optional[str] = None

    # ── lifecycle (mirrors GrootRobotController.connect/shutdown) ─────────
    def connect(self):
        self._link.start()
        # Make sure the loop starts in a held (idle) state, not executing.
        self._link.send_command("hold")
        log.info("G1ControlBridge connected — eval loop held (idle)")

    def shutdown(self):
        with _suppress():
            self._link.send_command("hold")
        self._link.stop()
        log.info("G1ControlBridge shut down")

    @property
    def link(self) -> G1Link:
        return self._link

    # ── state (read by PolicyRouter) ─────────────────────────────────────
    @property
    def state(self) -> str:
        return self._state

    @property
    def active_policy(self) -> Optional[str]:
        return self._active

    @property
    def current_hz(self) -> float:
        return self._link.loop_hz()

    # ── lifecycle commands (called by PolicyRouter._execute_policy_action) ─
    def _lang_for(self, policy_name: str) -> Optional[str]:
        task = self.tasks.get(policy_name)
        return task.lang if task is not None else None

    def start_policy(self, policy_name: str) -> dict:
        lang = self._lang_for(policy_name)
        if lang is None:
            return {"ok": False, "error": f"Unknown policy: {policy_name}"}
        with self._lock:
            if self._state == "running":
                return {"ok": False,
                        "error": "Already running. Call stop() or switch() first."}
            self._active = policy_name
            self._state = "running"
        self._link.send_command("run", lang)
        log.info("G1 start_policy → %s  ('%s')", policy_name, lang)
        return {"ok": True, "policy": policy_name, "state": "running"}

    def switch_policy(self, policy_name: str) -> dict:
        lang = self._lang_for(policy_name)
        if lang is None:
            return {"ok": False, "error": f"Unknown policy: {policy_name}"}
        with self._lock:
            prev = self._active
            self._active = policy_name
            self._state = "running"
        self._link.send_command("switch", lang)
        log.info("G1 switch_policy → %s  ('%s')  [was %s]", policy_name, lang, prev)
        return {"ok": True, "stopped": prev, "started": policy_name, "state": "running"}

    def stop(self) -> dict:
        with self._lock:
            if self._state != "running":
                return {"ok": True, "state": self._state, "msg": "Nothing to stop"}
            prev = self._active
            self._state = "idle"
            self._active = None
        self._link.send_command("hold")
        log.info("G1 stop → held (was %s)", prev)
        return {"ok": True, "stopped": prev, "state": "idle"}

    # ── completion / interrupt behaviors ─────────────────────────────────
    def go_home(self, duration_s: float = 1.2, steps: int = 30) -> None:
        # The eval loop moves both arms back to the pose captured at startup.
        if not self._home_on_complete:
            log.info("G1 go_home suppressed (--no-home)")
            return
        self._link.send_command("home")
        log.info("G1 go_home → eval returning arms to init pose")

    def place_object(self, duration_s: float = 0.8, steps: int = 24) -> None:
        # v1: no dexterous put-back on G1 yet. The preceding stop() already
        # issued "hold", so the arm is stationary; we simply don't trace the
        # object back to its grasp pose. See module docstring.
        log.info("G1 place_object → no-op (v1: put-back not ported to G1 dual-arm)")

    def clear_grasp_pose(self) -> None:
        # v1: no grasp-pose snapshot on G1 (no put-back). No-op.
        pass


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
