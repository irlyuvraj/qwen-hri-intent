"""
Bridge contract test — G1ControlBridge state machine + command emission.
Uses the real G1Link + HRIGate loopback so we assert the bridge's 9-method
contract produces the right ZMQ commands and state transitions. No robot/Qwen.

Run:  python3 test_g1_bridge.py
"""

import sys
import time
import importlib.util

from g1_link import G1Link
from g1_bridge import G1ControlBridge

GATE_PATH = ("/Users/yuvraj/G1 - codes/g1_imimtation_learning-main/"
             "unitree_lerobot/unitree_lerobot/eval_robot/utils/hri_gate.py")
_spec = importlib.util.spec_from_file_location("hri_gate", GATE_PATH)
hri_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hri_gate)
HRIGate = hri_gate.HRIGate


# Minimal stand-in for TaskRegistry (only .get(name).lang is used by the bridge).
class _Task:
    def __init__(self, name, lang):
        self.name, self.lang = name, lang


class _Reg:
    def __init__(self):
        self._t = {
            "pick_green_can": _Task("pick_green_can", "pick up the green monster can."),
            "pick_orange_can": _Task("pick_orange_can", "pick up the orange monster can."),
        }

    def get(self, name):
        return self._t.get(name)


PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    PASS += int(bool(cond)); FAIL += int(not cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def drain_latest(gate, tries=8):
    """Poll a few times and return the last non-None command."""
    last = None
    for _ in range(tries):
        c = gate.poll_command()
        if c is not None:
            last = c
        time.sleep(0.03)
    return last


def main():
    state_port, cmd_port = 5911, 5912
    gate = HRIGate(state_port=state_port, cmd_port=cmd_port, frame_hz=1000.0)
    link = G1Link(g1_host="127.0.0.1", state_port=state_port, cmd_port=cmd_port)
    reg = _Reg()
    bridge = G1ControlBridge(tasks=reg, link=link, home_on_complete=True)
    bridge.connect()  # starts link, sends initial 'hold'
    time.sleep(0.5)
    drain_latest(gate)  # consume the connect-time 'hold'

    print("[1] initial state")
    check("starts idle", bridge.state == "idle")
    check("no active policy", bridge.active_policy is None)

    print("[2] start_policy → running + 'run' command with lang")
    r = bridge.start_policy("pick_green_can")
    check("start_policy ok", r.get("ok") is True)
    check("state running", bridge.state == "running")
    check("active = green", bridge.active_policy == "pick_green_can")
    cmd = drain_latest(gate)
    check("emitted run", cmd and cmd.get("command") == "run")
    check("run carries green lang", cmd and cmd.get("task") == "pick up the green monster can.")

    print("[3] start while running is rejected")
    r2 = bridge.start_policy("pick_orange_can")
    check("second start rejected", r2.get("ok") is False)

    print("[4] switch_policy → stays running, new lang, 'switch' command")
    r3 = bridge.switch_policy("pick_orange_can")
    check("switch ok", r3.get("ok") is True)
    check("active = orange", bridge.active_policy == "pick_orange_can")
    cmd = drain_latest(gate)
    check("emitted switch", cmd and cmd.get("command") == "switch")
    check("switch carries orange lang", cmd and cmd.get("task") == "pick up the orange monster can.")

    print("[5] go_home → 'home' command")
    bridge.go_home()
    cmd = drain_latest(gate)
    check("emitted home", cmd and cmd.get("command") == "home")

    print("[6] stop → idle + 'hold' command")
    r4 = bridge.stop()
    check("stop ok", r4.get("ok") is True)
    check("state idle", bridge.state == "idle")
    check("active cleared", bridge.active_policy is None)
    cmd = drain_latest(gate)
    check("emitted hold", cmd and cmd.get("command") == "hold")

    print("[7] unknown policy rejected")
    check("unknown rejected", bridge.start_policy("nope").get("ok") is False)

    print("[8] go_home suppressed when home_on_complete=False")
    b2 = G1ControlBridge(tasks=reg, link=link, home_on_complete=False)
    drain_latest(gate)
    b2.go_home()
    cmd = drain_latest(gate)
    check("no home command emitted", cmd is None)

    bridge.shutdown()
    gate.close()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
