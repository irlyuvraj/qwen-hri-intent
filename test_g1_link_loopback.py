"""
Standalone loopback test for the G1 HRI ZMQ transport — NO robot, NO Qwen.

Exercises the real modules as written:
    HRIGate (eval side, G1 repo)  ◄── ZMQ ──►  G1Link (brain side, this repo)

Validates the full wire protocol:
  1. gate.publish_state(...)      → link.loop_state()/loop_hz()/is_loop_alive()
  2. gate.publish_frame(BGR)      → link.latest_frame()  (JPEG roundtrip, shape/color)
  3. link.send_command(cmd, task) → gate.poll_command()   (run/hold/switch/home)

Run:  python3 test_g1_link_loopback.py
Exit code 0 = all checks passed.
"""

import sys
import time
import importlib.util

import numpy as np

# Import the brain-side G1Link from this repo.
from g1_link import G1Link

# Import the eval-side HRIGate directly from the G1 repo by path (it lives in a
# package we don't have on sys.path; load the single file standalone).
GATE_PATH = ("/Users/yuvraj/G1 - codes/g1_imimtation_learning-main/"
             "unitree_lerobot/unitree_lerobot/eval_robot/utils/hri_gate.py")
_spec = importlib.util.spec_from_file_location("hri_gate", GATE_PATH)
hri_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hri_gate)
HRIGate = hri_gate.HRIGate


PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def main():
    state_port, cmd_port = 5901, 5902  # off the default ports to avoid clashes

    # Gate binds (the "G1 host"); link connects (the "brain").
    gate = HRIGate(state_port=state_port, cmd_port=cmd_port,
                   frame_hz=1000.0, jpeg_quality=90)  # high frame_hz so no throttle
    link = G1Link(g1_host="127.0.0.1", state_port=state_port, cmd_port=cmd_port)
    link.start()
    time.sleep(0.5)  # let PUB/SUB connections establish (slow-joiner)

    # ── 1. state channel ──────────────────────────────────────────────
    print("[1] state channel (gate → link)")
    for _ in range(10):
        gate.publish_state("running", "pick up the green monster can.", hz=29.5)
        time.sleep(0.02)
    time.sleep(0.2)
    st = link.loop_state()
    check("link received state", st.get("state") == "running")
    check("active_task propagated", st.get("active_task", "").startswith("pick up the green"))
    check("hz propagated (~29.5)", abs(link.loop_hz() - 29.5) < 0.01)
    check("is_loop_alive() True", link.is_loop_alive(max_age_s=2.0))

    # ── 2. frame channel (JPEG roundtrip, color/shape) ────────────────
    print("[2] frame channel (gate → link), BGR JPEG roundtrip")
    # Build a known BGR image: pure blue in BGR = (255,0,0). If color order is
    # preserved end-to-end, the decoded frame's blue channel dominates.
    h, w = 48, 64
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # B channel
    for _ in range(5):
        gate.publish_frame(bgr)
        time.sleep(0.02)
    time.sleep(0.3)
    got = link.latest_frame()
    check("frame received", got is not None)
    if got is not None:
        check("frame shape preserved (48x64x3)", got.shape == (h, w, 3))
        b, g, r = got[:, :, 0].mean(), got[:, :, 1].mean(), got[:, :, 2].mean()
        # JPEG is lossy; just assert blue dominates (color order intact).
        check(f"BGR color order intact (B={b:.0f}>>G={g:.0f},R={r:.0f})",
              b > 200 and g < 40 and r < 40)
    check("frame_age() fresh (<1s)", link.frame_age() < 1.0)

    # ── 3. command channel (link → gate) ──────────────────────────────
    print("[3] command channel (link → gate), latest-wins")
    # poll before sending: nothing
    check("poll_command() empty initially", gate.poll_command() is None)

    link.send_command("run", "pick up the orange monster can.")
    time.sleep(0.2)
    cmd = gate.poll_command()
    check("run command received", cmd is not None and cmd.get("command") == "run")
    check("run task propagated", cmd is not None
          and cmd.get("task", "").startswith("pick up the orange"))

    # latest-wins (CONFLATE): fire several, expect only the newest.
    link.send_command("hold")
    link.send_command("switch", "pick up the green monster can.")
    link.send_command("home")
    time.sleep(0.3)
    cmd = gate.poll_command()
    check("conflate keeps newest command (home)",
          cmd is not None and cmd.get("command") == "home")

    # ── teardown ──────────────────────────────────────────────────────
    link.stop()
    gate.close()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
