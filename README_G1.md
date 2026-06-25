# G1 HRI Integration — Anticipatory Qwen brain over Unitree GR00T eval loop

This wires the existing Qwen3-Omni HRI brain (intent prediction → stop / switch /
complete / home) onto the **Unitree G1** running the GR00T N1.7 monster-tray
pick-place policy — **without touching the working SO-101 system** and **without
rewriting Unitree's G1 driver**.

## Design in one picture

```
   BRAIN (native, this repo)                     G1 HOST (docker, Unitree repo)
   run_system_g1.py                              eval_g1_isaac_gr00t.py --hri_enable
   ├─ mic (local)                                ├─ owns the 30 Hz control loop
   ├─ Qwen vLLM client ──────► s99:8000          ├─ drives arms (DDS) + grippers
   ├─ FastQwenInferenceEngine  (REUSED)          ├─ reads cam_head (image_host)
   ├─ StreamingIntentPredictor (REUSED)          │
   ├─ PolicyRouter             (REUSED)          │   HRIGate (utils/hri_gate.py)
   ├─ G1ControlBridge ◄── 9-method contract      │   ├─ PUB state + cam_head  :5701 ─┐
   └─ G1Link  ──── ZMQ ──────────────────────────┼──►SUB                              │
        SUB state+frame  ◄───────────────────────┼───PUB :5701  (frames @10Hz)  ◄─────┘
        PUB run/hold/switch/home ───────────────►┼───SUB :5702  (gate the apply)
```

The brain owns the idle/running/active-task state machine (in `G1ControlBridge`,
identical semantics to `GrootRobotController`). The eval loop is a dumb executor:
each tick it publishes `cam_head` + state to the brain, polls the latest command,
and **holds in place** unless the brain says `run`.

## Why SO-101 still works unchanged

`run_system_groot.py` is not modified. The brain modules
(`FastQwenInferenceEngine`, `StreamingIntentPredictor`, `InterruptDetectionSystem`,
`PolicyRouter`, `TaskRegistry`) are robot-agnostic and shared. SO-101 uses
`GrootRobotController`; G1 uses `G1ControlBridge`. Both satisfy the same 9-method
router contract: `state`, `active_policy`, `current_hz`, `tasks`, `start_policy`,
`switch_policy`, `stop`, `go_home`, `place_object`, `clear_grasp_pose`.

## Files added

| File | Side | Purpose |
|---|---|---|
| `tasks_g1.yaml` | brain | green/orange monster-can tasks (exact trained lang strings) |
| `g1_link.py` | brain | ZMQ client: recv state+frames, send commands |
| `g1_bridge.py` | brain | `G1ControlBridge` (9-method contract → ZMQ) |
| `run_system_g1.py` | brain | brain entry point (mirror of `run_system_groot.run`) |
| `unitree_lerobot/.../utils/hri_gate.py` | G1 repo | eval-side ZMQ gate (self-contained) |
| `eval_g1_isaac_gr00t.py` (patched) | G1 repo | +camera tap +stop/switch/home gate, all behind `--hri_enable` |

## How to run

**1. Services on s99** (unchanged): vLLM Qwen on :8000, GR00T N1.7 policy server on :5555.
See `CLAUDE.md` and `project_g1_groot_setup` for the exact commands. Use the
checkpoint `monster-tray-debug/monster-tray-pickplace/checkpoint-30000`.

**2. G1 eval loop (docker)** — same as your working command, **plus** `--hri_enable`
and the ports. `pyzmq` must be in the container (`uv pip install pyzmq` once):

```
docker compose -f unitree_lerobot/docker-compose.gr00t.yml run --rm gr00t \
  python unitree_lerobot/eval_robot/eval_g1_isaac_gr00t.py \
    --repo_id=tysyuvraj/monster-tray-pickplace \
    --root=/workspace/Isaac-GR00T/monster-tray-pickplace \
    --episodes=0 --frequency=30 --arm=G1_29 --ee=dex1 --gripper_backend=dainamo_dds \
    --visualization=true --network_interface=enx00e04c1f1a38 \
    --image_host=192.168.123.200 --policy_host=127.0.0.1 --policy_port=5555 \
    --open_loop_horizon=8 --max_steps=0 --max_action_delta=0.0 \
    --hri_enable=true --hri_state_port=5701 --hri_cmd_port=5702 --hri_frame_hz=10
```

Press `s` as usual. The arm moves to its init pose and then **holds** (idle),
waiting for a spoken command. Without `--hri_enable` the script behaves exactly
as before.

**3. Brain (native, same venv as the SO-101 system)** — needs mic + reach to the
G1 host and to s99's vLLM:

```
python run_system_g1.py \
  --vllm-url http://192.168.2.25:8000/v1 \
  --tasks tasks_g1.yaml \
  --g1-host 192.168.123.200 \
  --state-port 5701 --cmd-port 5702 \
  --metrics ~/sessions/g1_metrics_$(date +%Y%m%d_%H%M%S).jsonl
```

Then speak: *"pick up the green can"* → cold-start; *"stop"* → halt; *"the orange
one"* → switch; the visual completion verifier auto-stops + homes when a can is on
the tray.

> The brain must run in the **same venv as `run_system_groot.py`** — it imports
> `PolicyRouter` from that module (which imports lerobot/gr00t at class-def time).
> Those deps already exist on the SO-101 host. The brain never calls them.

## On-hardware bring-up checklist (I could not test these — verify live)

These are the assumptions baked in that need a real-robot confirmation, in order:

1. **ZMQ reachability.** The G1 host binds :5701/:5702; the brain connects. If the
   brain logs "No state from G1 eval loop yet", check the host IP (`--g1-host`),
   that the container publishes on the host network (docker `network_mode: host`
   or port mapping for 5701/5702), and firewall.
2. **Frame color.** The gate sends `cam_head` as **BGR** (RGB→`[:, :, ::-1]`) to
   match SO-101. Confirm Qwen names the cans correctly ("green"/"orange"); if
   colors look swapped, the channel order is wrong — flip it in
   `eval_g1_isaac_gr00t.py` at the `publish_frame` call.
3. **Hold semantics.** When held/idle the loop re-commands the **current measured
   pose** every tick (`ctrl_dual_arm(current_arm_q, ...)`). Verify the arm freezes
   cleanly and does not sag or jitter. If the controller prefers no command while
   holding, change the held branch to `continue` without the `ctrl_dual_arm` call.
4. **Cold-start gating.** The brain starts the loop **held**; nothing moves until a
   spoken command. Confirm `run` un-holds and the policy executes the right task.
5. **Switch.** Mid-task "the orange one" → the brain sends `switch` with the new
   lang; the loop swaps the `task` string fed to `select_action`. Note: v1 switch
   just changes the task string — there is **no gripper-release / regrasp** on G1
   yet (see Known gaps).
6. **Completion + home.** The visual verifier ("is a can on the tray?") should
   auto-stop; the brain then sends `home` and the loop interpolates both arms to
   the init pose. Confirm the home motion is safe from a mid-task pose (it's a
   straight joint-space lerp — watch for self-collision; reduce speed via
   `frequency` or raise `steps` in `_hri_go_home` if needed).
7. **Latency of the anticipatory claim.** `cam_head` is network-streamed, so Qwen
   sees frames slightly later than on SO-101's local webcam. Measure
   `link.frame_age()` and the stop latency; if frame lag is large it weakens the
   "anticipatory" timing story and you may want to raise `--hri_frame_hz`.

## Known gaps (v1, intentional)

- **Put-back on interrupt is a no-op on G1.** `place_object`/`clear_grasp_pose`
  are SO-101 arm-interpolation behaviors not yet ported to the G1's 16-DoF
  dual-arm + dex grippers. On interrupt the arm holds in place (object not traced
  back to its grasp pose). `go_home` *is* supported.
- **Switch does not release/regrasp** — it only changes the task string. If GR00T
  keeps the held can across a switch, add a release step to the `switch` command
  handling in the eval loop (open the dex gripper before resuming).
- **`--no-vad` cold-start** relies on the WAITING-prompt `command_*` path exactly
  as on SO-101; default (VAD on) uses the energy-VAD → transcribe path. Both go
  through the same router; pick per demo.
