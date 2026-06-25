# Applying the HRI gate to Unitree's `eval_g1_isaac_gr00t.py`

The G1 runner is Unitree's repo (with NVIDIA/Unitree submodules) — not re-hosted
here. To reproduce the integration on a fresh clone:

1. Clone Unitree's `g1_imitation_learning` repo and set it up per their README
   (the one you already use to run the GR00T eval).
2. Copy `g1_eval/hri_gate.py` (this folder) into:
   ```
   unitree_lerobot/unitree_lerobot/eval_robot/utils/hri_gate.py
   ```
3. Install pyzmq in the gr00t container once: `uv pip install pyzmq`.
4. Apply the **5 edits** below to
   `unitree_lerobot/unitree_lerobot/eval_robot/eval_g1_isaac_gr00t.py`.

All edits are additive and gated by `cfg.hri_enable`; without `--hri_enable=true`
the script behaves exactly as upstream.

---

### Edit 1 — config fields (in `EvalIsaacGr00tRealConfig`)

Find:
```python
    max_action_delta: float = 0.0
    max_steps: int = 0

    def __post_init__(self):
```
Replace with:
```python
    max_action_delta: float = 0.0
    max_steps: int = 0
    # ── Qwen HRI bridge (optional; default off → vanilla eval unchanged) ──
    hri_enable: bool = False
    hri_state_port: int = 5701
    hri_cmd_port: int = 5702
    hri_frame_hz: float = 10.0

    def __post_init__(self):
```

### Edit 2 — the home helper (after `_maybe_clamp_action_delta`)

Append after that function:
```python
def _hri_go_home(arm_ctrl, arm_ik, current_q, target_q, frequency, steps=30):
    """Smoothly interpolate both arms from the current pose back to the pose
    captured at startup (init_arm_pose). Used by the HRI bridge's 'home'
    command after a confirmed task completion."""
    cur = np.asarray(current_q, dtype=float)
    tgt = np.asarray(target_q, dtype=float)
    dt = 1.0 / max(float(frequency), 1.0)
    for i in range(1, steps + 1):
        a = cur + (tgt - cur) * (i / steps)
        arm_ctrl.ctrl_dual_arm(a, arm_ik.solve_tau(a))
        time.sleep(dt)
```

### Edit 3 — declare the gate handle (in `eval_policy`, the None-init block)

Find:
```python
    client = None
    image_client = None
    robot_interface = None

    try:
```
Replace with:
```python
    client = None
    image_client = None
    robot_interface = None
    hri_gate = None

    try:
```

### Edit 4 — create the gate + state vars (after the init-pose move)

Find:
```python
        arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
        time.sleep(1.0)

        logger_mp.info(f"Starting Isaac-GR00T evaluation loop at {cfg.frequency} Hz.")
        idx = 0
```
Replace with:
```python
        arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
        time.sleep(1.0)

        # ── HRI bridge state machine (only active with --hri_enable) ──────
        hri_hold = False
        hri_task = task
        hri_home_request = False
        _loop_hz = 0.0
        if cfg.hri_enable:
            from unitree_lerobot.eval_robot.utils.hri_gate import HRIGate
            hri_gate = HRIGate(
                state_port=cfg.hri_state_port,
                cmd_port=cfg.hri_cmd_port,
                frame_hz=cfg.hri_frame_hz,
            )
            hri_hold = True  # held until the Qwen brain issues 'run'
            logger_mp.info(
                "[HRI] gate up — state/frame PUB :%d, command SUB :%d. "
                "Robot HELD at init pose until the brain sends a start command.",
                cfg.hri_state_port, cfg.hri_cmd_port,
            )

        logger_mp.info(f"Starting Isaac-GR00T evaluation loop at {cfg.frequency} Hz.")
        idx = 0
```

### Edit 5a — the in-loop tap + gate (right after the arm-state read)

Find:
```python
            if current_arm_q is None:
                raise RuntimeError("Failed to read current arm state from robot.")

            left_ee_state = right_ee_state = np.array([])
```
Replace with:
```python
            if current_arm_q is None:
                raise RuntimeError("Failed to read current arm state from robot.")

            # ── HRI bridge: forward cam_head + state to the brain, take its
            #    routing decisions. All no-ops unless --hri_enable. ──────────
            if hri_gate is not None:
                cam = observation.get("observation.images.cam_head")
                if cam is not None:
                    # cam_head is an RGB tensor; the brain expects BGR (the
                    # cv2/webcam convention SO-101 feeds the predictor).
                    cam_np = cam.numpy() if hasattr(cam, "numpy") else np.asarray(cam)
                    hri_gate.publish_frame(np.ascontiguousarray(cam_np[:, :, ::-1]))
                hri_gate.publish_state("idle" if hri_hold else "running",
                                       hri_task, _loop_hz)
                cmd = hri_gate.poll_command()
                if cmd:
                    c = cmd.get("command")
                    if c in ("run", "switch"):
                        if cmd.get("task"):
                            hri_task = cmd["task"]
                        hri_hold = False
                        logger_mp.info("[HRI] %s → task='%s'", c, hri_task)
                    elif c == "hold":
                        hri_hold = True
                        logger_mp.info("[HRI] hold")
                    elif c == "home":
                        hri_home_request = True
                # Home: interpolate both arms back to the captured init pose.
                if hri_home_request:
                    logger_mp.info("[HRI] homing to init pose")
                    _hri_go_home(arm_ctrl, arm_ik, current_arm_q, init_arm_pose, cfg.frequency)
                    hri_home_request = False
                    hri_hold = True
                    idx += 1
                    continue
                # Held: freeze in place; skip the policy query and the apply.
                if hri_hold:
                    arm_ctrl.ctrl_dual_arm(current_arm_q, arm_ik.solve_tau(current_arm_q))
                    time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
                    idx += 1
                    continue
                # Running under HRI: use the brain-provided task string.
                if hri_task:
                    task = hri_task

            left_ee_state = right_ee_state = np.array([])
```

### Edit 5b — loop-tail Hz tracking + gate cleanup

Find:
```python
            idx += 1
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    finally:
        if robot_interface:
```
Replace with:
```python
            idx += 1
            _elapsed = time.perf_counter() - loop_start_time
            if _elapsed > 0:
                _loop_hz = 0.9 * _loop_hz + 0.1 * (1.0 / _elapsed)
            time.sleep(max(0, (1.0 / cfg.frequency) - _elapsed))
    finally:
        if hri_gate is not None:
            hri_gate.close()
        if robot_interface:
```

---

After these edits, `python -m py_compile eval_g1_isaac_gr00t.py` should pass, and
launching with `--hri_enable=true` activates the bridge (see `../README_G1.md`).
