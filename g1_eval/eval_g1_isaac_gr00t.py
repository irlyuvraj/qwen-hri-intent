# ─────────────────────────────────────────────────────────────────────────────
# MODIFIED COPY — derived from Unitree's g1_imitation_learning
#   unitree_lerobot/unitree_lerobot/eval_robot/eval_g1_isaac_gr00t.py
# with the Qwen HRI bridge added (all additions gated behind --hri_enable; the
# default eval path is unchanged). Upstream copyright/license applies.
#
# Ready-to-use: if your Unitree checkout is the same g1_imimtation_learning
# version, drop THIS file and g1_eval/hri_gate.py straight into your repo (see
# g1_eval/APPLY_PATCH.md "ready-to-use" note). Otherwise apply the 5 edits in
# APPLY_PATCH.md to your own copy instead of overwriting.
# ─────────────────────────────────────────────────────────────────────────────
"""Real G1 evaluation using an Isaac-GR00T policy server."""

from contextlib import suppress
from dataclasses import asdict, dataclass
import logging
from pprint import pformat
import time

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging
from multiprocessing.sharedctypes import SynchronizedArray

from unitree_lerobot.eval_robot.make_robot import (
    process_images_and_observations,
    setup_image_client,
    setup_robot_interface,
)
from unitree_lerobot.eval_robot.utils.isaac_gr00t_adapter import (
    IsaacGr00tG1Adapter,
    IsaacGr00tZmqClient,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.ee_pose_utils import assert_lerobot_action_space
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig, to_list, to_scalar

import logging_mp


logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


@dataclass
class EvalIsaacGr00tRealConfig(EvalRealConfig):
    policy: str = ""
    policy_host: str = "127.0.0.1"
    policy_port: int = 5555
    policy_timeout_ms: int = 15000
    policy_api_token: str = ""
    open_loop_horizon: int = 8
    lang_instruction: str = ""
    gr00t_video_map: str = (
        "cam_head:observation.images.cam_head,"
        "cam_right_wrist:observation.images.cam_right_wrist"
    )
    gr00t_state_slices: str = (
        "left_arm:0:7,right_arm:7:14,left_ee:14:15,right_ee:15:16"
    )
    gr00t_action_keys: str = "left_arm,right_arm,left_ee,right_ee"
    gr00t_language_key: str = "annotation.human.task_description"
    max_action_delta: float = 0.0
    max_steps: int = 0
    # ── Qwen HRI bridge (optional; default off → vanilla eval unchanged) ──
    hri_enable: bool = False
    hri_state_port: int = 5701
    hri_cmd_port: int = 5702
    hri_frame_hz: float = 10.0

    def __post_init__(self):
        pass

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return []


def _setup_policy(cfg: EvalIsaacGr00tRealConfig) -> tuple[IsaacGr00tZmqClient, IsaacGr00tG1Adapter]:
    client = IsaacGr00tZmqClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        timeout_ms=cfg.policy_timeout_ms,
        api_token=cfg.policy_api_token or None,
    )
    client.ping()
    adapter = IsaacGr00tG1Adapter(
        client,
        video_map=cfg.gr00t_video_map,
        state_slices=cfg.gr00t_state_slices,
        action_keys=cfg.gr00t_action_keys,
        language_key=cfg.gr00t_language_key,
        open_loop_horizon=cfg.open_loop_horizon,
        arm_dof=14,
    )
    adapter.reset()
    logger_mp.info("Connected to Isaac-GR00T server at %s:%s", cfg.policy_host, cfg.policy_port)
    logger_mp.info("Server modality config: %s", adapter.modality_config)
    return client, adapter


def _maybe_clamp_action_delta(action: np.ndarray, current_state: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return action
    dim = min(len(action), len(current_state))
    clipped = action.copy()
    clipped[:dim] = np.clip(clipped[:dim], current_state[:dim] - max_delta, current_state[:dim] + max_delta)
    return clipped


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


def eval_policy(cfg: EvalIsaacGr00tRealConfig, dataset: LeRobotDataset) -> None:
    logger_mp.info(f"Arguments: {cfg}")
    rerun_logger = RerunLogger() if cfg.visualization else None
    client = None
    image_client = None
    robot_interface = None
    hri_gate = None

    try:
        client, policy = _setup_policy(cfg)
        image_client, camera_config = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        expected_action_dim = arm_dof + (2 * ee_dof if cfg.ee else 0)

        from_idx = int(dataset.meta.episodes["dataset_from_index"][cfg.episodes])
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()
        task = cfg.lang_instruction or step.get("task", "")

        user_input = input("Enter 's' to initialize the robot and start Isaac-GR00T eval: ")
        if user_input.lower() != "s":
            return

        logger_mp.info("Initializing robot to starting pose...")
        tau = arm_ik.solve_tau(init_arm_pose)
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
        while cfg.max_steps <= 0 or idx < cfg.max_steps:
            loop_start_time = time.perf_counter()
            observation, current_arm_q = process_images_and_observations(
                image_client, camera_config, arm_ctrl
            )
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
            full_state = None
            if cfg.ee:
                with ee_shared_mem["lock"]:
                    full_state = np.array(ee_shared_mem["state"][:])
                    left_ee_state = full_state[:ee_dof]
                    right_ee_state = full_state[ee_dof:]

            state_np = np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0).astype(np.float32)
            observation["observation.state"] = torch.from_numpy(state_np).float()

            action_np = policy.select_action(observation, task)
            if action_np.ndim != 1 or action_np.shape[0] < expected_action_dim:
                raise RuntimeError(
                    f"Isaac-GR00T action shape {action_np.shape} is too small for "
                    f"arm_dof={arm_dof}, ee_dof={ee_dof}."
                )
            action_np = action_np[:expected_action_dim].astype(np.float32, copy=False)
            if not np.isfinite(action_np).all():
                raise RuntimeError(f"Isaac-GR00T action contains non-finite values: {action_np}")
            action_np = _maybe_clamp_action_delta(action_np, state_np, cfg.max_action_delta)

            arm_action = action_np[:arm_dof]
            tau = arm_ik.solve_tau(arm_action)
            arm_ctrl.ctrl_dual_arm(arm_action, tau)

            if cfg.ee:
                ee_action_start_idx = arm_dof
                left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                right_ee_action = action_np[
                    ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof
                ]
                if isinstance(ee_shared_mem["left"], SynchronizedArray):
                    ee_shared_mem["left"][:] = to_list(left_ee_action)
                    ee_shared_mem["right"][:] = to_list(right_ee_action)
                elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                    ee_shared_mem["left"].value = to_scalar(left_ee_action)
                    ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if rerun_logger:
                visualization_data(idx, observation, state_np, action_np, rerun_logger)

            idx += 1
            _elapsed = time.perf_counter() - loop_start_time
            if _elapsed > 0:
                _loop_hz = 0.9 * _loop_hz + 0.1 * (1.0 / _elapsed)
            time.sleep(max(0, (1.0 / cfg.frequency) - _elapsed))
    finally:
        if hri_gate is not None:
            hri_gate.close()
        if robot_interface:
            ee_ctrl = robot_interface.get("ee_ctrl")
            if hasattr(ee_ctrl, "close"):
                ee_ctrl.close()
        if image_client:
            image_client.close()
        if client:
            with suppress(Exception):
                client.close()


@parser.wrap()
def eval_main(cfg: EvalIsaacGr00tRealConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root or None)
    assert_lerobot_action_space(dataset, "joint", context="eval_g1_isaac_gr00t.py")
    eval_policy(cfg, dataset)
    logging.info("End of Isaac-GR00T real eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
