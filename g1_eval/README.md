# `g1_eval/` — the eval-loop side of the G1 HRI bridge

These are the files that go into **Unitree's** `g1_imitation_learning` repo (the
G1 runner), kept here so the whole integration is reproducible from this repo
alone without re-hosting Unitree's 105 MB repo + submodules.

| File | What it is |
|---|---|
| `hri_gate.py` | The eval-side ZMQ gate (new file — yours). |
| `eval_g1_isaac_gr00t.py` | A **ready-to-use modified copy** of Unitree's eval script with the HRI bridge already applied. |
| `APPLY_PATCH.md` | The 5 manual edits, for when you'd rather patch your own copy than overwrite. |

## Two ways to install (pick one)

### A. Drop-in (easiest — same Unitree version)
If your G1 computer uses the **same `g1_imimtation_learning` version** these were
made against, just copy both files into the Unitree tree, overwriting:

```
g1_eval/hri_gate.py             → unitree_lerobot/unitree_lerobot/eval_robot/utils/hri_gate.py
g1_eval/eval_g1_isaac_gr00t.py  → unitree_lerobot/unitree_lerobot/eval_robot/eval_g1_isaac_gr00t.py
```

You can download them straight from GitHub on the G1 computer, e.g.:
```bash
cd /path/to/g1_imimtation_learning/unitree_lerobot/unitree_lerobot/eval_robot
RAW=https://raw.githubusercontent.com/irlyuvraj/qwen-hri-intent/main/g1_eval
curl -fsSL $RAW/eval_g1_isaac_gr00t.py -o eval_g1_isaac_gr00t.py
curl -fsSL $RAW/hri_gate.py            -o utils/hri_gate.py
```

### B. Patch (safer — different Unitree version)
If your Unitree checkout differs, don't overwrite — copy only `hri_gate.py`, then
apply the 5 small edits in `APPLY_PATCH.md` to your own `eval_g1_isaac_gr00t.py`.

## Then, regardless of A or B
1. Install pyzmq in the gr00t container once: `uv pip install pyzmq`
2. Make sure docker exposes ports **5701/5702** to the host (`network_mode: host`
   or `-p 5701:5701 -p 5702:5702`).
3. Launch the eval loop with `--hri_enable=true --hri_state_port=5701
   --hri_cmd_port=5702 --hri_frame_hz=10` (full command in `../README_G1.md`).

Without `--hri_enable`, the modified file behaves **exactly like the original**.
