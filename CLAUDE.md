# CLAUDE.md — Qwen3-Omni + GR00T N1.6 HRI System (SO-101)
> Master's Thesis: Human-Robot Interaction — Intent-Aware Robot Control
> Last updated: April 26, 2026

Paste this file at the start of a new Claude conversation to restore full project context instantly.

---

## Project Overview

Real-time **multimodal execution monitor** for a learned robot policy. Uses **Qwen3-Omni-30B** (via vLLM) to watch a camera feed + listen to a microphone and predict what a human will do in the **next 1-2 seconds** — which object they're reaching for, whether they're approaching/gesturing/withdrawing, and whether the task scene is complete.

The system acts as an **execution monitor** for a **GR00T N1.6** policy running on an **SO-101 follower arm**. When human intent diverges from the robot's active task, it fires an interrupt (STOP/switch signal) to GR00T in real time.

**Research contribution (revised, April 25):**
Qwen3-Omni serves as the **unified multimodal backbone** for continuous scene understanding AND in-task voice control. Mid-task stops and switches are extracted directly from Qwen's continuous prediction via the `predicted_intent` field (`interrupt` for stops, `change_target` for switches) — no separate ASR or transcription path needed during execution. A small energy-VAD (~30 lines) remains for cold-start task initiation only.

Replacement scope:
- **In-task control (stop/switch)**: ✅ Pure Qwen3-Omni — no VAD, no transcription
- **Cold-start command capture**: VAD + Qwen.transcribe_audio (small auxiliary)
- **Visual intent + target ID + scene completion**: Pure Qwen3-Omni
- **Audio-context awareness during prediction**: Pure Qwen3-Omni

This replaces what would otherwise be Whisper (ASR) + VAP (turn-taking) + a visual intent encoder + a fusion network with Qwen3-Omni doing all the semantic work, and ~30 lines of VAD handling discrete cold-start segmentation.

**Key finding from April 25 session:** Qwen3-Omni performs internal multimodal fusion that is observable in two places — (a) the `predicted_intent` field flips to `"interrupt"` when audio "stop" coincides with arm motion, providing a usable in-task stop signal; (b) the free-form `reason` field writes things like `"Hand mid-motion toward yellow ball, but audio command 'stop' interrupts"`, providing explainable evidence of fusion. The dedicated `spoken_command` field experiment failed (Qwen refuses field discipline) but the `interrupt` intent path made it unnecessary for in-task control. See [Multimodal control experiments](#multimodal-control-experiments-april-25).

**Current status:** Fully integrated and running live on SO-101. GR00T N1.6 connected, voice commands work via both VAD path and `--no-vad` Qwen-only path, Qwen continuous multimodal predictions drive the 3-tier auto-stop, recording with HUD overlay (intent + reason + Hz stats) works. Visual intent predictions (approach/gesture/withdraw/continue) confirmed showing during execution in `--no-vad` mode (April 26 session).

---

## Hardware

| Machine | Role | Specs |
|---|---|---|
| MacBook (robot host) | Runs `run_system_groot.py`, drives SO-101, captures camera+mic | Local |
| `yuvraj@s99` | vLLM (Qwen) + GR00T policy server | 2× NVIDIA RTX PRO 6000 Blackwell, 96GB VRAM each |

**s99 IP:** `192.168.2.25`
- vLLM Qwen on port `8000`
- GR00T policy server on port `5555`

**Robot:** SO-101 follower arm
- USB port: `/dev/tty.usbmodem5AE70452961`
- Camera: index `0` (shared — Qwen sees same feed as GR00T). Camera delivers **640×480** natively; both consumers must request 640×480 or LeRobot's read_loop crashes when one client calls `cap.set()` and reconfigures the AVFoundation device. Frames are center-cropped to **640×360** (the GR00T training resolution) inside `_obs_to_policy_inputs()` and `_crop_to_training()`.
- Calibration: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json`

**Model:** `Qwen/Qwen3-Omni-30B-A3B-Instruct` at `/home/yuvraj/qwen_data/models/` on s99

**GR00T checkpoint:** `/home/yuvraj/so101_training/outputs/groot_n16_so101/checkpoint-20000`

---

## How to Run

```bash
# Always use the shell script — never paste the python command directly.
# Blank lines after a backslash-continuation terminate the shell command early.
bash run.sh                    # no recording
bash run.sh --record           # records to ~/sessions/<timestamp>.mp4
```

Or manually (no blank lines after the backslashes):
```bash
python run_system_groot.py \
  --vllm-url http://192.168.2.25:8000/v1 \
  --tasks tasks.yaml \
  --robot-port /dev/tty.usbmodem5AE70452961 \
  --camera-index 0 \
  --robot-camera-index 0 \
  --policy-host 192.168.2.25 \
  --policy-port 5555
```

**Start services on s99 first:**
```bash
# GR00T policy server
ssh yuvraj@192.168.2.25 'cd ~/Isaac-GR00T && \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python gr00t/eval/run_gr00t_server.py \
    --embodiment_tag NEW_EMBODIMENT \
    --model_path /home/yuvraj/so101_training/outputs/groot_n16_so101/checkpoint-20000 \
    --device cuda:1 --host 0.0.0.0 --port 5555 --strict'

# vLLM (Qwen) — separate GPU
VLLM_V1_ENABLED=0 vllm serve "/home/yuvraj/qwen_data/models/Qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --api-key vllm-omni --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"audio":1,"video":1,"image":1}' \
    --trust-remote-code --served-model-name "qwen3-30b-a3b" --max-num-seqs 4
```

**Verify s99 services:**
```bash
curl -H "Authorization: Bearer vllm-omni" http://192.168.2.25:8000/v1/models
nc -zv 192.168.2.25 5555
```

---

## File Structure

```
~/qwen-hri-intent/
├── run_system_groot.py         ← Main entry point: Qwen + GR00T + SO-101 (CURRENT)
├── run.sh                      ← Shell wrapper — use this to launch (avoids blank-line bug)
├── tasks.yaml                  ← Task registry: 2 tasks (pick pink / yellow cotton ball)
├── task_registry.py            ← TaskRegistry class: YAML loader + keyword resolver
├── qwen_inference_engine.py    ← FastQwenInferenceEngine: task_complete, active_task_lang, spoken_command (failed exp)
├── streaming_intent_predictor.py ← StreamingIntentPredictor: PredictionOutput w/ task_complete + spoken_command
├── interrupt_detection_system.py ← InterruptDetectionSystem + AudioInterruptDetector (energy VAD)
├── recorder.py                 ← SystemRecorder: mp4 capture with HUD overlay (intent, reason, Hz, seq counter)
├── file_based_predictor.py     ← CLI tool for offline video testing
├── compare_ground_truth.py     ← Evaluation against ground truth JSON
├── interrupt_test_runner.py    ← Test runner for interrupt scenarios
├── visualize_predictions.py    ← Matplotlib timeline plot
├── realtime_dashboard.py       ← Live browser dashboard (MJPEG + timeline)
└── generate_report.py          ← Self-contained interactive HTML report
```

---

## Task Registry (tasks.yaml)

Two tasks. Extend by adding entries to `tasks.yaml` — no code changes needed.

```yaml
tasks:
  - name: pick_pink_ball
    lang: "Pick up the pink cotton ball and place it in the bowl"
    object: "pink cotton ball"
    keywords: [pink, "pink ball", "pink cotton", "pink one"]

  - name: pick_yellow_ball
    lang: "Pick up the yellow cotton ball and place it in the bowl"
    object: "yellow cotton ball"
    keywords: [yellow, "yellow ball", "yellow cotton", "yellow one"]
```

`TaskRegistry.resolve()` uses **longest-matching keyword** (not first-match) so "pink ball" beats "ball" when both appear.

---

## System Architecture

```
MacBook (robot host)
│
├─ Camera (index 0, 640×480 native, cropped to 640×360 for GR00T) ──────┐
│                                                                        │
├─ Microphone (sounddevice, 16kHz) ───────────────────────────────────┐ │
│                                                                     │ │
│  StreamingIntentPredictor (0.5s interval, 1 worker)                │ │
│  ├─ motion gate (optical flow, threshold=1.5) ←────────────── frame │
│  ├─ 2.0s rolling audio buffer ←────────────────── audio chunks ─────┘
│  └─ Qwen inference → PredictionOutput                              │
│       {predicted_intent, confidence, target_object,                │
│        task_complete, reason, spoken_command(unused, see exp)}     │
│                                                                    │
│  AudioInterruptDetector (ENERGY VAD — kept, see decision below)   │
│  └─ speech_end → Qwen.transcribe_audio() → _command_validator     │
│       → TaskRegistry.resolve() / _resolve_relative                 │
│                                                                    │
│  InterruptDetectionSystem                                          │
│  ├─ MismatchDetector (consecutive_required=3, grace_period=4s)    │
│  └─ InterruptEvent → PolicyRouter.handle_interrupt()              │
│                                                                    │
│  PolicyRouter                                                      │
│  ├─ handle_voice_command() — direct keyword or relative reference  │
│  ├─ handle_interrupt() — pre-asserts correct task target first     │
│  └─ handle_prediction() — multi-tier logic:                        │
│       0. spoken_command → start/switch policy   (UNUSED — Qwen     │
│          never populates the field; code is dead-but-harmless)     │
│       1. max_task_runtime_s=40s safety cap                        │
│       2. task_complete (Qwen visual, streak≥1, runtime≥15s)       │
│       3. withdraw heuristic (streak≥2, conf≥0.6, runtime≥15s)    │
│                                                                    │
│  SystemRecorder (when --record passed)                             │
│  └─ HUD card on left panel shows:                                  │
│      FUTURE INTENT PREDICTION  #N                                  │
│      INTENT  [conf%]                                               │
│      Target: <object>                                              │
│      [conf bar]                                                    │
│      Why: <reason from Qwen>                                       │
│      Qwen X.YHz · GR00T Y.ZHz                                      │
│                                                                    │
│  GrootRobotController                                              │
│  ├─ _control_loop (background thread, 30Hz)                       │
│  │   └─ get_action wrapped in ThreadPoolExecutor (5s timeout)     │
│  ├─ current_hz property — EMA over recent control cycles           │
│  └─ start_policy / switch_policy / stop                           │
│                                                                    │
└──────────────── s99 (192.168.2.25) ──────────────────────────────┘
                  ├─ vLLM Qwen on port 8000 (CUDA:0)
                  └─ GR00T policy server on port 5555 (CUDA:1)
```

**Audio pipeline detail:**
`sounddevice` callback → `predictor.add_audio()` (rolling 2.0s buffer for continuous prediction) + `interrupt_system.on_audio()` (energy VAD for command segmentation).
The two paths are independent — Qwen sees audio context continuously AND the VAD path captures discrete commands.

---

## Multimodal control experiments (April 25)

### Experiment 1: `spoken_command` field — FAILED, then bypassed

**What was tried:** Add a separate `spoken_command` field to Qwen's JSON output schema, decoupled from `predicted_intent`. Idea: vision fills `predicted_intent` (gesture/approach/etc.), audio fills `spoken_command` (verbatim command). PolicyRouter routes on `spoken_command` instead of going through VAD.

**What was actually changed:**
- Both prompts (`_FAST_PROMPT_EXECUTING`, `_FAST_PROMPT_WAITING`) rewritten to add a CRITICAL RULE forbidding speech mentions in `reason`, three concrete few-shot examples, field reordering with `spoken_command` first
- `PredictionOutput.spoken_command: str = ""` added
- Parser extracts `spoken_command` from JSON
- `PolicyRouter.handle_prediction()` got a top-priority branch routing on `spoken_command`
- Audio buffer for prediction calls bumped from 0.5s → 2.0s so multi-word commands fit
- Robot state mapping `idle → waiting` so the WAITING prompt is actually selected when idle

**Result across multiple test runs:** `spoken_command=""` in 100% of predictions, including ones where Qwen *clearly* heard the audio (its `reason` field literally writes things like `"Hand mid-motion toward yellow ball, but audio command 'stop' interrupts"`). Qwen ignores field discipline. Even with explicit prompts, examples, and field reordering, the model dumps audio observations into the free-form `reason` field.

**Diagnosis:** Model-behavior ceiling. Qwen3-Omni-30B treats schema as a soft suggestion when it has rich multimodal observations to communicate. Prompt engineering can't fix this.

**Code state:** All the plumbing is still in place but inert. The `spoken_command` field exists on `PredictionOutput`, the parser reads it, and PolicyRouter has the routing branch — they just never fire because Qwen always returns empty. The code is harmless and demonstrates the experiment was attempted.

**Defensible thesis finding:** Qwen's behavior in the `reason` field IS evidence of internal multimodal fusion — citable as "the model demonstrates audio-aware visual reasoning even when failing to externalize the command in a dedicated structured field." This is a real result.

### Experiment 2: `predicted_intent="interrupt"` for multimodal STOP — WORKING ✅

After the `spoken_command` field failed, we noticed Qwen was already emitting `predicted_intent="interrupt"` when it heard a verbal "stop" alongside arm motion (e.g. `Qwen #11: interrupt(pink cotton ball) conf=0.90 why=Hand is mid-motion toward ball, but audio command 'stop' int...`). Wired `PolicyRouter.handle_prediction()` to react to this intent:

```python
if intent == "interrupt" and conf >= 0.85 and self.state == "running":
    self._interrupt_streak += 1
    if self._interrupt_streak >= self._interrupt_count:  # ≥2 consecutive
        log.info("Multimodal STOP (Qwen interrupt) — stopping %s", self.active_policy)
        self.robot.stop()
        self._post_stop_quiet_until = time.time() + 3.0
        return
```

**Result:** confirmed working in `/Users/yuvraj/sessions/20260425_153047.mp4` (and earlier sessions). The robot stopped purely from Qwen's multimodal `interrupt` intent — the VAD path's `Speech detected → Transcribed` chain did not fire because Qwen got there first via fused understanding. Streak filter (≥2 consecutive `interrupt` predictions at conf≥0.85) prevents single-tick false positives.

**Side effect handled:** the `InterruptDetectionSystem` (legacy MismatchDetector) would fire a `visual_interrupt` ~500ms after the multimodal stop, immediately restarting the same task. Fix: `_post_stop_quiet_until` timestamp blocks `handle_interrupt()` callbacks for 3 seconds after a multimodal stop. Confirmed clean stop+stay-stopped behavior.

### Experiment 3: `predicted_intent="change_target"` for multimodal SWITCH — wired, untested

Same pattern for `change_target`: when Qwen emits this intent with a recognizable `target_object` for ≥2 consecutive ticks while running, hot-swap to that policy. Wired in PolicyRouter but not yet confirmed in a session log. To test: while pink-ball task is running, say "yellow ball" and watch for `Multimodal SWITCH (Qwen change_target) → pick_yellow_ball`.

### Where VAD remains (and why)

VAD is still required for **cold-start task initiation** (e.g. "pick up the pink ball" from idle). Reason: when idle, Qwen has no `predicted_intent` value that means "user just gave a fresh command" — visual intent classes describe motion (continue/approach/gesture/etc.), not command events. `target_object` populates from visual presence, so it can't be used as a start trigger without false positives (robot would start the moment a ball appears on the table). `spoken_command` doesn't work. So VAD remains as the cold-start command-capture mechanism — invoked once per task, not continuously.

**Default mode factoring:**
| Phase | Mechanism |
|---|---|
| Cold-start (idle → running) | Energy VAD → Qwen.transcribe_audio → TaskRegistry.resolve |
| In-task STOP | Qwen continuous prediction → `predicted_intent="interrupt"` |
| In-task SWITCH | Qwen continuous prediction → `predicted_intent="change_target"` |
| In-task SCENE COMPLETION | Qwen continuous prediction → `task_complete=true` |
| In-task SAFETY CAP | 40s timer in PolicyRouter |

---

## --no-vad Mode (April 26)

Pass `--no-vad` to `run_system_groot.py` (or `bash run.sh --no-vad`) to bypass energy VAD entirely. Cold-start task detection is handled by Qwen's WAITING prompt instead of the VAD→transcribe chain.

### How it works (simple version)

The system has two phases — **idle** and **running** — and Qwen sees a different prompt for each:

**Phase 1: Idle (robot not moving)**
- Qwen uses the **WAITING prompt**, which tells it to listen for task commands
- Every 0.5s, Qwen looks at the camera + mic and classifies what it hears into `command_pick_pink_ball`, `command_pick_yellow_ball`, or `none`
- When it hears the same command 2 times in a row at high confidence (conf≥0.85), the policy starts — this is the **cold-start streak**
- No VAD hardware needed — Qwen does the listening

**Phase 2: Running (robot arm moving)**
- Qwen switches to the **EXECUTING prompt**, which watches the arm motion AND listens for verbal stops/resumes
- Every 0.5s, Qwen reports `predicted_intent` (what the arm is doing: approach/gesture/withdraw/etc.) AND `spoken_command` (any words it hears)
- `interrupt` intent → robot stops; `spoken_command` containing resume words → robot resumes; `change_target` → policy switch

```
IDLE STATE                          RUNNING STATE
──────────────────────────────      ──────────────────────────────
Qwen uses WAITING prompt            Qwen uses EXECUTING prompt
↓                                   ↓
Outputs: command_pick_* / none      Outputs: predicted_intent + spoken_command
↓                                   ↓
streak≥2 at conf≥0.85 → START      interrupt → STOP
                                    resume word in spoken_command → RESUME
                                    change_target → SWITCH
                                    task_complete → DONE
```

### --no-vad mode factoring
| Phase | Mechanism |
|---|---|
| Cold-start (idle → running) | Qwen WAITING prompt → `command_pick_*` streak ≥2 at conf≥0.85 |
| In-task STOP | Qwen EXECUTING prompt → `predicted_intent="interrupt"` OR `spoken_command` contains stop word |
| In-task RESUME | Qwen EXECUTING prompt → `spoken_command` contains resume word (continue/resume/go on/etc.) |
| In-task SWITCH | Qwen EXECUTING prompt → `predicted_intent="change_target"` OR `spoken_command` task keyword |
| In-task SCENE COMPLETION | Qwen EXECUTING prompt → `task_complete=true` |
| In-task SAFETY CAP | 40s timer in PolicyRouter |

### Reliability notes for --no-vad
- **Cold-start**: confirmed working (command_pick_pink_ball streak 2/2 fires reliably)
- **Visual intent during execution**: confirmed (approach/gesture/withdraw/continue predictions now appear)
- **Policy switch via spoken_command**: confirmed ("maybe pick up another ball" → change_target → pick_yellow_ball)
- **task_complete**: confirmed (fires correctly after ball placed in bowl)
- **Stop reliability**: ~50% — depends on Qwen hearing the verbal stop in the 0.5s inference window. Audio encoder degeneration (motor noise) drops the audio+video path to video-only ~50% of requests, losing audio context. HPF filtering (200Hz) helps but doesn't fully solve it. **Recommended: use default VAD mode for reliable verbal stops.**
- **Resume ("continue")**: wired and working (bug was fixed April 26 — see bug #28)

---

## Intent Classes

`approach` | `gesture` | `withdraw` | `continue` | `point` | `change_target` | `interrupt` | `unknown` | `new_command` (defined but unused after spoken_command experiment)

State machine: `continue → approach → gesture → withdraw → continue`
- `gesture → withdraw` requires 2+ consecutive gestures
- 6 consecutive same-state → reset to `continue`

---

## Key Architecture Decisions

**[DECISION] TaskRegistry + YAML task definitions**
Tasks live in `tasks.yaml`. Each has a `lang` string (sent to GR00T per obs), `object` (target for mismatch detection), and `keywords` (voice matching). Swapping tasks = editing YAML only.

**[DECISION] PolicyRouter 3-tier stop**
GR00T has no internal "done" signal and loops forever. Three stops, in order of priority:
1. **Max runtime (40s)** — safety cap, always fires even if Qwen is blind
2. **task_complete** — Qwen visual signal: ball is visibly in bowl. Requires 1 confident frame + 15s min runtime. Ignores arm position (arm never withdraws while GR00T loops).
3. **Withdraw heuristic** — 2 consecutive `withdraw` predictions at conf≥0.6 + 15s min runtime

**[DECISION] 1 Qwen inference worker (was 2)**
Qwen vLLM and GR00T policy server share s99. 2 concurrent Qwen workers saturate the GPU and slow GR00T inference ~8x. Reduced to 1 worker: `predictor.start(num_workers=1)`. This serializes visual inference but keeps GR00T fast.

**[DECISION] GPU split on s99**
Qwen vLLM must run on CUDA:0, GR00T on CUDA:1. Sharing a GPU causes GR00T inference to slow ~8x because Qwen polls at 0.5s and blocks GR00T's GPU time slices.

**[DECISION] GR00T get_action in dedicated ThreadPoolExecutor with 5s timeout**
`get_action` blocks. If s99 is overloaded, it can stall the control loop indefinitely. Fix: submit to a 1-thread executor, `fut.result(timeout=5.0)`, skip tick on TimeoutError.

**[DECISION] task_complete ignores arm position**
Initial implementation required "robot arm withdrawn" as a visual condition. GR00T loops indefinitely so the arm never withdraws while running — task_complete would never fire. Fixed: only check if the target object is visibly at its destination (ball in bowl).

**[DECISION] _last_active_policy for relative reference resolution when idle**
After a task stops, `active_policy` becomes `None`. Saying "other ball" then would incorrectly flip to the first task in the registry. Fix: `PolicyRouter` tracks `_last_active_policy`, and `_resolve_relative` uses `active_policy or _last_active_policy` as the "current" task to flip away from.

**[DECISION] Pre-assert correct task target in handle_interrupt**
The interrupt system internally calls `task_monitor.set_task(intent="change_target", target_object=<raw_command_text>)` before firing the interrupt callback. This sets the target to the raw command string (e.g. "other ball instead") instead of the actual object name. Fix: `handle_interrupt` calls `task_monitor.set_task(target_object=task.object)` BEFORE `_execute_policy_action`, immediately overwriting the stale state and starting the 4s grace period with the correct target.

**[DECISION] _command_validator passes relative references**
`_command_validator` previously only passed stop-words and exact task keyword matches. "Other ball", "not this one", "switch to the other", etc. were silently dropped. Fixed: added relative-reference word check so they reach `handle_voice_command → _resolve_relative`.

**[DECISION] Keep Qwen audio (not Whisper)**
Qwen3-Omni processes audio + video in one unified call. The thesis contribution is a single multimodal model handling both streams. Splitting into Whisper+Qwen-video would be two models — less novel.

**[DECISION] Audio-interference retry (workaround, not a fix)**
Qwen3-Omni returns `'\n\n'` (empty) when audio contains speech — a vLLM+Qwen3-Omni quirk. Fix: retry video-only on empty response. Reduced unknowns from 31% → 1.6%.

**[DECISION] MismatchDetector consecutive_required = 3, grace_period = 4s**
Raised from 2/2s to reduce false positive object_mismatch interrupts. 4s physically motivated by robot reorientation time after task switch.

**[DECISION] Active task context injected into Qwen prompt**
`engine.active_task_lang` is set when a policy starts. Qwen's prompt includes the task being executed and is asked to evaluate `task_complete`. `engine.active_task_object` is set in `task_monitor.set_task` (via `connect_to_predictor` monkey-patch) to reduce object_mismatch FPs.

**[DECISION] Camera 640×480, crop to 640×360 for GR00T (April 25)**
Camera delivers 1280×720 by default and accepts 640×480 (per working teleop config) but does NOT support 640×360 — LeRobot's `_validate_width_and_height` rejects with `RuntimeError: failed to set capture_width=640 (actual_width=1280)` if you ask for 640×360. Fix: configure `OpenCVCameraConfig(width=640, height=480)` and center-crop to 640×360 inside `_obs_to_policy_inputs()` via `_crop_to_training()`. Both consumers (Qwen `cv2.VideoCapture` and LeRobot `OpenCVCamera`) must request the same resolution or one will reconfigure the AVFoundation device mid-stream and crash the other's read thread.

**[DECISION] Robot state "idle" mapped to "waiting" before sending to predictor (April 25)**
`qwen_inference_engine.py` selects the WAITING vs EXECUTING prompt based on `'state=waiting' in self._current_robot_state`. But `GrootRobotController.state` returns `"idle"` when not running, so the WAITING prompt was never selected. Fix in `run_system_groot.py` main loop: `_qwen_state = "waiting" if robot.state == "idle" else "executing"` before passing to `predictor.set_robot_state()`.

**[DECISION] Audio buffer 0.5s → 2.0s for prediction calls (April 25)**
`audio_max_duration` was 0.5s, far too short to capture full spoken commands ("pick up the yellow cotton ball" takes ~1.7s). Bumped to 2.0s to match the predictor's `audio_buffer_duration`. Cost: +50-100ms per Qwen inference. Confirmed Qwen now hears full commands (visible in `reason` field) — but still won't populate `spoken_command` due to the schema-discipline issue.

**[DECISION] Energy VAD reduced to cold-start only (April 25)**
Original ambition was to remove energy VAD entirely. Tested extensively. **Outcome: in-task control (stop / switch) is now Qwen-only via `predicted_intent="interrupt"` and `predicted_intent="change_target"`**, confirmed working on the robot. Cold-start task initiation still needs VAD because Qwen has no intent class for "user gave a fresh command from idle". So VAD stays as a small cold-start helper, fired once per task, not continuously. Thesis framing: "Qwen handles all in-task semantic control via fused multimodal prediction; VAD is a 30-line cold-start auxiliary."

**[DECISION] PolicyRouter reacts to `predicted_intent="interrupt"` for multimodal STOP (April 25)**
Streak ≥ 2 consecutive predictions at conf≥0.85 → `robot.stop()`. Arms a 3-second `_post_stop_quiet_until` timestamp to suppress legacy `InterruptDetectionSystem` follow-up callbacks (visual_interrupt / object_mismatch) that would otherwise immediately restart the same task because Qwen lingers on `interrupt` intent for 1-2 frames after the stop. Confirmed clean stop+stay-stopped behavior.

**[DECISION] PolicyRouter reacts to `predicted_intent="change_target"` for multimodal SWITCH (April 25)**
Streak ≥ 2 consecutive predictions at conf≥0.85 with a recognizable `target_object` → `_execute_policy_action`. Same code path, different intent. Wired but not yet confirmed in a session log.

**[DECISION] HUD card shows reason + Hz stats + sequence counter (April 25)**
Per sensei's request, the recorder's HUD card now displays Qwen's `reason` field (cropped to 55 chars), a prediction sequence counter (`#N`), and live Qwen/GR00T Hz stats. Added `SystemRecorder.push_stats(qwen_hz, groot_hz, pred_seq)` and `GrootRobotController.current_hz` property (EMA over control cycles). Card height 130px, anchored bottom-left.

**[DECISION] matplotlib graph panel disabled in recorder (April 25)**
`_draw_graph_panel_mpl` created a new figure per recorded frame at 10fps → 20+ figures warning + memory leak. `_draw_graph_panel` now always uses `_draw_graph_panel_cv2` (a pure-cv2 bar-chart renderer). matplotlib import is still tolerated for offline tools.

**[DECISION] `_FAST_PROMPT_VIDEO_ONLY` for video-only multi-frame inference (April 26)**
`predict_intent_video_only_multi_frame` must NOT use `_FAST_PROMPT_EXECUTING`. That prompt says "You receive BOTH video AND audio" — sending a request with no audio causes Qwen3-Omni to return empty content in ~88ms (model detects modality mismatch and aborts streaming). Fix: new `_FAST_PROMPT_VIDEO_ONLY` constant that describes frames as "NO audio" and omits all audio-related rules and fields. Used exclusively in `predict_intent_video_only_multi_frame`.

**[DECISION] HPF audio filtering on predictor buffer during execution (April 26)**
Motor noise (low-frequency rumble from SO-101 servos) corrupts the 2.0s rolling audio buffer during execution, causing Qwen3-Omni audio encoder to fail and return empty responses on ~50% of audio+video requests. Fix: in `audio_callback` in `run_system_groot.py`, audio added to the predictor is high-pass filtered (200Hz 4th-order Butterworth) when `robot.state != "idle"`. The 200Hz cutoff preserves the speech band (300Hz+) while suppressing servo rumble. Partial fix — ~50% retry rate persists (likely because some noise overlaps the speech band). Audio path during idle is unfiltered.

**[DECISION] Resume path wired into `handle_prediction()` spoken_command branch (April 26)**
`_RESUME_WORDS` = ("continue", "resume", "keep going", "go on", "carry on", "proceed"). Previously these were only checked in `handle_voice_command()`, which is only called from the VAD callback. In `--no-vad` mode, no VAD callback fires, so "continue" had no code path at all — user had to repeat the full task command. Fix: added resume branch directly in `handle_prediction()` inside the spoken_command routing block. If `robot.state == "idle"` and a resume word is detected and `_last_active_policy` exists, restart that policy.

**[DECISION] `--no-vad` mode: Qwen WAITING prompt as cold-start detector (April 26)**
Alternative to energy VAD for cold-start: pass `--no-vad` flag. Main loop sets `_qwen_state = "waiting"` when idle, which selects `_FAST_PROMPT_WAITING`. That prompt outputs `command_pick_*` intents. `handle_prediction()` counts a cold-start streak (≥2 consecutive `command_pick_*` at conf≥0.85) and starts the policy. Trade-off: more latency than VAD (Qwen inference ~500ms vs VAD ~30ms) but no hardware VAD required, and the cold-start can happen purely from Qwen's speech understanding.

---

## Current Engine (FastQwenInferenceEngine) Settings

```python
temperature        = 0.1   # near-greedy = fastest sampling
max_tokens         = 80    # keep at 80 — 50 caused empty stream aborts on scene transitions
image_quality      = 65    # JPEG q65
max_width          = 320   # 320×240 per frame
audio_max_duration = 2.0   # bumped from 0.5s on 2026-04-25 so Qwen sees full commands
```

---

## Latency Profile

| Mode | Latency |
|---|---|
| Audio + video, normal | ~230–530ms avg |
| Audio + video, with retry (audio interference) | ~350–800ms |
| Video only (retry fallback) | ~220–320ms |
| Network overhead (LAN) | ~10–30ms |

Avg latency across full run: **~360ms**. Well within the ~1.4s advance window.

GR00T control loop typically runs **1.7–2.0 Hz** end-to-end (obs ~1ms, inf ~200-300ms, motion ~290ms per 8-step horizon).

---

## Interrupt Detection Results — Best Offline Run (March 31, 2026)

**Video:** `interrupt_test_take2.mp4` — 30.9s, blue bottle + headphones + brown flask.

- Total predictions: 60
- Avg latency: **360ms** (min 216ms, max 783ms)
- Correct interrupts: **1/1 (100%)**
- False positives: **0**
- Unknown predictions: **1/60 (1.7%)**

---

## Bugs Fixed (cumulative)

**Offline / inference pipeline:**
1. Wrong vLLM URL (`localhost` vs `192.168.2.25`)
2. Parse errors — `_parse_prediction` now strips preamble before `{`
3. Audio interference empty stream → video-only retry
4. `max_tokens=50` caused mid-JSON aborts → kept at 80
5. Intent history not updating on single-frame paths
6. Streaming code duplication → consolidated to `_stream_until_json()`
7. Thread safety race in MismatchDetector → `threading.Lock()`
8. Confidence out of range → clamped to [0,1]
9. Active task object injection → eliminated last FP

**SO-101 / GR00T integration:**
10. Shell blank-line bug — blank line after `\` terminates command early; created `run.sh`
11. GR00T loops forever after task completion → 3-tier stop (task_complete + withdraw + max_runtime)
12. `task_complete` never fired — prompt incorrectly required arm withdrawal; fixed to check object position only
13. GPU contention slowing GR00T → reduced to 1 Qwen worker
14. GR00T `get_action` hangs → wrapped in ThreadPoolExecutor with 5s timeout
15. Relative references ("other ball") dropped by `_command_validator` → added relative-word check
16. `_resolve_relative` returned wrong task when robot idle (`active_policy=None`) → `_last_active_policy` fallback
17. Object_mismatch FPs after interrupt → `handle_interrupt` pre-asserts correct target before policy switch

**This session (April 25):**
18. Camera read-thread crash from dual cv2.VideoCapture + LeRobot OpenCVCamera fighting over AVFoundation device → both must request 640×480
19. Camera doesn't support 640×360 (training resolution) → configure 640×480, center-crop in `_obs_to_policy_inputs()` via `_crop_to_training()`
20. WAITING prompt never selected because `robot.state == "idle"` ≠ `"waiting"` → mapped in main loop before `set_robot_state()`
21. matplotlib figure leak in recorder (20+ figures warning at 10fps) → switched `_draw_graph_panel` to pure-cv2 implementation
22. Qwen audio truncated to 0.5s missed full commands → bumped `audio_max_duration` to 2.0s
23. RealSense camera grayscale on macOS via cv2 (delivers color in browser) → unfixable; pyrealsense2 segfaults on Darwin 25.4.0; **use a different camera**
24. Multimodal STOP via Qwen's `interrupt` intent caused infinite restart loop (legacy `visual_interrupt` re-fired ~500ms later) → `_post_stop_quiet_until` 3s cooldown in `handle_interrupt()`

**This session (April 26):**
25. Unknown intent predictions (approach/gesture/withdraw not showing) — `predict_intent_video_only_multi_frame` used `_FAST_PROMPT_EXECUTING` which says "You receive BOTH video AND audio" but sent no audio, causing Qwen3-Omni to return empty in ~88ms → added `_FAST_PROMPT_VIDEO_ONLY` (does NOT mention audio) and switched the method to use it
26. Audio encoder degeneration during execution — motor noise in the 2.0s predictor audio buffer causes Qwen3-Omni to fail and return empty on ~50% of audio+video requests → user added 200Hz high-pass Butterworth filter (`engine._highpass_filter`) applied in `audio_callback` to predictor audio when robot is running (partial fix: HPF reduces motor rumble but ~50% retries persist)
27. "Continue" command not restarting the robot in `--no-vad` mode — `_RESUME_WORDS` check only existed in `handle_voice_command()` (VAD callback path only); no code path existed for resume in `handle_prediction()` → added resume branch in `handle_prediction()` spoken_command section
28. Verbal stop `_post_stop_quiet_until` not armed in the verbal-stop branch of `handle_prediction()` → added `self._post_stop_quiet_until = time.time() + self._post_stop_quiet_seconds` after `robot.stop()` in that branch

---

## Pending / Future Work

- [ ] `'wise'` response — Qwen occasionally outputs a single word instead of JSON; add retry for non-empty non-JSON
- [ ] Async FrameRecorder — re-add recording on background thread (previous sync version slowed control loop) — partially done; recorder is async but control loop integration could be cleaner
- [ ] VAP-Realtime integration for turn-taking awareness (needs stereo audio) — note: VAD stays even after this
- [ ] Re-evaluate cleaning video accuracy (last measured at 57.1%) with updated engine
- [ ] Speech buffer unbounded growth fix in AudioInterruptDetector
- [ ] Audio reliability metrics logging
- [ ] Latency decomposition (network vs inference vs encoding)
- [ ] Decide whether to delete the `spoken_command` plumbing now that the experiment failed, OR leave it as a "tried and failed" code artifact (current: leaving it; it's harmless)
- [ ] Write thesis methodology section using the revised framing (Qwen as unified semantic backbone, not VAP+VAD replacement)
- [ ] Fix verbal stop reliability in `--no-vad` mode — ~50% retry rate due to audio encoder degeneration from motor noise despite 200Hz HPF. Options: raise HPF cutoff, gate predictor audio by energy threshold during execution, or use a dedicated short audio buffer for stop-word detection only
- [ ] Evaluate `--no-vad` stop reliability with higher HPF cutoff (e.g. 400Hz) — servo rumble may extend higher than 200Hz

---

## What to Tell Claude Next Session

Paste this file, then use one of these openers:

- "I want to write up the thesis methodology section" → start from this file's Architecture + revised research contribution; use the --no-vad mode as the "pure Qwen" configuration
- "Help me run an evaluation session" → start the system, capture log, analyse interrupts/false positives
- "Fix the speech buffer growth" → unbounded list in AudioInterruptDetector
- "Add VAP-Realtime" → TCP/IP integration, stereo audio, knowing VAD still stays
- "Delete the dead spoken_command code" → revert the failed experiment cleanly
- "Re-run the cleaning video evaluation" → check accuracy with updated engine (was 57.1%)
- "Fix stop reliability in --no-vad" → root cause is audio encoder degeneration (~50% retry rate despite HPF); options are: (a) increase HPF cutoff, (b) gate predictor audio by energy threshold during execution, (c) use a separate short audio buffer for stop detection
- "Add async recording back" → see memory note: re-add FrameRecorder on background thread; previous sync version slowed control loop
- Describe a new runtime bug → paste the relevant log lines and describe what you expected

---

## System Prompts (FastQwenInferenceEngine)

There are now **three** prompts. Both WAITING and EXECUTING include a `spoken_command` field and a CRITICAL RULE forbidding speech mentions in `reason`. **Qwen ignores the field discipline.** Kept anyway — harmless, and the WAITING/EXECUTING distinction is load-bearing.

**_FAST_PROMPT_WAITING** (state=waiting / robot idle, `--no-vad` mode): Classifies spoken commands into `command_pick_pink_ball`, `command_pick_yellow_ball`, or `none`. Emphasises listening for verbal commands. Used for cold-start task detection without VAD.

**_FAST_PROMPT_EXECUTING** (task active — selected when `'state=waiting' NOT in robot_state_string`): TWO JOBS format — (1) LISTEN: transcribe any spoken words verbatim into `spoken_command`; (2) WATCH: classify arm motion into `predicted_intent`. Contains STOP/INTERRUPT RULE (highest priority), SWITCH RULE, and 4 few-shot examples. Full intent vocabulary: continue/approach/gesture/withdraw/change_target/interrupt/unknown. Explicitly states "You receive BOTH video AND audio." **Only use when audio is actually included in the request.**

**_FAST_PROMPT_VIDEO_ONLY** (NEW, April 26): Video-only prompt for `predict_intent_video_only_multi_frame`. Does NOT mention audio at all ("NO audio." in frame description). Contains the same intent vocabulary as EXECUTING but no `spoken_command` field, no audio-related rules. **Critical: if the EXECUTING prompt (which says "You receive BOTH video AND audio") is used for a video-only request, Qwen3-Omni returns empty content in ~88ms.** This prompt was added specifically to fix that.

JSON output schema (EXECUTING + WAITING):
```json
{"spoken_command":"<words or empty>",
 "predicted_intent":"<class>","confidence":0.0-1.0,
 "target_object":"<object color or none>",
 "task_complete":false,
 "reason":"<15 words max>"}
```

JSON output schema (VIDEO_ONLY — no spoken_command):
```json
{"predicted_intent":"<class>","confidence":0.0-1.0,
 "target_object":"<object color or none>",
 "task_complete":false,
 "reason":"<15 words max>"}
```

Reality: `spoken_command` is always empty regardless of audio content. `reason` ends up containing audio observations like `"audio command 'stop'"` even though forbidden. During execution, ~50% of requests fall back to video-only due to audio encoder degeneration from motor noise.

**InterruptReason types:** `VERBAL_STOP`, `VISUAL_INTERRUPT`, `CHANGE_TARGET`, `OBJECT_MISMATCH`, `TRAJECTORY_CHANGE`
