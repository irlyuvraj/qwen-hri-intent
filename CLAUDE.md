# CLAUDE.md — Qwen3-Omni + GR00T N1.6 HRI System (SO-101)
> Master's Thesis: Human-Robot Interaction — Intent-Aware Robot Control
> Last updated: May 26, 2026

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

**Current status (May 19 evening):** Demoable end-to-end on SO-101. Parse-failure rate down to ~0% (was ~50%) after discovering [vLLM Issue #18819](https://github.com/vllm-project/vllm/issues/18819) — Qwen3 + `enable_thinking=False` + guided_json produces gibberish; fix is to append `/no_think` to user prompts instead of using the chat_template kwarg. All five primitives now fire reliably via multimodal Qwen path:
- **Cold-start** (`command_pick_*` streak 2/2)
- **Stop** (`predicted_intent="interrupt"`, multimodal hit — no fast-lane fallback needed)
- **Switch** (`predicted_intent="change_target"` → hard switch with gripper release)
- **Resume** (`command_resume` streak 2/2 → restart `_last_active_policy`)
- **Auto-complete** (placing-phase fallback: ≥2 consecutive `predicted_phase=placing` past 15s runtime → STOP)

HPF default flipped — `--no-vad` runs without the 200Hz audio HPF by default (Whisper-community consensus). MetricsLogger and telemetry dashboard running. Recorder updated (full-height log panel, no more confidence bar chart).

May 18 session added: `RobotProfile` hardware abstraction, hard-switch gripper release on policy switch, fast-lane audio-only command classifier, `predicted_phase` (3-5s horizon), structured `MetricsLogger` JSONL output, speech-burst accumulator pattern.

May 19 session added: `/no_think` workaround (THE big win — parse rate ~50% → ~0%), HPF default off, placing-phase fallback for auto-complete, telemetry dashboard + publisher, max_tokens 80→160, raw-response logging on parse failure.

**Current status (May 21):** Confirmed working on SO-101 with three new capabilities, all task-agnostic. (1) **Visual grounding gate** (cold-start only) — refuses to start a pick into an empty workspace. (2) **Visual completion verifier** — a dedicated focused Qwen call ("is a ball resting inside the bowl?") that reliably auto-stops the robot once the object is placed; **this closed the long-standing yellow auto-complete gap** (yellow now auto-stops at ~15s instead of looping to the 40s cap — confirmed live May 21). (3) **Registry-driven prompts** — the EXECUTING/VIDEO/AUDIO/WAITING prompts are now templated from `tasks.yaml`; no object name is hardcoded, so adding a pick task needs no prompt edits. Also: `eval_metrics.py` baseline tool (0% parse failures across 18 sessions / 1473 predictions; executing latency median 471ms / p95 606ms).

**Key design principle (validated May 21):** Qwen3-Omni is unreliable at structured fields, object/color naming, and gripper-state, but reliable at a single focused yes/no visual question. The grounding and completion features both work *because* they ask one dedicated question (vision-only, `/no_think`, 2-consecutive-confirmation guard) rather than reading a field from the multi-task streaming prediction or string-matching Qwen's free-form names.

**Current status (May 26):** Stable and demo-ready across many live runs. Full cycle works repeatedly: cold-start → pick → visual-completion confirm → stop → **return to home pose**. Refinements since May 21:
- **Completion verifier reworked** to ask about the BOWL ("is a ball resting inside the bowl?"), dropping the earlier "gripper empty" criterion (the post-place empty-but-closed gripper read as "holding") and the phase gate (the post-place GR00T loop is mislabeled "approaching"). Every check is now logged (`Completion check: complete=... why=...`). Confirmed working for both balls live.
- **Return-to-home** added — after a confirmed completion the arm smoothly interpolates back to the pose captured at startup. Robot-agnostic (auto-captured pose). `--no-home` disables.
- **8s request timeout** (was 30s) — fixed the "startup hang": an occasional server-side vLLM stall froze the single-worker pipeline for 30s; now it recovers in ~8s.
- **Lenient grounding** — grounding now allows the pick if Qwen's `seen` list names any graspable object even when it answered `pickable_present=false` (it was false-refusing "seen: purple yarn ball"); refuses only a truly empty workspace.

**New findings (May 26 runs):**
- **Out-of-vocabulary rejection works now** — "pick the red ball" is refused at cold-start (`why=Command for 'red ball' not in known objects` → `none`) instead of being mapped onto pink/yellow. Emergent side-benefit of the registry-driven WAITING prompt (it enumerates only known commands). This *mitigates* the May 21 color-grounding limitation at the cold-start stage.
- **Object-dependent completion latency** — identical control logic, but **pink verifies/stops faster than yellow**. Cause is perceptual, not code: Qwen confirms "ball in bowl" later and less often for yellow (lower contrast; placing-phase rate ~5% vs pink ~7%), so the 2-consecutive completion confirmation lands later → stop later → home later. The home motion itself is fixed (~1.2s). Good thesis point: perception quality, not the controller, sets per-object latency. (Clean A/B not yet measured — today's durations are confounded by mid-pick verbal stops.)

**Known limitations (May 26):** (1) mid-task SWITCH to a nonexistent color can still map onto a present object (grounding is cold-start-only and a ball IS present). (2) verbal "stop" can be missed if spoken below the 0.012 audio gate (quiet/far from mic). (3) LeRobot occasionally shows a calibration-mismatch prompt at startup (`Press ENTER to use provided calibration file`) that blocks until you press ENTER — re-run SO-101 calibration if it recurs; unrelated to our pipeline.

**Git note (May 26):** all of the above (grounding, completion verifier, return-to-home, prompt templating, timeout fix, lenient grounding, and the new files `eval_metrics.py` / `metrics_logger.py` / `telemetry_*.py`) is **uncommitted** in the working tree — last commit is `78e11e6` which predates all of it. Commit before the teleop/retrain phase.

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
bash run.sh                    # no recording; always writes ~/sessions/metrics_<ts>.jsonl
bash run.sh --record           # records to ~/sessions/<timestamp>.mp4 + metrics file
bash run.sh --no-vad           # bypass energy VAD; Qwen WAITING prompt handles cold-start
bash run.sh --no-hard-switch   # revert policy switch to lang-swap only (legacy)
bash run.sh --gripper-open-pos 45.0  # override SO-101 gripper release position (degrees)
bash run.sh --hpf              # re-enable 200Hz HPF (default is OFF as of May 19)
bash run.sh --no-grounding         # disable cold-start visual grounding gate (default ON)
bash run.sh --no-completion-check  # disable visual completion verifier (default ON)
bash run.sh --no-home              # disable return-to-home after completion (default ON)
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
│                                  Contains: RobotProfile, SO101_PROFILE, PolicyRouter,
│                                  GrootRobotController, speech-onset fast lane, audio_callback
├── run.sh                      ← Shell wrapper — use this to launch (avoids blank-line bug)
│                                  Now always writes metrics_<ts>.jsonl to ~/sessions/
├── tasks.yaml                  ← Task registry: 2 tasks; supports per-task max_runtime_s
├── task_registry.py            ← TaskRegistry class: YAML loader + keyword resolver
│                                  Task dataclass now has optional max_runtime_s field
├── qwen_inference_engine.py    ← FastQwenInferenceEngine: task_complete, active_task_lang,
│                                  classify_command() (audio-only fast lane),
│                                  _build_classifier_prompt(), predicted_phase support
├── streaming_intent_predictor.py ← StreamingIntentPredictor: PredictionOutput w/ task_complete,
│                                  spoken_command, predicted_phase; is_fast_lane InferenceInput flag;
│                                  _immediate_request Event; vLLM backoff; _command_to_prediction()
├── metrics_logger.py           ← MetricsLogger: per-prediction JSONL structured logging
├── eval_metrics.py             ← Offline baseline report over ~/sessions/metrics_*.jsonl (May 21)
├── interrupt_detection_system.py ← InterruptDetectionSystem + AudioInterruptDetector (energy VAD)
├── recorder.py                 ← SystemRecorder: mp4 capture with HUD overlay
│                                  (intent, phase, reason, Hz, seq counter; card 144px)
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
    # max_runtime_s: 60  # optional override of PolicyRouter's global 40s cap

  - name: pick_yellow_ball
    lang: "Pick up the yellow cotton ball and place it in the bowl"
    object: "yellow cotton ball"
    keywords: [yellow, "yellow ball", "yellow cotton", "yellow one"]
```

`TaskRegistry.resolve()` uses **longest-matching keyword** (not first-match) so "pink ball" beats "ball" when both appear.

`Task.max_runtime_s` (optional, May 18) — per-task safety cap overriding PolicyRouter's global 40s default. Useful for multi-step tasks (e.g. pour + place) that take longer. Leave unset to use the global cap.

---

## System Architecture

```
MacBook (robot host)
│
├─ Camera (index 0, 640×480 native, cropped to 640×360 for GR00T) ──────┐
│                                                                        │
├─ Microphone (sounddevice, 16kHz) ───────────────────────────────────┐ │
│   │                                                                 │ │
│   │  audio_callback ──────────────────────────────────────────────┐│ │
│   │   ├─ HPF 200Hz (always) → energy VAD → interrupt_system      ││ │
│   │   ├─ Speech-burst accumulator (May 18):                      ││ │
│   │   │   voiced blocks (post-HPF RMS ≥ 0.012) → _speech_burst   ││ │
│   │   │   on ≥500ms voiced → predictor buffer REPLACED w/ burst  ││ │
│   │   │                      + _immediate_request.set()          ││ │
│   │   │   on ≥400ms silence → burst clears, _onset_armed resets  ││ │
│   │   └─ Fast lane: bypasses 0.5s poll — fires Qwen immediately  ││ │
│                                                                   ││ │
│  StreamingIntentPredictor (0.5s interval, 1 worker)              ││ │
│  ├─ motion gate (optical flow, threshold=1.5) ←──────── frame ───┘│ │
│  ├─ 2.0s rolling audio buffer ←───────────────── audio chunks ─────┘
│  ├─ is_fast_lane flag on InferenceInput (from _immediate_request)  │
│  └─ Qwen inference → PredictionOutput                             │
│       {predicted_intent, predicted_phase(NEW), confidence,        │
│        target_object, task_complete, reason,                      │
│        spoken_command(unused in main path, see exp)}              │
│                                                                   │
│  FastQwenInferenceEngine — fast-lane path (NEW May 18):          │
│  └─ classify_command(audio_window) → audio-only Qwen call        │
│       output: {command: stop|command_pick_*|switch_*|none,        │
│               confidence, heard}                                  │
│       → _command_to_prediction() → PredictionOutput              │
│         (maps stop→interrupt, switch_*→change_target, etc.)      │
│                                                                   │
│  AudioInterruptDetector (ENERGY VAD — kept, see decision below)  │
│  └─ speech_end → Qwen.transcribe_audio() → _command_validator    │
│       → TaskRegistry.resolve() / _resolve_relative               │
│                                                                   │
│  InterruptDetectionSystem                                         │
│  ├─ MismatchDetector (consecutive_required=3, grace_period=4s)   │
│  └─ InterruptEvent → PolicyRouter.handle_interrupt()             │
│                                                                   │
│  PolicyRouter                                                     │
│  ├─ handle_voice_command() — direct keyword or relative reference │
│  ├─ handle_interrupt() — pre-asserts correct task target first    │
│  └─ handle_prediction() — multi-tier logic (priority order):      │
│       0. spoken_command (Qwen-heard verbatim) — stop/resume/start │
│          (active in EXECUTING prompt; gated by command_validator) │
│       1. max_task_runtime_s (per-task or global 40s) safety cap  │
│       2. task_complete (Qwen visual, streak≥1, runtime≥15s)      │
│       3. withdraw heuristic (streak≥2, conf≥0.6, runtime≥15s)   │
│                                                                   │
│  MetricsLogger (NEW May 18)                                       │
│  └─ ~/sessions/metrics_<ts>.jsonl — one JSONL line per prediction │
│       {ts, seq, intent, phase, confidence, target_object,         │
│        task_complete, spoken_command, reason, latency_ms,         │
│        robot_state, active_policy, audio_rms_pre/post,            │
│        parse_failed, raw_response_len}                            │
│                                                                   │
│  SystemRecorder (when --record passed)                            │
│  └─ HUD card on left panel shows (144px tall):                    │
│      FUTURE INTENT PREDICTION  #N                                 │
│      INTENT  [conf%]                                              │
│      Phase:  <phase> (NEW — 3-5s horizon)                        │
│      Target: <object>                                             │
│      [conf bar]                                                   │
│      Reason: <reason from Qwen>                                   │
│      Qwen X.YHz · GR00T Y.ZHz                                     │
│                                                                   │
│  GrootRobotController (robot-agnostic via RobotProfile, NEW)      │
│  ├─ profile: RobotProfile = SO101_PROFILE                        │
│  ├─ _control_loop (background thread, 30Hz)                      │
│  │   └─ get_action wrapped in ThreadPoolExecutor (5s timeout)    │
│  ├─ current_hz property — EMA over recent control cycles          │
│  └─ start_policy / switch_policy(hard_switch=True) / stop        │
│       hard-switch path: stop loop → apply release_overrides      │
│       (open gripper) → start new policy fresh                    │
│                                                                   │
└──────────────── s99 (192.168.2.25) ──────────────────────────────┘
                  ├─ vLLM Qwen on port 8000 (CUDA:0)
                  └─ GR00T policy server on port 5555 (CUDA:1)
```

**Audio pipeline detail:**
`sounddevice` callback → `predictor.add_audio()` (rolling 2.0s buffer for continuous prediction) + `interrupt_system.on_audio()` (energy VAD for command segmentation) + speech-onset fast-lane trigger (new May 18: sets `_immediate_request` event when RMS crosses threshold, causing the scheduler to fire an inference immediately bypassing the 0.5s poll).
The three paths are independent — Qwen sees audio context continuously, the VAD path captures discrete commands, and the fast lane reduces stop-command latency.

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

### --no-vad mode factoring (updated May 18)
| Phase | Mechanism |
|---|---|
| Cold-start (idle → running) | Qwen WAITING prompt → `command_pick_*` streak ≥2 at conf≥0.85 (or fast-lane classifier) |
| Resume (idle → running, replay last task) | Qwen WAITING prompt → `command_resume` streak ≥2 at conf≥0.85 → restart `_last_active_policy` |
| In-task STOP | Qwen EXECUTING prompt → `predicted_intent="interrupt"` (streak ≥2) OR fast-lane classifier `command="stop"` |
| In-task SWITCH | Qwen EXECUTING prompt → `predicted_intent="change_target"` (streak ≥2) OR fast-lane classifier `command="switch_<color>"` |
| In-task SCENE COMPLETION | Qwen EXECUTING prompt → `task_complete=true` |
| In-task SAFETY CAP | 40s timer in PolicyRouter (or per-task `max_runtime_s`) |

### Reliability notes for --no-vad
- **Cold-start**: confirmed working (command_pick_pink_ball streak 2/2 fires reliably)
- **Visual intent during execution**: confirmed (approach/gesture/withdraw/continue predictions now appear)
- **Policy switch via spoken_command**: confirmed ("maybe pick up another ball" → change_target → pick_yellow_ball)
- **task_complete**: confirmed (fires correctly after ball placed in bowl)
- **Stop reliability**: ~50% on the main path (April 26 issue), but **fast-lane classifier (May 18)** + smart audio-only retry have largely solved this in practice. May 18 session demonstrated reliable stop via fast-lane path. HPF filtering (200Hz) remains for the main multimodal path.
- **Resume ("continue")**: confirmed working via new `command_resume` intent class (May 18). Goes through cold-start streak filter — same path as initial task start. Session log Qwen #45-46.
- **Switch ("other ball", "another")**: confirmed working via `change_target` predicted_intent + hard-switch path. Session log Qwen #57.

---

## Intent Classes

**`predicted_intent` (1-2s horizon):**
`approach` | `gesture` | `withdraw` | `continue` | `point` | `change_target` | `interrupt` | `unknown` | `new_command` (defined but unused after spoken_command experiment)
Also used for cold-start / resume in `--no-vad` mode: `command_pick_pink_ball` | `command_pick_yellow_ball` | `command_resume` (NEW May 18 — for "continue"/"resume"/"keep going" etc.) | `none`

**`predicted_phase` (3-5s horizon, NEW May 18):**
`approaching` | `grasping` | `transporting` | `placing` | `retracting` | `idle` | `unknown`
Stable over multi-step subactions — same physical phase persists across several 0.5s prediction ticks. Complements `predicted_intent` which can flip rapidly during sub-second actions. Shown in HUD card and logged to MetricsLogger.

**State machine: REMOVED (May 18)**
The `apply_state_machine()` method in `qwen_inference_engine.py` was deleted along with `_sm_state`, `_sm_state_count`, `_sm_gesture_count`. The SM was masking real predictions (holding state on invalid transitions). With the fast-lane classifier and streak filters in PolicyRouter, the SM's false-positive protection is no longer needed — Qwen's raw predictions are used directly.

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

**[DECISION] Audio-interference retry (workaround, not a fix; updated May 18 to smart-retry)**
Qwen3-Omni returns `'\n\n'` (empty) when audio contains speech — a vLLM+Qwen3-Omni quirk. Original fix: retry video-only on empty response. Reduced unknowns from 31% → 1.6%. **May 18 update:** video-only retry was silently losing spoken commands (user says "stop", audio choke, retry drops audio, command lost). Replaced with **smart retry**: branch on audio RMS — speech present (≥0.005) → audio-only retry with `_FAST_PROMPT_AUDIO_ONLY`; silent → legacy video-only retry. See "Smart retry" decision below.

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

**[DECISION] `RobotProfile` hardware abstraction (May 18)**
All robot-specific constants (`_ROBOT_STATE_KEYS`, `_CAMERA_KEYS`, native/training resolution, gripper release pose) extracted into a `RobotProfile` dataclass. `SO101_PROFILE` defined as a module-level constant. `_obs_to_policy_inputs()`, `_decode_chunk()`, `_crop_to_training()`, and `GrootRobotController.__init__` accept a `profile` parameter. Everything else (Qwen perception, PolicyRouter, TaskRegistry, MetricsLogger) is fully robot-agnostic. A commented `G1_PROFILE` sketch is included as a porting guide. To port to a new embodiment: define a new profile + write a LeRobot-compatible driver; no other code changes needed.

**[DECISION] Hard-switch on policy switch (May 18)**
`switch_policy()` previously hot-swapped only the `lang` string, letting GR00T finish its current trajectory before "switching" — in practice it kept executing the old task for 1-2 more full horizons. New default: **hard switch** — stop the control loop, apply `profile.release_overrides` (open gripper for SO-101) to release any held object, then start the new policy from a clean state. Avoids GR00T's "finish current trajectory" inertia. `release_overrides` is robot-specific: SO-101 sets `{"gripper.pos": 50.0}` (open); a G1 would list all finger joints. CLI: `--no-hard-switch` reverts to lang-swap-only; `--gripper-open-pos <deg>` overrides the SO-101 gripper value at runtime without editing the profile.

**[DECISION] Fast-lane audio-only command classifier (May 18)**
The main Qwen path (audio+video, 0.5s poll) has high worst-case stop latency: up to `500ms polling gap + 350-800ms inference = ~1.3s`. Fix: speech-onset detector in `audio_callback` sets `predictor._immediate_request` Event when RMS crosses `_SPEECH_RMS_GATE` for `_SPEECH_ONSET_BLOCKS` consecutive blocks. The scheduler checks this event before its polling clock and fires an inference immediately. That inference uses `classify_command()` — a new **audio-only** Qwen call with a minimal prompt (`stop|command_pick_*|switch_*|none`). Audio-only eliminates the dominant failure mode (audio+video fusion failure on short utterances). `_command_to_prediction()` maps the output to a `PredictionOutput` the router understands (stop→interrupt, switch_*→change_target). One fast-lane fire per speech burst (`_onset_armed` flag reset after each firing).

**[DECISION] `predicted_phase` for longer-horizon prediction (May 18)**
`predicted_intent` (1-2s) flips rapidly during sub-second subactions. Added `predicted_phase` (3-5s): `approaching|grasping|transporting|placing|retracting|idle|unknown`. Phase is stable over several consecutive ticks and represents the high-level manipulation stage. Added to `PredictionOutput`, JSON schema (guided), `MetricsLogger`, and HUD card. Not yet wired into PolicyRouter routing logic — currently informational only (and logged). Thesis value: demonstrates Qwen can do two-level temporal prediction simultaneously.

**[DECISION] `MetricsLogger` structured JSONL logging (May 18)**
Previous logging was ad-hoc log lines only. Added `metrics_logger.py`: writes one JSONL record per Qwen prediction with full schema (ts, seq, intent, phase, confidence, target, task_complete, spoken_command, reason, latency_ms, robot_state, active_policy, audio_rms pre/post HPF, parse_failed, raw_response_len). `run.sh` always creates `~/sessions/metrics_<ts>.jsonl` — the `--metrics` flag is now always passed, not optional. Designed for post-hoc evaluation: precision/recall per intent class, latency distributions, parse failure breakdown, threshold sweeps.

**[DECISION] State machine removed from inference engine (May 18)**
`apply_state_machine()` in `qwen_inference_engine.py` was deleting valid predictions by holding state on "invalid" transitions (e.g. blocking `withdraw` until 2 gestures). With the fast-lane classifier providing a reliable stop path, the SM's false-positive protection was redundant and caused more harm (suppressed real withdraw/interrupt signals) than good. Deleted: `_sm_state`, `_sm_state_count`, `_sm_gesture_count`, `apply_state_machine()`. Streak filters in PolicyRouter (≥2 consecutive interrupt/change_target at conf≥0.85) still guard against single-tick FPs.

**[DECISION] Voice vocabulary centralized at module level (May 18)**
`RELATIVE_WORDS`, `RESUME_WORDS`, `STOP_WORDS` were duplicated as class constants in `PolicyRouter` and inline strings in the command validator. Centralized as module-level tuples in `run_system_groot.py`. `PolicyRouter._RELATIVE_WORDS` and `_RESUME_WORDS` now reference these. Single source of truth for all command matching logic.

**[DECISION] `/no_think` workaround for vLLM Issue #18819 (May 19) ⭐ ROOT-CAUSE FIX**
Persistent ~50% parse-failure rate during execution turned out to be a **documented vLLM bug**, not motor noise or audio encoder degeneration: [Issue #18819](https://github.com/vllm-project/vllm/issues/18819) reports that **Qwen3 models + `chat_template_kwargs={"enable_thinking": False}` + guided JSON output produces gibberish JSON** (extra `{` or `[`, "```" prefix, or complete random tokens). Our symptoms matched exactly — random fragments like `'wise'`, `'ously'`, `'user'`, `'boot'`, `'.com'`, `'-by'`, plus prompt regurgitation (`'{"spoken_command":"Listen to the audio and classify per...'`). **Fix:** removed `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` from all 9 inference call sites; appended `/no_think` to every user-prompt body (`build_user_prompt`, `classify_command`, audio-only retry, video-only multi-frame, transcribe_audio). The `/no_think` soft-switch tells the model to skip its thinking step without triggering the reasoning-parser interaction that breaks guided_json. **Result:** parse-failure rate dropped from ~50% to **effectively 0%** across two consecutive confirmation sessions (May 19 19:29 and 19:33). All garbage-token failure modes disappeared. This was the single biggest improvement in the entire debug cycle — and it made all the earlier hypotheses (token cutoff, sampling tightening, audio encoder degeneration, stale-buffer) effectively moot for the parse-failure dimension. The May 18 silence-buffer fix is still kept (different fix for the stale-RMS pattern); the May 19 max_tokens=160 is kept (harmless defensive padding); the temperature/top_p revert is kept (the 0.01/0.1 attempt actively hurt).

**[DECISION] HPF default flipped — `--hpf` now opt-in (May 19)**
A/B testing confirmed the [Whisper-community consensus](https://github.com/openai/whisper/discussions/2125): denoising before a Whisper-style audio encoder (Qwen3-Omni's AuT is architecturally similar — 128-channel mel-spectrogram) HURTS more than it helps. Session 18:08 (HPF on) vs session 18:28 (HPF off) showed: (a) stop detection went from "fast-lane fallback after ~28s" to "multimodal direct hit at ~8s"; (b) speech RMS peaks ~50% higher (0.05–0.18 vs 0.04–0.12); (c) silence floor unchanged (~0.003 both ways — there was less low-frequency motor rumble than we'd assumed; the HPF was solving a problem that didn't really exist). Flipped the flag: was `--no-hpf` to disable (default on); now `--hpf` to enable (default off). CLI on startup logs `HPF ENABLED (--hpf)` only when explicitly turned back on for comparison runs. **Stays 100% Qwen-driven** — only changes preprocessing, all decisions still made by the model.

**[DECISION] Placing-phase fallback for auto-complete (May 19)**
Qwen reliably reports `predicted_phase="placing"` when the arm is putting an object down, but is conservative about setting `task_complete=true` (fires ~50% of the time it should). Added a tertiary stop in PolicyRouter: ≥2 consecutive `predicted_phase=placing` predictions where `predicted_intent ∈ {gesture, continue, approach}` and `conf ≥ 0.85` and runtime ≥ 15s → fire COMPLETE. New state: `_placing_complete_count = 2`, `_placing_streak = 0`. Streak resets on every state transition (stop, switch, complete, idle, max-runtime). **One follow-up bug:** initially the `predicted_phase` field wasn't being passed in the prediction dict to `router.handle_prediction()`, so the branch never triggered. Fixed by adding `"predicted_phase": pred.predicted_phase` to the dict at the call site. **Status (May 19 session):** confirmed working for pink_ball (auto-stops at runtime ~16s). Does NOT fire for yellow_ball — Qwen never labels yellow placement as `placing` (visual perception issue; classifies as `transporting` or `approaching` instead). Yellow currently relies on the 40s max_runtime safety cap.

**[DECISION] Telemetry dashboard + ZMQ publisher (May 19)**
Built a separate live dashboard (`telemetry_dashboard.py`) and helper module (`telemetry_publisher.py`). One-way ZMQ PUB/SUB from the robot to the dashboard, completely decoupled (publisher is a no-op if the flag is missing). Two panels (confidence with 0.85 streak gate; audio RMS post-HPF with 0.012 speech gate) + status bar (intent/phase/target/policy/state) + event markers (color-coded vertical lines for STOP/SWITCH/RESUME/COMPLETE/COLD-START on both plots). Latency, Hz, and pre-HPF RMS panels were deliberately removed — those are post-hoc analysis metrics, better read from JSONL/CSV than watched live. CSV save on close (full schema retained). CLI: `--telemetry-port 5601`. Dashboard auto-connects; safe to open/close mid-session.

**[DECISION] Recorder slimmed — confidence bar chart removed (May 19)**
Removed the bottom-right scrolling confidence bar chart from the recorded MP4 (`recorder.py`). The HUD card on the camera panel already shows intent + phase + target + confidence + bar, and the live telemetry dashboard is where temporal plots belong. Scrolling log panel now full-height on the right side (was 55% / 45% split). Layout: `camera (60%) | log (40%, full height)`. Also dropped the matplotlib import from the recorder entirely (it was already disabled per April 25 figure-leak fix; now cleaned up).

**[DECISION] `max_tokens` raised 80 → 160 (May 19)**
First real-data session with telemetry showed ~50% parse-failure rate during execution — Qwen returning empty/truncated JSON on roughly half of all inference calls, even on the new audio-only smart retry. CLAUDE.md previously framed this as audio-encoder degeneration (bug #26). **Investigation on May 19 found the more likely cause is mid-JSON truncation at the 80-token cap.** Two May 18 changes pushed typical output above 80 tokens: (a) added `predicted_phase` field (+5-8 tokens for name+value), (b) the rewritten EXECUTING prompt now actually elicits long `spoken_command` values (10-15 tokens for phrases like "okay can you pick up the pink ball"). Typical response budget: ~30 (structure) + 10-15 (spoken_command) + 4-6 (intent+phase) + 8-12 (other values) + 20 (reason 15-word cap) = **70-85 tokens**. Cap was 80 → cut off at the closing brace → empty/truncated parses. Raised to 160 for comfortable headroom. Streaming stops at the JSON `}` via `_stream_until_json`, so the new ceiling costs ~nothing in the happy path. **External evidence supports this isn't purely a Qwen3-Omni quirk** — see [Qwen3-Omni issue #139](https://github.com/QwenLM/Qwen3-Omni/issues/139) and [vLLM issue #27906 fixed in PR #27920](https://github.com/vllm-project/vllm/issues/27906), but those describe different symptoms (worker crashes, streaming bugs), not the mid-JSON cutoff pattern we observed.

**[DECISION] Audio buffer must refresh during silence (May 19) — real bug, but NOT the actual root cause**
*(Note: this was initially diagnosed as the root cause of the ~50% parse-failure rate, but the **actual** root cause turned out to be vLLM Issue #18819. See the `/no_think` DECISION above. This silence-branch fix is still a legitimate bug fix — the stale-RMS pattern was real — but it became cosmetic next to /no_think.)*

Investigation after the May 19 17:50 session (with raw-response logging) surfaced a separate audio-buffer issue. The token-budget hypothesis was wrong; raw responses showed the vast majority of failures are truly empty (`''`), not mid-JSON truncations. Trace: when `robot.state == "running"`, the `audio_callback` silence-branch was a no-op — silent blocks were dropped, **and nothing else was added to the predictor's audio buffer**. During the brief ~358 ms window where `robot.state` flips to `"idle"` during a hard-switch's "stop, release, restart" sequence, the idle-branch (no gate) adds raw mic audio to the buffer. Once running resumes, the silence gate drops every block (background noise ~0.009 RMS, just below the 0.012 gate) — so the buffer keeps the stale 358 ms idle-period audio for many seconds. Symptom in the log: 20+ consecutive parse failures with identical `RMS=0.0092`. Qwen's AuT encoder sees the same low-energy stale fragment over and over and emits empty streams (`''`) or random garbage tokens (`'boot'`, `'.com'`, `'-by-robot'`). **Fix:** silence-branch in `audio_callback` now also calls `predictor.add_audio(filtered)`, so the buffer rolls forward naturally with silence. Engine's `audio_energy < 1e-6` check now correctly trips during true silence → forces video-only mode. Speech bursts are also surrounded by natural silence padding, which matches Qwen3-Omni's training distribution. The explicit `predictor.audio_buffer.clear()` on burst-end still fires to flush residual speech content.

**[DECISION] Sampling tightened 0.1/0.9 → 0.01/0.1 — REVERTED same day (May 19)**
[Qwen3-Omni Issue #139](https://github.com/QwenLM/Qwen3-Omni/issues/139) reference code uses `temperature=0.01, top_p=0.1` as the stable deterministic config for audio input. Tried this briefly — **made things worse**. Two new failure modes emerged: (a) deterministic prose responses — the model emitted 549+ char natural-language paragraphs instead of JSON when the AuT audio encoder was confused (visible as session log raw response starting `'itionally, the robot arm is currently in the process of placing the yellow cotton ball...'`); (b) stop detection during long-running tasks got noticeably worse — the model commits early to a wrong path on ambiguous audio rather than recovering probabilistically. **Reverted to 0.1/0.9.** Lesson: the Issue #139 config is probably tuned for their offline batch use case; for live streaming with motor-noise-corrupted audio, the slight extra entropy at 0.1/0.9 lets the sampler escape degenerate paths and find JSON structure.

**[DECISION] Raw-response logging on parse failure (May 19)**
When `_parse_prediction` falls into the regex-fallback branch, we now also log the raw response text (truncated to 240 chars). Previously the log only said "Empty response" / "No JSON object found" — uninformative because we couldn't tell apart (a) truly empty streams, (b) mid-JSON token cutoffs, (c) malformed content. The raw text immediately distinguishes these. After the max_tokens bump, if any parse failures remain, the raw log will tell us what kind to chase.

**[DECISION] Focused-question pattern: dedicated yes/no Qwen calls beat structured fields (May 21) ⭐ DESIGN PRINCIPLE**
The grounding gate and completion verifier (below) are both built on a principle proven repeatedly this project: **Qwen3-Omni is unreliable at structured fields, object/color naming, and gripper-state, but reliable at a single focused yes/no visual question.** Evidence: the inline `task_complete` field fires <0.3% of the time; Qwen labels the same balls "purple"/"cyan"/"ball"/"yarn" across frames; a closed-but-empty gripper reads as "holding." Both new features failed when they string-matched Qwen's names or trusted a buried field, and became reliable once reframed as a dedicated vision-only call asking ONE question, with `/no_think` and a 2-consecutive-confirmation guard. Apply this for any future perception-driven behavior. Also a methodology note: this was found by **instrumenting (logging every call's raw result) and reading the data, not by guessing prompts** — the per-call `Completion check: complete=... why=...` log is what cracked it.

**[DECISION] Visual grounding gate — cold-start only, pickability-based (May 21)**
Before STARTING a pick from idle, `verify_object_present()` (vision-only Qwen call) checks "is there any small graspable object on the table?" and the router refuses the start if the workspace is empty (prevents reaching into empty space / acting on a non-present object). Two failed earlier designs taught the final shape: (a) exact-color match ("is a *pink* ball present?") false-vetoed valid picks because Qwen's color naming is unreliable; (b) object-noun match ("ball" in seen-list) false-vetoed because Qwen flips "ball"↔"yarn". Final: trust Qwen's boolean answer to the generic pickability question — do NOT string-match its words. **Cold-start only** (`self.state != "running"`): the gate is skipped on mid-task switches, because the switch-time frame (arm holding the current object) makes the pickability judgment misfire, and a ball is obviously present mid-task anyway. CLI: `--no-grounding`. Honest limit: cannot catch "pick red when only yellow present" (a ball IS present) — color grounding is beyond the model.

**[DECISION] Visual completion verifier — the reliable auto-stop (May 21) ⭐**
GR00T loops forever; the inline `task_complete` field and placing-phase fallback both missed most real completions (esp. yellow), so the robot kept picking at the empty table until the 40s cap. Fix: `verify_task_complete()` — a dedicated, throttled (every 2.0s past 8s runtime), vision-only Qwen call asking **"is a ball now resting INSIDE the bowl/plate?"**. Two consecutive confirmations (conf ≥ 0.7) → stop. Key design choices, each from an observed failure: (a) **ask about the bowl contents, NOT gripper state** — after placing, GR00T closes the empty gripper on the table and Qwen reads it as "holding," which defeated an earlier "gripper empty?" criterion; (b) **no phase gate** — the post-placement loop is mislabeled "approaching/grasping," so gating on phase skipped the check exactly when needed; (c) **2-consecutive guard** absorbs Qwen's frame-to-frame noise (e.g. a transient false `complete=True` during transport, or GR00T re-grabbing the ball mid-loop). **Confirmed live May 21: yellow auto-stops at ~15s** (the long-standing gap) and pink stops via verifier or placing-fallback. Every check is logged (`Completion check: complete=... conf=... why=...`). Fails safe (returns not-complete on error → max-runtime backstop). CLI: `--no-completion-check`. Does NOT yet return the arm to a home pose — stop = idle/freeze in place; home-return is a possible follow-up.

**[DECISION] Registry-driven prompts — no object name hardcoded (May 21)**
The EXECUTING / VIDEO_ONLY / AUDIO_ONLY prompts (previously hardcoded "Cotton balls (pink, yellow)" + pink/yellow few-shot examples) are now built by `_build_executing_prompt()` / `_build_video_only_prompt()` / `_build_audio_only_prompt()`, templated from the task registry (`_cold_start_choices` → object names), matching the already-dynamic `_build_waiting_prompt()`. Object list and few-shot examples derive from `tasks.yaml` objects (obj1/obj2, first-word as color). Cached as instance attrs in `__init__` (shadow the old class constants). WAITING example B also made dynamic. Result: adding a pick task is a pure `tasks.yaml` edit. The pick-and-place *phase vocabulary* (approaching/grasping/transporting/placing/retracting) remains as a stated domain assumption — a structurally different task type (pour, stack) would still need new phases. `_FAST_PROMPT_EXECUTING_LEGACY` kept for revert.

**[DECISION] Placing-phase fallback → sliding window + 12s gate (May 21)**
The strict "2 consecutive placing" auto-complete (May 19) rarely tripped — eval_metrics showed yellow hits `placing` at half pink's rate and almost never twice in a row, AND the placement happens at ~7-13s but the old gate was 15s (suppressing nearly every placing frame). Changed to: (a) sliding window — ≥2 `placing` hits within the last 4 gated predictions (robust to the placing→transporting oscillation); (b) decoupled `placing_min_runtime_s = 12.0` (placement clusters at 10-13s; sweep showed 12s doubles pink fire rate, no gain below). Now a constructor param, not a magic constant. This is the secondary completion path; the visual verifier is primary.

**[DECISION] eval_metrics.py baseline tool (May 21)**
`eval_metrics.py` aggregates `~/sessions/metrics_*.jsonl` into a baseline report: parse-failure rate, latency (waiting/executing/timeouts), intent & phase distributions, task_complete count, placing-coverage per policy. Re-runnable for the G1 comparison later. Frozen SO-101 baseline (18 post-/no_think sessions, 1473 predictions): **0.0% parse failures**, executing latency median 471ms / p95 606ms, `task_complete` fired only 4× total (confirming the inline field is unusable — motivated the visual verifier).

**[DECISION] Completion verifier asks about the BOWL, not the gripper or phase (May 21, evening) ⭐**
The first completion verifier required "(a) ball in bowl AND (b) gripper empty" and was gated to placing/transporting phases. Both failed in practice: after a place, GR00T loops back and closes the empty gripper on the table, which Qwen reads as "holding" (defeating criterion b); and the post-place loop is mislabeled "approaching/grasping" so the phase gate skipped the check. Reworked: ask ONLY "is a ball now resting INSIDE the bowl/plate?", drop the gripper criterion, and remove the phase gate (check every 2.0s past 8s runtime regardless of phase). Every check logs `Completion check: complete=... conf=... why=...` for auditability. Confirmed live: both balls auto-complete and stop. This is the reliable auto-complete path; placing-phase fallback + max-runtime remain as backups.

**[DECISION] Return-to-home on completion (May 21) — robot-agnostic**
After a confirmed completion stop, `GrootRobotController.go_home()` smoothly interpolates each joint (30 steps over ~1.2s) from its current pose back to the pose captured at `connect()`. The home pose is **auto-captured at startup** (not hardcoded), so it is robot-agnostic and works unchanged on a G1. Triggered only on the goal-confirming stops (visual verifier, inline task_complete, placing-fallback) — NOT on verbal stop or switch (you don't want it homing mid-redirect). `--no-home` disables. Caveat: "home" = whatever pose the arm is in at launch, so start it in a good ready pose.

**[DECISION] 8s request timeout — fixes the startup "hang" (May 21) ⭐**
The httpx client timeout was 30s. An occasional server-side vLLM/Qwen3-Omni stall (stops streaming mid-response, more likely on idle/silent input) leaves the single inference worker stuck; with queue backpressure that freezes the ENTIRE prediction pipeline until the timeout fires — i.e. a 30s "hang" where the robot appears dead (these were the 30008ms outliers in the baseline). Dropped the read timeout to 8s (`httpx.Timeout(8.0, connect=5.0)`): a normal call is <1s, so 8s cleanly separates legit from stalled; a stall now raises in 8s and the worker recovers/continues instead of freezing. Resilience, not prevention — the stalls are server-side (vLLM 0.12.0; a future upgrade may reduce them).

**[DECISION] Lenient grounding — allow if `seen` lists a graspable object (May 22)**
The cold-start grounding gate was false-refusing resume/re-pick in a loop: Qwen answered `pickable_present=false` while its own `seen` list said "purple yarn ball" — a graspable object. Fix: when grounding would refuse, first check the `seen` list for any graspable noun (ball/yarn/pom/cotton/cube/block/object/toy/sphere); if present, override and allow. Refuse only when the scene truly shows nothing graspable. Biases toward letting the robot try — the empty-table safety is preserved, but it stops fighting valid resume/re-pick. (Given how often Qwen's pickability judgment caused false refusals, leniency is the right trade-off.)

**[DECISION] Out-of-vocabulary command rejection — emergent from registry prompts (May 26)**
With the WAITING prompt now enumerating only the registry's known commands (registry-driven prompts, May 21), Qwen rejects a command for an object not in the set: "pick the red ball" → `predicted_intent="none"`, `why="Command for 'red ball' not in known objects"`. So a nonexistent color no longer silently starts the nearest real task **at cold-start** (it still can't be caught mid-task switch — grounding is cold-start-only and a ball is present). Not a code change — a tested consequence of the prompt templating, confirmed in the May 26 runs.

**[DECISION] vLLM worker exponential backoff on connection errors (May 18)**
Previously, if vLLM was unreachable (cold start, network blip), the inference worker logged an error and immediately retried — hammering the server or hanging the worker for 30s on httpx's default timeout. Fix: catch connection/timeout errors and set `backoff_until = time.time() + 2.0`; worker sleeps in 0.2s increments until the backoff expires. Prevents log spam and avoids blocking the worker queue.

**[DECISION] `_FAST_PROMPT_EXECUTING` rewritten as future-prediction prompt (May 18)**
Old EXECUTING prompt framed Qwen as a multi-job classifier ("LISTEN + WATCH"). New prompt explicitly frames the model's role as **predicting the next 1-2s (predicted_intent) and 3-5s (predicted_phase)** using both video AND audio. The SWITCH rule now says explicitly that on a switch command, the arm SHOULD drop the current object and redirect — `predicted_phase="retracting"` — because the task is to **predict the future, not describe the past**. Tightened from ~520 to ~330 tokens. Legacy prompt preserved as `_FAST_PROMPT_EXECUTING_LEGACY` for one-line revert. Sets up the conceptual framing for the upcoming cross-start dataset.

**[DECISION] Smart retry: audio-only fallback when audio had speech (May 18)**
Previously, any empty/parse-failed response triggered a video-only retry — which **lost the spoken command** in the dominant failure case (user says "stop", audio encoder chokes, retry drops audio, command is silently lost). Fix in `predict_intent_multi_frame`: compute audio RMS on failure; if `RMS ≥ 0.005` (had speech) → retry **audio-only** with the new `_FAST_PROMPT_AUDIO_ONLY` prompt, preserving the command. Else → legacy video-only retry. The audio-only prompt has a small command-oriented intent vocabulary (interrupt | change_target | continue | unknown) since it can't classify motion without video.

**[DECISION] `command_resume` explicit cold-start intent class (May 18)**
April 26 added a resume path that checked `spoken_command` for resume words. But `spoken_command` is unreliable (the schema-discipline issue). New approach: WAITING prompt now classifies `"continue" / "resume" / "keep going" / "go on" / "carry on" / "proceed"` into a dedicated `command_resume` intent. PolicyRouter routes it through the same `_cold_start_streak` filter (≥2 consecutive at conf≥0.85) as `command_pick_*`, but instead of resolving a task by name, it restarts `_last_active_policy`. If no last policy exists, logs "command_resume heard but no last policy to resume" and drops it. This is the path that fired cleanly in the May 18 session (`Qwen #45-46 → Multimodal RESUME → pick_pink_ball`).

**[DECISION] Speech-burst accumulator replaces rolling 2s audio buffer at onset (May 18)**
Previously, the predictor's 2s rolling buffer accumulated whatever audio was active — old phrases lingered for 2 seconds and got replayed on subsequent inferences (visible as "yellow ball instead" replay spam after a stop). New pattern: `audio_callback` maintains a `_speech_burst` list that only fills with **voiced blocks** (RMS ≥ `_SPEECH_RMS_GATE = 0.012`, post-HPF). When the burst reaches `_SPEECH_ONSET_BLOCKS = 5` (~500 ms of voiced energy), the predictor's audio buffer is **replaced** with just the burst — not appended. After `_BURST_END_SILENCE_BLOCKS = 4` (~400 ms of silence) the burst is cleared and `_onset_armed` resets. Result: Qwen sees a clean ~500 ms burst of just the command, not 2s of mixed silence+command+noise. One fast-lane fire per speech burst.

**[DECISION] `_clear_audio_buffer()` on stop/resume/switch/complete (May 18)**
Even with the burst accumulator, the predictor's audio buffer can hold stale audio across a routing decision. After every router action (stop, resume, switch, complete), `PolicyRouter._clear_audio_buffer()` is called to wipe `predictor.audio_buffer`. This is what prevents the "yellow ball instead" replay spam visible in earlier sessions — see session log Qwen #11-33 (silence respected for ~20s after stop with no spurious replays).

**[DECISION] `request_immediate_inference()` public API (May 18)**
The scheduler's fast-lane trigger (`_immediate_request` Event) is wrapped by a public method on `StreamingIntentPredictor`. Thread-safe, single-shot. Called from the `audio_callback` when speech onset is detected. Keeps the threading.Event encapsulated and gives the call site a self-documenting name.

---

## Current Engine (FastQwenInferenceEngine) Settings

```python
temperature        = 0.1   # tried 0.01 on 2026-05-19 — reverted (broke JSON discipline)
top_p              = 0.9   # tried 0.1  on 2026-05-19 — reverted (broke JSON discipline)
max_tokens         = 160   # raised 80→160 on 2026-05-19 (defensive; no_think is the actual fix)
image_quality      = 65    # JPEG q65
max_width          = 320   # 320×240 per frame
audio_max_duration = 2.0   # bumped from 0.5s on 2026-04-25 so Qwen sees full commands
# All user prompts end with "/no_think" (May 19 — vLLM Issue #18819 workaround).
# Replaces extra_body={"chat_template_kwargs": {"enable_thinking": False}} which
# was triggering a documented Qwen3 + guided_json bug producing gibberish JSON.
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

**This session (May 18):**
29. Hard-switch on task switch was missing — `switch_policy()` only hot-swapped the lang string; GR00T continued its current trajectory for 1-2 more horizons → added hard-switch mode (default on): stop loop + apply release_overrides + restart. `--no-hard-switch` flag reverts to legacy behavior.
30. State machine in `qwen_inference_engine.py` was suppressing valid interrupt/withdraw predictions — SM blocked `gesture→withdraw` until 2 prior gestures and `interrupt` was unreachable from several states → removed `apply_state_machine()` entirely; streak filters in PolicyRouter provide sufficient FP protection.
31. vLLM worker hammering server on cold-start / network blip — connection errors caused immediate retries in a tight loop → added 2s `backoff_until` in `_inference_worker` on connect/timeout errors.
32. Robot-specific constants scattered through `run_system_groot.py` made porting impossible — `_ROBOT_STATE_KEYS`, `_CAMERA_KEYS`, resolutions all hardcoded → extracted into `RobotProfile` dataclass + `SO101_PROFILE` instance.
33. Audio buffer stale-replay spam during silence — after a stop command, 2.0s rolling audio buffer contained old speech ("yellow ball instead") that would be replayed to Qwen on the next inference, causing false `change_target` after the robot had already stopped → added `_clear_audio_buffer()` method on PolicyRouter, called after stop/resume/switch/complete to purge stale audio. Also added speech-burst accumulator in `audio_callback`: only voiced blocks (post-HPF RMS ≥ 0.012) accumulate; on onset (≥500ms voiced) buffer is **replaced** with just the burst; on sustained silence (≥400ms) burst clears. Result: Qwen sees a clean ~500ms burst rather than 2s of mixed audio.
34. Video-only retry silently dropping spoken commands — when audio+video failed, the retry forced video-only, **losing the audio that was the whole point** (user said "stop", encoder choked, retry dropped the audio, command was silently lost) → smart retry in `predict_intent_multi_frame`: branch on audio RMS. RMS ≥ 0.005 → audio-only retry with new `_FAST_PROMPT_AUDIO_ONLY`. RMS < 0.005 → legacy video-only retry.
35. Resume via spoken_command was unreliable — April 26's resume path checked `spoken_command` for resume words, but `spoken_command` is empty in ~100% of predictions (schema-discipline issue) → added explicit `command_resume` intent class to the WAITING prompt. PolicyRouter routes it through the cold-start streak filter (≥2 consecutive at conf≥0.85) and restarts `_last_active_policy`. Confirmed working in May 18 session (Qwen #45-46 → RESUME → pick_pink_ball).
36. EXECUTING prompt was "describe the past" framed (TWO JOBS: LISTEN, WATCH) — model produced reactive predictions rather than predictive ones, and the SWITCH rule didn't tell the model the arm SHOULD drop the current object on switch → completely rewrote `_FAST_PROMPT_EXECUTING` as future-prediction (predict next 1-2s + 3-5s); SWITCH rule now explicitly says `predicted_phase="retracting"` so the model commits to the drop-and-redirect future. Legacy version kept as `_FAST_PROMPT_EXECUTING_LEGACY`.

**This session (May 19) — multi-day debug chronicle, in order of discovery:**

37. **Initial misdiagnosis: thought parse failures were token-budget cutoff.** Raised `max_tokens` 80 → 160. Added raw-response logging on parse failure (24-char truncated). The logging was the right move — it revealed the actual symptom. Token bump was harmless but didn't help.

38. **Second misdiagnosis: stale audio buffer.** Raw responses showing identical `RMS=0.0092` across many consecutive predictions led to discovering that `audio_callback`'s silence-branch was not adding to the predictor buffer — leftover audio from the brief `robot.state="idle"` window during a hard-switch persisted for many seconds. **Fix:** silence-branch now calls `predictor.add_audio(filtered)` regardless of energy. This was a real bug but not the dominant cause of parse failures.

39. **Third misdiagnosis: sampling temperature too high.** Tried tightening from (0.1 / 0.9) to (0.01 / 0.1) per [Qwen3-Omni Issue #139](https://github.com/QwenLM/Qwen3-Omni/issues/139) reference config. Made things worse — produced 549+ char natural-language paragraphs instead of JSON when the encoder was confused. **Reverted.**

40. **HPF tested and removed as default.** A/B compared HPF on vs `--no-hpf` flag in two consecutive sessions. Without HPF: stop went from "fast-lane fallback after ~28s" to "multimodal direct hit at ~8s." Confirmed [Whisper-community consensus](https://github.com/openai/whisper/discussions/2125). Silence floor was identical with/without HPF, so the HPF was solving a non-problem. Flag inverted: `--no-hpf` was opt-out → `--hpf` now opt-in.

41. **ACTUAL ROOT CAUSE: vLLM Issue #18819 ⭐** — Qwen3 + `enable_thinking=False` + guided_json produces "complete gibberish." Exactly matched all our symptoms (random fragments, prompt regurgitation, empty streams). Workaround: append `/no_think` to user prompts instead of using `chat_template_kwargs.enable_thinking=False`. Applied across all 9 inference call sites + 5 user-prompt construction sites. **Parse-failure rate dropped from ~50% to ~0%** in two consecutive confirmation sessions. The earlier fixes (silence-branch buffer feed, max_tokens bump) became cosmetic — they don't hurt but they were never the actual cause.

42. **Placing-phase fallback for auto-complete (May 19 evening).** Qwen reliably reports `predicted_phase="placing"` but is conservative about `task_complete=true`. Added a third stop tier in PolicyRouter: ≥2 consecutive placing-phase predictions past 15s runtime → COMPLETE. Confirmed firing reliably for pink_ball (auto-stops at runtime ~16s).

43. **Placing fallback wired to dead dict key.** First version of the placing fallback never fired in production because `predicted_phase` wasn't being included in the prediction dict passed from the main loop to `router.handle_prediction()` — only the `pred` dataclass had it. **Fix:** added `"predicted_phase": pred.predicted_phase` to the dict at the call site. After this fix, pink-ball auto-complete fired cleanly.

44. **Yellow-ball auto-complete doesn't fire (known limitation).** In the same session that confirmed pink, yellow_ball ran for 65+ predictions without Qwen ever labeling the placement as `predicted_phase="placing"`. All predictions stayed in `approaching` / `transporting` / `grasping`. Likely a perceptual issue — yellow may contrast less with the bowl than pink, or GR00T's yellow trajectory may not produce a cleanly observable "placing" motion. Currently falls back to the 40s max_runtime safety cap. Yellow placement is the last functional gap in the SO-101 system.

---

## Pending / Future Work

**Active — top of mind:**
- [ ] **Commit the working tree (May 26)** — everything since `78e11e6` is uncommitted (grounding, completion verifier, return-to-home, prompt templating, timeout fix, lenient grounding, eval_metrics/metrics_logger/telemetry files). Commit before teleop/retrain.
- [ ] **Record teleop data + retrain GR00T** — the user's chosen next step (before G1 code work). Goal: more consistent placement (esp. yellow), which directly improves the perception-driven completion latency.
- [ ] **Port to G1** — `RobotProfile` covers geometry but the modality grouping (`single_arm`/`gripper` in `_obs_to_policy_inputs`/`_decode_chunk`) and the LeRobot driver class (`SOFollowerRobotConfig` in `GrootRobotController.__init__`) are still hardcoded to SO-101 — move these into the profile. Also need a G1-trained GR00T checkpoint + `G1_PROFILE`. The Qwen/PolicyRouter/TaskRegistry stack above the profile is robot-agnostic.
- [ ] (Optional) Clean A/B of pink-vs-yellow completion latency — ~5 uninterrupted completions each; today's numbers are confounded by mid-pick verbal stops.
- [ ] (Optional) Lower the 0.012 audio gate or use a closer mic — quiet "stop" commands can fall below it.

**Confirmed done (closing out — kept here as history):**
- [x] **Yellow-ball auto-complete — SOLVED (May 21)** via the visual completion verifier. Both balls auto-stop reliably.
- [x] **Return-to-home on completion (May 21)** — `go_home()` interpolates to the auto-captured startup pose after a confirmed completion. `--no-home` to disable.
- [x] **Startup "hang" fixed (May 21)** — 8s request timeout; a server stall now recovers in ~8s instead of freezing 30s.
- [x] **Resume/re-pick false-refusal fixed (May 22)** — lenient grounding override.
- [x] Run MetricsLogger evaluation — `eval_metrics.py` built; SO-101 baseline frozen (0% parse failures, latency p95 606ms). Re-run on G1 for comparison.

**Confirmed done (closing out — kept here as history):**
- [x] Confirm hard-switch gripper release on live robot — May 19 sessions show repeated clean hard switches with `(stop, release, restart)` log lines.
- [x] Confirm fast-lane stop latency improvement — visible in May 19 sessions, ~7-11s stop latency on long executions via fast-lane.
- [x] **Fix verbal stop reliability in `--no-vad` mode** — was ~50% retry rate, now ~0% after /no_think fix. Stop fires via direct multimodal hit in most cases.
- [x] **Confirm max_tokens=160 fix** — confirmed in May 19 19:29 and 19:33 sessions; ~0% parse failures. (The /no_think workaround did most of the work; max_tokens is harmless padding.)
- [x] **Test HPF-disabled run** — confirmed better; HPF default flipped.

**Lower priority / future:**
- [ ] Check vLLM version on s99 (confirmed 0.12.0 — older than 0.18.0 where Qwen3-Omni audio crash was fixed). Not urgent now that /no_think solves the dominant failure mode, but a future upgrade window could clean up the residual issues.
- [ ] `'wise'`-style residual responses — Qwen still occasionally outputs a single word instead of JSON (rare now; was common before /no_think); the regex fallback catches most.
- [ ] Async FrameRecorder — recorder is async but control loop integration could be cleaner.
- [ ] VAP-Realtime integration for turn-taking awareness (needs stereo audio) — note: VAD stays even after this.
- [ ] Re-evaluate cleaning video accuracy (last measured at 57.1%) with the post-/no_think engine.
- [ ] Speech buffer unbounded growth fix in AudioInterruptDetector.
- [ ] Latency decomposition (network vs inference vs encoding).
- [ ] Decide whether to delete the `spoken_command` plumbing now that the experiment failed, OR leave it as a "tried and failed" code artifact (currently leaving it; harmless).
- [ ] Write thesis methodology section using the revised framing (Qwen as unified semantic backbone, not VAP+VAD replacement). The May 19 debug chronicle is rich thesis content.
- [ ] Wire `predicted_phase=retracting` into PolicyRouter routing as additional task-complete signal (alongside placing-phase fallback).
- [ ] Mask parse-failed predictions on the dashboard confidence panel (low priority now that parse failures are rare).
- [ ] Delete legacy code now safe to remove: `_FAST_PROMPT_EXECUTING_LEGACY`, the inert `spoken_command` field plumbing, the disabled HPF code path (or keep as `--hpf` opt-in for A/B research runs).

---

## What to Tell Claude Next Session

Paste this file, then use one of these openers:

**System work:**
- "Fix yellow-ball auto-complete" → pick one of (A) lower `max_task_runtime_s` to 25s, (B) sliding-window placing-OR-transporting fallback, (C) document as known limitation. Pink works via placing-phase fallback at runtime ~16s; yellow falls back to max_runtime.
- "Port to G1" → define `G1_PROFILE`, write LeRobot-compatible driver shim, train GR00T on G1 data. `RobotProfile` abstraction is the entry point; everything above it is robot-agnostic.
- "Analyse metrics logs" → load `~/sessions/metrics_*.jsonl` (post-May-19 19:00 sessions, since parse rate is ~0%). Compute: precision/recall by intent, fast-lane hit rate, latency distributions, placing-fallback fire rate.
- "Improve recorder / HUD" → currently camera (60%) + scrolling log (40%); could add a small dashboard-style sparkline.

**Thesis / writeup:**
- "Help me write the methodology section" → use Architecture + research contribution sections. May 19 debug chronicle (bugs #37-44) is rich content for a "failure analysis and root cause" subsection.
- "Help me write the failure modes analysis" → the spoken_command discipline experiment (April 25), the 0.01/0.1 sampling reversal (May 19), and the long /no_think hunt are all defensible "model-behavior ceilings" content.
- "Re-evaluate cleaning video accuracy" — was 57.1% with old engine; retest with current /no_think-clean engine.

**Cleanup:**
- "Delete legacy code" → `_FAST_PROMPT_EXECUTING_LEGACY`, the inert `spoken_command` plumbing, the HPF path (or keep behind `--hpf` flag).
- "Refactor: drop the dead spoken_command code" → revert the April 25 failed experiment cleanly.

**Debugging:**
- Describe a new runtime bug → paste the relevant log lines and describe what you expected. The raw-response logging in `_parse_prediction` (May 19) makes parse-failure diagnosis fast.

---

## System Prompts (FastQwenInferenceEngine)

There are now **five** prompt constants (May 18 update). Each is purpose-built for a specific inference path. The WAITING/EXECUTING distinction (selected by `'state=waiting' in robot_state_string`) is still load-bearing.

**_FAST_PROMPT_WAITING** (state=waiting / robot idle, `--no-vad` mode): Classifies spoken commands into `command_pick_pink_ball`, `command_pick_yellow_ball`, **`command_resume`** (NEW May 18 — for "continue" / "resume" / "keep going" etc.), or `none`. Emphasises listening for verbal commands. Used for cold-start task detection AND resume after a stop. Includes four few-shot examples (A: command, B: silence, C: unrelated speech, D: resume).

**_FAST_PROMPT_EXECUTING** (task active — selected when `'state=waiting' NOT in robot_state_string`): **Rewritten May 18** as a future-prediction prompt. Old format ("TWO JOBS — LISTEN + WATCH") was reframed to: predict the next 1-2s (`predicted_intent`) and 3-5s (`predicted_phase`) using video AND audio together. The SWITCH RULE explicitly states the arm SHOULD drop the current object and redirect (`predicted_phase="retracting"`) — predict the future, not describe the past. Tightened to ~330 tokens. Three few-shot examples (silent grasp, "wait stop" interrupt, "no the yellow one" change_target). Legacy version preserved as `_FAST_PROMPT_EXECUTING_LEGACY` for one-line revert.

**_FAST_PROMPT_VIDEO_ONLY** (April 26, updated May 18): Video-only prompt for `predict_intent_video_only_multi_frame` AND the video-only branch of the smart-retry path. Does NOT mention audio. Now includes `predicted_phase` field. Critical: if EXECUTING prompt is used for a video-only request, Qwen3-Omni returns empty in ~88ms.

**_FAST_PROMPT_AUDIO_ONLY** (NEW May 18): Audio-only fallback prompt used by `predict_intent_multi_frame`'s smart-retry path when the audio+video call failed AND audio had voiced energy (RMS≥0.005). Small command-oriented intent vocabulary (interrupt | change_target | continue | unknown) since the model can't classify motion without video. Critical for preserving stop/switch commands that would otherwise be lost on video-only retry. Also used by the fast-lane `classify_command()` via a separate dynamically-built classifier prompt.

**Classifier prompt** (NEW May 18, built dynamically by `_build_classifier_prompt()` for `classify_command()`): Minimal prompt enumerating `stop | command_pick_<task> | switch_<color> | none`. Task names enumerated dynamically from `_cold_start_choices` so adding a task to tasks.yaml requires no engine change. Output is JSON `{"command": "...", "confidence": 0.0-1.0, "heard": "<verbatim>"}`.

JSON output schema (EXECUTING + WAITING):
```json
{"spoken_command":"<words or empty>",
 "predicted_intent":"<class>","predicted_phase":"<phase>",
 "confidence":0.0-1.0,
 "target_object":"<object color or none>",
 "task_complete":false,
 "reason":"<15 words max>"}
```

JSON output schema (VIDEO_ONLY — no spoken_command):
```json
{"predicted_intent":"<class>","predicted_phase":"<phase>",
 "confidence":0.0-1.0,
 "target_object":"<object color or none>",
 "task_complete":false,
 "reason":"<15 words max>"}
```

JSON output schema (fast-lane audio classifier — `classify_command()`):
```json
{"command":"stop|command_pick_<task>|switch_<color>|none",
 "confidence":0.0-1.0,
 "heard":"<verbatim words or empty>"}
```

Reality: `spoken_command` is always empty regardless of audio content. `reason` ends up containing audio observations like `"audio command 'stop'"` even though forbidden. During execution, ~50% of requests fall back to video-only due to audio encoder degeneration from motor noise. The fast-lane `classify_command()` path uses audio-only to sidestep this failure mode.

**InterruptReason types:** `VERBAL_STOP`, `VISUAL_INTERRUPT`, `CHANGE_TARGET`, `OBJECT_MISMATCH`, `TRAJECTORY_CHANGE`
