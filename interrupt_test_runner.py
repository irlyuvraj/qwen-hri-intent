#!/usr/bin/env python3
"""
Interrupt Detection Test Runner
================================
Tests the full interrupt detection system using just a video file.
No GR00T, no robot, no microphone required.

HOW TO USE
----------
1. Record your test video (phone or webcam, top-down view):
   - Do a task with your hands (pick up objects)
   - Midway, say "stop" or "not that one, the other one" out loud
   - Redirect to a different object

2. Run this script:
   python interrupt_test_runner.py \\
     --video my_interrupt_test.mp4 \\
     --task "pick up the cup" \\
     --task-object "cup" \\
     --output results/interrupt_test.jsonl

3. It will print STOP events as they are detected in real time.

HOW IT WORKS (no GR00T version)
---------------------------------
  FileBasedCapture reads your video frame by frame
  StreamingIntentPredictor feeds frames to Qwen every 0.5s
  InterruptDetectionSystem watches predictions for mismatches
  When interrupt detected → prints STOP event + logs to file
  
  Audio from the video is sent to Qwen as part of each inference call
  so Qwen hears both "stop" commands AND sees the visual trajectory change.

SIMULATING INTERRUPTS WITHOUT AUDIO
-------------------------------------
If your test video has no audio or you want to test visual-only:
  Use --inject-interrupt to manually trigger an interrupt at a timestamp:
  
  python interrupt_test_runner.py \\
    --video my_test.mp4 \\
    --task "pick up the cup" --task-object "cup" \\
    --inject-interrupt 8.0 "stop, pick up the bottle instead" \\
    --output results/test.jsonl

  This injects a verbal command at t=8.0s as if the human said it.
"""

import argparse
import json
import time
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from streaming_intent_predictor import StreamingIntentPredictor, StreamConfig
from qwen_inference_engine import FastQwenInferenceEngine
from interrupt_detection_system import (
    InterruptDetectionSystem,
    InterruptEvent,
    connect_to_predictor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


class InterruptTestLogger:
    """Logs all predictions and interrupt events to JSONL."""

    def __init__(self, output_file: str):
        self.output_file = output_file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        # Clear file
        open(output_file, 'w').close()
        self.interrupt_count = 0
        self.prediction_count = 0
        self._start_time = time.time()

    def log_prediction(self, pred, elapsed: float):
        self.prediction_count += 1
        entry = {
            'type':             'prediction',
            'elapsed_s':        round(elapsed, 2),
            'predicted_intent': pred.predicted_intent,
            'confidence':       pred.confidence,
            'target_object':    pred.target_object,
            'reason':           pred.reason,
            'latency_ms':       pred.latency_ms,
        }
        self._write(entry)

    def log_interrupt(self, event: InterruptEvent, elapsed: float):
        self.interrupt_count += 1
        entry = {
            'type':            'interrupt',
            'elapsed_s':       round(elapsed, 2),
            'reason':          event.reason.value,
            'confidence':      event.confidence,
            'predicted_intent': event.predicted_intent,
            'predicted_object': event.predicted_object,
            'raw_command':     event.raw_command,
            'new_task':        event.new_task,
        }
        self._write(entry)
        # Print clearly to terminal
        print(f"\n{'='*60}")
        print(f"🛑 INTERRUPT at t={elapsed:.1f}s")
        print(f"   Reason:    {event.reason.value}")
        print(f"   Triggered: {event.predicted_intent}({event.predicted_object})")
        if event.raw_command:
            print(f"   Command:   '{event.raw_command}'")
        if event.new_task:
            print(f"   New task:  '{event.new_task}'")
        print(f"{'='*60}\n")

    def _write(self, entry: dict):
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def summary(self):
        elapsed = time.time() - self._start_time
        print(f"\n{'─'*60}")
        print(f"TEST SUMMARY")
        print(f"{'─'*60}")
        print(f"  Runtime:         {elapsed:.1f}s")
        print(f"  Predictions:     {self.prediction_count}")
        print(f"  Interrupts:      {self.interrupt_count}")
        print(f"  Output:          {self.output_file}")
        print(f"{'─'*60}")


def run_interrupt_test(
    video_path:        str,
    task_command:      str,
    task_object:       str,
    output_file:       str,
    vllm_url:          str,
    model_name:        str,
    interval:          float,
    fps:               float,
    inject_interrupts: list,   # list of (timestamp_s, command_text)
    consecutive:       int,
    no_audio:          bool,
):
    print(f"\n{'='*60}")
    print(f"INTERRUPT DETECTION TEST")
    print(f"{'='*60}")
    print(f"  Video:       {video_path}")
    print(f"  Task:        '{task_command}' → object='{task_object}'")
    print(f"  Interval:    {interval}s")
    print(f"  FPS target:  {fps}")
    print(f"  Consecutive: {consecutive} mismatches to trigger")
    if inject_interrupts:
        print(f"  Injected interrupts:")
        for t, cmd in inject_interrupts:
            print(f"    t={t:.1f}s → '{cmd}'")
    print(f"{'='*60}\n")

    # ── Set up video capture ───────────────────────────────────────────────
    if not _HAS_CV2:
        raise RuntimeError("OpenCV required: pip install opencv-python")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / native_fps
    logger.info(f"Video: {total_frames} frames, {native_fps:.1f}fps, {duration:.1f}s")

    # Extract audio from video if available and not disabled
    audio_data   = None
    audio_rate   = 16000
    if not no_audio:
        audio_data, audio_rate = _extract_audio(video_path)

    skip_n       = max(1, round(native_fps / fps))
    effective_fps = native_fps / skip_n
    logger.info(f"FPS: native={native_fps:.1f} → effective={effective_fps:.1f} (skip {skip_n})")

    # ── Set up streaming predictor ─────────────────────────────────────────
    config = StreamConfig(
        inference_interval=interval,
        vllm_url=vllm_url,
        model_name=model_name,
        log_file=str(Path(output_file).parent / 'predictions_raw.jsonl'),
        video_only=no_audio or audio_data is None,
        motion_threshold=1.5,
    )
    predictor = StreamingIntentPredictor(config)
    engine    = FastQwenInferenceEngine(
        vllm_url=vllm_url, model_name=model_name
    )
    predictor.scheduler.set_inference_engine(engine)

    # ── Set up interrupt detection ─────────────────────────────────────────
    interrupt_system = InterruptDetectionSystem(
        use_vap=False,
        consecutive_required=consecutive,
        audio_sample_rate=audio_rate,
    )
    connect_to_predictor(interrupt_system, predictor)

    # Set the initial task
    interrupt_system.task_monitor.set_task(task_command, "approach", task_object)

    # ── Logger ─────────────────────────────────────────────────────────────
    test_logger = InterruptTestLogger(output_file)

    def on_interrupt(event: InterruptEvent):
        elapsed = time.time() - start_time
        test_logger.log_interrupt(event, elapsed)

    interrupt_system.on_interrupt(on_interrupt)

    # ── Sort injected interrupts by timestamp ──────────────────────────────
    inject_queue = sorted(inject_interrupts, key=lambda x: x[0])
    inject_idx   = 0

    # ── Start ──────────────────────────────────────────────────────────────
    predictor.start(num_workers=1)
    start_time     = time.time()
    frame_interval = 1.0 / effective_fps
    next_frame_time = time.time()
    frames_read    = 0
    frames_fed     = 0
    audio_idx      = 0

    print(f"Running... (video duration {duration:.1f}s, Ctrl+C to stop early)\n")

    try:
        while True:
            elapsed = time.time() - start_time

            # Stop when video exceeds duration + 1s grace
            if elapsed > duration + 1.0:
                logger.info(f"Video complete at t={elapsed:.1f}s")
                break

            # ── Inject verbal interrupts at scheduled times ────────────────
            while inject_idx < len(inject_queue):
                t_inject, cmd = inject_queue[inject_idx]
                if elapsed >= t_inject:
                    print(f"\n[t={elapsed:.1f}s] Injecting verbal command: '{cmd}'")
                    interrupt_system.on_verbal_command(cmd)
                    inject_idx += 1
                else:
                    break

            # ── Feed frames ────────────────────────────────────────────────
            if time.time() >= next_frame_time:
                # Read skip_n frames, keep last
                frame = None
                for _ in range(skip_n):
                    ret, bgr = cap.read()
                    if not ret:
                        frame = None
                        break
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames_read += 1

                if frame is None:
                    logger.info(f"End of video ({frames_read} frames read)")
                    break

                predictor.add_frame(frame)

                # Feed audio chunk if available
                if audio_data is not None:
                    audio_duration = skip_n / native_fps
                    n_samples = int(audio_duration * audio_rate)
                    chunk = audio_data[audio_idx:audio_idx + n_samples]
                    if len(chunk) < n_samples:
                        chunk = np.pad(chunk, (0, n_samples - len(chunk)))
                    predictor.add_audio(chunk)
                    # Also feed to fast audio path
                    interrupt_system.on_audio(chunk)
                    audio_idx += n_samples
                else:
                    predictor.add_audio(np.zeros(int(audio_rate / effective_fps),
                                                  dtype=np.float32))

                frames_fed += 1
                next_frame_time += frame_interval

                # Print progress every 5s
                if frames_fed % max(1, int(effective_fps * 5)) == 0:
                    pct = elapsed / duration * 100
                    print(f"  t={elapsed:.1f}s / {duration:.1f}s ({pct:.0f}%)",
                          end='\r', flush=True)

            # ── Drain predictions ──────────────────────────────────────────
            predictions = predictor.get_all_predictions()
            for pred in predictions:
                elapsed_pred = pred.timestamp - start_time
                test_logger.log_prediction(pred, elapsed_pred)
                print(f"  t={elapsed_pred:5.1f}s  "
                      f"{pred.predicted_intent:15s}  "
                      f"conf={pred.confidence:.2f}  "
                      f"obj={pred.target_object[:20]:20s}  "
                      f"{pred.latency_ms:.0f}ms")

            time.sleep(0.001)

        # Wait for final predictions
        logger.info("Waiting for final predictions...")
        time.sleep(3.0)
        predictions = predictor.get_all_predictions()
        for pred in predictions:
            elapsed_pred = pred.timestamp - start_time
            test_logger.log_prediction(pred, elapsed_pred)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        predictor.stop()
        cap.release()

    # Print motion gate stats
    stats = predictor.get_runtime_stats()
    if stats.get('motion_gate_skipped', 0) > 0:
        print(f"\n  Motion gate: skipped {stats['motion_gate_skipped']} calls "
              f"({stats.get('motion_gate_pct',0):.0f}% of ticks)")

    print(f"\n  Interrupt system stats: {interrupt_system.stats()}")
    test_logger.summary()
    _print_analysis(output_file)


def _extract_audio(video_path: str):
    """Extract audio from video using ffmpeg. Returns (array, sample_rate)."""
    import subprocess, tempfile, os
    tmp = tempfile.mktemp(suffix='.wav')
    try:
        result = subprocess.run([
            'ffmpeg', '-i', video_path,
            '-ar', '16000', '-ac', '1', '-f', 'wav', tmp,
            '-y', '-loglevel', 'error'
        ], capture_output=True, timeout=30)
        if result.returncode != 0 or not Path(tmp).exists():
            logger.info("No audio track in video — running video-only")
            return None, 16000
        with wave.open(tmp, 'rb') as wf:
            rate = wf.getframerate()
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            return data.astype(np.float32) / 32768.0, rate
    except Exception as e:
        logger.info(f"Audio extraction failed ({e}) — video-only mode")
        return None, 16000
    finally:
        if Path(tmp).exists():
            os.unlink(tmp)


def _print_analysis(output_file: str):
    """Print a quick analysis of the test results."""
    try:
        entries = [json.loads(l) for l in open(output_file) if l.strip()]
        preds   = [e for e in entries if e['type'] == 'prediction']
        interrupts = [e for e in entries if e['type'] == 'interrupt']

        if not preds:
            return

        print(f"\n{'─'*60}")
        print("RESULTS ANALYSIS")
        print(f"{'─'*60}")

        # Intent distribution
        dist = {}
        for p in preds:
            i = p['predicted_intent']
            dist[i] = dist.get(i, 0) + 1
        print(f"  Intent distribution: {dist}")

        # Latency
        lats = [p['latency_ms'] for p in preds if p.get('latency_ms')]
        if lats:
            print(f"  Avg latency: {sum(lats)/len(lats):.0f}ms  "
                  f"Min: {min(lats):.0f}ms  Max: {max(lats):.0f}ms")

        # Interrupts
        print(f"  Total interrupts fired: {len(interrupts)}")
        for ev in interrupts:
            print(f"    t={ev['elapsed_s']:.1f}s  {ev['reason']:20s}  "
                  f"→ {ev.get('new_task','(stop only)')}")
        print(f"{'─'*60}")
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Test interrupt detection with a video file (no GR00T needed)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic test — system watches for mismatches with active task
  python interrupt_test_runner.py \\
    --video my_test.mp4 \\
    --task "pick up the cup" --task-object cup \\
    --output results/interrupt_test.jsonl

  # Inject a verbal interrupt at 8 seconds
  python interrupt_test_runner.py \\
    --video my_test.mp4 \\
    --task "pick up the cup" --task-object cup \\
    --inject-interrupt 8.0 "stop, pick up the bottle instead" \\
    --output results/interrupt_test.jsonl

  # Multiple injected interrupts
  python interrupt_test_runner.py \\
    --video my_test.mp4 \\
    --task "pick up the red can" --task-object "red can" \\
    --inject-interrupt 5.0 "stop" \\
    --inject-interrupt 7.0 "pick up the blue bottle instead" \\
    --output results/interrupt_test.jsonl

  # Video only (no audio even if present)
  python interrupt_test_runner.py \\
    --video my_test.mp4 --no-audio \\
    --task "pick up the scissors" --task-object scissors \\
    --output results/interrupt_test.jsonl
"""
    )

    parser.add_argument('--video',      required=True, help='Path to test video (.mp4)')
    parser.add_argument('--task',       required=True, help='Active task command e.g. "pick up the cup"')
    parser.add_argument('--task-object',required=True, dest='task_object',
                        help='Target object name e.g. "cup"')
    parser.add_argument('--output',     default='results/interrupt_test.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--vllm-url',   default='http://localhost:8000/v1')
    parser.add_argument('--model-name', default='qwen3-30b-a3b')
    parser.add_argument('--interval',   type=float, default=0.5,
                        help='Inference interval in seconds (default: 0.5)')
    parser.add_argument('--fps',        type=float, default=5.0,
                        help='Target FPS for frame feeding (default: 5)')
    parser.add_argument('--consecutive', type=int, default=3,
                        help='Consecutive mismatches before firing interrupt (default: 3)')
    parser.add_argument('--no-audio',   action='store_true',
                        help='Disable audio — use visual prediction only')
    parser.add_argument('--inject-interrupt', nargs=2, action='append',
                        metavar=('TIMESTAMP', 'COMMAND'),
                        default=[],
                        help='Inject a verbal command at TIMESTAMP seconds. '
                             'Can be used multiple times.')

    args = parser.parse_args()

    # Parse injected interrupts
    injected = []
    for ts_str, cmd in args.inject_interrupt:
        try:
            injected.append((float(ts_str), cmd))
        except ValueError:
            print(f"Warning: invalid timestamp '{ts_str}', skipping")

    run_interrupt_test(
        video_path=args.video,
        task_command=args.task,
        task_object=args.task_object,
        output_file=args.output,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        interval=args.interval,
        fps=args.fps,
        inject_interrupts=injected,
        consecutive=args.consecutive,
        no_audio=args.no_audio,
    )


if __name__ == '__main__':
    main()
