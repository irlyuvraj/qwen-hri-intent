#!/usr/bin/env python3
"""
File-based Intent Prediction  —  with Simulated-Streaming mode
==============================================================

Reads pre-recorded frames (directory or .mp4) and optionally audio,
then feeds them through the full streaming pipeline at **real-time pace**
so inference triggers just as it would on a live stream.

Usage examples:

    # Sparse mode (inference every 2 s — legacy behaviour):
    python file_based_predictor.py \
        --frames-dir test_data/cleaning/frames/ --no-audio \
        --output results/cleaning_sparse.jsonl

    # Simulated-streaming from .mp4 (inference every 0.5 s):
    python file_based_predictor.py \
        --video data/cleaning_h264.mp4 --no-audio --interval 0.5 \
        --output results/cleaning_stream.jsonl

    # With audio:
    python file_based_predictor.py \
        --frames-dir test_data/pointing/frames/ \
        --audio-file test_data/pointing/audio.wav \
        --interval 1.0 --output predictions.jsonl
"""

import argparse
import time
import wave
import numpy as np
from pathlib import Path
from PIL import Image
import logging

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from streaming_intent_predictor import StreamingIntentPredictor, StreamConfig
from qwen_inference_engine import FastQwenInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileBasedCapture:
    """Simulates capture from pre-recorded files.

    Supports two video sources:
     1. **frames-dir** – sorted .jpg/.png frames on disk (legacy)
     2. **video file** – an .mp4 (or any cv2-readable) file

    If *audio_file* is None the capture runs in video-only mode:
    get_audio_chunk() returns silence so the rest of the pipeline
    still works without special-casing.
    """

    def __init__(self, frames_dir=None, video_file=None, audio_file=None, fps=30):
        if frames_dir is None and video_file is None:
            raise ValueError("Either frames_dir or video_file must be provided")

        self.fps = fps
        self.has_audio = audio_file is not None
        self._use_cv2 = video_file is not None

        # ---------- video source ----------
        if self._use_cv2:
            if not _HAS_CV2:
                raise RuntimeError("OpenCV (cv2) is required for --video mode")
            self._cap = cv2.VideoCapture(video_file)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_file}")
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or fps
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_files = None  # not used
            logger.info(
                f"Opened video {video_file}: {self.total_frames} frames, "
                f"{self.fps:.1f} fps, "
                f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )
        else:
            self._cap = None
            self.frames_dir = Path(frames_dir)
            self.frame_files = sorted(self.frames_dir.glob("*.jpg"))
            if not self.frame_files:
                self.frame_files = sorted(self.frames_dir.glob("*.png"))
            self.total_frames = len(self.frame_files)
            logger.info(f"Loaded {self.total_frames} frames from {frames_dir}")

        # ---------- audio source ----------
        if self.has_audio:
            audio_path = Path(audio_file)
            with wave.open(str(audio_path), 'rb') as wf:
                self.audio_rate = wf.getframerate()
                self.audio_channels = wf.getnchannels()
                audio_bytes = wf.readframes(wf.getnframes())
                audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
                self.audio_data = audio_int.astype(np.float32) / 32768.0
            logger.info(f"Loaded audio: {len(self.audio_data)} samples @ {self.audio_rate}Hz")
        else:
            self.audio_rate = 16000
            self.audio_data = None
            logger.info("No audio file -- running in VIDEO-ONLY mode")

        self.frame_index = 0
        self.audio_index = 0

    # ---- video ----
    def get_next_frame(self):
        """Get next video frame as RGB numpy array (None when exhausted)."""
        if self._use_cv2:
            ret, bgr = self._cap.read()
            if not ret:
                return None
            self.frame_index += 1
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            if self.frame_index >= len(self.frame_files):
                return None
            frame_path = self.frame_files[self.frame_index]
            img = Image.open(frame_path)
            frame = np.array(img)
            self.frame_index += 1
            return frame

    # ---- audio ----
    def get_audio_chunk(self, duration=0.033):
        """Get audio chunk. Returns silence when no audio file was loaded."""
        samples_needed = int(self.audio_rate * duration)

        if self.audio_data is None:
            return np.zeros(samples_needed, dtype=np.float32)

        if self.audio_index + samples_needed > len(self.audio_data):
            remaining = len(self.audio_data) - self.audio_index
            chunk = np.concatenate([
                self.audio_data[self.audio_index:],
                np.zeros(samples_needed - remaining, dtype=np.float32)
            ])
        else:
            chunk = self.audio_data[self.audio_index:self.audio_index + samples_needed]

        self.audio_index += samples_needed
        return chunk

    def release(self):
        """Release cv2 capture if open."""
        if self._cap is not None:
            self._cap.release()


def run_file_based_prediction(
    frames_dir, audio_file, output_file, vllm_url, model_name,
    video_only=False, scene_objects=None, video_file=None,
    inference_interval=2.0,
):
    """Run prediction on pre-recorded files (real-time pace).

    Parameters
    ----------
    video_file : str or None
        Path to an .mp4 file.  If provided, *frames_dir* is ignored.
    inference_interval : float
        How often (seconds) the scheduler triggers inference.
        Lower → more dense predictions (0.5 s → ~70 predictions for a 35 s clip).
    """

    effective_audio = None if video_only else audio_file
    capture = FileBasedCapture(
        frames_dir=frames_dir if video_file is None else None,
        video_file=video_file,
        audio_file=effective_audio,
    )

    config = StreamConfig(
        inference_interval=inference_interval,
        vllm_url=vllm_url,
        model_name=model_name,
        log_file=output_file,
        video_only=video_only or (not capture.has_audio),
    )

    predictor = StreamingIntentPredictor(config)

    engine = FastQwenInferenceEngine(
        vllm_url=vllm_url,
        model_name=model_name,
        scene_objects=scene_objects,
    )

    predictor.scheduler.set_inference_engine(engine)
    predictor.start(num_workers=1)

    mode_str = "VIDEO-ONLY" if config.video_only else "AUDIO+VIDEO"
    src_str = video_file or frames_dir
    logger.info(
        f"Processing {src_str} ({mode_str}), "
        f"inference every {inference_interval:.2f}s ..."
    )
    start_time = time.time()

    frame_interval = 1.0 / capture.fps
    next_frame_time = time.time()
    total_frames_fed = 0

    try:
        while True:
            current_time = time.time()

            if current_time >= next_frame_time:
                frame = capture.get_next_frame()
                if frame is None:
                    logger.info(
                        f"Reached end of video ({total_frames_fed} frames fed)"
                    )
                    break

                audio_chunk = capture.get_audio_chunk(duration=frame_interval)

                predictor.add_frame(frame)
                predictor.add_audio(audio_chunk)
                total_frames_fed += 1

                next_frame_time += frame_interval

            # Drain and display predictions as they arrive
            predictions = predictor.get_all_predictions()
            for pred in predictions:
                elapsed = pred.timestamp - start_time
                logger.info(
                    f"[t={elapsed:6.2f}s seq={pred.sequence_id:4d}] "
                    f"{pred.predicted_intent:15s} "
                    f"(conf={pred.confidence:.2f}, obj={pred.target_object}) "
                    f"- {pred.reason[:60]}"
                )

            time.sleep(0.001)

        # Wait for in-flight inference to finish
        logger.info("Waiting for final predictions...")
        time.sleep(4.0)

        predictions = predictor.get_all_predictions()
        for pred in predictions:
            elapsed = pred.timestamp - start_time
            logger.info(
                f"[t={elapsed:6.2f}s seq={pred.sequence_id:4d}] "
                f"{pred.predicted_intent:15s} "
                f"(conf={pred.confidence:.2f}, obj={pred.target_object}) "
                f"- {pred.reason[:60]}"
            )
    finally:
        predictor.stop()
        capture.release()

    runtime = time.time() - start_time
    stats = predictor.get_runtime_stats()
    logger.info(
        f"\nDone — {runtime:.1f}s wall-clock, "
        f"{stats.get('total_predictions', '?')} predictions, "
        f"interval={inference_interval:.2f}s"
    )
    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="File-based Intent Prediction (real-time streaming simulation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # From frames directory (legacy 2 s interval)
  %(prog)s --frames-dir test_data/cleaning/frames/ --no-audio

  # From .mp4, dense streaming (0.5 s interval)
  %(prog)s --video data/cleaning_h264.mp4 --no-audio --interval 0.5

  # Custom interval + scene objects
  %(prog)s --video data/cleaning_h264.mp4 --no-audio --interval 1.0 \\
           --scene-objects green_plate tissue_paper
""",
    )

    # --- video source (mutually exclusive) ---
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--frames-dir', help='Directory with JPEG/PNG frame images')
    src.add_argument('--video', dest='video_file',
                     help='Path to .mp4 video file (H.264 recommended)')

    # --- audio ---
    parser.add_argument('--audio-file', default=None,
                        help='Audio WAV file (omit for video-only)')
    parser.add_argument('--no-audio', action='store_true',
                        help='Run in video-only mode (no audio)')

    # --- inference timing ---
    parser.add_argument('--interval', type=float, default=None,
                        help='Inference interval in seconds '
                             '(default: 0.5 for --video, 2.0 for --frames-dir)')

    # --- output / server ---
    parser.add_argument('--output', default='predictions.jsonl', help='Output JSONL file')
    parser.add_argument('--vllm-url', default='http://192.168.2.25:8000/v1')
    parser.add_argument('--model-name', default='qwen3-30b-a3b')
    parser.add_argument('--scene-objects', nargs='+', default=None,
                        help='List of object names visible in the scene '
                             '(e.g. green_plate sunglasses tissue_paper)')

    args = parser.parse_args()

    video_only = args.no_audio or (args.audio_file is None)

    if not video_only and args.audio_file and not Path(args.audio_file).exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        logger.info("Hint: use --no-audio for video-only mode")
        return

    # Default interval depends on source type
    if args.interval is not None:
        interval = args.interval
    else:
        interval = 0.5 if args.video_file else 2.0

    run_file_based_prediction(
        frames_dir=args.frames_dir,
        audio_file=args.audio_file,
        output_file=args.output,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        video_only=video_only,
        scene_objects=args.scene_objects,
        video_file=args.video_file,
        inference_interval=interval,
    )


if __name__ == "__main__":
    main()
