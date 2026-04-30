"""
Streaming Multimodal Intent Prediction System
==============================================
For Master's Thesis: Human-Robot Interaction with Unitree G1

PHASE 2: Real-time streaming inference for near-future intention prediction

This module handles:
- Continuous audio/video streaming
- Temporal buffering with sliding windows
- Incremental inference scheduling
- Low-latency intent prediction
"""

import time
import threading
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import json
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass

@dataclass
class StreamConfig:
    """Configuration for streaming system"""
    # Audio settings
    audio_sample_rate: int = 16000
    audio_buffer_duration: float = 2.0  # seconds of audio to keep
    
    # Video settings
    video_fps: int = 30
    video_buffer_size: int = 90  # keep ~3s of frames for motion context
    
    # Inference settings
    inference_interval: float = 0.2  # run inference every 200ms
    
    # Prediction horizon
    prediction_horizon: float = 2.0  # predict next 2 seconds
    
    # vLLM server settings
    vllm_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "vllm-omni"
    model_name: str = "qwen3-30b-a3b"
    
    # Output settings
    log_predictions: bool = True
    log_file: str = "predictions.jsonl"

    # Optional: save every captured frame to disk for replay/evaluation.
    # Set to a directory path string to enable, None to disable.
    frame_log_dir: Optional[str] = None

    # Video-only mode: skip audio in inference calls
    video_only: bool = False

    # Optical flow motion gating — skip inference when arms are idle.
    # motion_threshold: mean absolute pixel difference between frames
    #   that must be exceeded to trigger inference.
    # 0 = disabled (always infer), recommended starting value: 1.5
    # Tune upward if too sensitive, downward if missing fast motions.
    motion_threshold: float = 1.5


@dataclass
class InferenceInput:
    """Data structure for a single inference call"""
    timestamp: float
    audio_window: np.ndarray  # shape: (n_samples,)
    video_frame: np.ndarray   # shape: (H, W, 3) — latest frame (backwards compat)
    robot_state: Optional[str] = None
    sequence_id: int = 0
    video_frames: Optional[List[np.ndarray]] = None  # multiple frames for motion context


@dataclass
class PredictionOutput:
    """Structured output from Qwen3-Omni"""
    timestamp: float
    sequence_id: int
    predicted_intent: str  # "interrupt" | "continue" | "change_target" | "unknown"
    confidence: float = 0.0
    target_object: str = "none"  # which object is being interacted with
    reason: str = ""
    latency_ms: float = 0.0
    raw_response: str = ""
    task_complete: bool = False  # Qwen visual scene completion signal
    spoken_command: str = ""     # verbatim command Qwen heard in audio (replaces VAD path)


class RingBuffer:
    """Thread-safe ring buffer for audio streaming.

    Uses a pre-allocated numpy array instead of a Python deque-of-floats.
    This avoids the expensive .tolist() / list() round-trips that added
    5-15 ms per audio extraction on 32 000-sample buffers.
    """

    def __init__(self, max_duration: float, sample_rate: int):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self._buf = np.zeros(self.max_samples, dtype=np.float32)
        self._write_pos = 0   # next write index (wraps around)
        self._count = 0       # how many samples currently stored
        self.lock = threading.Lock()

    def append(self, audio_chunk: np.ndarray):
        """Add audio samples to buffer (bulk, zero-copy when possible)."""
        chunk = np.asarray(audio_chunk, dtype=np.float32).ravel()
        n = len(chunk)
        if n == 0:
            return
        with self.lock:
            if n >= self.max_samples:
                # Chunk larger than buffer — keep only the tail
                self._buf[:] = chunk[-self.max_samples:]
                self._write_pos = 0
                self._count = self.max_samples
            else:
                end = self._write_pos + n
                if end <= self.max_samples:
                    self._buf[self._write_pos:end] = chunk
                else:
                    first = self.max_samples - self._write_pos
                    self._buf[self._write_pos:] = chunk[:first]
                    self._buf[:n - first] = chunk[first:]
                self._write_pos = end % self.max_samples
                self._count = min(self._count + n, self.max_samples)

    def get_last_n_seconds(self, duration: float) -> np.ndarray:
        """Extract last N seconds of audio as a contiguous numpy array."""
        with self.lock:
            n_samples = min(int(duration * self.sample_rate), self._count)
            if n_samples == 0:
                return np.zeros(int(duration * self.sample_rate), dtype=np.float32)
            start = (self._write_pos - n_samples) % self.max_samples
            if start + n_samples <= self.max_samples:
                return self._buf[start:start + n_samples].copy()
            else:
                first = self.max_samples - start
                return np.concatenate([
                    self._buf[start:],
                    self._buf[:n_samples - first]
                ])

    def clear(self):
        """Clear buffer"""
        with self.lock:
            self._count = 0
            self._write_pos = 0


class FrameBuffer:
    """Thread-safe buffer for video frames.

    In-memory behaviour is unchanged: keeps the last *max_size* frames,
    oldest evicted automatically.

    Set *log_dir* to a directory path and every frame that arrives will
    ALSO be written to disk as a JPEG (for thesis replay / evaluation).
    """

    def __init__(self, max_size: int = 90, log_dir: Optional[str] = None):
        self.max_size = max_size
        self.frames = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.lock = threading.Lock()

        # --- optional disk logging ---
        self.log_dir = log_dir
        if log_dir:
            import os
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"FrameBuffer: logging frames to {log_dir}")

    def append(self, frame: np.ndarray, timestamp: float):
        """Add frame to buffer (and optionally log to disk)."""
        with self.lock:
            self.frames.append(frame.copy())
            self.timestamps.append(timestamp)

        # disk log is outside the lock — I/O should not block the ring
        if self.log_dir:
            self._save_frame(frame, timestamp)

    def _save_frame(self, frame: np.ndarray, timestamp: float):
        """Write frame as JPEG + companion timestamp file."""
        import os
        from PIL import Image as _Image          # local import; PIL is already a dep

        fname_base = os.path.join(self.log_dir, f"frame_{timestamp:.6f}")
        try:
            _Image.fromarray(frame).save(fname_base + ".jpg", quality=90)
            with open(fname_base + ".txt", "w") as f:
                f.write(f"{timestamp}\n")
        except Exception as e:
            logger.warning(f"FrameBuffer: failed to log frame: {e}")

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get most recent frame."""
        with self.lock:
            if len(self.frames) == 0:
                return None, None
            return self.frames[-1].copy(), self.timestamps[-1]

    def get_recent_frames(self, n: int = 3) -> List[np.ndarray]:
        """Get *n* evenly-spaced frames from the buffer (oldest-first).

        Instead of just returning the last *n* frames (which at 30 fps
        would span only ~100 ms), this picks frames spread across the
        full buffer duration so the model can perceive real motion.

        Example: buffer has 60 frames (2 s), n=3
          -> picks frame 0, 29, 59  (t-2s, t-1s, t-now)
        """
        with self.lock:
            available = list(self.frames)  # oldest first
            total = len(available)
            if total == 0:
                return []
            if total <= n:
                return [f.copy() for f in available]
            # Evenly-spaced indices: first, ..., last
            indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
            return [available[idx].copy() for idx in indices]

    def clear(self):
        """Clear in-memory buffer (logged frames on disk are kept)."""
        with self.lock:
            self.frames.clear()
            self.timestamps.clear()


class StreamingInferenceScheduler:
    """
    Manages periodic inference on buffered audio/video data
    
    Responsibilities:
    - Extract audio windows at fixed intervals
    - Grab latest video frame
    - Schedule async inference calls
    - Handle results asynchronously
    """
    
    def __init__(
        self,
        config: StreamConfig,
        audio_buffer: RingBuffer,
        frame_buffer: FrameBuffer
    ):
        self.config = config
        self.audio_buffer = audio_buffer
        self.frame_buffer = frame_buffer
        
        self.is_running = False
        self.scheduler_thread = None
        self.sequence_counter = 0
        self._current_robot_state = None   # set via set_robot_state()
        
        # Queue sized to num_workers + small buffer; backpressure in
        # the scheduler loop keeps it from ever filling.
        self.inference_queue = queue.Queue(maxsize=16)
        
        # Queue for prediction outputs
        self.prediction_queue = queue.Queue(maxsize=100)
        
        # Worker threads
        self.inference_workers = []

        # Optical flow motion gate — skip inference on idle frames
        self._last_gate_frame: Optional[np.ndarray] = None
        self._motion_skip_count: int = 0
        self._motion_fire_count: int = 0
        
    def start(self, num_workers: int = 2):
        """Start the scheduler and worker threads"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        
        # Start scheduler thread (produces inference inputs)
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self.scheduler_thread.start()
        logger.info("Scheduler thread started")
        
        # Start inference worker threads (consume inference inputs)
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._inference_worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.inference_workers.append(worker)
        logger.info(f"Started {num_workers} inference workers")
    
    def stop(self):
        """Stop all threads"""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2.0)
        
        for worker in self.inference_workers:
            worker.join(timeout=2.0)
        
        logger.info("Scheduler stopped")
    
    def _has_motion(self, current_frame: np.ndarray) -> bool:
        """
        Optical flow motion gate — returns True if arms are moving.

        Computes mean absolute pixel difference between the current frame
        and the last frame that was sent for inference.  If below
        config.motion_threshold, the arms are considered idle and inference
        is skipped entirely.

        Why this works for robot arms:
        - Arms are the ONLY moving objects in the scene
        - When idle, pixel difference ≈ 0 (static camera + static arms)
        - When moving, pixel difference is large and consistent

        Runs in ~0.5ms on a 320x240 frame — negligible overhead.
        Disabled when config.motion_threshold == 0.
        """
        if self.config.motion_threshold == 0:
            return True

        # Downsample to 160x120 for speed — motion detection doesn't need full res
        try:
            import cv2 as _cv2
            small = _cv2.resize(current_frame, (160, 120),
                                interpolation=_cv2.INTER_AREA).astype(float)
        except ImportError:
            # No cv2 — use PIL
            from PIL import Image as _Img
            small = np.array(
                _Img.fromarray(current_frame).resize((160, 120))
            ).astype(float)

        if self._last_gate_frame is None:
            self._last_gate_frame = small
            return True  # first frame — always fire

        diff = np.mean(np.abs(small - self._last_gate_frame))
        motion_detected = diff >= self.config.motion_threshold

        if motion_detected:
            self._last_gate_frame = small
            self._motion_fire_count += 1
        else:
            self._motion_skip_count += 1

        return motion_detected

    def _scheduler_loop(self):
        """
        Main loop that creates inference inputs at fixed intervals.

        Backpressure logic: if the queue already has as many items as there
        are workers, we skip this tick entirely and reset the timer so we
        don't try to "catch up" later.  This is the single change that
        eliminates the "queue full / dropping" flood.

        Optical flow gate: if motion_threshold > 0 and no motion is detected,
        the inference call is skipped — saving GPU time and reducing FPs on
        idle frames.
        """
        next_inference_time = time.time()

        while self.is_running:
            current_time = time.time()

            effective_interval = self.config.inference_interval

            if current_time >= next_inference_time:
                # --- backpressure: skip if workers are already saturated ---
                if self.inference_queue.qsize() >= len(self.inference_workers):
                    # Don't increment counter, don't enqueue.
                    # Reset timer to NOW so we don't pile up catch-up ticks.
                    next_inference_time = current_time + effective_interval
                    time.sleep(0.01)
                    continue

                # Extract data from buffers
                audio_window = self.audio_buffer.get_last_n_seconds(
                    self.config.audio_buffer_duration
                )
                video_frame, frame_timestamp = self.frame_buffer.get_latest()
                # Grab up to 3 recent frames for motion context
                recent_frames = self.frame_buffer.get_recent_frames(3)

                if video_frame is not None:
                    # Optical flow gate — skip if arms are idle
                    if not self._has_motion(video_frame):
                        next_inference_time = current_time + effective_interval
                        time.sleep(0.01)
                        continue

                    inference_input = InferenceInput(
                        timestamp=current_time,
                        audio_window=audio_window,
                        video_frame=video_frame,
                        robot_state=self._current_robot_state,
                        sequence_id=self.sequence_counter,
                        video_frames=recent_frames if len(recent_frames) > 1 else None,
                    )

                    try:
                        self.inference_queue.put_nowait(inference_input)
                        self.sequence_counter += 1
                    except queue.Full:
                        pass

                    # Only advance the timer when we actually enqueued.
                    next_inference_time = current_time + effective_interval
                # else: video not ready yet — don't advance timer, retry in 10 ms

            time.sleep(0.01)
    
    def _inference_worker(self, worker_id: int):
        """Worker thread that processes inference inputs"""
        logger.info(f"Inference worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get input from queue (with timeout)
                inference_input = self.inference_queue.get(timeout=0.5)
                
                # Run inference (this is where Qwen is called)
                start_time = time.time()
                prediction = self._run_inference(inference_input)
                latency_ms = (time.time() - start_time) * 1000
                
                prediction.latency_ms = latency_ms
                
                # Add to output queue
                self.prediction_queue.put(prediction)
                
                logger.debug(
                    f"Worker {worker_id}: seq={prediction.sequence_id}, "
                    f"intent={prediction.predicted_intent}, "
                    f"latency={latency_ms:.1f}ms"
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def set_inference_engine(self, engine):
        """Set the Qwen inference engine"""
        self.inference_engine = engine
        logger.info("Inference engine set")
    
    def _run_inference(self, input_data: InferenceInput) -> PredictionOutput:
        """
        Run Qwen3-Omni inference
        
        This is where you'll call the vLLM server with:
        - Audio data
        - Video frame
        - Robot state text
        """
        if not hasattr(self, 'inference_engine') or self.inference_engine is None:
            # Fallback to dummy prediction if no engine set
            logger.warning("No inference engine set, using dummy prediction")
            return PredictionOutput(
                timestamp=input_data.timestamp,
                sequence_id=input_data.sequence_id,
                predicted_intent="unknown",
                confidence=0.0,
                reason="No inference engine available",
                raw_response=""
            )
        
        try:
            # Use video-only inference when configured or audio is silent
            use_video_only = getattr(self.config, 'video_only', False)
            if not use_video_only:
                # Auto-detect: if audio is all zeros / near-silent, skip it
                audio_energy = np.abs(input_data.audio_window).max()
                if audio_energy < 1e-6:
                    use_video_only = True
                # During execution, audio is high-pass filtered (200 Hz cutoff)
                # before being added to the buffer — motor-noise rumble is
                # suppressed while the speech band (300Hz+) is preserved.
                # Use the full multimodal path so Qwen can detect stop/switch
                # commands via predicted_intent=interrupt/change_target.
                # Video-only is only forced when audio energy is essentially
                # zero (silence gate above) or audio is absent entirely.

            # Prefer multi-frame methods when we have multiple frames
            frames = input_data.video_frames  # List or None

            if use_video_only:
                if frames and hasattr(self.inference_engine, 'predict_intent_video_only_multi_frame'):
                    result = self.inference_engine.predict_intent_video_only_multi_frame(
                        video_frames=frames,
                        robot_state=input_data.robot_state or "",
                    )
                elif hasattr(self.inference_engine, 'predict_intent_video_only'):
                    result = self.inference_engine.predict_intent_video_only(
                        video_frame=input_data.video_frame,
                        robot_state=input_data.robot_state or "",
                    )
                else:
                    result = self.inference_engine.predict_intent(
                        audio_window=input_data.audio_window,
                        video_frame=input_data.video_frame,
                        robot_state=input_data.robot_state,
                        sample_rate=self.config.audio_sample_rate
                    )
            else:
                if frames and hasattr(self.inference_engine, 'predict_intent_multi_frame'):
                    result = self.inference_engine.predict_intent_multi_frame(
                        audio_window=input_data.audio_window,
                        video_frames=frames,
                        robot_state=input_data.robot_state,
                        sample_rate=self.config.audio_sample_rate
                    )
                else:
                    result = self.inference_engine.predict_intent(
                        audio_window=input_data.audio_window,
                        video_frame=input_data.video_frame,
                        robot_state=input_data.robot_state,
                        sample_rate=self.config.audio_sample_rate
                    )
            
            # Convert to PredictionOutput
            return PredictionOutput(
                timestamp=input_data.timestamp,
                sequence_id=input_data.sequence_id,
                predicted_intent=result.get('predicted_intent', 'unknown'),
                confidence=result.get('confidence', 0.0),
                target_object=result.get('target_object', 'none'),
                reason=result.get('reason', ''),
                raw_response=result.get('raw_response', ''),
                task_complete=bool(result.get('task_complete', False)),
                spoken_command=str(result.get('spoken_command', '') or '').strip(),
            )
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return PredictionOutput(
                timestamp=input_data.timestamp,
                sequence_id=input_data.sequence_id,
                predicted_intent="unknown",
                confidence=0.0,
                reason=f"Error: {str(e)}",
                raw_response=""
            )
    
    def get_latest_prediction(self) -> Optional[PredictionOutput]:
        """Get most recent prediction (non-blocking)"""
        try:
            return self.prediction_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_predictions(self) -> List[PredictionOutput]:
        """Get all pending predictions"""
        predictions = []
        while True:
            try:
                predictions.append(self.prediction_queue.get_nowait())
            except queue.Empty:
                break
        return predictions


class PredictionLogger:
    """Logs predictions with timestamps for evaluation"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.lock = threading.Lock()
    
    def log_prediction(self, prediction: PredictionOutput):
        """Append prediction to JSONL file"""
        with self.lock:
            with open(self.log_file, 'a') as f:
                log_entry = {
                    'timestamp': prediction.timestamp,
                    'sequence_id': prediction.sequence_id,
                    'predicted_intent': prediction.predicted_intent,
                    'confidence': prediction.confidence,
                    'target_object': prediction.target_object,
                    'reason': prediction.reason,
                    'latency_ms': prediction.latency_ms,
                    'datetime': datetime.fromtimestamp(prediction.timestamp).isoformat()
                }
                f.write(json.dumps(log_entry) + '\n')


class StreamingIntentPredictor:
    """
    Main interface for the streaming intent prediction system
    
    Usage:
        predictor = StreamingIntentPredictor(config)
        predictor.start()
        
        # Feed audio
        predictor.add_audio(audio_chunk)
        
        # Feed video
        predictor.add_frame(frame)
        
        # Get predictions
        prediction = predictor.get_latest_prediction()
        
        predictor.stop()
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        
        # Buffers
        self.audio_buffer = RingBuffer(
            max_duration=self.config.audio_buffer_duration,
            sample_rate=self.config.audio_sample_rate
        )
        self.frame_buffer = FrameBuffer(
            max_size=self.config.video_buffer_size,
            log_dir=self.config.frame_log_dir
        )
        
        # Scheduler
        self.scheduler = StreamingInferenceScheduler(
            config=self.config,
            audio_buffer=self.audio_buffer,
            frame_buffer=self.frame_buffer
        )
        
        # Logger
        if self.config.log_predictions:
            self.logger = PredictionLogger(self.config.log_file)
        else:
            self.logger = None
        
        self.start_time = None
        self.is_running = False
    
    def start(self, num_workers: int = 2):
        """Start the system"""
        self.start_time = time.time()
        self.scheduler.start(num_workers=num_workers)
        self.is_running = True
        logger.info("StreamingIntentPredictor started")
    
    def stop(self):
        """Stop the system"""
        self.scheduler.stop()
        self.is_running = False
        logger.info("StreamingIntentPredictor stopped")
    
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio samples to buffer (call this continuously from mic)"""
        if not self.is_running:
            logger.warning("System not running, audio dropped")
            return
        self.audio_buffer.append(audio_chunk)
    
    def add_frame(self, frame: np.ndarray):
        """Add video frame to buffer (call this from camera)"""
        if not self.is_running:
            logger.warning("System not running, frame dropped")
            return
        timestamp = time.time()
        self.frame_buffer.append(frame, timestamp)
    
    def set_robot_state(self, state: str):
        """Update robot state description — passed into every inference call"""
        self.scheduler._current_robot_state = state
    
    def get_latest_prediction(self) -> Optional[PredictionOutput]:
        """Get most recent prediction"""
        prediction = self.scheduler.get_latest_prediction()
        if prediction and self.logger:
            self.logger.log_prediction(prediction)
        return prediction
    
    def get_all_predictions(self) -> List[PredictionOutput]:
        """Get all pending predictions"""
        predictions = self.scheduler.get_all_predictions()
        if self.logger:
            for pred in predictions:
                self.logger.log_prediction(pred)
        return predictions
    
    def get_runtime_stats(self) -> Dict[str, Any]:
        """Get runtime statistics (safe to call after stop)"""
        if not self.start_time:
            return {}

        fired  = self.scheduler._motion_fire_count
        skipped = self.scheduler._motion_skip_count
        total   = fired + skipped
        return {
            'runtime_seconds':   time.time() - self.start_time,
            'total_predictions': self.scheduler.sequence_counter,
            'pending_inferences': self.scheduler.inference_queue.qsize(),
            'pending_predictions': self.scheduler.prediction_queue.qsize(),
            'motion_gate_fired':   fired,
            'motion_gate_skipped': skipped,
            'motion_gate_pct':     round(skipped / total * 100, 1) if total else 0,
        }


if __name__ == "__main__":
    # Example usage
    config = StreamConfig(
        inference_interval=0.2,
        audio_buffer_duration=2.0,
        log_predictions=True
    )
    
    predictor = StreamingIntentPredictor(config)
    predictor.start(num_workers=2)
    
    try:
        # Simulate streaming for 10 seconds
        print("Simulating streaming for 10 seconds...")
        for i in range(100):
            # Simulate audio chunk (20ms of audio at 16kHz)
            audio_chunk = np.random.randn(320).astype(np.float32)
            predictor.add_audio(audio_chunk)
            
            # Simulate video frame (every 33ms for ~30fps)
            if i % 3 == 0:
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                predictor.add_frame(frame)
            
            # Check for predictions
            prediction = predictor.get_latest_prediction()
            if prediction:
                print(f"Prediction {prediction.sequence_id}: {prediction.predicted_intent}")
            
            time.sleep(0.01)
        
        # Print stats
        stats = predictor.get_runtime_stats()
        print("\nRuntime stats:", stats)
        
    finally:
        predictor.stop()
