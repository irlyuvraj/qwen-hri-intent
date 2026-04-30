"""
Interrupt Detection & Task Monitoring System
=============================================
Master's Thesis: Human-Robot Interaction with Unitree G1 / GR00T N1.6

HOW THE SYSTEM WORKS
---------------------
This system sits between your intent predictor and GR00T N1.6.

Normal execution:
  Human: "pick up the cup"
  → GR00T starts task: {intent: approach, object: cup}
  → Qwen predicts: approach → cup  ✓ MATCH → do nothing

Interrupt scenario:
  Human: "stop, not that one, the other can"
  → FAST PATH: VAD detects speech onset → AudioInterruptDetector fires
  → SLOW PATH: Qwen predicts change_target / interrupt
  → MismatchDetector: predicted_object (other can) ≠ active_object (first can)
  → InterruptEvent fired → send STOP to GR00T immediately
  → Parse new command → send new task to GR00T

TWO SIGNAL PATHS (why both matter):
  Fast path  < 50ms:  Audio energy / keyword detection
                      Catches "stop/no/wait" before arm moves far
  Slow path  ~323ms:  Qwen visual intent prediction
                      Catches trajectory change, pointing, object mismatch
                      Works even if human is silent (just redirects physically)

VAP NOTE:
  VAP (Voice Activity Projection) would replace the simple audio VAD here.
  VAP runs on CPU, handles noise, predicts speech 200-600ms in advance.
  To use VAP: install VAP-Realtime and set USE_VAP=True in config.
  Without VAP: falls back to energy-based VAD (still works, less robust).

GROOTN1.6 INTERFACE:
  GR00T takes text commands. This system sends:
  - STOP signal: halt current execution immediately
  - New task string: "pick up the [new object]"
  GR00T integration is via the CommandInterface class — swap in your
  actual GR00T API when ready. Currently prints to console for testing.
"""

import time
import threading
import queue
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class InterruptReason(Enum):
    VERBAL_STOP       = "verbal_stop"        # heard "stop / no / wait / other"
    VISUAL_INTERRUPT  = "visual_interrupt"   # Qwen predicted interrupt
    CHANGE_TARGET     = "change_target"      # Qwen predicted change_target
    OBJECT_MISMATCH   = "object_mismatch"    # arm heading to wrong object
    TRAJECTORY_CHANGE = "trajectory_change"  # arm direction reversed


@dataclass
class ActiveTask:
    """The task GR00T is currently executing."""
    command:        str             # raw command e.g. "pick up the cup"
    intent:         str             # e.g. "approach", "gesture"
    target_object:  str             # e.g. "cup", "red can"
    start_time:     float = field(default_factory=time.time)
    completed:      bool  = False

    def matches_prediction(self, predicted_object: str) -> bool:
        """
        Check if a predicted target object matches the active task object.
        Uses fuzzy matching — 'red can', 'can', 'the can' all match 'can'.
        """
        if not predicted_object or predicted_object == 'none':
            return True  # no object predicted — can't mismatch
        a = self.target_object.lower().strip()
        b = predicted_object.lower().strip()
        # Direct match or substring match
        return a in b or b in a or _word_overlap(a, b) >= 0.5


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets."""
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclass
class InterruptEvent:
    """Fired when an interrupt is detected."""
    reason:          InterruptReason
    confidence:      float
    timestamp:       float = field(default_factory=time.time)
    predicted_intent: str  = ""
    predicted_object: str  = ""
    raw_command:      str  = ""      # verbal command if available
    new_task:         str  = ""      # parsed new task if available


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO INTERRUPT DETECTOR  (fast path, <50ms)
# ══════════════════════════════════════════════════════════════════════════════

# Keywords that always trigger immediate stop
STOP_KEYWORDS    = {'stop', 'no', 'wait', 'halt', 'cancel', 'abort'}
REDIRECT_KEYWORDS = {'other', 'different', 'instead', 'not that', 'wrong',
                     'not this', 'another', 'change'}
# Keywords that resume the most recently paused task
CONTINUE_KEYWORDS = {'continue', 'resume', 'go', 'go ahead', 'proceed',
                     'keep going', 'carry on'}

# Single-word transcripts we accept (verbal stop / barge-in / resume tokens)
_ALLOWED_SINGLE_WORDS = STOP_KEYWORDS | CONTINUE_KEYWORDS | {'pause'}


def _is_valid_command_transcript(text: str) -> bool:
    """
    Filter junk transcripts produced by Qwen on noise / partial utterances.
    Drops:
      - Non-ASCII-dominant strings (Qwen sometimes hallucinates Chinese on noise, e.g. '我。')
      - Single-word fragments that aren't a recognized stop/control word (e.g. 'Pick', 'Continue')
    """
    if not text or not text.strip():
        return False
    cleaned = text.strip()

    # ASCII fraction — drop transcripts dominated by non-Latin characters
    ascii_chars = sum(1 for c in cleaned if ord(c) < 128)
    if ascii_chars / max(len(cleaned), 1) < 0.6:
        return False

    # Single-word fragments are usually mid-utterance bleed; only allow control words
    words = [w for w in cleaned.lower().replace('.', ' ').replace(',', ' ').split() if w]
    if len(words) <= 1:
        return bool(words) and words[0] in _ALLOWED_SINGLE_WORDS

    return True

class AudioInterruptDetector:
    """
    Fast audio path for interrupt detection.

    Two modes:
    1. Energy VAD (default): detects speech onset in <20ms, triggers slow
       path (Qwen) to parse the command. Simple but no noise robustness.

    2. VAP mode (recommended): uses VAP-Realtime for noise-robust,
       predictive speech activity detection. Install VAP-Realtime first:
       pip install git+https://github.com/inokoj/VAP-Realtime.git
       Then set use_vap=True.

    Why use VAP here:
    - VAP predicts speech 200-600ms in advance → faster interrupt detection
    - Handles background noise (important for real robot environments)
    - Detects turn-taking cues (human starting to speak mid-task)
    - CPU-only, <50ms latency
    """

    def __init__(
        self,
        sample_rate:        int   = 16000,
        energy_threshold:   float = 0.02,   # RMS energy for VAD trigger
        use_vap:            bool  = False,  # set True if VAP-Realtime installed
        silence_hangover_ms: int  = 150,    # require this much silence before declaring utterance ended
        min_utterance_ms:   int   = 200,    # drop utterances shorter than this (likely noise / single phoneme)
        max_utterance_ms:   int   = 8000,   # force-flush very long utterances (safety)
    ):
        self.sample_rate         = sample_rate
        self.energy_threshold    = energy_threshold
        self.use_vap             = use_vap
        self.silence_hangover_ms = silence_hangover_ms
        self.min_utterance_ms    = min_utterance_ms
        self.max_utterance_ms    = max_utterance_ms
        self._vap_model          = None
        self._speech_active      = False
        self._speech_buffer      = []   # accumulate audio while speech detected
        self._silence_ms         = 0    # consecutive silence accumulated while in active utterance
        self._utterance_ms       = 0    # total length of current utterance
        self._on_speech_end_callbacks: List[Callable] = []

        if use_vap:
            self._init_vap()

    def _init_vap(self):
        """Try to load VAP-Realtime model."""
        try:
            from vap_realtime import VAPRealtime
            self._vap_model = VAPRealtime()
            logger.info("✓ VAP-Realtime loaded — using predictive speech detection")
        except ImportError:
            logger.warning(
                "VAP-Realtime not installed — falling back to energy VAD.\n"
                "To install: pip install git+https://github.com/inokoj/VAP-Realtime.git"
            )
            self.use_vap = False

    def on_speech_end(self, callback: Callable[[np.ndarray], None]):
        """Register callback — called with audio buffer when speech ends."""
        self._on_speech_end_callbacks.append(callback)

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Process one audio chunk. Returns True if speech activity detected.
        Call this every 20ms from your audio capture loop.
        """
        if self.use_vap and self._vap_model:
            return self._process_vap(audio_chunk)
        return self._process_energy_vad(audio_chunk)

    def _process_energy_vad(self, audio_chunk: np.ndarray) -> bool:
        chunk_ms = len(audio_chunk) / self.sample_rate * 1000
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))
        is_speech = rms > self.energy_threshold

        if is_speech:
            if not self._speech_active:
                logger.debug(f"Speech onset (RMS={rms:.3f})")
                self._utterance_ms = 0
            self._speech_active = True
            self._silence_ms = 0
            self._speech_buffer.append(audio_chunk)
            self._utterance_ms += chunk_ms

            # Safety: cap runaway utterances
            if self._utterance_ms >= self.max_utterance_ms:
                self._flush_utterance(reason="max_duration")
                return True

        elif self._speech_active:
            # Inside an utterance but this chunk is silent — keep buffering through
            # short pauses (between syllables / words), only flush after sustained silence.
            self._speech_buffer.append(audio_chunk)
            self._silence_ms += chunk_ms
            self._utterance_ms += chunk_ms
            if self._silence_ms >= self.silence_hangover_ms:
                self._flush_utterance(reason="silence_hangover")

        return is_speech

    def _flush_utterance(self, reason: str = ""):
        """End-of-utterance: hand the accumulated buffer to callbacks, reset state."""
        if not self._speech_buffer:
            self._reset_utterance_state()
            return
        full_audio = np.concatenate(self._speech_buffer)
        duration_ms = len(full_audio) / self.sample_rate * 1000
        self._reset_utterance_state()

        if duration_ms < self.min_utterance_ms:
            logger.debug(f"Dropped short utterance ({duration_ms:.0f}ms < min "
                         f"{self.min_utterance_ms}ms, reason={reason})")
            return

        for cb in self._on_speech_end_callbacks:
            cb(full_audio)

    def _reset_utterance_state(self):
        self._speech_active = False
        self._speech_buffer = []
        self._silence_ms = 0
        self._utterance_ms = 0

    def _process_vap(self, audio_chunk: np.ndarray) -> bool:
        """VAP-based speech detection (higher quality). Same hangover/min-duration rules."""
        try:
            chunk_ms = len(audio_chunk) / self.sample_rate * 1000
            result = self._vap_model.process(audio_chunk)
            is_speech = result.get('p_now', 0) > 0.5

            if is_speech:
                if not self._speech_active:
                    logger.debug(f"VAP speech onset: p_now={result.get('p_now',0):.2f}")
                    self._utterance_ms = 0
                self._speech_active = True
                self._silence_ms = 0
                self._speech_buffer.append(audio_chunk)
                self._utterance_ms += chunk_ms
                if self._utterance_ms >= self.max_utterance_ms:
                    self._flush_utterance(reason="max_duration")
            elif self._speech_active:
                self._speech_buffer.append(audio_chunk)
                self._silence_ms += chunk_ms
                self._utterance_ms += chunk_ms
                if self._silence_ms >= self.silence_hangover_ms:
                    self._flush_utterance(reason="silence_hangover")
            return is_speech
        except Exception as e:
            logger.error(f"VAP error: {e}")
            return self._process_energy_vad(audio_chunk)

    def parse_keyword(self, text: str) -> Optional[str]:
        """
        Check if transcribed text contains stop/redirect/continue keywords.
        Returns 'stop', 'redirect', 'continue', or None.
        Stop wins over continue when both appear ("stop, then continue" → stop).
        """
        text_lower = text.lower()
        if any(kw in text_lower for kw in STOP_KEYWORDS):
            return 'stop'
        if any(kw in text_lower for kw in REDIRECT_KEYWORDS):
            return 'redirect'
        if any(kw in text_lower for kw in CONTINUE_KEYWORDS):
            return 'continue'
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MISMATCH DETECTOR  (slow path — uses Qwen predictions)
# ══════════════════════════════════════════════════════════════════════════════

class MismatchDetector:
    """
    Detects when Qwen's visual predictions diverge from the active task.

    Three mismatch conditions:

    1. Intent mismatch:
       Active task = approach(cup), Qwen predicts = interrupt/change_target
       → Arm stopped or redirected

    2. Object mismatch:
       Active task = approach(cup), Qwen predicts = approach(bottle)
       → Arm heading toward wrong object

    3. Consecutive mismatches (robustness):
       Require N consecutive mismatching predictions before firing.
       Prevents single noisy predictions from triggering false interrupts.
       N=2 is recommended — balances speed vs robustness.
    """

    def __init__(self, consecutive_required: int = 3, grace_period: float = 4.0):
        self.consecutive_required = consecutive_required
        self.grace_period = grace_period          # seconds after new task — skip mismatch checks
        self._lock = threading.Lock()
        self._mismatch_count = 0
        self._last_mismatch: Optional[InterruptEvent] = None
        self._task_start_time: float = 0.0        # reset when a new task is registered

    def check(
        self,
        prediction: Dict[str, Any],
        active_task: Optional[ActiveTask],
    ) -> Optional[InterruptEvent]:
        """
        Check one Qwen prediction against the active task.
        Returns InterruptEvent if interrupt should fire, else None.
        Thread-safe: guards _mismatch_count and _task_start_time.
        """
        with self._lock:
            if active_task is None or active_task.completed:
                self._mismatch_count = 0
                return None

            # Grace period: skip mismatch checks for N seconds after a new task.
            # Prevents false positives while the arm is still orienting.
            if (self.grace_period > 0
                    and (time.time() - self._task_start_time) < self.grace_period):
                logger.debug("Mismatch check skipped — within grace period")
                return None

            intent = prediction.get('predicted_intent', 'unknown')
            obj    = prediction.get('target_object', 'none')
            conf   = prediction.get('confidence', 0.0)

            # Unknown predictions don't trigger mismatches
            if intent == 'unknown' or conf < 0.7:
                self._mismatch_count = 0
                return None

            # ── Check 1: explicit interrupt/change_target from Qwen ──────────
            if intent == 'interrupt':
                event = InterruptEvent(
                    reason=InterruptReason.VISUAL_INTERRUPT,
                    confidence=conf,
                    predicted_intent=intent,
                    predicted_object=obj,
                )
                return self._count_and_fire(event)

            if intent == 'change_target':
                event = InterruptEvent(
                    reason=InterruptReason.CHANGE_TARGET,
                    confidence=conf,
                    predicted_intent=intent,
                    predicted_object=obj,
                    new_task=f"pick up the {obj}" if obj != 'none' else "",
                )
                return self._count_and_fire(event)

            # ── Check 2: object mismatch ──────────────────────────────────────
            if intent in ('approach', 'gesture') and obj != 'none':
                if not active_task.matches_prediction(obj):
                    event = InterruptEvent(
                        reason=InterruptReason.OBJECT_MISMATCH,
                        confidence=conf,
                        predicted_intent=intent,
                        predicted_object=obj,
                        new_task=f"pick up the {obj}",
                    )
                    return self._count_and_fire(event)

            # No mismatch — reset counter
            self._mismatch_count = 0
            self._last_mismatch = None
            return None

    def _count_and_fire(self, event: InterruptEvent) -> Optional[InterruptEvent]:
        """Only fire after N consecutive mismatches."""
        self._mismatch_count += 1
        self._last_mismatch = event
        if self._mismatch_count >= self.consecutive_required:
            self._mismatch_count = 0
            return event
        logger.debug(
            f"Mismatch {self._mismatch_count}/{self.consecutive_required}: "
            f"{event.reason.value}"
        )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TASK MONITOR  (tracks active GR00T task)
# ══════════════════════════════════════════════════════════════════════════════

class TaskMonitor:
    """
    Tracks what GR00T is currently doing.

    You update this whenever GR00T starts a new task.
    The MismatchDetector reads from it to know what to compare against.

    In real deployment: hook this into GR00T's task callback.
    For testing: call set_task() manually when you give a command.
    """

    def __init__(self):
        self._active: Optional[ActiveTask] = None
        self._paused: Optional[ActiveTask] = None
        self._lock = threading.Lock()
        self._history: List[ActiveTask] = []

    def set_task(self, command: str, intent: str = "approach",
                 target_object: str = "unknown"):
        """Call this when GR00T starts a new task."""
        with self._lock:
            if self._active:
                self._history.append(self._active)
            task = ActiveTask(
                command=command,
                intent=intent,
                target_object=target_object,
            )
            self._active = task
            logger.info(f"📋 New task: '{command}' → {intent}({target_object})")
        return task

    def complete_task(self):
        """Call this when GR00T finishes the current task."""
        with self._lock:
            if self._active:
                self._active.completed = True
                logger.info(f"✅ Task completed: '{self._active.command}'")

    def get_active(self) -> Optional[ActiveTask]:
        with self._lock:
            return self._active

    def pause_active(self) -> Optional[ActiveTask]:
        """
        Move the active task to the paused slot. Returns the paused task (or None).
        Called on VERBAL_STOP so a later 'continue' can resume the same task.
        """
        with self._lock:
            if self._active is None or self._active.completed:
                return None
            self._paused = self._active
            self._active = None
            logger.info(f"⏸️  Paused task: '{self._paused.command}'")
            return self._paused

    def take_paused(self) -> Optional[ActiveTask]:
        """Pop the paused task (resume path). Returns it, or None if nothing paused."""
        with self._lock:
            t = self._paused
            self._paused = None
            return t

    def clear_paused(self):
        """Discard any paused task (called when human picks a different task instead)."""
        with self._lock:
            if self._paused is not None:
                logger.info(f"🗑️  Discarded paused task: '{self._paused.command}'")
                self._paused = None

    def parse_command(self, command: str) -> Dict[str, str]:
        """
        Parse a natural language command into intent + target_object.
        Simple rule-based parser — replace with LLM parser for production.

        Examples:
          "pick up the cup"      → {intent: approach, object: cup}
          "grab the red bottle"  → {intent: approach, object: red bottle}
          "place it on the table"→ {intent: gesture,  object: table}
          "stop"                 → {intent: interrupt, object: none}
        """
        cmd = command.lower().strip()

        # Stop commands
        if any(w in cmd for w in ['stop', 'halt', 'cancel', 'abort']):
            return {'intent': 'interrupt', 'object': 'none'}

        # Extract object — everything after pick/grab/get/take/move/place
        obj = 'unknown'
        patterns = [
            r'(?:pick up|grab|get|take|pick)\s+(?:the\s+)?(.+)',
            r'(?:place|put|move|bring)\s+(?:it\s+)?(?:on|to|in|onto)\s+(?:the\s+)?(.+)',
            r'(?:not|other|different|instead)\s+(?:the\s+)?(.+)',
        ]
        for pat in patterns:
            m = re.search(pat, cmd)
            if m:
                obj = m.group(1).strip().rstrip('.,!?')
                break

        # Determine intent from verb
        if any(w in cmd for w in ['place', 'put', 'set down', 'drop']):
            intent = 'gesture'
        elif any(w in cmd for w in ['not', 'other', 'instead', 'different']):
            intent = 'change_target'
        else:
            intent = 'approach'

        return {'intent': intent, 'object': obj}


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND INTERFACE  (sends signals to GR00T)
# ══════════════════════════════════════════════════════════════════════════════

class CommandInterface:
    """
    Interface to GR00T N1.6.

    GR00T takes text commands. This interface:
    1. Sends STOP to halt current execution
    2. Sends new task string for re-planning

    For real deployment: replace the _send_* methods with your GR00T API calls.
    Currently logs to console + calls registered callbacks for testing.

    GR00T API (when available):
      groot.stop()
      groot.execute("pick up the red can")
    """

    def __init__(self, groot_api=None):
        self._groot = groot_api
        self._on_stop_callbacks:     List[Callable] = []
        self._on_new_task_callbacks: List[Callable] = []
        self._stop_count   = 0
        self._redirect_count = 0

    def on_stop(self, callback: Callable):
        """Register callback for stop events."""
        self._on_stop_callbacks.append(callback)

    def on_new_task(self, callback: Callable[[str], None]):
        """Register callback for new task events."""
        self._on_new_task_callbacks.append(callback)

    def send_stop(self, reason: InterruptReason, confidence: float):
        """Send STOP signal to GR00T immediately."""
        self._stop_count += 1
        msg = f"🛑 STOP [{reason.value}] conf={confidence:.2f}"
        logger.warning(msg)
        print(f"\n{'='*60}\n{msg}\n{'='*60}")

        if self._groot:
            try:
                self._groot.stop()
            except Exception as e:
                logger.error(f"GR00T stop failed: {e}")

        for cb in self._on_stop_callbacks:
            cb(reason, confidence)

    def send_new_task(self, task: str):
        """Send new task command to GR00T."""
        if not task:
            return
        self._redirect_count += 1
        msg = f"🔄 NEW TASK: '{task}'"
        logger.info(msg)
        print(f"\n{msg}")

        if self._groot:
            try:
                self._groot.execute(task)
            except Exception as e:
                logger.error(f"GR00T execute failed: {e}")

        for cb in self._on_new_task_callbacks:
            cb(task)

    def stats(self) -> Dict[str, int]:
        return {
            'stops_sent':    self._stop_count,
            'redirects_sent': self._redirect_count,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SYSTEM — ties everything together
# ══════════════════════════════════════════════════════════════════════════════

class InterruptDetectionSystem:
    """
    Main system — integrates audio fast path + Qwen slow path.

    Usage:
        system = InterruptDetectionSystem()

        # When GR00T starts a task:
        system.task_monitor.set_task("pick up the cup", "approach", "cup")

        # Feed Qwen predictions (from StreamingIntentPredictor):
        system.on_prediction(pred_dict)

        # Feed audio chunks (from microphone):
        system.on_audio(audio_chunk)

        # Register callbacks:
        system.on_interrupt(lambda event: print(f"INTERRUPT: {event.reason}"))

    The system handles everything else automatically.
    """

    def __init__(
        self,
        use_vap:              bool  = False,
        consecutive_required: int   = 3,
        grace_period:         float = 4.0,
        audio_sample_rate:    int   = 16000,
        groot_api             = None,
        verbal_stop_dedup_s:  float = 2.0,
    ):
        # Components
        self.task_monitor    = TaskMonitor()
        self.audio_detector  = AudioInterruptDetector(
            sample_rate=audio_sample_rate,
            use_vap=use_vap,
        )
        self.mismatch_detector = MismatchDetector(
            consecutive_required=consecutive_required,
            grace_period=grace_period,
        )
        self.command_interface = CommandInterface(groot_api=groot_api)

        # Optional caller-supplied transcript validator. Returns True to keep,
        # False to drop. Set this to a closure over your TaskRegistry so junk
        # transcripts ("Thank you.", "Shh.") never reach the task monitor.
        self.command_validator: Optional[Callable[[str], bool]] = None

        # Verbal-stop dedup: suppress duplicate VERBAL_STOP within this window.
        # Without this the same "stop" word can fire 2-3 interrupts back-to-back.
        self._verbal_stop_dedup_s = verbal_stop_dedup_s
        self._last_verbal_stop_time = 0.0

        # Patch set_task to reset the mismatch grace period timer on every new task
        # and inject active task object into the inference engine prompt.
        _orig_set_task = self.task_monitor.set_task
        _md = self.mismatch_detector
        _self = self
        def _set_task_with_grace(command, intent="approach", target_object="unknown"):
            with _md._lock:
                _md._task_start_time = time.time()
                _md._mismatch_count = 0
            if _self._inference_engine and hasattr(_self._inference_engine, 'active_task_object'):
                _self._inference_engine.active_task_object = target_object
            return _orig_set_task(command, intent, target_object)
        self.task_monitor.set_task = _set_task_with_grace

        # Optional reference to inference engine — used to inject active task
        # object into Qwen prompts for FP reduction.
        self._inference_engine = None

        # Interrupt callbacks
        self._interrupt_callbacks: List[Callable[[InterruptEvent], None]] = []

        # Wire audio → interrupt
        self.audio_detector.on_speech_end(self._on_speech_captured)

        # Stats
        self._prediction_count = 0
        self._interrupt_count  = 0
        self._start_time       = time.time()

        logger.info(
            f"InterruptDetectionSystem ready "
            f"(VAP={'on' if use_vap else 'off'}, "
            f"consecutive={consecutive_required})"
        )

    def on_interrupt(self, callback: Callable[[InterruptEvent], None]):
        """Register callback — called whenever an interrupt is detected."""
        self._interrupt_callbacks.append(callback)

    # ── Input methods ──────────────────────────────────────────────────────

    def on_prediction(self, prediction: Dict[str, Any]):
        """
        Feed a Qwen prediction into the system.
        Call this from your StreamingIntentPredictor callback.

        prediction should be the dict from qwen_inference_engine._parse_prediction
        containing: predicted_intent, confidence, target_object, reason
        """
        self._prediction_count += 1
        active = self.task_monitor.get_active()

        event = self.mismatch_detector.check(prediction, active)
        if event:
            self._fire_interrupt(event)

    def on_audio(self, audio_chunk: np.ndarray):
        """
        Feed a raw audio chunk into the fast path.
        Call this from your audio capture loop (every 20ms).
        """
        self.audio_detector.process_chunk(audio_chunk)

    def on_verbal_command(self, command: str):
        """
        Feed a transcribed verbal command directly.
        Call this from your ASR pipeline if you have one.
        """
        keyword_type = self.audio_detector.parse_keyword(command)

        if keyword_type == 'stop':
            event = InterruptEvent(
                reason=InterruptReason.VERBAL_STOP,
                confidence=0.95,
                raw_command=command,
            )
            self._fire_interrupt(event)

        elif keyword_type == 'continue':
            # Resume the most recently paused task, if any.
            paused = self.task_monitor.take_paused()
            if paused is None:
                logger.info(f"'{command}' → no paused task to resume; ignoring")
                return
            logger.info(f"▶️  Resuming paused task: '{paused.command}'")
            self.command_interface.send_new_task(paused.command)
            self.task_monitor.set_task(
                command=paused.command,
                intent=paused.intent,
                target_object=paused.target_object,
            )

        elif keyword_type == 'redirect':
            # Human chose a different task — drop any paused task first.
            self.task_monitor.clear_paused()
            parsed = self.task_monitor.parse_command(command)
            new_task = f"pick up the {parsed['object']}" if parsed['object'] != 'unknown' else ""
            event = InterruptEvent(
                reason=InterruptReason.CHANGE_TARGET,
                confidence=0.90,
                raw_command=command,
                new_task=new_task,
                predicted_object=parsed['object'],
            )
            self._fire_interrupt(event)

        else:
            # New task command (not an interrupt) — notify the robot via command_interface.
            # A new pick command supersedes any paused task.
            self.task_monitor.clear_paused()
            parsed = self.task_monitor.parse_command(command)
            self.command_interface.send_new_task(command)
            self.task_monitor.set_task(
                command=command,
                intent=parsed['intent'],
                target_object=parsed['object'],
            )

    # ── Internal ───────────────────────────────────────────────────────────

    def _on_speech_captured(self, audio: np.ndarray):
        """
        Called when a speech segment ends (from audio_detector).
        Pipes audio to Qwen for transcription, then parses the command.
        """
        duration_ms = len(audio) / self.audio_detector.sample_rate * 1000
        logger.info(f"Speech detected ({duration_ms:.0f}ms) — transcribing via Qwen...")

        engine = self._inference_engine
        if engine is None or not hasattr(engine, 'transcribe_audio'):
            logger.warning("No transcription engine available — speech ignored")
            return

        def _transcribe_and_dispatch():
            try:
                text = engine.transcribe_audio(
                    audio, sample_rate=self.audio_detector.sample_rate
                )
                if not text:
                    logger.debug("Transcription empty — no command")
                    return
                if not _is_valid_command_transcript(text):
                    logger.info(f"Dropped junk transcript: '{text}'")
                    return
                # Control keywords (stop / continue / redirect) are not tasks —
                # they must bypass the task-registry validator.
                is_control = self.audio_detector.parse_keyword(text) is not None
                if (not is_control) and self.command_validator and not self.command_validator(text):
                    logger.info(f"Dropped non-actionable transcript: '{text}'")
                    return
                logger.info(f"Transcribed: '{text}'")
                self.on_verbal_command(text)
            except Exception as e:
                logger.error(f"Transcription error: {e}")

        # Run in background thread — audio callback must not block
        threading.Thread(target=_transcribe_and_dispatch, daemon=True).start()

    def _fire_interrupt(self, event: InterruptEvent):
        """Fire an interrupt event — send stop, then new task if available."""
        # Dedup repeated VERBAL_STOPs from the same word ("stop", "stop.").
        if event.reason == InterruptReason.VERBAL_STOP:
            now = time.time()
            since = now - self._last_verbal_stop_time
            if since < self._verbal_stop_dedup_s:
                logger.info(
                    f"Suppressed duplicate verbal stop ({since:.2f}s since last)"
                )
                return
            self._last_verbal_stop_time = now

        self._interrupt_count += 1
        logger.warning(
            f"⚡ INTERRUPT #{self._interrupt_count}: {event.reason.value} "
            f"conf={event.confidence:.2f} obj={event.predicted_object}"
        )

        # 1. Stop immediately
        self.command_interface.send_stop(event.reason, event.confidence)

        # 2a. VERBAL_STOP with no redirect → stash the active task so 'continue' can resume it.
        if event.reason == InterruptReason.VERBAL_STOP and not event.new_task:
            self.task_monitor.pause_active()

        # 2b. Send new task if we know what it is (CHANGE_TARGET / visual redirect).
        if event.new_task:
            # Redirect supersedes any paused task — human picked something different.
            self.task_monitor.clear_paused()
            self.command_interface.send_new_task(event.new_task)
            parsed = self.task_monitor.parse_command(event.new_task)
            self.task_monitor.set_task(
                command=event.new_task,
                intent=parsed['intent'],
                target_object=parsed['object'],
            )

        # 3. Notify registered callbacks
        for cb in self._interrupt_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Interrupt callback error: {e}")

    def stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            'runtime_s':        round(elapsed, 1),
            'predictions_seen': self._prediction_count,
            'interrupts_fired': self._interrupt_count,
            **self.command_interface.stats(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER — connects to StreamingIntentPredictor
# ══════════════════════════════════════════════════════════════════════════════

def connect_to_predictor(
    interrupt_system: InterruptDetectionSystem,
    predictor,         # StreamingIntentPredictor instance
):
    """
    Wire InterruptDetectionSystem to an existing StreamingIntentPredictor.

    After calling this, predictions from the predictor automatically
    flow into the interrupt detection system.

    Usage:
        from streaming_intent_predictor import StreamingIntentPredictor, StreamConfig
        from interrupt_detection_system import InterruptDetectionSystem, connect_to_predictor

        predictor = StreamingIntentPredictor(config)
        interrupt_system = InterruptDetectionSystem()
        connect_to_predictor(interrupt_system, predictor)

        # Register what happens on interrupt:
        interrupt_system.on_interrupt(lambda e: print(f"STOP: {e.reason}"))

        # Set active task:
        interrupt_system.task_monitor.set_task("pick up the cup", "approach", "cup")
    """
    # Wire inference engine reference so set_task() can inject active_task_object
    engine = getattr(predictor, 'inference_engine', None)
    if engine is None and hasattr(predictor, 'scheduler'):
        engine = getattr(predictor.scheduler, 'inference_engine', None)
    if engine is not None:
        interrupt_system._inference_engine = engine

    original_get_all = predictor.get_all_predictions

    def patched_get_all():
        predictions = original_get_all()
        for pred in predictions:
            # Convert PredictionOutput dataclass → dict
            pred_dict = {
                'predicted_intent': pred.predicted_intent,
                'confidence':       pred.confidence,
                'target_object':    pred.target_object,
                'reason':           pred.reason,
                'timestamp':        pred.timestamp,
            }
            interrupt_system.on_prediction(pred_dict)
        return predictions

    predictor.get_all_predictions = patched_get_all
    logger.info("✓ InterruptDetectionSystem connected to StreamingIntentPredictor")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / TEST SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    print("="*65)
    print("INTERRUPT DETECTION SYSTEM — Demo")
    print("="*65)
    print()
    print("This demo simulates your research scenario:")
    print("  Robot is given a task, human interrupts mid-execution.")
    print()

    system = InterruptDetectionSystem(
        use_vap=False,           # set True if VAP-Realtime installed
        consecutive_required=3,  # require 3 consecutive mismatches (was 2)
        grace_period=4.0,        # skip mismatch checks for 4s after new task
    )

    # Log all interrupts
    def on_interrupt(event: InterruptEvent):
        print(f"\n  ✅ Interrupt handled:")
        print(f"     Reason:     {event.reason.value}")
        print(f"     Confidence: {event.confidence:.2f}")
        if event.raw_command:
            print(f"     Command:    '{event.raw_command}'")
        if event.new_task:
            print(f"     New task:   '{event.new_task}'")

    system.on_interrupt(on_interrupt)

    # ── SCENARIO 1: Verbal stop ────────────────────────────────────────────
    print("─"*65)
    print("SCENARIO 1: Verbal stop mid-task")
    print("─"*65)
    print("  GR00T task: 'pick up the cup'")
    system.task_monitor.set_task("pick up the cup", "approach", "cup")
    time.sleep(0.1)

    print("  Qwen predictions: approach(cup), approach(cup) ← match, no interrupt")
    for _ in range(2):
        system.on_prediction({'predicted_intent':'approach','confidence':0.9,'target_object':'cup','reason':'moving toward cup'})
    time.sleep(0.1)

    print("  Human says: 'stop'")
    system.on_verbal_command("stop")
    time.sleep(0.3)

    # ── SCENARIO 2: Object mismatch ────────────────────────────────────────
    print()
    print("─"*65)
    print("SCENARIO 2: Arm heading to wrong object (visual mismatch)")
    print("─"*65)
    print("  GR00T task: 'pick up the red can'")
    system.task_monitor.set_task("pick up the red can", "approach", "red can")
    time.sleep(0.1)

    print("  Qwen: approach(red can) ← match")
    system.on_prediction({'predicted_intent':'approach','confidence':0.9,'target_object':'red can','reason':'moving toward red can'})

    print("  Qwen: approach(blue bottle) ← mismatch #1")
    system.on_prediction({'predicted_intent':'approach','confidence':0.9,'target_object':'blue bottle','reason':'moving toward blue bottle'})

    print("  Qwen: approach(blue bottle) ← mismatch #2 → INTERRUPT")
    system.on_prediction({'predicted_intent':'approach','confidence':0.9,'target_object':'blue bottle','reason':'still moving toward blue bottle'})
    time.sleep(0.3)

    # ── SCENARIO 3: Verbal redirect ────────────────────────────────────────
    print()
    print("─"*65)
    print("SCENARIO 3: Verbal redirect — 'not that one, the other can'")
    print("─"*65)
    print("  GR00T task: 'pick up the can'")
    system.task_monitor.set_task("pick up the can", "approach", "can")
    time.sleep(0.1)

    print("  Human says: 'no, not that one, the other can'")
    system.on_verbal_command("no, not that one, the other can")
    time.sleep(0.3)

    # ── SCENARIO 4: Visual change_target from Qwen ────────────────────────
    print()
    print("─"*65)
    print("SCENARIO 4: Qwen detects change_target visually")
    print("─"*65)
    print("  GR00T task: 'pick up the scissors'")
    system.task_monitor.set_task("pick up the scissors", "approach", "scissors")
    time.sleep(0.1)

    print("  Qwen: approach(scissors) ← match")
    system.on_prediction({'predicted_intent':'approach','confidence':0.9,'target_object':'scissors','reason':'moving toward scissors'})

    print("  Qwen: change_target(screwdriver) ← mismatch #1")
    system.on_prediction({'predicted_intent':'change_target','confidence':0.92,'target_object':'screwdriver','reason':'arm redirecting to screwdriver'})

    print("  Qwen: change_target(screwdriver) ← mismatch #2 → INTERRUPT")
    system.on_prediction({'predicted_intent':'change_target','confidence':0.92,'target_object':'screwdriver','reason':'arm redirecting to screwdriver'})
    time.sleep(0.3)

    print()
    print("─"*65)
    print("FINAL STATS:")
    stats = system.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("─"*65)
