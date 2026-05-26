"""
Qwen3-Omni Inference Engine for Streaming Intent Prediction
============================================================

This module handles:
- Converting audio/video data to Qwen3-Omni input format
- Making API calls to vLLM server
- Parsing structured outputs from Qwen
- Optimized for low-latency streaming inference
"""

import base64
import io
import json
import re
import sys
import tempfile
import os
import struct
import wave
from typing import Optional, Dict, Any, Tuple
import numpy as np
import soundfile as sf
from PIL import Image
import httpx
from openai import OpenAI
import logging

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class Qwen3OmniInferenceEngine:
    """
    Handles Qwen3-Omni inference for streaming intent prediction
    
    Key optimizations:
    - Minimal audio/video preprocessing
    - Direct base64 encoding without file I/O
    - Structured output parsing
    - Async-ready design
    """
    
    def __init__(
        self,
        vllm_url: str = "http://192.168.2.25:8000/v1",
        api_key: str = "vllm-omni",
        model_name: str = "qwen3-30b-a3b",
        temperature: float = 0.3,
        top_p: float = 0.8,
        max_tokens: int = 160,  # raised May 19 — see FastQwenInferenceEngine
        system_prompt: Optional[str] = None,
        scene_objects: Optional[list] = None,
    ):
        self.vllm_url = vllm_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Optional list of known object names in the scene.
        # When provided, the model uses these names consistently.
        self.scene_objects = scene_objects or []

        # Intent history for temporal context in prompts
        self.intent_history: list = []
        self.max_history: int = 3

        # Active task context — set by the interrupt system so Qwen
        # can deprioritise non-target objects in its predictions.
        self.active_task_object: Optional[str] = None

        # Full task instruction — injected so Qwen can signal task_complete
        # when the scene shows the task is already finished.
        self.active_task_lang: Optional[str] = None

        # Initialize client with connection pooling for low-latency
        try:
            # Tight read timeout (was 30s). A normal inference returns in
            # <1s (baseline p95 ~600ms); anything past a few seconds is a
            # server-side stall (vLLM/Qwen3-Omni occasionally stops responding
            # mid-stream, esp. on idle/silent input). With 1 worker + queue
            # backpressure, one stuck call freezes the WHOLE prediction pipeline
            # until this timeout fires — so 30s meant a 30s "hang". 8s read
            # timeout makes a stall raise quickly; the worker then backs off 2s
            # and resumes listening, so the system recovers instead of hanging.
            http_client = httpx.Client(
                timeout=httpx.Timeout(8.0, connect=5.0),
                limits=httpx.Limits(
                    max_keepalive_connections=4,  # reuse TCP connections
                    max_connections=8,            # allow parallel workers
                    keepalive_expiry=120,         # keep alive for 2 min
                ),
                follow_redirects=True,
                http2=False,                      # HTTP/1.1 is faster for single requests
            )
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=vllm_url,
                http_client=http_client,
                max_retries=0
            )
            logger.info(f"✅ Initialized Qwen3-Omni engine: {vllm_url}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            self.client = None
        
        # System prompt — accept override or build default
        self.system_prompt = system_prompt or self._build_system_prompt()

        # vLLM guided JSON schema. Grammar-constrains generation so the model
        # CANNOT emit prose, thinking text, or chat-template fragments — only
        # valid JSON conforming to this schema. Fixes the failure mode where
        # Qwen3-30B-Omni dumps reasoning ("Additionally, the robot arm...")
        # instead of structured output during execution.
        self._guided_json_schema = {
            "type": "object",
            "properties": {
                "predicted_intent": {"type": "string"},
                "predicted_phase":  {"type": "string"},
                "confidence":       {"type": "number", "minimum": 0, "maximum": 1},
                "target_object":    {"type": "string"},
                "task_complete":    {"type": "boolean"},
                "reason":           {"type": "string", "maxLength": 200},
                "spoken_command":   {"type": "string"},
            },
            "required": ["predicted_intent", "confidence", "target_object", "reason"],
            "additionalProperties": False,
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for intent prediction task.

        Designed to be scene-agnostic: works for robot arms, human hands,
        simulation, or real hardware.  The model auto-detects actors and
        objects — no hardcoded scene vocabulary.
        """
        return """
You are a multimodal intent prediction system for human-robot interaction.

YOUR TASK: Predict the NEXT 2-second intention of the agent (human or robot) visible in the scene.

You may receive:
- Video frames (single or multi-frame composite)
- Audio (speech / ambient sound) — may be absent
- Optional context about the scene

CRITICAL RULES:
1. PREDICT the future, do NOT describe the present.
2. Identify the specific object involved.
3. If nothing is changing between frames → predict "continue".

─── INTENT CLASSES ───

"approach"
  The agent's hand/arm is MOVING TOWARD an object but has NOT yet touched it.
  Key evidence: closing distance between hand and object across frames.

"gesture"
  The agent is ACTIVELY MANIPULATING an object: grasping, holding, lifting,
  placing, pushing, rotating, or depositing it.
  Key evidence: hand is IN CONTACT with or very close to an object AND the
  object is being moved, held, or released. This is the phase BETWEEN
  approach and withdraw.

"withdraw"
  The agent's hand/arm is MOVING AWAY from the workspace or an object
  after completing an action.
  Key evidence: increasing distance between hand and last-touched object.

"point"
  The agent is pointing at an object with an extended finger or tool.
  Key evidence: static pose with finger/tool aimed at a specific object.

"continue"
  No significant motion change. The agent is idle, waiting, or holding
  a steady position.
  Key evidence: hand positions are NEARLY IDENTICAL across frames.
  IMPORTANT: Use "continue" when frames show little to no movement —
  do NOT force an action prediction when nothing is happening.

"change_target"
  The agent's trajectory is redirecting toward a different object.
  Key evidence: hand was heading toward object A, now curves toward B.

"interrupt"
  The current action is stopping abruptly.
  Key evidence: sudden deceleration, verbal "stop", freezing mid-motion.

"unknown"
  Insufficient or contradictory evidence.

─── HOW TO DECIDE ───

Step 1: Compare frame panels (oldest → newest). Measure position change.
Step 2: If positions barely changed → "continue".
Step 3: If hand is moving TOWARD an object → is it already touching/holding?
        • Not touching → "approach"
        • Touching / holding / placing → "gesture"
Step 4: If hand is moving AWAY from an object after interaction → "withdraw".
Step 5: If audio contains a command, factor it in.

─── OBJECT IDENTIFICATION ───

Describe targets by visible properties (color, shape, type).
Examples: "red cube", "green plate", "tissue paper", "left arm".
If no specific object is targeted, use "none".

─── OUTPUT FORMAT (JSON only) ───
{
  "predicted_intent": "<class>",
  "confidence": 0.0-1.0,
  "target_object": "<object description or none>",
  "reason": "<brief motion-trend explanation, max 20 words>"
}
"""
    
    def encode_audio_to_base64(
        self,
        audio_window: np.ndarray,
        sample_rate: int = 16000,
        max_duration: Optional[float] = None,
    ) -> str:
        """
        Encode audio waveform to base64 WAV format.

        If *max_duration* is given, only the last *max_duration* seconds
        of the window are encoded — smaller payload = faster transfer
        and fewer audio tokens for the model to process.

        Uses raw struct packing (4x faster than soundfile for small buffers).
        """
        try:
            if max_duration is not None:
                max_samples = int(max_duration * sample_rate)
                if len(audio_window) > max_samples:
                    audio_window = audio_window[-max_samples:]

            # Ensure int16
            if audio_window.dtype != np.int16:
                audio_window = (audio_window * 32767).astype(np.int16)

            # Build WAV in-memory with struct (avoids soundfile overhead)
            pcm_bytes = audio_window.tobytes()
            num_channels = 1
            sample_width = 2  # int16
            byte_rate = sample_rate * num_channels * sample_width
            block_align = num_channels * sample_width
            data_size = len(pcm_bytes)

            buf = io.BytesIO()
            # RIFF header
            buf.write(b'RIFF')
            buf.write(struct.pack('<I', 36 + data_size))
            buf.write(b'WAVE')
            # fmt chunk
            buf.write(b'fmt ')
            buf.write(struct.pack('<IHHIIHH',
                16, 1, num_channels, sample_rate,
                byte_rate, block_align, 16))
            # data chunk
            buf.write(b'data')
            buf.write(struct.pack('<I', data_size))
            buf.write(pcm_bytes)

            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Audio encoding failed: {e}")
            return ""
    
    def encode_image_to_base64(
        self,
        video_frame: np.ndarray,
        format: str = 'JPEG',
        quality: int = 85,
        max_width: int = 320,
        max_height: int = 240
    ) -> str:
        """
        Encode image frame to base64.

        Uses OpenCV when available (2-3x faster than PIL for resize+encode).
        Falls back to PIL otherwise.
        """
        try:
            if video_frame.dtype == np.float32 or video_frame.dtype == np.float64:
                video_frame = (video_frame * 255).astype(np.uint8)

            if _CV2_AVAILABLE:
                # OpenCV path — much faster than PIL
                h, w = video_frame.shape[:2]
                if w > max_width or h > max_height:
                    scale = min(max_width / w, max_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    # INTER_AREA is best for downscaling
                    video_frame = _cv2.resize(video_frame, (new_w, new_h),
                                              interpolation=_cv2.INTER_AREA)
                # OpenCV expects BGR for imencode; input is RGB
                bgr = _cv2.cvtColor(video_frame, _cv2.COLOR_RGB2BGR)
                _, img_bytes = _cv2.imencode(
                    '.jpg', bgr,
                    [_cv2.IMWRITE_JPEG_QUALITY, quality])
                return base64.b64encode(img_bytes.tobytes()).decode('utf-8')
            else:
                # PIL fallback
                image = Image.fromarray(video_frame)
                image.thumbnail((max_width, max_height), Image.BILINEAR)
                with io.BytesIO() as buf:
                    image.save(buf, format=format, quality=quality)
                    buf.seek(0)
                    img_bytes = buf.read()
                return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return ""
    
    def _update_history(self, intent: str):
        if intent and intent != "unknown":
            self.intent_history.append(intent)
            if len(self.intent_history) > self.max_history:
                self.intent_history.pop(0)

    def clear_history(self):
        """Reset — call when starting a new scene/video."""
        self.intent_history.clear()

    def build_user_prompt(
        self,
        robot_state: Optional[str] = None
    ) -> str:
        """Build user prompt with intent history and scene context."""
        parts = []
        if self.active_task_object:
            parts.append(
                f"Robot is currently targeting: {self.active_task_object}. "
                f"Always describe objects by color and type (e.g. 'blue bottle', 'brown flask'). "
                f"Only report a different target if the hand is clearly moving toward it."
            )
        if self.active_task_lang:
            parts.append(
                f"Task being executed: \"{self.active_task_lang}\". "
                f"Set task_complete:true if the TARGET OBJECT is already at its destination "
                f"(e.g. ball is visibly inside the bowl/plate). "
                f"Ignore the robot arm position — only check whether the object reached its goal."
            )
        if self.scene_objects:
            parts.append(f"Known objects in scene: {', '.join(self.scene_objects)}")
        if self.intent_history:
            parts.append(f"Previous intents: {' → '.join(self.intent_history)}")
        if robot_state:
            parts.append(f"Robot state: {robot_state}")
        parts.append(
            "Predict the near-future intention of the agent in the scene. "
            "Identify which object is targeted."
        )
        # Append /no_think to skip Qwen3's thinking mode without triggering
        # the broken guided-decoding-with-enable_thinking=False bug (vLLM
        # Issue #18819). With chat_template_kwargs.enable_thinking=False the
        # reasoning parser is bypassed and the guided_json output becomes
        # gibberish (random fragments, prompt regurgitation, empty streams).
        # The "/no_think" soft-switch produces valid JSON instead.
        parts.append("/no_think")
        return "\n".join(parts)
    
    def predict_intent(
        self,
        audio_window: np.ndarray,
        video_frame: np.ndarray,
        robot_state: str = "",
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Predict user intent from audio and video stream"""
        if self.client is None:
            return {
                'predicted_intent': 'unknown',
                'confidence': 0.0,
                'reason': 'OpenAI client not initialized',
                'error': 'Client initialization failed'
            }
        
        try:
            # Encode audio and video
            audio_b64 = self.encode_audio_to_base64(audio_window, sample_rate)
            frame_b64 = self.encode_image_to_base64(video_frame)
            
            if not audio_b64 or not frame_b64:
                return {
                    'predicted_intent': 'unknown',
                    'confidence': 0.0,
                    'reason': 'Failed to encode media',
                    'error': 'Encoding failed'
                }
            
            # Build the multimodal prompt for vLLM
            prompt = self.build_user_prompt(robot_state)
            
            # Create message content
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                }
            ]
            
            # Only add audio if we have it
            if audio_b64:
                content.append({
                    "type": "input_audio", 
                    "input_audio": {"data": audio_b64, "format": "wav"}
                })
            
            # Use streaming to get the response as soon as JSON closes
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True,
                            )
            result_text = self._stream_until_json(stream)

            # Retry video-only if audio caused empty response
            if not result_text.strip() and audio_b64:
                logger.warning("predict_intent: empty response with audio — retrying video-only")
                content_retry = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                ]
                stream_retry = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content_retry},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stream=True,
                                    )
                result_text = self._stream_until_json(stream_retry)

            # Parse JSON output
            parsed = self._parse_prediction(result_text)
            parsed['raw_response'] = result_text
            self._update_history(parsed.get('predicted_intent'))

            return parsed

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'predicted_intent': 'unknown',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}',
                'raw_response': '',
                'error': str(e)
            }
    
    def predict_intent_video_only(
        self,
        video_frame: np.ndarray,
        robot_state: str = ""
    ) -> Dict[str, Any]:
        """
        Predict user intent from video only (no audio)
        Useful for simulation environments without audio
        """
        if self.client is None:
            return {
                'predicted_intent': 'unknown',
                'confidence': 0.0,
                'reason': 'OpenAI client not initialized',
                'error': 'Client initialization failed'
            }
        
        try:
            frame_b64 = self.encode_image_to_base64(video_frame)
            
            if not frame_b64:
                return {
                    'predicted_intent': 'unknown',
                    'confidence': 0.0,
                    'reason': 'Failed to encode frame',
                    'error': 'Encoding failed'
                }
            
            # Build prompt that emphasizes video-only analysis
            parts = []
            if self.scene_objects:
                parts.append(f"Known objects in scene: {', '.join(self.scene_objects)}")
            if robot_state:
                parts.append(f"Robot state: {robot_state}")
            parts.append(
                "No audio is available. Predict the near-future intention "
                "of the agent based only on the video. Identify which object "
                "is targeted."
            )
            prompt = "\n".join(parts)
            
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                            }
                        ]
                    }
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                                stream=True,
            )
            result_text = self._stream_until_json(stream)

            parsed = self._parse_prediction(result_text)
            parsed['raw_response'] = result_text
            self._update_history(parsed.get('predicted_intent'))
            return parsed

        except Exception as e:
            logger.error(f"Video-only inference error: {e}")
            return {
                'predicted_intent': 'unknown',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }

    # ------------------------------------------------------------------
    # Multi-frame methods  (stitch frames into one composite image)
    # ------------------------------------------------------------------
    # vLLM with --limit-mm-per-prompt image=1 only allows ONE image
    # per prompt.  We work around this by stitching 2-3 frames
    # side-by-side into a single composite image with labels
    # ("t-2s | t-1s | t-now").  The model sees temporal progression
    # in one image, enabling motion reasoning.
    # ------------------------------------------------------------------

    def _stream_until_json(self, stream) -> str:
        """Consume a streaming response, returning text once a complete
        JSON object ``{...}`` is found (then closes the stream early).
        Returns empty string if Qwen produces no content (only newlines).
        """
        result_text = ""
        brace_depth = 0
        json_started = False
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                result_text += delta.content
                for ch in delta.content:
                    if ch == '{':
                        json_started = True
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1
                        if json_started and brace_depth == 0:
                            stream.close()
                            return result_text
        stripped = result_text.strip()
        # Demoted from warning → debug: these fire on every degenerate response
        # during execution and flood the log. The audio-encoder degeneration
        # is a known server-side issue; we just want the log to remain
        # readable for actual events (cold-start, stop, switch, task_complete).
        if not stripped:
            logger.debug("_stream_until_json: empty response (audio encoder degeneration)")
        elif '{' not in stripped:
            logger.debug(f"_stream_until_json: no JSON: {stripped!r}")
        return result_text

    def _stitch_frames(self, frames: list) -> np.ndarray:
        """Stitch multiple frames side-by-side into a single composite.

        Adds a small text label at the top of each panel so the model
        knows the temporal order.  Returns an RGB numpy array.
        """
        n = len(frames)
        if n == 1:
            return frames[0]

        # Resize all frames to same height (use the smallest)
        target_h = min(f.shape[0] for f in frames)
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            if h != target_h:
                scale = target_h / h
                new_w = int(w * scale)
                if _CV2_AVAILABLE:
                    f = _cv2.resize(f, (new_w, target_h), interpolation=_cv2.INTER_AREA)
                else:
                    from PIL import Image as _Img
                    f = np.array(_Img.fromarray(f).resize((new_w, target_h)))
            resized.append(f)

        # Add temporal labels
        labels = self._make_labels(n)
        labeled = []
        for i, (frame, label) in enumerate(zip(resized, labels)):
            frame = frame.copy()
            if _CV2_AVAILABLE:
                # White text with black outline for visibility
                pos = (5, 22)
                font = _cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                _cv2.putText(frame, label, pos, font, scale, (0, 0, 0), 3)
                _cv2.putText(frame, label, pos, font, scale, (255, 255, 255), 1)
            labeled.append(frame)

        # Horizontal concatenation
        composite = np.concatenate(labeled, axis=1)
        return composite

    @staticmethod
    def _make_labels(n: int) -> list:
        """Generate temporal labels for n frames."""
        if n == 2:
            return ["t-2s", "t-now"]
        elif n == 3:
            return ["t-2s", "t-1s", "t-now"]
        else:
            return [f"t-{n-1-i}s" if i < n - 1 else "t-now" for i in range(n)]

    # ──────────────────────────────────────────────────────────────
    # Fast-lane command classifier (audio-only Qwen call)
    #
    # Purpose: a SEPARATE Qwen inference path specialised for reactive
    # command detection (stop, switch, cold-start). Sits alongside the
    # main future-intent prediction (which keeps running every 250 ms
    # with full audio+video for the thesis contribution).
    #
    # Design rationale:
    #   - Audio-only payload eliminates the dominant failure mode
    #     (audio+video fusion failure on short utterances).
    #   - Minimal prompt eliminates JSON parse-failure noise.
    #   - Short audio (clean recent speech) keeps the encoder happy.
    #   - Different prompt prefix → independent vLLM cache → no
    #     interference with the main prediction stream.
    #
    # Robot-agnostic: only the command enum mentions the current task
    # names. For G1 or any other robot, the task list is built from
    # the TaskRegistry — no other change needed.
    # ──────────────────────────────────────────────────────────────

    def _build_classifier_prompt(self) -> str:
        """Build the command-classifier system prompt.

        Enumerates available task names so the model can emit
        ``command_pick_<name>`` and ``switch_<color>`` values that the
        router consumes directly. Rebuilt on demand (cheap, ~few hundred
        bytes) so adding a task to tasks.yaml requires no engine change.
        """
        task_lines = []
        switch_lines = []
        for label, obj in (self._cold_start_choices or []):
            # label is "command_pick_<name>"; we want a hint about the
            # spoken trigger
            task_lines.append(f'- "{label}" — heard a fresh task command for {obj}')
            color = obj.split()[0] if obj else label
            switch_lines.append(f'- "switch_{color}" — heard "{color}" only (no full command), redirect mid-task')

        prompt = (
            'You are a fast audio-only command classifier for a robot. '
            'You hear ~0.5-2 s of audio. Output a single JSON object — no extra text.\n'
            '\n'
            'Classify the audio into ONE of these classes:\n'
            '- "stop" — heard stop / halt / wait / cancel / abort\n'
        )
        if task_lines:
            prompt += '\n'.join(task_lines) + '\n'
        if switch_lines:
            prompt += '\n'.join(switch_lines) + '\n'
        prompt += (
            '- "none" — silence, background noise, or no clear command\n'
            '\n'
            'Output ONLY JSON in this exact shape:\n'
            '{"command":"<class>","confidence":0.0-1.0,"heard":"<verbatim words or empty>"}\n'
        )
        return prompt

    def classify_command(
        self,
        audio_window: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Audio-only command classification — used by the speech-onset
        fast lane in StreamingIntentPredictor.

        Returns ``{"command": "<class>", "confidence": float, "heard": "<text>"}``.
        """
        if self.client is None:
            return {"command": "none", "confidence": 0.0, "heard": "",
                    "error": "no client"}
        # Keep only the most recent ~1 s of audio — that's where the
        # speech burst lives after the audio_callback's onset clear.
        if len(audio_window) > sample_rate:
            audio_window = audio_window[-sample_rate:]
        try:
            audio_b64 = self.encode_audio_to_base64(audio_window, sample_rate)
            if not audio_b64:
                return {"command": "none", "confidence": 0.0, "heard": "",
                        "error": "no audio"}

            system_prompt = self._build_classifier_prompt()
            user_text = (
                "Classify this audio. Output one JSON object exactly as specified.\n"
                "/no_think"  # see Issue #18819 workaround — keeps guided JSON valid
            )
            content: list = [
                {"type": "text", "text": user_text},
                {"type": "input_audio",
                 "input_audio": {"data": audio_b64, "format": "wav"}},
            ]

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=60,   # tighter cap — output is ~30 tokens
                stream=True,
                            )
            result_text = self._stream_until_json(stream)

            # Best-effort JSON parse with robust fallbacks. The model
            # occasionally wraps in code fences or adds preamble.
            import json
            import re
            text = result_text.strip()
            if not text:
                return {"command": "none", "confidence": 0.0, "heard": "",
                        "error": "empty"}
            # Strip ```json fences
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            # Take first { ... } block
            m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
            try:
                parsed = json.loads(text)
            except Exception:
                # Fallback: keyword detection in raw text
                low = result_text.lower()
                if any(w in low for w in ("stop", "halt", "wait", "cancel")):
                    parsed = {"command": "stop", "confidence": 0.7,
                              "heard": result_text[:60]}
                else:
                    parsed = {"command": "none", "confidence": 0.0,
                              "heard": ""}
            return {
                "command": str(parsed.get("command", "none")).strip().lower(),
                "confidence": float(parsed.get("confidence", 0.0)),
                "heard": str(parsed.get("heard", "")).strip(),
            }
        except Exception as e:
            logger.error(f"classify_command error: {e}")
            return {"command": "none", "confidence": 0.0, "heard": "",
                    "error": str(e)}

    def verify_object_present(
        self,
        object_name: str,
        video_frame: np.ndarray,
    ) -> Dict[str, Any]:
        """Visual feasibility check: is there a graspable object to pick at all?

        A focused, vision-only Qwen call used as a feasibility gate before the
        robot starts or switches to a pick task. This is SayCan-style "world
        grounding": the perception model can veto a verbal command when the
        workspace has nothing pickable in it (e.g. "pick the ball" at an empty
        table). It does NOT use audio — it judges purely by what is in frame.

        Why a generic "is anything pickable here" question instead of "is the
        EXACT object present": Qwen's naming of these small objects is
        unreliable — it labels the same pink/yellow cotton balls "purple/cyan/
        blue" and flips between "cotton ball" and "yarn" frame to frame. Both
        exact-color and exact-noun matching produced false vetoes on valid
        picks. So we ask the question the model answers reliably — "is there a
        small graspable object on the table?" — and trust its boolean rather
        than string-matching its words. Honest trade-off: this catches the
        empty-workspace case but cannot distinguish "pick red when only yellow
        is present" — robust per-object color/identity grounding is beyond what
        this model does dependably.

        Returns ``{present, seen, grounded}``. Fails OPEN (present=True,
        grounded=False) on any error so a grounding outage never makes the
        system worse than having no gate at all.
        """
        fail_open = {"present": True, "seen": [], "grounded": False}
        if self.client is None or video_frame is None:
            return fail_open
        try:
            frame_b64 = self.encode_image_to_base64(video_frame)
            if not frame_b64:
                return fail_open
            prompt = (
                "Look ONLY at the image (there is no audio). The robot is about "
                f"to try to pick up a small object (a \"{object_name}\").\n"
                "Question: is there AT LEAST ONE small, graspable object (the one "
                "described above, or any similar small item) actually sitting in "
                "the workspace that the arm could pick up? Ignore the robot arm, "
                "the plate/bowl, and the bare table surface. Do NOT worry about "
                "exact color or what it is called — judge only whether a pickable "
                "object is physically present.\n"
                "Reply with JSON only:\n"
                '{"pickable_present": true or false, '
                '"seen": ["small objects you see, by color and type"]}\n'
                "/no_think"  # Issue #18819 workaround — keeps JSON valid
            )
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
            ]
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content":
                     "You are a visual feasibility checker for a robot arm. You "
                     "report only what is physically visible in the image, never "
                     "what you were asked to find."},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True,
            )
            text = self._stream_until_json(stream)
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                logger.warning("verify_object_present: no JSON (%r) — failing open",
                               text[:80])
                return fail_open
            data = json.loads(match.group(0))
            return {
                "present": bool(data.get("pickable_present", True)),
                "seen": [str(s) for s in data.get("seen", []) if s],
                "grounded": True,
            }
        except Exception as e:
            logger.warning("verify_object_present error: %s — failing open", e)
            return fail_open

    def verify_task_complete(
        self,
        object_name: str,
        video_frame: np.ndarray,
    ) -> Dict[str, Any]:
        """Visual completion check: is the pick-and-place task actually DONE?

        GR00T has no internal "done" signal and loops forever, and the inline
        ``task_complete`` field in the streaming prediction fires <0.3% of the
        time (it is buried in a 7-field multi-task JSON, which the model handles
        unreliably). This dedicated, vision-only call asks the focused question
        the model answers reliably — "is an object now in the bowl AND the
        gripper empty?" — the same pattern that made the grounding gate work.

        The two-part criterion (object placed AND gripper empty) is what stops
        the robot from halting mid-transport: while the gripper still holds the
        object, this returns False. It judges by the image, not by the phase
        label, so it works even when Qwen mislabels the phase (e.g. yellow,
        which the model never tags "placing").

        Returns ``{complete, confidence, grounded}``. Fails SAFE (complete=False)
        on any error so a glitch never stops the robot prematurely — the
        max-runtime cap remains the ultimate backstop.
        """
        fail = {"complete": False, "confidence": 0.0, "reason": "", "grounded": False}
        if self.client is None or video_frame is None:
            return fail
        try:
            frame_b64 = self.encode_image_to_base64(video_frame)
            if not frame_b64:
                return fail
            # Focus the model on the BOWL contents, not the gripper. After a
            # successful place, GR00T loops back and closes the (empty) gripper
            # on the table — which looks "holding" to Qwen and defeats any
            # gripper-state criterion. "Is a ball resting inside the bowl" is the
            # signal that actually defines completion and avoids that trap.
            prompt = (
                "Look ONLY at the image (there is no audio). A robot was asked "
                'to pick up a "' + object_name + '" and place it in the '
                "bowl/plate.\n"
                "Look SPECIFICALLY at the bowl/plate. Is a small ball now "
                "resting INSIDE the bowl/plate — already dropped in and released "
                "(NOT still being carried or held in the gripper above it)?\n"
                "Answer true only if a ball is clearly sitting inside the "
                "bowl/plate. If the bowl is empty, or the ball is still being "
                "carried/lifted, answer false. Ignore exact color.\n"
                "Reply with JSON only:\n"
                '{"complete": true or false, "confidence": 0.0-1.0, '
                '"reason": "<8 words>"}\n'
                "/no_think"  # Issue #18819 workaround — keeps JSON valid
            )
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
            ]
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content":
                     "You are a task-completion checker for a robot arm. You "
                     "report only what is physically visible in the image."},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True,
            )
            text = self._stream_until_json(stream)
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                logger.warning("verify_task_complete: no JSON (%r) — failing safe",
                               text[:80])
                return fail
            data = json.loads(match.group(0))
            return {
                "complete": bool(data.get("complete", False)),
                "confidence": float(data.get("confidence", 0.0)),
                "reason": str(data.get("reason", "")),
                "grounded": True,
            }
        except Exception as e:
            logger.warning("verify_task_complete error: %s — failing safe", e)
            return fail

    def predict_intent_multi_frame(
        self,
        audio_window: np.ndarray,
        video_frames: list,
        robot_state: str = "",
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Predict intent using multiple video frames + audio.

        Frames are stitched side-by-side into one composite image
        (to comply with vLLM's 1-image-per-prompt limit).
        """
        if self.client is None:
            return {'predicted_intent': 'unknown', 'confidence': 0.0,
                    'reason': 'Client not initialized', 'error': 'no client'}
        self._select_system_prompt(robot_state)
        try:
            audio_b64 = self.encode_audio_to_base64(audio_window, sample_rate)
            composite = self._stitch_frames(video_frames)
            frame_b64 = self.encode_image_to_base64(composite)

            if not frame_b64:
                return {'predicted_intent': 'unknown', 'confidence': 0.0,
                        'reason': 'Failed to encode composite', 'error': 'encoding'}

            n = len(video_frames)
            is_waiting = 'state=waiting' in (robot_state or '')
            if is_waiting:
                prompt = (
                    f"The image contains {n} consecutive frames stitched "
                    f"side-by-side (oldest→newest, left to right). The robot "
                    f"is IDLE. Listen to the audio and classify per the "
                    f"system prompt's enum.\n"
                )
            else:
                prompt = (
                    f"The image contains {n} consecutive frames stitched "
                    f"side-by-side (labeled oldest→newest, left to right).\n"
                    f"1. Compare hand/arm positions across panels to detect motion.\n"
                    f"2. If positions barely changed → predict 'continue'.\n"
                    f"3. If hand is touching/holding an object → predict 'gesture'.\n"
                )
            if self.active_task_object and not is_waiting:
                prompt += (
                    f"Robot is currently targeting: {self.active_task_object}. "
                    f"Always describe objects by color and type (e.g. 'blue bottle', 'brown flask'). "
                    f"Only report a different target if the hand is clearly moving toward it.\n"
                )
            if self.scene_objects:
                prompt += f"Known objects: {', '.join(self.scene_objects)}\n"
            if robot_state:
                prompt += f"Robot state: {robot_state}\n"
            if not is_waiting:
                prompt += "Predict the agent's near-future intention and identify which object is targeted.\n"
            prompt += "/no_think"  # Issue #18819 workaround — keeps guided JSON valid

            content: list = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
            ]
            if audio_b64:
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"}
                })

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True,
                            )
            result_text = self._stream_until_json(stream)
            parsed = self._parse_prediction(result_text)

            # Smart retry on empty response / parse failure: branch on
            # whether the audio actually had speech in it.
            #
            # - Speech present → AUDIO-ONLY retry. Preserves the spoken
            #   command (stop, switch keyword) which is what we care about
            #   most in this case. Drops video — but motion can be inferred
            #   from the next inference tick. Critical: fixes the "stop was
            #   in audio but got dropped on retry" failure mode.
            # - Audio essentially silent → VIDEO-ONLY retry (legacy path).
            #   Audio had nothing useful anyway; visual scene is what we
            #   need to know.
            #
            # The threshold is generous (~RMS 0.005, well below the gate's
            # 0.012). If anything voiced got through, we treat it as speech.
            if audio_b64 and (not result_text.strip() or parsed.get('_parse_failed')):
                audio_rms = float(np.sqrt(np.mean(audio_window.astype(np.float32) ** 2)))
                had_speech = audio_rms >= 0.005

                if had_speech and hasattr(self, '_FAST_PROMPT_AUDIO_ONLY'):
                    logger.warning("Multi-frame: degenerate response with audio "
                                   "— retrying AUDIO-only (RMS=%.4f, speech present)",
                                   audio_rms)
                    content_retry: list = [
                        {"type": "text", "text":
                         "Listen to the audio and classify per the system prompt.\n"
                         "/no_think"},  # Issue #18819 workaround
                        {"type": "input_audio",
                         "input_audio": {"data": audio_b64, "format": "wav"}},
                    ]
                    stream_retry = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self._FAST_PROMPT_AUDIO_ONLY},
                            {"role": "user", "content": content_retry},
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        stream=True,
                                            )
                    result_text = self._stream_until_json(stream_retry)
                    parsed = self._parse_prediction(result_text)
                else:
                    logger.warning("Multi-frame: degenerate response — retrying video-only "
                                   "(RMS=%.4f, silent)", audio_rms)
                    content_retry: list = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                    ]
                    stream_retry = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self._FAST_PROMPT_VIDEO_ONLY},
                            {"role": "user", "content": content_retry},
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        stream=True,
                                            )
                    result_text = self._stream_until_json(stream_retry)
                    parsed = self._parse_prediction(result_text)

            parsed['raw_response'] = result_text
            return parsed
        except Exception as e:
            logger.error(f"Multi-frame inference error: {e}")
            return {'predicted_intent': 'unknown', 'confidence': 0.0,
                    'reason': f'Error: {e}', 'raw_response': '', 'error': str(e)}

    def predict_intent_video_only_multi_frame(
        self,
        video_frames: list,
        robot_state: str = "",
    ) -> Dict[str, Any]:
        """Predict intent from multiple video frames (no audio).

        Frames are stitched side-by-side into a single composite image.
        This gives the model temporal context to perceive motion —
        critical for distinguishing approach/withdraw/continue from
        a single still frame.
        """
        if self.client is None:
            return {'predicted_intent': 'unknown', 'confidence': 0.0,
                    'reason': 'Client not initialized', 'error': 'no client'}
        # Must use VIDEO-ONLY prompt here. _FAST_PROMPT_EXECUTING says
        # "You receive BOTH video AND audio" — sending no audio with that
        # prompt causes Qwen3-Omni to return empty content in ~88ms.
        self.system_prompt = self._FAST_PROMPT_VIDEO_ONLY
        try:
            composite = self._stitch_frames(video_frames)
            frame_b64 = self.encode_image_to_base64(composite)

            if not frame_b64:
                return {'predicted_intent': 'unknown', 'confidence': 0.0,
                        'reason': 'Failed to encode composite', 'error': 'encoding'}

            n = len(video_frames)
            prompt = (
                f"The image contains {n} consecutive frames stitched "
                f"side-by-side (labeled oldest→newest, left to right). "
                f"No audio is available.\n"
                f"1. Compare hand/arm positions across panels to detect motion.\n"
                f"2. If positions barely changed → predict 'continue'.\n"
                f"3. If hand is touching/holding an object → predict 'gesture'.\n"
            )
            if self.active_task_object:
                prompt += (
                    f"Robot is currently targeting: {self.active_task_object}. "
                    f"Always describe objects by color and type (e.g. 'blue bottle', 'brown flask'). "
                    f"Only report a different target if the hand is clearly moving toward it.\n"
                )
            if self.scene_objects:
                prompt += f"Known objects: {', '.join(self.scene_objects)}\n"
            if robot_state:
                prompt += f"Robot state: {robot_state}\n"
            prompt += "Predict the agent's near-future intention and identify which object is targeted.\n"
            prompt += "/no_think"  # Issue #18819 workaround — keeps guided JSON valid

            content: list = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
            ]

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True,
                            )
            result_text = self._stream_until_json(stream)
            parsed = self._parse_prediction(result_text)
            parsed['raw_response'] = result_text
            return parsed
        except Exception as e:
            logger.error(f"Multi-frame video-only inference error: {e}")
            return {'predicted_intent': 'unknown', 'confidence': 0.0,
                    'reason': f'Error: {e}', 'raw_response': '', 'error': str(e)}

    def _parse_prediction(self, response: str) -> Dict[str, Any]:
        """
        Parse Qwen output into structured format
        
        Handles both clean JSON and JSON embedded in text
        """
        try:
            response = response.strip()
            if not response:
                raise ValueError("Empty response")

            # Strip preamble before first '{'
            first_brace = response.find('{')
            if first_brace > 0:
                response = response[first_brace:]

            # Remove markdown code fences
            if response.startswith('```'):
                lines = response.split('\n')
                if len(lines) > 2:
                    response = '\n'.join(lines[1:-1])

            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Validate required fields
                if 'predicted_intent' not in parsed:
                    parsed['predicted_intent'] = 'unknown'
                if 'confidence' not in parsed:
                    parsed['confidence'] = 0.5
                else:
                    parsed['confidence'] = max(0.0, min(1.0, float(parsed['confidence'])))
                if 'target_object' not in parsed:
                    parsed['target_object'] = 'none'
                if 'reason' not in parsed:
                    parsed['reason'] = 'No reason provided'
                parsed['task_complete'] = bool(parsed.get('task_complete', False))
                parsed['predicted_phase'] = str(
                    parsed.get('predicted_phase', 'unknown') or 'unknown'
                ).strip().lower()

                return parsed
            else:
                raise ValueError("No JSON object found in response")
                
        except Exception as e:
            logger.warning(f"JSON parse failed, trying regex fallback: {e}")
            # Log the raw response so we can diagnose whether failures are
            # token-budget truncation (response ends mid-JSON, no closing
            # brace), server-side aborts (truly empty / whitespace), or
            # malformed content. Added May 19 during parse-failure
            # investigation. Length-capped to keep log lines readable.
            _raw = (response or "").replace("\n", "\\n")
            if len(_raw) > 240:
                _raw = _raw[:240] + f"...(+{len(response) - 240} chars)"
            logger.warning(f"  raw response was: {_raw!r}")

        # ── Regex fallback ──────────────────────────────────────────────
        # If the model produced text that looks like intent output but
        # isn't valid JSON (e.g. truncated, extra commas, markdown),
        # extract what we can.
        intent_match = re.search(
            r'"predicted_intent"\s*:\s*"(\w+)"', response
        )
        phase_match = re.search(
            r'"predicted_phase"\s*:\s*"(\w+)"', response
        )
        conf_match = re.search(
            r'"confidence"\s*:\s*([\d.]+)', response
        )
        obj_match = re.search(
            r'"target_object"\s*:\s*"([^"]+)"', response
        )
        tc_match = re.search(
            r'"task_complete"\s*:\s*(true|false)', response, re.IGNORECASE
        )
        reason_match = re.search(
            r'"reason"\s*:\s*"([^"]+)"', response
        )

        if intent_match:
            conf = max(0.0, min(1.0, float(conf_match.group(1)))) if conf_match else 0.5
            return {
                'predicted_intent': intent_match.group(1),
                'predicted_phase': phase_match.group(1).lower() if phase_match else 'unknown',
                'confidence': conf,
                'target_object': obj_match.group(1) if obj_match else 'none',
                'task_complete': tc_match.group(1).lower() == 'true' if tc_match else False,
                'reason': reason_match.group(1) if reason_match else 'Parsed via regex fallback',
            }

        return {
            'predicted_intent': 'unknown',
            'confidence': 0.0,
            'target_object': 'none',
            'reason': 'Parse error: no valid prediction in response',
            '_parse_failed': True,
        }
    
    def predict_intent_batch(
        self,
        audio_windows: list,
        video_frames: list,
        robot_states: Optional[list] = None,
        sample_rate: int = 16000
    ) -> list[Dict[str, Any]]:
        """
        Batch inference (for evaluation purposes)
        
        Note: vLLM doesn't natively support batching multimodal,
        so this runs sequential calls
        """
        if robot_states is None:
            robot_states = [""] * len(audio_windows)
        
        results = []
        for audio_window, video_frame, state in zip(audio_windows, video_frames, robot_states):
            result = self.predict_intent(audio_window, video_frame, state, sample_rate)
            results.append(result)
        
        return results


class FastQwenInferenceEngine(Qwen3OmniInferenceEngine):
    """
    Optimized for low latency with multi-frame motion context.

    Key settings:
    - 320×240 images at JPEG q65 (good balance of detail vs payload)
    - Up to 3 frames per call (motion context is critical for accuracy)
    - 1-second audio window
    - 64-token max output
    - Near-greedy sampling (temperature 0.1)
    - Concise system prompt that emphasises frame-to-frame comparison
    """

    # Legacy prompt — kept for revert. To roll back, set:
    #     _FAST_PROMPT_EXECUTING = _FAST_PROMPT_EXECUTING_LEGACY
    _FAST_PROMPT_EXECUTING_LEGACY = (
        'Camera watching a 6-DOF SO101 robot arm with gripper on a workspace.\n'
        'Objects on workspace: cotton balls (pink, yellow).\n'
        'Frames oldest→newest (composite panels left→right).\n'
        'You receive BOTH video AND audio. Audio may contain human voice commands.\n'
        '\n'
        'THREE JOBS — do all three every call:\n'
        '1. LISTEN: Transcribe any spoken words verbatim into spoken_command.\n'
        '   spoken_command = "" if silent or only background/motor noise.\n'
        '2. WATCH (short horizon, ~1-2s): Classify arm motion into predicted_intent.\n'
        '3. WATCH (longer horizon, ~3-5s): Classify the current task PHASE into\n'
        '   predicted_phase — what subaction the arm is in / will be in next.\n'
        '\n'
        'STOP/INTERRUPT RULE (highest priority):\n'
        'If you hear "stop", "halt", "wait", "cancel" — set\n'
        'predicted_intent="interrupt" AND spoken_command=<the word heard>.\n'
        '\n'
        'SWITCH RULE:\n'
        'If you hear a command to switch objects ("yellow one", "other ball", etc.)\n'
        'set predicted_intent="change_target" AND spoken_command=<words heard>.\n'
        '\n'
        'predicted_intent values (next 1-2s):\n'
        '- "continue" — gripper barely moving\n'
        '- "approach" — gripper moving toward object, claws open\n'
        '- "gesture" — gripper closed on object, holding/moving it\n'
        '- "withdraw" — gripper moving away after release\n'
        '- "interrupt" — heard stop/halt OR arm froze mid-motion\n'
        '- "change_target" — heard switch command OR trajectory redirecting\n'
        '- "unknown" — unclear\n'
        '\n'
        'predicted_phase values (next 3-5s subaction):\n'
        '- "approaching" — moving toward target, not yet contacting\n'
        '- "grasping"    — closing gripper on target object\n'
        '- "transporting"— holding object, moving it toward destination\n'
        '- "placing"     — releasing object at destination\n'
        '- "retracting"  — moving back to home/idle pose after task\n'
        '- "idle"        — no active subaction\n'
        '- "unknown"     — unclear\n'
        '\n'
        'task_complete: true ONLY if target object is visibly at its destination '
        '(ball in bowl/plate). Ignore arm position. Default false.\n'
        '\n'
        'EXAMPLE 1 — silence, arm grasping pink ball:\n'
        '{"spoken_command":"","predicted_intent":"gesture","predicted_phase":"grasping",'
        '"confidence":0.95,"target_object":"pink cotton ball","task_complete":false,'
        '"reason":"Gripper closed on pink ball, lifting it"}\n'
        '\n'
        'EXAMPLE 2 — human says "stop" while arm moves:\n'
        '{"spoken_command":"stop","predicted_intent":"interrupt","predicted_phase":"idle",'
        '"confidence":0.95,"target_object":"yellow cotton ball","task_complete":false,'
        '"reason":"Verbal stop command heard, arm mid-motion"}\n'
        '\n'
        'EXAMPLE 3 — human says "pick up the yellow one" while arm holds pink:\n'
        '{"spoken_command":"pick up the yellow one","predicted_intent":"change_target",'
        '"predicted_phase":"approaching","confidence":0.9,"target_object":"yellow cotton ball",'
        '"task_complete":false,"reason":"Switch command, redirecting to yellow ball"}\n'
        '\n'
        'EXAMPLE 4 — silence, arm approaching pink ball:\n'
        '{"spoken_command":"","predicted_intent":"approach","predicted_phase":"approaching",'
        '"confidence":0.9,"target_object":"pink cotton ball","task_complete":false,'
        '"reason":"Gripper moving toward pink ball, claws open"}\n'
        '\n'
        'Reply ONLY with JSON in this exact field order: '
        '{"spoken_command":"<verbatim words or empty>",'
        '"predicted_intent":"<class>","predicted_phase":"<phase>",'
        '"confidence":0.0-1.0,"target_object":"<object color or none>",'
        '"task_complete":false,"reason":"<15 words max>"}'
    )

    # The EXECUTING / VIDEO_ONLY / AUDIO_ONLY prompts are now built from the
    # task registry at construction time — see _build_executing_prompt,
    # _build_video_only_prompt, _build_audio_only_prompt above. They are cached
    # as instance attributes (self._FAST_PROMPT_*) in __init__. The legacy
    # hardcoded EXECUTING constant is preserved below for one-line revert.

    # WAITING prompt — Qwen-only cold-start experiment (April 25).
    # Encodes cold-start commands AS predicted_intent enum values
    # (command_pick_pink_ball / command_pick_yellow_ball / none) instead of
    # asking Qwen to fill a free-form spoken_command field (which it ignores).
    # This rides the same classification mechanism that already works for
    # interrupt and change_target. Built dynamically from task list — see
    # _build_waiting_prompt below.
    _FAST_PROMPT_WAITING = ''  # populated dynamically per task registry

    def _build_waiting_prompt(self) -> str:
        choices = self._cold_start_choices or [
            ('command_pick_pink_ball', 'pink cotton ball'),
            ('command_pick_yellow_ball', 'yellow cotton ball'),
        ]
        bullets = '\n'.join(
            f'  - "{intent}" — heard a verbal command to pick up the {obj}'
            for intent, obj in choices
        )
        example_intent, example_obj = choices[0]
        example_color = example_obj.split()[0]
        return (
            'Camera watching a 6-DOF SO101 robot arm (idle) on a workspace.\n'
            'The robot is IDLE waiting for a verbal task command from the human.\n'
            'You receive BOTH video AND audio.\n'
            '\n'
            'YOUR JOB: classify whether the human just spoke a verbal command\n'
            'to start (or RESUME) a task. Output one of these values in\n'
            'predicted_intent:\n'
            f'{bullets}\n'
            '  - "command_resume" — heard "continue" / "resume" / "keep going" /\n'
            '     "go on" / "carry on" / "proceed" (user wants to RESUME the last\n'
            '     paused task, not start a new one)\n'
            '  - "none" — silence, background noise, or speech unrelated to a task\n'
            '\n'
            'CRITICAL RULES:\n'
            '1. Only emit a command_* value if you HEARD a clear verbal command\n'
            '   in the audio. Visible objects on the table do NOT count — silence\n'
            '   with objects visible MUST be classified as "none".\n'
            '2. If you hear unrelated speech (chatter, "hello", "thanks"), output "none".\n'
            '3. Be conservative — when in doubt, output "none".\n'
            '\n'
            'EXAMPLE A — silence, both balls visible on workspace:\n'
            '{"predicted_intent":"none","confidence":0.95,'
            '"target_object":"none","reason":"Workspace silent, no verbal command"}\n'
            '\n'
            'EXAMPLE B — human says "pick up the ' + example_color + ' one":\n'
            '{"predicted_intent":"' + example_intent + '","confidence":0.9,'
            '"target_object":"' + example_obj + '","reason":"Heard verbal command for ' + example_color + ' object"}\n'
            '\n'
            'EXAMPLE C — human says "hello there" (unrelated chatter):\n'
            '{"predicted_intent":"none","confidence":0.9,'
            '"target_object":"none","reason":"Unrelated speech, no task command"}\n'
            '\n'
            'EXAMPLE D — human says "continue" (after a previous stop):\n'
            '{"predicted_intent":"command_resume","confidence":0.95,'
            '"target_object":"none","reason":"Heard resume command"}\n'
            '\n'
            'Reply ONLY with JSON: '
            '{"predicted_intent":"<value>","confidence":0.0-1.0,'
            '"target_object":"<object color or none>",'
            '"reason":"<short, 15 words max>"}'
        )

    def _task_object_names(self) -> list:
        """Object names for the active task set, taken from the task registry
        (cold_start_choices), falling back to scene_objects then a generic
        default. This is the single source the EXECUTING / VIDEO / AUDIO prompts
        template from, so adding a task to tasks.yaml needs NO prompt edits and
        no object name is hardcoded in any prompt."""
        if self._cold_start_choices:
            objs = [obj for (_, obj) in self._cold_start_choices if obj]
            if objs:
                return objs
        if getattr(self, 'scene_objects', None):
            return list(self.scene_objects)
        return ['pink cotton ball', 'yellow cotton ball']

    def _build_executing_prompt(self) -> str:
        objs = self._task_object_names()
        obj1 = objs[0]
        obj2 = objs[1] if len(objs) > 1 else objs[0]
        c1 = obj1.split()[0]
        c2 = obj2.split()[0]
        object_list = ", ".join(objs)
        colors = [o.split()[0] for o in objs]
        switch_kw = ", ".join('"' + c + '"' for c in colors) + ', "other", "not this", "instead"'
        return (
            'SO101 robot arm + gripper on workspace. Objects: ' + object_list + '; plus a bowl/plate.\n'
            'Frames are oldest→newest panels. You receive video AND audio.\n'
            '\n'
            'JOB: Predict what happens in the NEXT 1-2 seconds (predicted_intent) and\n'
            'the NEXT 3-5 seconds subaction (predicted_phase). Audio is a strong\n'
            'predictor: what the human SAYS is what is about to happen.\n'
            '\n'
            'Also fill spoken_command with verbatim words heard ("" if silent/noise).\n'
            '\n'
            'PRIORITY RULES:\n'
            '1. Hear stop/halt/wait/cancel → predicted_intent="interrupt",\n'
            '   predicted_phase="retracting"\n'
            '2. Hear switch (' + switch_kw + ') →\n'
            '   predicted_intent="change_target", target_object=<new object>,\n'
            '   predicted_phase="retracting" (arm SHOULD drop current and redirect —\n'
            '   predict the future, not the past)\n'
            '3. Target object visibly inside bowl/plate → task_complete=true\n'
            '\n'
            'predicted_intent (1-2s): continue | approach | gesture | withdraw |\n'
            '  change_target | interrupt | unknown\n'
            'predicted_phase (3-5s): approaching | grasping | transporting | placing |\n'
            '  retracting | idle | unknown\n'
            '\n'
            'EX1 silent, arm grasping ' + c1 + ':\n'
            '{"spoken_command":"","predicted_intent":"gesture","predicted_phase":"transporting",'
            '"confidence":0.95,"target_object":"' + obj1 + '","task_complete":false,'
            '"reason":"Holding ' + c1 + ', will move to bowl"}\n'
            '\n'
            'EX2 user says "wait stop" mid-pickup:\n'
            '{"spoken_command":"wait stop","predicted_intent":"interrupt","predicted_phase":"retracting",'
            '"confidence":0.95,"target_object":"' + obj1 + '","task_complete":false,'
            '"reason":"Stop heard, arm should retract"}\n'
            '\n'
            'EX3 user says "no the ' + c2 + ' one" while holding ' + c1 + ':\n'
            '{"spoken_command":"no the ' + c2 + ' one","predicted_intent":"change_target",'
            '"predicted_phase":"retracting","confidence":0.9,"target_object":"' + obj2 + '",'
            '"task_complete":false,"reason":"Drop ' + c1 + ', redirect to ' + c2 + '"}\n'
            '\n'
            'Reply ONLY with JSON: {"spoken_command":"...","predicted_intent":"...",'
            '"predicted_phase":"...","confidence":0.0-1.0,"target_object":"...",'
            '"task_complete":false,"reason":"<15 words>"}'
        )

    def _build_video_only_prompt(self) -> str:
        objs = self._task_object_names()
        obj1 = objs[0]
        obj2 = objs[1] if len(objs) > 1 else objs[0]
        c1 = obj1.split()[0]
        c2 = obj2.split()[0]
        object_list = ", ".join(objs)
        return (
            'Camera watching a 6-DOF SO101 robot arm with gripper on a workspace.\n'
            'Objects on workspace: ' + object_list + '.\n'
            'Frames oldest→newest (composite panels left→right). NO audio.\n'
            '\n'
            'predicted_intent values (next 1-2s):\n'
            '- "continue" — gripper barely moving\n'
            '- "approach" — gripper moving toward object, claws open\n'
            '- "gesture" — gripper closed on object, holding/moving it\n'
            '- "withdraw" — gripper moving away after release\n'
            '- "change_target" — trajectory redirecting to different object\n'
            '- "interrupt" — sudden freeze mid-motion\n'
            '- "unknown" — unclear\n'
            '\n'
            'predicted_phase values (next 3-5s subaction):\n'
            '- "approaching" | "grasping" | "transporting" | "placing" | "retracting" |\n'
            '  "idle" | "unknown"\n'
            '\n'
            'task_complete: true ONLY if target object is visibly at its destination '
            '(object in bowl/plate). Default false.\n'
            '\n'
            'EXAMPLE 1:\n'
            '{"spoken_command":"","predicted_intent":"approach","predicted_phase":"approaching",'
            '"confidence":0.9,"target_object":"' + obj1 + '","task_complete":false,'
            '"reason":"Gripper moving toward ' + c1 + '"}\n'
            '\n'
            'EXAMPLE 2:\n'
            '{"spoken_command":"","predicted_intent":"gesture","predicted_phase":"grasping",'
            '"confidence":0.95,"target_object":"' + obj2 + '","task_complete":false,'
            '"reason":"Gripper closed, holding ' + c2 + '"}\n'
            '\n'
            'Reply ONLY with JSON: '
            '{"spoken_command":"","predicted_intent":"<class>","predicted_phase":"<phase>",'
            '"confidence":0.0-1.0,"target_object":"<object or none>",'
            '"task_complete":false,"reason":"<15 words max>"}'
        )

    def _build_audio_only_prompt(self) -> str:
        objs = self._task_object_names()
        colors = [o.split()[0] for o in objs]
        color_list = ", ".join(colors)
        color_hint = " / ".join('"' + c + '"' for c in colors)
        return (
            'Audio-only mode (no video). You hear ~2 s of audio that may contain '
            'a spoken command for a robot.\n'
            '\n'
            'spoken_command: fill with the verbatim words heard ("" if silent/noise).\n'
            '\n'
            'predicted_intent values (audio-derived):\n'
            '- "interrupt"     — heard stop / halt / wait / cancel / abort\n'
            '- "change_target" — heard a different object/target (' + color_hint + ', "other")\n'
            '- "continue"      — silence or non-command speech\n'
            '- "unknown"       — unclear\n'
            '\n'
            'target_object: if you heard a known object color (' + color_list + '), set '
            'the matching "<color> ..." object name; otherwise "none".\n'
            '\n'
            'predicted_phase: "retracting" if interrupt or change_target, else "unknown".\n'
            '\n'
            'Reply ONLY with JSON: '
            '{"spoken_command":"<verbatim or empty>","predicted_intent":"<class>",'
            '"predicted_phase":"<phase>","confidence":0.0-1.0,'
            '"target_object":"<object or none>","task_complete":false,'
            '"reason":"<10 words max>"}'
        )

    def __init__(
        self,
        vllm_url: str = "http://192.168.2.25:8000/v1",
        api_key: str = "vllm-omni",
        model_name: str = "qwen3-30b-a3b",
        system_prompt: Optional[str] = None,
        scene_objects: Optional[list] = None,
        cold_start_choices: Optional[list] = None,
    ):
        # cold_start_choices: list of (intent_value, object_description) tuples
        # used to build the WAITING prompt's enum. Populated from TaskRegistry
        # in run_system_groot.py main.
        self._cold_start_choices = cold_start_choices
        super().__init__(
            vllm_url=vllm_url,
            api_key=api_key,
            model_name=model_name,
            # Reverted May 19 evening from (0.01 / 0.1) back to (0.1 / 0.9).
            # The Issue #139 reference config (0.01/0.1) was tried briefly
            # but caused two new failure modes:
            #   (a) deterministic prose responses — model emitted 549+ char
            #       natural-language paragraphs instead of JSON when the
            #       audio encoder was confused (visible in session log #58)
            #   (b) stop detection during long executions got noticeably
            #       worse — model commits early to a wrong path on
            #       ambiguous audio rather than recovering probabilistically
            # At 0.1/0.9 the sampler has enough entropy to find the JSON
            # structure even when the encoder output is borderline.
            temperature=0.1,
            top_p=0.9,
            # Token budget raised 80 → 160 on May 19 after diagnosing ~50%
            # parse-failure rate in --no-vad sessions. Root cause: the May 18
            # changes added `predicted_phase` (+5-8 tokens) AND the rewritten
            # EXECUTING prompt now actually elicits long `spoken_command`
            # values (10-15 tokens for phrases like "can you pick up the
            # yellow ball"). Typical full response: 70-85 tokens — right at
            # the old 80-token cap, causing mid-JSON truncation → empty
            # streams. 160 gives generous headroom. Streaming stops at the
            # closing `}` via _stream_until_json, so extra cap costs ~nothing
            # in the happy path.
            max_tokens=160,
            system_prompt=system_prompt,
            scene_objects=scene_objects,
        )

        # Balanced image settings — enough detail to see grippers + objects
        self.image_quality = 65
        self.image_format = 'JPEG'
        self.max_width = 320
        self.max_height = 240
        # For composite (stitched) images: allow wider to preserve per-panel detail
        self.composite_max_width = 960   # 320px × 3 panels
        self.composite_max_height = 240

        # Audio window length sent to Qwen on each prediction tick.
        # Was 0.5s — too short for Qwen to recognise multi-word commands like
        # "pick up the yellow cotton ball" (~1.7s). Bumped to 2.0s to match the
        # streaming predictor's audio_buffer_duration so spoken_command can fire.
        self.audio_max_duration = 2.0

        # Robot state for dynamic prompt switching
        self._current_robot_state = ''

        # Build the task-specific prompts once from the registry (object names
        # come from cold_start_choices). These instance attributes shadow the
        # old hardcoded class constants, so every prompt now adapts to whatever
        # tasks are in tasks.yaml with no per-task editing. The WAITING prompt
        # is built per-call in _select_system_prompt (it already was dynamic).
        self._FAST_PROMPT_EXECUTING = self._build_executing_prompt()
        self._FAST_PROMPT_VIDEO_ONLY = self._build_video_only_prompt()
        self._FAST_PROMPT_AUDIO_ONLY = self._build_audio_only_prompt()

    def encode_image_to_base64(self, video_frame: np.ndarray) -> str:
        """Override with balanced compression.

        Auto-detects composite (stitched) images by aspect ratio and
        uses wider max dimensions to preserve per-panel detail.
        """
        h, w = video_frame.shape[:2]
        # If the image is very wide (aspect > 2:1), it's a multi-frame composite
        if w / max(h, 1) > 2.0:
            max_w, max_h = self.composite_max_width, self.composite_max_height
        else:
            max_w, max_h = self.max_width, self.max_height

        return super().encode_image_to_base64(
            video_frame,
            format=self.image_format,
            quality=self.image_quality,
            max_width=max_w,
            max_height=max_h,
        )

    # High-pass filter coefficients (Butterworth order 4, cutoff 400 Hz @ 16 kHz).
    # Raised from 200→400 Hz: motor harmonics extend past 200 Hz; 400 Hz removes
    # more rumble while keeping speech intelligibility (formants start above 300 Hz).
    # Computed dynamically so the cutoff is easy to change; falls back to hardcoded
    # 200 Hz coefficients if scipy is unavailable (should never happen).
    try:
        from scipy.signal import butter as _scipy_butter
        _HPF_B, _HPF_A = _scipy_butter(4, 400.0 / 8000.0, btype='highpass')
        del _scipy_butter
    except ImportError:
        _HPF_B = np.array([
            0.90244446, -3.60977783,  5.41466674, -3.60977783,  0.90244446])
        _HPF_A = np.array([
            1.0,        -3.7947911,   5.40516686, -3.42474735,  0.814406  ])

    # RMS amplitude cap applied after HPF. Prevents loud motor-noise bursts
    # (which survive the frequency filter as harmonics) from causing encoder
    # overload in Qwen3-Omni. Only scales DOWN — never amplifies quiet audio.
    _HPF_RMS_CAP: float = 0.04

    def _highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply a 400 Hz Butterworth high-pass to suppress GR00T motor rumble.
        Zero-phase via lfilter forward-then-reverse (cheap filtfilt).
        Also caps RMS amplitude so loud motor-noise bursts don't overload
        Qwen3-Omni's audio encoder. Only scales down, never amplifies.
        """
        if audio.size < 30:
            return audio
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x = x / 32768.0
        from scipy.signal import lfilter
        y = lfilter(self._HPF_B, self._HPF_A, x)
        y = lfilter(self._HPF_B, self._HPF_A, y[::-1])[::-1]
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms > self._HPF_RMS_CAP:
            y = y * (self._HPF_RMS_CAP / rms)
        return y.astype(np.float32)

    def encode_audio_to_base64(
        self,
        audio_window: np.ndarray,
        sample_rate: int = 16000,
        max_duration: Optional[float] = None,
    ) -> str:
        """Override to cap audio duration. NOTE: high-pass filter was tested
        (April 26) but removed — it didn't reduce prediction-path failures
        and degraded transcribe_audio because speech fundamentals (85–250 Hz)
        overlap the filter cutoff."""
        return super().encode_audio_to_base64(
            audio_window,
            sample_rate=sample_rate,
            max_duration=max_duration or self.audio_max_duration,
        )

    def _select_system_prompt(self, robot_state: Optional[str]) -> None:
        """Switch self.system_prompt between WAITING and EXECUTING based on
        robot_state. Must be called at the top of every predict_* method —
        the multi-frame paths build their own user prompt and bypass
        build_user_prompt, so without this call the WAITING prompt would
        never be selected even when the robot is idle."""
        self._current_robot_state = robot_state or ''
        if 'state=waiting' in self._current_robot_state:
            self.system_prompt = self._build_waiting_prompt()
        else:
            self.system_prompt = self._FAST_PROMPT_EXECUTING

    def build_user_prompt(self, robot_state: Optional[str] = None) -> str:
        """Override to store robot_state for dynamic system prompt switching."""
        self._select_system_prompt(robot_state)
        return super().build_user_prompt(robot_state)

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a speech segment using Qwen's audio capability.
        Returns the transcribed text, or empty string on failure.
        Used by the interrupt system to parse verbal stop/redirect commands.
        """
        if self.client is None:
            return ""
        try:
            # Pad short utterances with leading silence so Qwen3-Omni's audio
            # encoder has enough context. Without this, sub-500ms clips like a
            # quick "stop" tend to come back as empty transcriptions.
            min_samples = int(1.0 * sample_rate)
            if len(audio) < min_samples:
                pad = np.zeros(min_samples - len(audio), dtype=audio.dtype)
                audio = np.concatenate([pad, audio])
            audio_b64 = self.encode_audio_to_base64(
                audio, sample_rate=sample_rate, max_duration=5.0
            )
            if not audio_b64:
                return ""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Transcribe the speech exactly as spoken. Reply with only the transcribed words, nothing else. If there is no speech, reply with an empty string.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe:"},
                            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=60,
                            )
            text = response.choices[0].message.content or ""
            # Strip vLLM chat-template echo artifacts (e.g. "user\nStop", "assistant\nTranscribe:")
            _noise = {'user', 'assistant', 'system', 'transcribe:', 'transcribe'}
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            lines = [l for l in lines if l.lower() not in _noise]
            return ' '.join(lines).strip()
        except Exception as e:
            logger.warning(f"transcribe_audio failed: {e}")
            return ""


if __name__ == "__main__":
    import time
    import statistics

    logging.basicConfig(level=logging.INFO)

    VLLM_URL = "http://192.168.2.25:8000/v1"
    MODEL    = "qwen3-30b-a3b"
    N_WARM   = 1   # warm-up calls (excluded from stats)
    N_BENCH  = 5   # timed calls

    # --- Synthetic test data ---
    audio_window = np.random.randn(32000).astype(np.float32) * 0.001  # 2s @ 16kHz
    video_frame  = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # =========================================================
    # 1. LOCAL ENCODING BENCHMARK  (no server needed)
    # =========================================================
    print("=" * 64)
    print("1) ENCODING BENCHMARK  (no vLLM required)")
    print("=" * 64)

    std_engine  = Qwen3OmniInferenceEngine(vllm_url=VLLM_URL, model_name=MODEL)
    fast_engine = FastQwenInferenceEngine(vllm_url=VLLM_URL, model_name=MODEL)

    for label, eng in [("Standard", std_engine), ("Fast", fast_engine)]:
        # Image
        times_img = []
        for _ in range(20):
            t0 = time.perf_counter()
            b64 = eng.encode_image_to_base64(video_frame)
            times_img.append((time.perf_counter() - t0) * 1000)
        img_kb = len(b64) * 3 / 4 / 1024  # approximate decoded size

        # Audio
        times_aud = []
        for _ in range(20):
            t0 = time.perf_counter()
            b64a = eng.encode_audio_to_base64(audio_window, 16000)
            times_aud.append((time.perf_counter() - t0) * 1000)
        aud_kb = len(b64a) * 3 / 4 / 1024

        print(f"\n  [{label}]")
        print(f"    Image encode:  {statistics.median(times_img):.2f} ms median  "
              f"(payload ~{img_kb:.0f} KB)")
        print(f"    Audio encode:  {statistics.median(times_aud):.2f} ms median  "
              f"(payload ~{aud_kb:.0f} KB)")

    # Prompt token estimate
    print(f"\n  System prompt tokens (approx):")
    print(f"    Standard: ~{len(std_engine.system_prompt.split())} words")
    print(f"    Fast:     ~{len(fast_engine.system_prompt.split())} words")

    # =========================================================
    # 2. END-TO-END INFERENCE BENCHMARK  (requires vLLM server)
    # =========================================================
    print("\n" + "=" * 64)
    print("2) END-TO-END INFERENCE BENCHMARK  (requires vLLM at", VLLM_URL + ")")
    print("=" * 64)

    # Quick connectivity check
    try:
        import httpx as _hx
        r = _hx.get(VLLM_URL.replace("/v1", "/health"), timeout=3.0)
        server_up = r.status_code == 200
    except Exception:
        server_up = False

    if not server_up:
        print("\n  ⚠️  vLLM server not reachable — skipping inference benchmark.")
        print("     Start the server and re-run, or just check encoding results above.")
        sys.exit(0)

    for label, eng in [("Standard", std_engine), ("Fast", fast_engine)]:
        print(f"\n  --- {label} Engine ---")

        # Warm-up
        for _ in range(N_WARM):
            eng.predict_intent(audio_window, video_frame)

        latencies = []
        for i in range(N_BENCH):
            t0 = time.perf_counter()
            result = eng.predict_intent(audio_window, video_frame)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            intent = result.get("predicted_intent", "?")
            conf   = result.get("confidence", 0)
            print(f"    [{i+1}/{N_BENCH}] {lat:7.1f} ms  →  {intent} (conf={conf:.2f})")

        print(f"\n    Median: {statistics.median(latencies):.1f} ms")
        print(f"    Mean:   {statistics.mean(latencies):.1f} ms")
        print(f"    Min:    {min(latencies):.1f} ms")
        print(f"    Max:    {max(latencies):.1f} ms")

    print("\n" + "=" * 64)
    print("Done!")
    print("=" * 64)
