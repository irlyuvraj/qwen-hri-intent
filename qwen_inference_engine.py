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
        max_tokens: int = 80,
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

        # State machine fields
        self._sm_state: Optional[str] = None      # current confirmed state
        self._sm_state_count: int = 0             # consecutive predictions in this state
        self._SM_MAX_CONSECUTIVE: int = 6         # max before forcing reset to continue
        # Minimum consecutive predictions needed before certain transitions are accepted
        # gesture→withdraw needs 2 gestures first (prevents single-prediction locks)
        self._sm_gesture_count: int = 0

        # Initialize client with connection pooling for low-latency
        try:
            http_client = httpx.Client(
                timeout=30.0,
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
        self._sm_state = None
        self._sm_state_count = 0
        self._sm_gesture_count = 0

    def apply_state_machine(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter predictions through a valid-transition state machine.

        The physical manipulation cycle is universal to pick-and-place:
            continue → approach → gesture → withdraw → continue → ...

        Two safeguards prevent the "stuck state" problem:

        1. TRANSITION GUARD on gesture → withdraw:
           Requires at least 2 consecutive gesture predictions before
           allowing a withdraw. Prevents a single stray withdraw from
           locking the state machine into withdraw for the whole video.

        2. STATE TIMEOUT:
           Any non-continue state held for more than _SM_MAX_CONSECUTIVE
           predictions is forcibly reset to continue.
           This is the universal escape valve for any stuck state.

        Blocked predictions are held at the current state (not converted),
        so the FP budget stays neutral rather than shifting to another class.
        """
        raw   = parsed.get('predicted_intent', 'unknown')
        state = self._sm_state

        # ── State timeout: force reset if stuck too long ──────────────────
        if (self._sm_state_count >= self._SM_MAX_CONSECUTIVE
                and state not in ('continue', 'unknown', None)):
            logger.debug(f"SM timeout: {state} × {self._sm_state_count} → continue")
            self._sm_state = 'continue'
            self._sm_state_count = 0
            self._sm_gesture_count = 0
            state = 'continue'

        # ── Valid transition table ────────────────────────────────────────
        VALID = {
            None:            {'continue', 'approach', 'unknown'},
            'continue':      {'continue', 'approach', 'unknown'},
            'approach':      {'approach', 'gesture', 'continue', 'unknown'},
            'gesture':       {'gesture', 'unknown'},   # withdraw gated separately
            'withdraw':      {'withdraw', 'continue', 'approach', 'unknown'},
            'unknown':       {'continue', 'approach', 'gesture',
                              'withdraw', 'point', 'change_target', 'interrupt', 'unknown'},
            'point':         {'point', 'continue', 'approach', 'unknown'},
            'change_target': {'approach', 'gesture', 'unknown'},
            'interrupt':     {'continue', 'unknown'},
        }

        allowed = VALID.get(state, VALID[None])

        # ── Special gate: gesture → withdraw requires 2+ gestures first ──
        if raw == 'withdraw' and state == 'gesture':
            if self._sm_gesture_count >= 2:
                allowed = allowed | {'withdraw'}
            # else: withdraw is NOT allowed yet — will be blocked below

        if raw in allowed:
            # Valid — accept
            new_state = raw
        else:
            # Invalid — hold current state silently
            new_state = state or 'continue'
            logger.debug(f"SM blocked {state}→{raw}, holding {new_state}")
            parsed['predicted_intent'] = new_state
            parsed['reason'] = f"[sm:{raw}→{new_state}] " + parsed.get('reason','')

        # ── Update counters ───────────────────────────────────────────────
        if new_state == self._sm_state:
            self._sm_state_count += 1
        else:
            self._sm_state_count = 1

        if new_state == 'gesture':
            self._sm_gesture_count += 1
        else:
            self._sm_gesture_count = 0

        self._sm_state = new_state
        return parsed

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
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
        if not stripped:
            logger.warning("_stream_until_json: Qwen returned only whitespace/newlines — likely audio interference")
        elif '{' not in stripped:
            logger.warning(f"_stream_until_json: no JSON in response: {stripped!r}")
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
        try:
            audio_b64 = self.encode_audio_to_base64(audio_window, sample_rate)
            composite = self._stitch_frames(video_frames)
            frame_b64 = self.encode_image_to_base64(composite)

            if not frame_b64:
                return {'predicted_intent': 'unknown', 'confidence': 0.0,
                        'reason': 'Failed to encode composite', 'error': 'encoding'}

            n = len(video_frames)
            prompt = (
                f"The image contains {n} consecutive frames stitched "
                f"side-by-side (labeled oldest→newest, left to right).\n"
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
            prompt += "Predict the agent's near-future intention and identify which object is targeted."

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
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            result_text = self._stream_until_json(stream)

            # Retry without audio if Qwen returned empty (audio interference)
            if not result_text.strip() and audio_b64:
                logger.warning("Multi-frame: empty response with audio — retrying video-only")
                content_retry: list = [
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
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
            prompt += "Predict the agent's near-future intention and identify which object is targeted."

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
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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

                return parsed
            else:
                raise ValueError("No JSON object found in response")
                
        except Exception as e:
            logger.warning(f"JSON parse failed, trying regex fallback: {e}")

        # ── Regex fallback ──────────────────────────────────────────────
        # If the model produced text that looks like intent output but
        # isn't valid JSON (e.g. truncated, extra commas, markdown),
        # extract what we can.
        intent_match = re.search(
            r'"predicted_intent"\s*:\s*"(\w+)"', response
        )
        conf_match = re.search(
            r'"confidence"\s*:\s*([\d.]+)', response
        )
        obj_match = re.search(
            r'"target_object"\s*:\s*"([^"]+)"', response
        )
        reason_match = re.search(
            r'"reason"\s*:\s*"([^"]+)"', response
        )

        if intent_match:
            conf = max(0.0, min(1.0, float(conf_match.group(1)))) if conf_match else 0.5
            return {
                'predicted_intent': intent_match.group(1),
                'confidence': conf,
                'target_object': obj_match.group(1) if obj_match else 'none',
                'reason': reason_match.group(1) if reason_match else 'Parsed via regex fallback',
            }

        return {
            'predicted_intent': 'unknown',
            'confidence': 0.0,
            'target_object': 'none',
            'reason': f'Parse error: no valid prediction in response'
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

    _FAST_PROMPT_EXECUTING = (
        'Camera watching a 6-DOF SO101 robot arm with gripper on a workspace.\n'
        'Objects on workspace: cotton balls (pink, yellow).\n'
        'Frames oldest→newest (composite panels left→right).\n'
        'Compare GRIPPER position across panels to detect motion.\n'
        'Predict the NEXT 2-second intent. Name the target object by color.\n'
        'ALSO LISTEN for verbal commands in the audio.\n'
        'DECISION RULE:\n'
        '- Gripper at frame edge OR barely moved → "continue"\n'
        '- Gripper moving TOWARD object, claws open → "approach"\n'
        '- Gripper claws CLOSED ON object, arm moving/holding → "gesture"\n'
        '- Gripper moving AWAY after releasing → "withdraw"\n'
        '- Trajectory redirecting to different object → "change_target"\n'
        '- Verbal "stop"/"no"/"wait" or sudden stop → "interrupt"\n'
        '- Verbal command to pick a different object → "new_command"\n'
        '- Unclear → "unknown"\n'
        'Reply ONLY with JSON: '
        '{"predicted_intent":"<class>","confidence":0.0-1.0,'
        '"target_object":"<object description by color or none>",'
        '"reason":"<20 words max describing motion or voice command>"}'
    )

    _FAST_PROMPT_WAITING = (
        'Camera watching a 6-DOF SO101 robot arm (idle) on a workspace.\n'
        'Objects on workspace: cotton balls (pink, yellow).\n'
        'The robot is IDLE and waiting for a voice command.\n'
        'LISTEN carefully for verbal commands in the audio.\n'
        'If you hear a command like "pick up the [color] ball", report it.\n'
        'DECISION RULE:\n'
        '- No voice command heard → "continue"\n'
        '- Voice command to pick an object → "new_command"\n'
        '- Unclear speech → "unknown"\n'
        'Reply ONLY with JSON: '
        '{"predicted_intent":"<class>","confidence":0.0-1.0,'
        '"target_object":"<object mentioned in command or none>",'
        '"reason":"<what you heard or saw>"}'
    )

    def __init__(
        self,
        vllm_url: str = "http://192.168.2.25:8000/v1",
        api_key: str = "vllm-omni",
        model_name: str = "qwen3-30b-a3b",
        system_prompt: Optional[str] = None,
        scene_objects: Optional[list] = None,
    ):
        super().__init__(
            vllm_url=vllm_url,
            api_key=api_key,
            model_name=model_name,
            temperature=0.1,   # near-greedy = fastest sampling
            top_p=0.9,
            max_tokens=80,     # JSON output is ~40-60 tokens with target_object
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

        # Only send last 0.5 seconds of audio (reduces audio tokens)
        self.audio_max_duration = 0.5

        # Robot state for dynamic prompt switching
        self._current_robot_state = ''

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

    def encode_audio_to_base64(
        self,
        audio_window: np.ndarray,
        sample_rate: int = 16000,
        max_duration: Optional[float] = None,
    ) -> str:
        """Override to cap audio at 1 second."""
        return super().encode_audio_to_base64(
            audio_window,
            sample_rate=sample_rate,
            max_duration=max_duration or self.audio_max_duration,
        )

    def build_user_prompt(self, robot_state: Optional[str] = None) -> str:
        """Override to store robot_state for dynamic system prompt switching."""
        # Store robot_state so system_prompt property can read it
        self._current_robot_state = robot_state or ''
        # Dynamically update system prompt based on robot state
        if 'state=waiting' in self._current_robot_state:
            self.system_prompt = self._FAST_PROMPT_WAITING
        else:
            self.system_prompt = self._FAST_PROMPT_EXECUTING
        return super().build_user_prompt(robot_state)


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
