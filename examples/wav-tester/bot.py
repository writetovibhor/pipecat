#!/usr/bin/env python3
#
# WAV Tester Bot — FastAPI + Pipecat pipeline
#
# Serves a web frontend that lets you select WAV files from the input directory,
# stream them as audio input to a Pipecat STT→LLM→TTS pipeline via WebSocket,
# and hear the TTS response back in the browser.
#
# Required env vars (copy .env.example → .env):
#   PIPER_BASE_URL   (default: http://localhost:5000)
#   OLLAMA_BASE_URL  (default: http://localhost:11434)
#   OLLAMA_MODEL     (default: llama3.1:8b)
#   INPUT_DIR        (default: input)
#
# STT: WhisperSTTService (local faster-whisper, no API key needed)
# VAD: SileroVADAnalyzer (local, no API key needed)
#
# Run:
#   source .venv/bin/activate
#   uvicorn bot:app --host 0.0.0.0 --port 7860 --reload
#

import asyncio
import json
import math
import os
import re
import time
import wave
from datetime import datetime

import aiohttp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

try:
    import torch
    from df.enhance import enhance as df_enhance
    from df.enhance import init_df as df_init_df
    _DEEPFILTER_AVAILABLE = True
except ModuleNotFoundError:
    _DEEPFILTER_AVAILABLE = False

try:
    from pipecat.audio.filters.aic_filter import AICFilter
    _AIC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _AIC_AVAILABLE = False

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Pipecat imports
# ---------------------------------------------------------------------------
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    FilterControlFrame,
    FilterEnableFrame,
    FilterUpdateSettingsFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame
from pipecat.utils.time import time_now_iso8601
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.piper.tts import PiperHttpTTSService
from pipecat.services.stt_latency import WHISPER_TTFS_P99
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "input"))
RECORDINGS_DIR = Path(os.environ.get("RECORDINGS_DIR", "recordings"))


def _iter_wav_files(root: Path) -> Iterator[Path]:
    """Walk root recursively following symlinks and yield .wav files."""
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                yield Path(dirpath) / fname

# ---------------------------------------------------------------------------
# DeepFilterNet noise suppression filter
# ---------------------------------------------------------------------------


class DeepFilterNetFilter(BaseAudioFilter):
    """Real-time noise suppression using DeepFilterNet.

    Wraps the `deepfilternet` Python package. Requires 48 kHz internally;
    resamples automatically when the transport runs at a different rate.
    Toggle at runtime via FilterEnableFrame.
    """

    def __init__(
        self,
        atten_lim_db: Optional[float] = None,
        post_filter: bool = False,
        default_model: str = "DeepFilterNet3",
        resampler_quality: str = "QQ",
    ) -> None:
        self._filtering = False  # off by default; UI toggles it on
        self._atten_lim_db = atten_lim_db
        self._post_filter = post_filter
        self._default_model = default_model
        self._resampler_quality = resampler_quality
        self._sample_rate = 0
        self._model_sample_rate = 48000
        self._model = None
        self._df_state = None
        self._model_ready = False
        self._resampler_in = None
        self._resampler_out = None
        # Stats for DeepFilterNet comparison
        self._latency_sum: float = 0.0
        self._latency_count: int = 0
        self._noise_reduction_sum: float = 0.0
        self._noise_reduction_count: int = 0

    async def start(self, sample_rate: int):
        self._sample_rate = sample_rate

        if not _DEEPFILTER_AVAILABLE:
            logger.warning(
                "DeepFilterNet not installed — noise suppression unavailable. "
                "Run: pip install deepfilternet"
            )
            return

        logger.info("DeepFilterNet: loading model (may download on first run) …")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: df_init_df(
                    log_level="WARNING",
                    post_filter=False,
                    default_model=self._default_model,
                ),
            )
            self._model, self._df_state, model_base_dir, *_ = result
            self._model_sample_rate = int(self._df_state.sr())
            self._model_ready = True
            logger.info(
                f"DeepFilterNet ready — model: {model_base_dir}, "
                f"model sr={self._model_sample_rate} Hz, transport sr={sample_rate} Hz"
            )
        except Exception as e:
            logger.error(f"DeepFilterNet init failed: {e}")
            return

        if sample_rate != self._model_sample_rate:
            try:
                from pipecat.audio.resamplers.soxr_stream_resampler import (
                    SOXRStreamAudioResampler,
                )

                self._resampler_in = SOXRStreamAudioResampler(
                    quality=self._resampler_quality
                )
                self._resampler_out = SOXRStreamAudioResampler(
                    quality=self._resampler_quality
                )
                logger.info(
                    f"DeepFilterNet: resampling {sample_rate} ↔ {self._model_sample_rate} Hz"
                )
            except ImportError as e:
                logger.error(f"DeepFilterNet: resampler not available: {e}")
                self._model_ready = False

    async def stop(self):
        self._model = None
        self._df_state = None
        self._model_ready = False
        self._model_sample_rate = 48000
        self._resampler_in = None
        self._resampler_out = None

    def get_stats(self) -> dict:
        """Return average filter stats since last reset_stats() call."""
        avg_latency = (
            round(self._latency_sum / self._latency_count, 2) if self._latency_count else None
        )
        avg_noise_reduction = (
            round(self._noise_reduction_sum / self._noise_reduction_count, 2)
            if self._noise_reduction_count
            else None
        )
        return {
            "enabled": self._filtering,
            "model": self._default_model,
            "post_filter": self._post_filter,
            "atten_lim_db": self._atten_lim_db,
            "avg_latency_ms": avg_latency,
            "avg_noise_reduction_db": avg_noise_reduction,
        }

    def reset_stats(self):
        self._latency_sum = 0.0
        self._latency_count = 0
        self._noise_reduction_sum = 0.0
        self._noise_reduction_count = 0

    def set_atten_lim(self, value: Optional[float]):
        """Set attenuation limit at runtime (no model reload needed)."""
        self._atten_lim_db = value
        logger.info(f"DeepFilterNet: atten_lim_db set to {value}")

    async def process_frame(self, frame):
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable
            logger.info(
                f"DeepFilterNet: {'enabled' if frame.enable else 'disabled'}"
            )
        elif isinstance(frame, FilterUpdateSettingsFrame):
            if "atten_lim_db" in frame.settings:
                self.set_atten_lim(frame.settings["atten_lim_db"])

    def _run_enhance(self, audio_tensor):
        return df_enhance(
            self._model,
            self._df_state,
            audio_tensor,
            atten_lim_db=self._atten_lim_db,
        )

    async def filter(self, audio: bytes) -> bytes:
        if not self._model_ready or not self._filtering:
            return audio

        # Resample to model sample rate if needed
        in_audio = audio
        if self._sample_rate != self._model_sample_rate and self._resampler_in:
            in_audio = await self._resampler_in.resample(
                audio, self._sample_rate, self._model_sample_rate
            )

        if not in_audio:
            return b""

        # int16 bytes → float32 tensor [1, T]
        audio_np = np.frombuffer(in_audio, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        try:
            enhanced = await loop.run_in_executor(None, self._run_enhance, audio_tensor)
        except Exception as e:
            logger.warning(f"DeepFilterNet enhance error: {e}")
            return audio
        self._latency_sum += (time.monotonic() - t0) * 1000
        self._latency_count += 1

        # float32 tensor → int16 bytes
        out_np = np.clip(enhanced.squeeze().detach().cpu().numpy(), -1.0, 1.0)
        out = (out_np * 32767).astype(np.int16).tobytes()

        # Track noise reduction (dB difference between input and output RMS)
        rms_in = float(np.sqrt(np.mean(audio_np**2))) if len(audio_np) > 0 else 0.0
        rms_out = float(np.sqrt(np.mean(out_np**2))) if len(out_np) > 0 else 0.0
        if rms_in > 1e-9 and rms_out > 1e-9:
            self._noise_reduction_sum += 20 * math.log10(rms_in / rms_out)
            self._noise_reduction_count += 1

        # Resample back if needed
        if self._sample_rate != self._model_sample_rate and self._resampler_out:
            return await self._resampler_out.resample(out, self._model_sample_rate, self._sample_rate)

        return out


# ---------------------------------------------------------------------------
# Quail (ai|coustics) speech enhancement filter
# ---------------------------------------------------------------------------


class QuailFilter(BaseAudioFilter):
    """Speech enhancement for ASR using the ai|coustics Quail SDK (AICFilter).

    Wraps pipecat's built-in AICFilter. Requires the `aic_sdk` package and a
    valid license key (AIC_LICENSE_KEY env var or passed at runtime via
    filter_select message). Toggle at runtime via FilterEnableFrame.
    """

    def __init__(
        self,
        license_key: str = "",
        model_id: str = "quail-vf-l-16khz",
    ) -> None:
        self._license_key = license_key
        self._model_id = model_id
        self._inner: Optional[Any] = None  # AICFilter instance
        self._filtering = False
        self._sample_rate = 0
        self._latency_sum: float = 0.0
        self._latency_count: int = 0

    async def start(self, sample_rate: int):
        self._sample_rate = sample_rate
        logger.info(f"QuailFilter: starting (aic_available={_AIC_AVAILABLE}, key={'set' if self._license_key else 'not set'}, model={self._model_id})")
        if _AIC_AVAILABLE and self._license_key:
            await self._init()
        # If the filter was enabled before start() ran (race condition with DeepFilterNet loading),
        # perform the deferred initialisation now.
        if self._filtering and self._inner is None and _AIC_AVAILABLE and self._license_key:
            logger.info("QuailFilter: performing deferred init (was enabled before start completed)")
            await self._init()

    async def _init(self):
        try:
            if self._inner is not None:
                await self._inner.stop()
            self._inner = AICFilter(
                license_key=self._license_key,
                model_id=self._model_id,
            )
            await self._inner.start(self._sample_rate)
            logger.info(f"QuailFilter ready — model: {self._model_id}")
        except Exception as e:
            logger.error(f"QuailFilter init failed: {e}")
            self._inner = None

    async def stop(self):
        if self._inner:
            await self._inner.stop()
        self._inner = None
        self._filtering = False

    async def process_frame(self, frame):
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable
            if frame.enable and self._inner is None and _AIC_AVAILABLE and self._license_key:
                if self._sample_rate == 0:
                    logger.warning("QuailFilter: enabled before start() completed — will init when start() finishes")
                else:
                    await self._init()
            status = "ready" if (self._inner and self._inner._aic_ready) else "NOT READY"
            logger.info(f"QuailFilter: {'enabled' if frame.enable else 'disabled'} (inner={status})")
            if frame.enable and (self._inner is None or not self._inner._aic_ready):
                if not self._license_key:
                    logger.error("QuailFilter: AIC_LICENSE_KEY is not set — filtering will be a no-op!")
                elif not _AIC_AVAILABLE:
                    logger.error("QuailFilter: aic_sdk not installed — filtering will be a no-op!")
                else:
                    logger.error("QuailFilter: filter not ready — filtering will be a no-op!")
        elif isinstance(frame, FilterUpdateSettingsFrame):
            updated = False
            if "quail_license_key" in frame.settings and frame.settings["quail_license_key"]:
                # Only overwrite the key if a non-empty one was explicitly provided.
                # Empty means "keep using the server-side key (from env var)".
                self._license_key = frame.settings["quail_license_key"]
                updated = True
            if "quail_model_id" in frame.settings:
                self._model_id = frame.settings["quail_model_id"]
                updated = True
            if updated and self._filtering and _AIC_AVAILABLE:
                await self._init()

    def get_stats(self) -> dict:
        avg_latency = (
            round(self._latency_sum / self._latency_count, 2) if self._latency_count else None
        )
        return {
            "enabled": self._filtering,
            "model": self._model_id,
            "avg_latency_ms": avg_latency,
        }

    def reset_stats(self):
        self._latency_sum = 0.0
        self._latency_count = 0

    async def filter(self, audio: bytes) -> bytes:
        if not self._filtering:
            return audio
        if self._inner is None:
            logger.warning(
                "QuailFilter: filtering requested but filter is not initialised "
                f"(aic_available={_AIC_AVAILABLE}, key={'set' if self._license_key else 'NOT SET'}). "
                "Audio passed through unchanged."
            )
            return audio
        if not self._inner._aic_ready:
            logger.warning("QuailFilter: AICFilter not ready (init may have failed). Audio passed through unchanged.")
            return audio
        t0 = time.monotonic()
        try:
            result = await self._inner.filter(audio)
        except Exception as e:
            logger.warning(f"QuailFilter enhance error: {e}")
            return audio
        self._latency_sum += (time.monotonic() - t0) * 1000
        self._latency_count += 1
        return result


# ---------------------------------------------------------------------------
# Filter mux — routes audio to None / DeepFilterNet / Quail
# ---------------------------------------------------------------------------


class FilterMux(BaseAudioFilter):
    """Multiplexes audio through one of several noise-suppression backends.

    The active backend is selected at runtime via FilterUpdateSettingsFrame
    carrying ``{"active_filter": "none"|"deepfilter"|"quail"}``.
    All sub-filters are initialised at start; only the active one processes audio.
    """

    def __init__(
        self,
        deepfilter: Optional[DeepFilterNetFilter] = None,
        quail: Optional[QuailFilter] = None,
        recorder: Optional["SessionRecorder"] = None,
        on_status=None,
    ) -> None:
        self._filters: dict[str, Optional[BaseAudioFilter]] = {
            "deepfilter": deepfilter,
            "quail": quail,
        }
        self._active = "none"
        self._recorder = recorder
        self._on_status = on_status  # async (filter_name, state, detail) callback

    async def _emit(self, filter_name: str, state: str, detail: str = "") -> None:
        if self._on_status:
            try:
                await self._on_status(filter_name, state, detail)
            except Exception as e:
                logger.debug(f"FilterMux status callback error: {e}")

    async def _start_one(self, name: str, f: BaseAudioFilter, sample_rate: int) -> None:
        await self._emit(name, "loading")
        try:
            await f.start(sample_rate)
            # Determine effective ready state after start().
            if name == "deepfilter":
                ready = getattr(f, "_model_ready", True)
                err = "" if ready else "model load failed"
            elif name == "quail":
                inner = getattr(f, "_inner", None)
                if inner is None and not getattr(f, "_license_key", ""):
                    ready, err = False, "AIC_LICENSE_KEY not set"
                elif inner is None:
                    ready, err = False, "init failed (check logs)"
                else:
                    ready = getattr(inner, "_aic_ready", False)
                    err = "" if ready else "AICFilter init failed"
            else:
                ready, err = True, ""
            await self._emit(name, "ready" if ready else "error", err)
        except Exception as exc:
            await self._emit(name, "error", str(exc))

    async def start(self, sample_rate: int):
        # Start all filters in parallel so a slow-loading filter (e.g. DeepFilterNet
        # downloading/loading its model) doesn't block QuailFilter from starting.
        coros = []
        for name, f in self._filters.items():
            if f is None:
                continue
            coros.append(self._start_one(name, f, sample_rate))
        await asyncio.gather(*coros)

    async def stop(self):
        for f in self._filters.values():
            if f is not None:
                await f.stop()

    def get_active_name(self) -> str:
        return self._active

    def get_deepfilter(self) -> Optional[DeepFilterNetFilter]:
        return self._filters.get("deepfilter")  # type: ignore[return-value]

    def get_active_filter(self) -> Optional[BaseAudioFilter]:
        return self._filters.get(self._active)

    async def process_frame(self, frame):
        if isinstance(frame, FilterUpdateSettingsFrame):
            # Forward settings to sub-filters first so they update config before enable/disable.
            for f in self._filters.values():
                if f is not None:
                    await f.process_frame(frame)
            # Handle active_filter switch.
            if "active_filter" in frame.settings:
                new_active = frame.settings["active_filter"]
                if new_active in ("none", "deepfilter", "quail"):
                    old_active = self._active
                    self._active = new_active
                    if old_active != "none" and old_active != new_active:
                        old_f = self._filters.get(old_active)
                        if old_f:
                            await old_f.process_frame(FilterEnableFrame(enable=False))
                        await self._emit(old_active, "inactive")
                    if new_active != "none":
                        new_f = self._filters.get(new_active)
                        if new_f:
                            await new_f.process_frame(FilterEnableFrame(enable=True))
                        await self._emit(new_active, "active")
                    elif old_active != "none":
                        # switched back to "none" — already emitted inactive above
                        pass
                    logger.info(f"FilterMux: switched to '{new_active}'")
        elif isinstance(frame, FilterEnableFrame):
            active = self.get_active_filter()
            if active:
                await active.process_frame(frame)
        else:
            for f in self._filters.values():
                if f is not None:
                    await f.process_frame(frame)

    def get_stats(self) -> dict:
        active = self.get_active_filter()
        base: dict = {"active": self._active}
        if active and hasattr(active, "get_stats"):
            base.update(active.get_stats())  # type: ignore[union-attr]
        return base

    def reset_stats(self):
        for f in self._filters.values():
            if f and hasattr(f, "reset_stats"):
                f.reset_stats()  # type: ignore[union-attr]

    async def filter(self, audio: bytes) -> bytes:
        if self._recorder:
            self._recorder.record_original(audio)
        active = self.get_active_filter()
        result = await active.filter(audio) if active else audio
        if self._recorder:
            self._recorder.record_filtered(result)
        return result


# ---------------------------------------------------------------------------
# Simple binary audio serializer
#
# Client → Server: raw PCM Int16 bytes @ 16 kHz mono
# Server → Client (binary):  raw PCM Int16 bytes @ 16 kHz mono  (TTS audio)
# Server → Client (text):    JSON status/transcription messages
# ---------------------------------------------------------------------------


class RawAudioSerializer(FrameSerializer):
    """Minimal serializer: raw PCM int16 in both directions + JSON text messages."""

    async def serialize(self, frame: Frame) -> bytes | str | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio  # raw int16 PCM bytes
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if not self.should_ignore_frame(frame):
                return json.dumps(frame.message)  # JSON text message
        return None

    async def deserialize(self, data: bytes | str) -> Frame | None:
        if isinstance(data, bytes) and len(data) > 0:
            return InputAudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
        if isinstance(data, str):
            try:
                msg = json.loads(data)
                mtype = msg.get("type")
                if mtype == "filter_select":
                    # { type: "filter_select", filter: "none"|"deepfilter"|"quail",
                    #   license_key?: str, model_id?: str }
                    settings: dict = {"active_filter": msg.get("filter", "none")}
                    if "license_key" in msg:
                        settings["quail_license_key"] = msg["license_key"]
                    if "model_id" in msg:
                        settings["quail_model_id"] = msg["model_id"]
                    return FilterUpdateSettingsFrame(settings=settings)
                if mtype == "deepfilter_atten_lim":
                    # value=null means "no limit"; otherwise float dB
                    raw = msg.get("value")
                    atten = float(raw) if raw is not None else None
                    return FilterUpdateSettingsFrame(settings={"atten_lim_db": atten})
                if mtype == "quail_update":
                    # Re-configure Quail while it is active.
                    settings = {"active_filter": "quail"}
                    if "license_key" in msg:
                        settings["quail_license_key"] = msg["license_key"]
                    if "model_id" in msg:
                        settings["quail_model_id"] = msg["model_id"]
                    return FilterUpdateSettingsFrame(settings=settings)
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# Status notifier — intercepts pipeline events and forwards them to the
# browser as OutputTransportMessageFrame (→ JSON text over WebSocket).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Metrics data model
# ---------------------------------------------------------------------------


@dataclass
class TurnData:
    """Timing checkpoints for a single user→bot conversation turn.

    Written by both MetricsTracker (upstream events) and BotTextNotifier
    (downstream LLM events) via a shared SessionMetrics object.
    """

    turn_id: int
    t_user_started: Optional[float] = None
    t_user_stopped: Optional[float] = None
    t_transcription: Optional[float] = None
    t_llm_first_token: Optional[float] = None
    t_llm_complete: Optional[float] = None
    t_bot_started: Optional[float] = None
    t_bot_stopped: Optional[float] = None
    user_text: str = ""
    bot_text: str = ""
    interruptions: int = 0
    interrupted_bot_text: Optional[str] = None      # partial bot text spoken before interruption
    bot_speaking_ms_at_interruption: Optional[float] = None  # ms bot had been speaking when interrupted
    interruption_session_elapsed_ms: Optional[float] = None  # ms since stream start when interruption fired
    stt_confidence: Optional[float] = None    # avg word confidence 0–1 (provider-specific)
    speaking_rate_wpm: Optional[float] = None # words per minute (provider-specific)
    bot_word_count: int = 0                   # word count of bot response
    stt_latency_ms: Optional[float] = None    # wall-clock time from user stopped to transcript

    def _diff(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        return round((b - a) * 1000, 1) if a is not None and b is not None else None

    @property
    def stt_ms(self) -> Optional[float]:
        return self.stt_latency_ms

    @property
    def llm_ms(self) -> Optional[float]:
        return self._diff(self.t_transcription, self.t_llm_first_token)

    @property
    def tts_ms(self) -> Optional[float]:
        return self._diff(self.t_llm_complete, self.t_bot_started)

    @property
    def rtt_ms(self) -> Optional[float]:
        return self._diff(self.t_user_stopped, self.t_bot_started)

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "user_text": self.user_text,
            "bot_text": self.bot_text,
            "stt_ms": self.stt_ms,
            "llm_ms": self.llm_ms,
            "tts_ms": self.tts_ms,
            "rtt_ms": self.rtt_ms,
            "interruptions": self.interruptions,
            "interrupted_bot_text": self.interrupted_bot_text,
            "bot_speaking_ms_at_interruption": self.bot_speaking_ms_at_interruption,
            "interruption_session_elapsed_ms": self.interruption_session_elapsed_ms,
            "stt_confidence": self.stt_confidence,
            "speaking_rate_wpm": self.speaking_rate_wpm,
            "bot_word_count": self.bot_word_count,
        }


class SessionMetrics:
    """Session-level metrics container shared between MetricsTracker and BotTextNotifier."""

    def __init__(self):
        self.current: TurnData = TurnData(turn_id=0)
        self.history: list[TurnData] = []
        self.total_interruptions: int = 0
        self.bot_speaking: bool = False
        self._next_id: int = 1
        self.t_stream_start: Optional[float] = None  # wall-clock time of first audio frame to STT

    def start_turn(self) -> TurnData:
        self.current = TurnData(turn_id=self._next_id)
        self._next_id += 1
        return self.current

    def finish_turn(self) -> TurnData:
        self.history.append(self.current)
        return self.current

    def _avg(self, attr: str, turns: Optional[list] = None) -> Optional[float]:
        source = turns if turns is not None else self.history
        vals = [getattr(t, attr) for t in source if getattr(t, attr) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    def _percentile(self, attr: str, pct: float, turns: Optional[list] = None) -> Optional[float]:
        source = turns if turns is not None else self.history
        vals = sorted(v for t in source if (v := getattr(t, attr)) is not None)
        if not vals:
            return None
        idx = (len(vals) - 1) * pct / 100
        lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
        return round(vals[lo] + (vals[hi] - vals[lo]) * (idx - lo), 1)

    def _all_turns(self) -> list:
        """Completed turns + current turn, for live aggregate computation."""
        return self.history + [self.current]

    def session_dict(self) -> dict:
        turns = self._all_turns()
        return {
            "turn_count": len(self.history),
            "total_interruptions": self.total_interruptions,
            # Latency aggregates: mean, p50, p99
            "rtt_mean": self._avg("rtt_ms", turns),
            "rtt_p50":  self._percentile("rtt_ms", 50, turns),
            "rtt_p99":  self._percentile("rtt_ms", 99, turns),
            "stt_mean": self._avg("stt_ms", turns),
            "stt_p50":  self._percentile("stt_ms", 50, turns),
            "stt_p99":  self._percentile("stt_ms", 99, turns),
            "llm_mean": self._avg("llm_ms", turns),
            "llm_p50":  self._percentile("llm_ms", 50, turns),
            "llm_p99":  self._percentile("llm_ms", 99, turns),
            "tts_mean": self._avg("tts_ms", turns),
            "tts_p50":  self._percentile("tts_ms", 50, turns),
            "tts_p99":  self._percentile("tts_ms", 99, turns),
            # Other session aggregates
            "avg_stt_confidence": self._avg("stt_confidence", turns),
            "avg_speaking_rate_wpm": self._avg("speaking_rate_wpm", turns),
            "avg_bot_word_count": self._avg("bot_word_count", turns),
        }


# ---------------------------------------------------------------------------
# LLM quality judge — async Ollama call, runs outside the pipeline
# ---------------------------------------------------------------------------


class QualityJudge:
    """Scores a full conversation session using a local Ollama model as an impartial evaluator."""

    ISSUES = [
        "none", "context_loss", "repetitive", "wrong_entity",
        "unhelpful", "irrelevant", "escalation_failure", "too_long", "error",
    ]

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def _build_session_prompt(self, history: list) -> str:
        # Use concatenation — NOT str.format() — so braces in user/bot text
        # (e.g. booking codes like {WJ4721X}) never cause a KeyError.
        parts = [
            "You are evaluating an airline customer service call handled by an AI bot. "
            "Review the full conversation transcript and rate the bot's overall performance.\n"
            "Return JSON only — no explanation, no markdown.\n"
            "JSON schema:\n"
            '{"overall_score": <1-5>, '
            '"overall_issue": "<none|context_loss|repetitive|wrong_entity|unhelpful|irrelevant|escalation_failure|too_long>", '
            '"summary": "<1-2 sentences on the main finding>", '
            '"flagged_turns": [{"turn_id": <n>, "issue": "<type>", "note": "<brief note>"}]}\n'
            "\n[Conversation transcript]",
        ]
        # Cap at 15 turns to stay within context window
        for t in history[-15:]:
            if t.user_text or t.bot_text:
                parts.append("Turn " + str(t.turn_id) + ":")
                if t.user_text:
                    parts.append("  Customer: " + t.user_text)
                if t.bot_text:
                    parts.append("  Agent: " + t.bot_text)
        return "\n".join(parts)

    async def score_session(self, history: list) -> dict:
        """Evaluate the full session transcript.

        Returns dict with overall_score, overall_issue, summary, flagged_turns.
        Returns a zeroed-out error dict on failure.
        """
        logger.info(f"QualityJudge: scoring session with {len(history)} turns")
        fallback = {"overall_score": 0, "overall_issue": "error", "summary": "", "flagged_turns": []}
        if not history:
            return fallback
        try:
            prompt = self._build_session_prompt(history)
            payload = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            response_text = data.get("response", "")
            m = re.search(r"\{[\s\S]*\}", response_text)
            if m:
                parsed = json.loads(m.group())
                score = int(parsed.get("overall_score", 3))
                issue = str(parsed.get("overall_issue", "none"))
                summary = str(parsed.get("summary", ""))
                flagged = parsed.get("flagged_turns", [])
                if not isinstance(flagged, list):
                    flagged = []
                logger.info(f"QualityJudge: overall_score={score} issue={issue}")
                return {
                    "overall_score": score,
                    "overall_issue": issue,
                    "summary": summary,
                    "flagged_turns": flagged,
                }
            logger.warning(f"QualityJudge: no JSON found in response: {response_text!r}")
        except Exception as e:
            logger.warning(f"QualityJudge session error: {e}")
        return fallback


# ---------------------------------------------------------------------------
# WhisperSTT subclass that extracts word confidence from faster-whisper segments
# ---------------------------------------------------------------------------


class WhisperConfidenceSTTService(WhisperSTTService):
    """Extends WhisperSTTService to report per-segment confidence and STT latency.

    STT latency is measured from the moment VADUserStoppedSpeakingFrame is
    received (speech segment end) to the moment the TranscriptionFrame is
    yielded (Whisper inference complete). This is more reliable than measuring
    across async pipeline paths.

    faster-whisper returns avg_logprob per segment. We convert it to a 0–1
    confidence score using exp(avg_logprob) and average across accepted segments.
    """

    def __init__(self, on_confidence=None, on_stt_latency=None, **kwargs):
        super().__init__(**kwargs)
        self._on_confidence = on_confidence
        self._on_stt_latency = on_stt_latency
        self._segment_stop_time: Optional[float] = None

    async def _handle_user_stopped_speaking(self, frame):
        """Record wall-clock time when VAD signals end of speech, then run Whisper."""
        self._segment_stop_time = time.monotonic()
        await super()._handle_user_stopped_speaking(frame)

    async def run_stt(self, audio: bytes):
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        whisper_lang = self.language_to_service_language(self._settings["language"])
        segments, _ = await asyncio.to_thread(
            self._model.transcribe, audio_float, language=whisper_lang
        )

        accepted = []
        for segment in segments:
            if segment.no_speech_prob < self._no_speech_prob:
                accepted.append(segment)

        await self.stop_processing_metrics()

        if accepted:
            text = "".join(s.text for s in accepted).strip()
            if text:
                # avg_logprob is typically in (-5, 0]; exp maps it to (0, 1]
                confidence = sum(math.exp(max(s.avg_logprob, -5.0)) for s in accepted) / len(
                    accepted
                )
                if self._on_confidence:
                    await self._on_confidence(round(min(confidence, 1.0), 3))
                if self._on_stt_latency and self._segment_stop_time is not None:
                    latency_ms = round((time.monotonic() - self._segment_stop_time) * 1000, 1)
                    await self._on_stt_latency(latency_ms)
                await self._handle_transcription(text, True, self._settings["language"])
                logger.debug(f"Transcription: [{text}] confidence={confidence:.3f}")
                yield TranscriptionFrame(
                    text, self._user_id, time_now_iso8601(), self._settings["language"]
                )


# ---------------------------------------------------------------------------
# Pipeline processors
# ---------------------------------------------------------------------------


class AudioStreamTracker(FrameProcessor):
    """Records the wall-clock time of the first audio frame sent to STT.

    Must be placed immediately before the STT service in the pipeline so that
    t_stream_start aligns with the origin of Deepgram's word timestamps.
    """

    def __init__(self, session: SessionMetrics):
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, InputAudioRawFrame)
            and self._session.t_stream_start is None
        ):
            self._session.t_stream_start = time.monotonic()


class BotTextNotifier(FrameProcessor):
    """Accumulates LLM TextFrames; records LLM timing into the shared TurnData."""

    def __init__(self, session: SessionMetrics):
        super().__init__()
        self._session = session
        self._chunks: list[str] = []
        self._first_token_seen = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            return

        if isinstance(frame, TextFrame) and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)) and frame.text:
            if not self._first_token_seen:
                self._session.current.t_llm_first_token = time.monotonic()
                self._first_token_seen = True
            self._chunks.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._session.current.t_llm_complete = time.monotonic()
            text = "".join(self._chunks).strip()
            self._chunks = []
            self._first_token_seen = False
            if text:
                self._session.current.bot_text = text
                self._session.current.bot_word_count = len(text.split())
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(message={"type": "bot_text", "text": text}),
                )

    async def _start_interruption(self):
        """Reset accumulation on interruption so next turn starts clean."""
        partial = "".join(self._chunks).strip()
        if partial:
            self._session.current.interrupted_bot_text = partial
        self._chunks = []
        self._first_token_seen = False
        await super()._start_interruption()


class MetricsTracker(FrameProcessor):
    """Tracks turn timing and emits metrics events over the WebSocket."""

    def __init__(self, session: SessionMetrics, audio_filter=None):
        super().__init__()
        self._s = session
        self._audio_filter = audio_filter

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        now = time.monotonic()

        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, UserStartedSpeakingFrame):
                was_bot_speaking = self._s.bot_speaking
                bot_ms = (
                    round((now - self._s.current.t_bot_started) * 1000, 1)
                    if was_bot_speaking and self._s.current.t_bot_started is not None
                    else None
                )
                self._s.start_turn()
                self._s.current.t_user_started = now
                if was_bot_speaking:
                    self._s.current.interruptions += 1
                    self._s.total_interruptions += 1
                    self._s.current.bot_speaking_ms_at_interruption = bot_ms
                    session_elapsed_ms = (
                        round((now - self._s.t_stream_start) * 1000, 1)
                        if self._s.t_stream_start is not None
                        else None
                    )
                    self._s.current.interruption_session_elapsed_ms = session_elapsed_ms
                    await self._send({
                        "type": "interruption",
                        "bot_speaking_ms": bot_ms,
                        "session_elapsed_ms": session_elapsed_ms,
                    })
                await self._send({"type": "user_speaking", "value": True})

            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._s.current.t_user_stopped = now
                await self._send({"type": "user_speaking", "value": False})

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TranscriptionFrame):
                if not self._s.current.t_transcription:
                    self._s.current.t_transcription = now
                    self._s.current.user_text = frame.text
                    # Note: stt_latency_ms is set directly by WhisperConfidenceSTTService
                    # callback (measured from VAD speech-stop to Whisper completing),
                    # which is more reliable than computing it across async pipeline paths.
                    # Speaking rate: word count / VAD duration in minutes
                    t_start = self._s.current.t_user_started
                    t_stop = self._s.current.t_user_stopped
                    if t_start is not None and t_stop is not None and t_stop > t_start:
                        words = len(frame.text.split())
                        duration_min = (t_stop - t_start) / 60.0
                        if duration_min > 0:
                            self._s.current.speaking_rate_wpm = round(words / duration_min, 1)
                await self._send({"type": "transcription", "text": frame.text, "user_id": frame.user_id})

        elif direction == FrameDirection.UPSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._s.bot_speaking = True
                self._s.current.t_bot_started = now
                # All latencies are now computable — emit metrics update
                msg = {
                    "type": "metrics_update",
                    "turn": self._s.current.to_dict(),
                    "session": self._s.session_dict(),
                }
                if self._audio_filter is not None:
                    msg["filter"] = self._audio_filter.get_stats()
                await self._send(msg)
                await self._send({"type": "bot_speaking", "value": True})

            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._s.bot_speaking = False
                self._s.current.t_bot_stopped = now
                self._s.finish_turn()
                await self._send({"type": "bot_speaking", "value": False})

    async def _send(self, msg: dict[str, Any]):
        await self.push_frame(OutputTransportMessageUrgentFrame(message=msg))


# ---------------------------------------------------------------------------
# Session recorder — captures pre-filter, post-filter, and bot audio to WAVs
# ---------------------------------------------------------------------------


class SessionRecorder:
    """Records three audio streams for a session to WAV files.

    Files are written on ``save()`` with the session's start timestamp:
      ``{YYYYMMDD_HHMMSS}_original_input.wav``   — raw input before filter (opt-in)
      ``{YYYYMMDD_HHMMSS}_filtered_input.wav``   — input after active filter
      ``{YYYYMMDD_HHMMSS}_bot_output.wav``        — TTS output sent to client

    All files share the same start time (WebSocket connect). The period before
    the first audio arrives is padded with silence so timestamps are aligned.
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 16000,
        record_original: bool = False,
    ) -> None:
        self._dir = output_dir
        self._sr = sample_rate
        self._record_original = record_original
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._t_start = time.monotonic()

        self._t_first_input: Optional[float] = None
        self._t_first_output: Optional[float] = None

        self._original_chunks: list[bytes] = []
        self._filtered_chunks: list[bytes] = []
        self._output_chunks: list[bytes] = []

    # ── Recording callbacks ──────────────────────────────────────────────────

    def record_original(self, audio: bytes) -> None:
        """Raw input before the audio filter."""
        if not self._record_original:
            return
        if self._t_first_input is None:
            self._t_first_input = time.monotonic()
        self._original_chunks.append(audio)

    def record_filtered(self, audio: bytes) -> None:
        """Input after the audio filter (what STT sees)."""
        if self._t_first_input is None:
            self._t_first_input = time.monotonic()
        self._filtered_chunks.append(audio)

    def record_output(self, audio: bytes) -> None:
        """Bot TTS audio sent to the client."""
        if self._t_first_output is None:
            self._t_first_output = time.monotonic()
        self._output_chunks.append(audio)

    # ── Save ────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write WAV files to disk. Call once on disconnect."""
        if not (self._filtered_chunks or self._output_chunks):
            logger.info("SessionRecorder: no audio captured — skipping save")
            return

        self._dir.mkdir(parents=True, exist_ok=True)

        input_silence_s = max(
            0.0,
            (self._t_first_input - self._t_start) if self._t_first_input is not None else 0.0,
        )
        output_silence_s = max(
            0.0,
            (self._t_first_output - self._t_start) if self._t_first_output is not None else 0.0,
        )

        if self._filtered_chunks:
            self._write_wav(
                self._dir / f"{self._timestamp}_filtered_input.wav",
                self._silence(input_silence_s) + b"".join(self._filtered_chunks),
            )

        if self._record_original and self._original_chunks:
            self._write_wav(
                self._dir / f"{self._timestamp}_original_input.wav",
                self._silence(input_silence_s) + b"".join(self._original_chunks),
            )

        if self._output_chunks:
            self._write_wav(
                self._dir / f"{self._timestamp}_bot_output.wav",
                self._silence(output_silence_s) + b"".join(self._output_chunks),
            )

    def _silence(self, duration_s: float) -> bytes:
        return b"\x00" * (int(duration_s * self._sr) * 2)  # int16 = 2 bytes/sample

    def _write_wav(self, path: Path, pcm: bytes) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sr)
            wf.writeframes(pcm)
        logger.info(f"Saved recording: {path} ({len(pcm) // 2 / self._sr:.1f}s)")


# ---------------------------------------------------------------------------
# Filter router — intercepts FilterControlFrame flowing downstream from the
# WebSocket receive task and routes them to the filter mux.
#
# Background: FastAPIWebsocketInputTransport._receive_messages() calls
# push_frame(frame) (downstream) for non-audio frames, which bypasses the
# transport's own process_frame() handler for FilterControlFrame. This
# processor, placed right after transport.input(), intercepts those frames
# and forwards them to the filter mux directly.
# ---------------------------------------------------------------------------


class FilterRouter(FrameProcessor):
    def __init__(self, filter_mux: "FilterMux"):
        super().__init__()
        self._filter_mux = filter_mux

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, FilterControlFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._filter_mux.process_frame(frame)
            return  # consumed — don't push further downstream
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Recording processor — sits in the pipeline and captures bot TTS output
# ---------------------------------------------------------------------------


class RecordingProcessor(FrameProcessor):
    """Captures OutputAudioRawFrame frames and forwards them to SessionRecorder."""

    def __init__(self, recorder: SessionRecorder):
        super().__init__()
        self._recorder = recorder

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            self._recorder.record_output(frame.audio)
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Pipecat WAV Tester")


@app.get("/")
async def index():
    return FileResponse("index.html")


@app.get("/files")
async def list_files():
    """Return a list of all .wav files found recursively under INPUT_DIR."""
    if not INPUT_DIR.exists():
        return JSONResponse(
            {
                "files": [],
                "error": f"INPUT_DIR not found: {INPUT_DIR.resolve()}",
            }
        )
    files = []
    for f in sorted(_iter_wav_files(INPUT_DIR)):
        try:
            rel = f.relative_to(INPUT_DIR)
            files.append(
                {
                    "name": f.name,
                    "path": str(rel),
                    "url": f"/audio/{rel}",
                    "dir": str(rel.parent) if rel.parent != Path(".") else "",
                    "size": f.stat().st_size,
                }
            )
        except Exception:
            pass
    return JSONResponse({"files": files})


@app.get("/audio/{path:path}")
async def serve_audio(path: str):
    """Serve a WAV file from INPUT_DIR."""
    # Security: reject any path containing ".." to prevent directory traversal.
    # We intentionally do NOT resolve symlinks here — the input directory uses
    # symlinks and we want those to work normally.
    if ".." in Path(path).parts:
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    file_path = INPUT_DIR / path
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(file_path), media_type="audio/wav")


# ---------------------------------------------------------------------------
# WebSocket endpoint — one pipeline per connection
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, record_original: bool = False):
    await websocket.accept()
    logger.info(f"WebSocket client connected (record_original={record_original})")

    # --- session recorder ---
    recorder = SessionRecorder(
        output_dir=RECORDINGS_DIR,
        sample_rate=16000,
        record_original=record_original,
    )

    # --- audio filters (switched from UI via filter_select message) ---
    async def send_filter_status(filter_name: str, state: str, detail: str = "") -> None:
        try:
            await websocket.send_text(json.dumps(
                {"type": "filter_status", "filter": filter_name, "state": state, "detail": detail}
            ))
        except Exception:
            pass  # client disconnected

    # Emit disabled state immediately for filters whose packages are not installed.
    if not _DEEPFILTER_AVAILABLE:
        await send_filter_status("deepfilter", "disabled", "deepfilternet not installed")
    if not _AIC_AVAILABLE:
        await send_filter_status("quail", "disabled", "aic_sdk not installed")

    deep_filter = DeepFilterNetFilter(
        post_filter=False,
        default_model="DeepFilterNet3",
        atten_lim_db=float(v) if (v := os.getenv("DEEPFILTER_ATTEN_LIM_DB")) else None,
    )
    quail_filter = QuailFilter(
        license_key=os.getenv("AIC_LICENSE_KEY", ""),
        model_id=os.getenv("QUAIL_MODEL_ID", "quail-vf-l-16khz"),
    )
    filter_mux = FilterMux(
        deepfilter=deep_filter,
        quail=quail_filter,
        recorder=recorder,
        on_status=send_filter_status,
    )

    # --- transport ---
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_in_channels=1,
            audio_out_channels=1,
            audio_in_filter=filter_mux,
            serializer=RawAudioSerializer(),
        ),
    )

    # --- session metrics (created early so the STT confidence callback can reference it) ---
    session = SessionMetrics()
    judge = QualityJudge(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    # --- services ---
    # WhisperConfidenceSTTService measures latency from VAD speech-stop to transcript,
    # and extracts per-segment avg_logprob → confidence 0–1.
    async def _on_whisper_confidence(confidence: float):
        session.current.stt_confidence = confidence

    async def _on_whisper_stt_latency(latency_ms: float):
        session.current.stt_latency_ms = latency_ms

    stt = WhisperConfidenceSTTService(
        on_confidence=_on_whisper_confidence,
        on_stt_latency=_on_whisper_stt_latency,
        model=Model.DISTIL_MEDIUM_EN,
        device="auto",
        ttfs_p99_latency=WHISPER_TTFS_P99,
    )

    llm = OLLamaLLMService(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )

    piper_session = aiohttp.ClientSession()
    tts = PiperHttpTTSService(
        base_url=os.getenv("PIPER_BASE_URL", "http://localhost:5000"),
        aiohttp_session=piper_session,
        voice_id="en_US-hfc_female-medium",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an airline customer care agent for WestJet. "
                "Your job is to handle passenger inquiries over the phone in a realistic, natural way.\n\n"

                "IMPORTANT BEHAVIORAL RULES:\n"
                "- Respond exactly as a real airline IVR/agent system would — with authentic airline terminology, "
                "confirmation codes, flight numbers, gate info, timestamps, policy references, etc.\n"
                "- For every interaction, INVENT plausible mock data on the fly: "
                "random 6-char booking references (e.g. WJ4721X), realistic flight numbers (WJ101–WJ999), "
                "airports (use real IATA codes), dates/times, seat numbers, baggage allowances, fees, etc.\n"
                "- Dynamically pick a scenario based on what the caller says and randomly vary the outcome "
                "(success, failure, partial info, error, policy block, hold, escalation, etc.).\n\n"

                "SCENARIO BANK — pick and respond realistically for whichever matches the caller's intent:\n"
                "• Flight status: on-time / delayed (give reason: ATC, weather, crew, aircraft) / cancelled / diverted\n"
                "• Booking lookup: found with full itinerary / not found / multiple matches needing verification\n"
                "• Check-in: successful with boarding pass details / window closed / seat already taken / upgrade offered\n"
                "• Seat change: confirmed / unavailable / chargeable upgrade / waitlisted\n"
                "• Baggage: allowance info / overweight fee quoted / lost bag claim opened / delayed bag traced\n"
                "• Cancellation/refund: full refund / credit voucher only / non-refundable fare / 24-hr grace period\n"
                "• Rebooking: new flight offered with options / sold out / next available in 2 days / standby offered\n"
                "• Special requests: meal preference confirmed / wheelchair assistance booked / unaccompanied minor flagged\n"
                "• Loyalty/miles: balance read out / tier status / points redemption success or shortfall\n"
                "• Payment: charge confirmed / card declined / price difference quoted\n"
                "• System errors: booking system down / retry after hold / escalate to supervisor\n\n"

                "VOICE STYLE:\n"
                "- Greet with your name and airline on first turn (e.g. 'Thank you for calling WestJet, "
                "this is Maya. How can I assist you today?').\n"
                "- Use natural call-centre phrases: 'Let me pull up your booking...', 'Just a moment while I check that...', "
                "'I can see your reservation here...', 'I do apologise for the inconvenience...'\n"
                "- Keep responses concise — 1-3 sentences for simple answers, up to 5 for complex ones.\n"
                "- Read out codes character by character when appropriate (e.g. 'Your reference is Sierra-Bravo-4-7-2-1-X').\n"
                "- Always offer a follow-up: 'Is there anything else I can help you with?'"
            ),
        }
    ]
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                # Lower min_volume so telephony/quiet recordings trigger VAD.
                # Default is 0.6 which is too high for many WAV files.
                params=VADParams(min_volume=0.0)
            )
        ),
    )

    audio_stream_tracker = AudioStreamTracker(session=session)
    metrics_tracker = MetricsTracker(session=session, audio_filter=filter_mux)
    bot_text_notifier = BotTextNotifier(session=session)
    recording_processor = RecordingProcessor(recorder=recorder)
    filter_router = FilterRouter(filter_mux=filter_mux)

    # --- pipeline ---
    pipeline = Pipeline(
        [
            transport.input(),
            filter_router,          # routes FilterControlFrame to filter_mux
            audio_stream_tracker,  # records t_stream_start for STT latency baseline
            stt,
            metrics_tracker,    # status notifications + latency tracking
            user_aggregator,
            llm,
            bot_text_notifier,  # LLM text + timing
            tts,
            recording_processor,  # captures bot TTS audio before it leaves the pipeline
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat pipeline client connected")
        # Ollama requires at least one user message in the context to generate a response.
        # Append an empty user turn and run the LLM to trigger the initial greeting.
        await task.queue_frames([
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": ""}], run_llm=True)
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected — cancelling pipeline task")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
        # Pipeline finished — run a single holistic quality evaluation over the full transcript.
        if session.history:
            sq = await judge.score_session(session.history)
            try:
                await websocket.send_text(json.dumps({"type": "session_quality", **sq}))
            except Exception:
                pass  # client already disconnected
    finally:
        await piper_session.close()
        recorder.save()


# ---------------------------------------------------------------------------
# Static files (served last so they don't shadow API routes)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
