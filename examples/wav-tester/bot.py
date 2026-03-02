#!/usr/bin/env python3
#
# WAV Tester Bot — FastAPI + Pipecat pipeline
#
# Serves a web frontend that lets you select WAV files from the input directory,
# stream them as audio input to a Pipecat STT→LLM→TTS pipeline via WebSocket,
# and hear the TTS response back in the browser.
#
# Required env vars (copy .env.example → .env):
#   DEEPGRAM_API_KEY
#   OPENAI_API_KEY
#   CARTESIA_API_KEY
#   INPUT_DIR  (default: input)
#
# Run:
#   source .venv/bin/activate
#   uvicorn bot:app --host 0.0.0.0 --port 7860 --reload
#

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Pipecat imports
# ---------------------------------------------------------------------------
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "input"))


def _iter_wav_files(root: Path) -> Iterator[Path]:
    """Walk root recursively following symlinks and yield .wav files."""
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                yield Path(dirpath) / fname

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
    t_user_stopped: Optional[float] = None
    t_transcription: Optional[float] = None
    t_llm_first_token: Optional[float] = None
    t_llm_complete: Optional[float] = None
    t_bot_started: Optional[float] = None
    t_bot_stopped: Optional[float] = None
    user_text: str = ""
    bot_text: str = ""
    interruptions: int = 0
    stt_confidence: Optional[float] = None    # avg word confidence 0–1
    speaking_rate_wpm: Optional[float] = None # words per minute (from Deepgram word timestamps)
    bot_word_count: int = 0                   # word count of bot response

    def _diff(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        return round((b - a) * 1000, 1) if a is not None and b is not None else None

    @property
    def stt_ms(self) -> Optional[float]:
        return self._diff(self.t_user_stopped, self.t_transcription)

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
# LLM quality judge — async Gemini call, runs outside the pipeline
# ---------------------------------------------------------------------------


class QualityJudge:
    """Scores a user/bot exchange using Gemini as an impartial evaluator."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_prompt(self, user_text: str, bot_text: str, history: list) -> str:
        # Use concatenation — NOT str.format() — so braces in user/bot text
        # (e.g. booking codes like {WJ4721X}) never cause a KeyError.
        parts = [
            "Rate the quality of the LATEST exchange in this airline customer service conversation. "
            "Consider the full conversation context (repetition, consistency, context retention). "
            "Score 1-5 where 5=excellent, 1=bad. Reply with JSON only.",
        ]
        # Include up to 5 prior turns for context (avoid token bloat)
        prior = [t for t in history[-5:] if t.user_text and t.bot_text]
        if prior:
            parts.append("\n[Prior turns — context only]")
            for t in prior:
                parts.append("Customer: " + t.user_text)
                parts.append("Agent: " + t.bot_text)
        parts.append("\n[Latest turn to evaluate]")
        parts.append("Customer: " + user_text)
        parts.append("Agent: " + bot_text)
        parts.append('\n{"score": <1-5>, "issue": "none|irrelevant|repetitive|too_long|unhelpful|off_topic"}')
        return "\n".join(parts)

    async def score(self, user_text: str, bot_text: str, history: Optional[list] = None) -> tuple[int, str]:
        """Return (score 1-5, issue string). Returns (0, 'error') on failure."""
        logger.info(f"QualityJudge: scoring turn — user={user_text[:60]!r}")
        try:
            prompt = self._build_prompt(user_text, bot_text, history or [])
            client = self._client_instance()
            # Use the native async API — avoids asyncio.to_thread conflicts
            response = await client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            m = re.search(r"\{[^}]+\}", response.text or "")
            if m:
                data = json.loads(m.group())
                score = int(data.get("score", 3))
                issue = str(data.get("issue", "none"))
                logger.info(f"QualityJudge: score={score} issue={issue}")
                return score, issue
            logger.warning(f"QualityJudge: no JSON found in response: {response.text!r}")
        except Exception as e:
            logger.warning(f"QualityJudge error: {e}")
        return 0, "error"


# ---------------------------------------------------------------------------
# Pipeline processors
# ---------------------------------------------------------------------------


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
        self._chunks = []
        self._first_token_seen = False
        await super()._start_interruption()


class MetricsTracker(FrameProcessor):
    """Replaces StatusNotifier. Tracks turn timing, emits metrics events, runs quality judge."""

    def __init__(self, session: SessionMetrics, judge: QualityJudge):
        super().__init__()
        self._s = session
        self._judge = judge

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        now = time.monotonic()

        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, UserStartedSpeakingFrame):
                if self._s.bot_speaking:
                    self._s.current.interruptions += 1
                    self._s.total_interruptions += 1
                self._s.start_turn()
                await self._send({"type": "user_speaking", "value": True})

            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._s.current.t_user_stopped = now
                await self._send({"type": "user_speaking", "value": False})

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TranscriptionFrame):
                if not self._s.current.t_transcription:
                    self._s.current.t_transcription = now
                    self._s.current.user_text = frame.text
                    # Extract STT confidence and speaking rate from Deepgram word-level data
                    if frame.result is not None:
                        try:
                            alt = frame.result.channel.alternatives[0]
                            words = alt.words or []
                            # Average per-word confidence; fall back to alternative-level confidence
                            word_confs = [w.confidence for w in words if w.confidence > 0]
                            if word_confs:
                                self._s.current.stt_confidence = round(
                                    sum(word_confs) / len(word_confs), 3
                                )
                            elif alt.confidence > 0:
                                self._s.current.stt_confidence = round(alt.confidence, 3)
                            # Speaking rate: words / speech_duration * 60
                            if len(words) >= 2:
                                speech_dur = words[-1].end - words[0].start
                                if speech_dur > 0:
                                    self._s.current.speaking_rate_wpm = round(
                                        len(words) / speech_dur * 60, 1
                                    )
                        except Exception:
                            pass
                await self._send({"type": "transcription", "text": frame.text, "user_id": frame.user_id})

        elif direction == FrameDirection.UPSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._s.bot_speaking = True
                self._s.current.t_bot_started = now
                # All latencies are now computable — emit metrics update
                await self._send({
                    "type": "metrics_update",
                    "turn": self._s.current.to_dict(),
                    "session": self._s.session_dict(),
                })
                await self._send({"type": "bot_speaking", "value": True})
                # Start judge now — bot_text is already set (LLM finished before TTS started),
                # so the Gemini call runs in parallel with TTS playback instead of after it.
                if self._s.current.user_text and self._s.current.bot_text:
                    self.create_task(self._judge_turn(self._s.current), "quality_judge")

            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._s.bot_speaking = False
                self._s.current.t_bot_stopped = now
                self._s.finish_turn()
                await self._send({"type": "bot_speaking", "value": False})

    async def _judge_turn(self, turn: TurnData):
        # Pass completed history (excluding the current turn) as conversation context
        history = [t for t in self._s.history if t.turn_id != turn.turn_id]
        score, issue = await self._judge.score(turn.user_text, turn.bot_text, history)
        await self._send({
            "type": "turn_quality",
            "turn_id": turn.turn_id,
            "score": score,
            "issue": issue,
        })

    async def _send(self, msg: dict[str, Any]):
        await self.push_frame(OutputTransportMessageUrgentFrame(message=msg))


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
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

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
            serializer=RawAudioSerializer(),
        ),
    )

    # --- services ---
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))

    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        model="gemini-2.5-flash",
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
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

    session = SessionMetrics()
    judge = QualityJudge(api_key=os.getenv("GEMINI_API_KEY", ""))
    metrics_tracker = MetricsTracker(session=session, judge=judge)
    bot_text_notifier = BotTextNotifier(session=session)

    # --- pipeline ---
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            metrics_tracker,    # status notifications + latency tracking
            user_aggregator,
            llm,
            bot_text_notifier,  # LLM text + timing
            tts,
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
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected — cancelling pipeline task")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


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
