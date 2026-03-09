#!/usr/bin/env python3
"""Unattended batch runner for the WAV-Tester pipeline.

Replays every WAV file in the input directory through the WebSocket pipeline
twice (no filter, then Quail), collects structured metrics, and saves a
JSON results file.

Usage:
    python batch_test.py [options]

    python batch_test.py --url ws://localhost:7860/ws --input-dir ../../input
    python batch_test.py --quail-license-key <key> --quail-model-id quail-vf-l-16khz
"""

import argparse
import asyncio
import fnmatch
import json
import os
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from df.enhance import load_audio as df_load_audio
    _DEEPFILTER_LOAD_AUDIO_AVAILABLE = True
except ModuleNotFoundError:
    _DEEPFILTER_LOAD_AUDIO_AVAILABLE = False

try:
    import scipy.signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

import websockets


# ---------------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------------

def _tensor_to_pcm16(audio_tensor) -> np.ndarray:
    """Convert DeepFilterNet-style float tensor to PCM16 mono numpy array."""
    audio_f32 = audio_tensor.squeeze().detach().cpu().numpy()
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    return (audio_f32 * 32767.0).astype(np.int16)


def load_wav_pcm16(path: Path) -> np.ndarray:
    """Load a WAV file as int16 mono PCM at 16000 Hz.

    Handles:
    - DeepFilterNet loader path (preferred, when available)
    - 16-bit or 32-bit input (converts to 16-bit)
    - Stereo (averaged to mono)
    - Any sample rate (resampled to 16000 Hz via scipy or nearest-power method)

    Returns:
        int16 numpy array at 16000 Hz mono.
    """
    if _DEEPFILTER_LOAD_AUDIO_AVAILABLE:
        audio_tensor, _ = df_load_audio(str(path), sr=16000)
        return _tensor_to_pcm16(audio_tensor)

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32)
        audio = (audio >> 16).astype(np.int16)
    elif sampwidth == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        audio = (audio - 128) * 256
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Stereo → mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        audio = audio.mean(axis=1).astype(np.int16)

    # Resample to 16000 Hz
    target_rate = 16000
    if framerate != target_rate:
        if _SCIPY_AVAILABLE:
            n_target = int(len(audio) * target_rate / framerate)
            audio_f = scipy.signal.resample(audio.astype(np.float32), n_target)
            audio = np.clip(audio_f, -32768, 32767).astype(np.int16)
        else:
            # Naive integer resampling (lower quality but no dependency)
            ratio = target_rate / framerate
            indices = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
            indices = np.clip(indices, 0, len(audio) - 1)
            audio = audio[indices]

    return audio


# ---------------------------------------------------------------------------
# Session data model
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn_id: int
    user_text: str = ""
    bot_text: str = ""
    stt_ms: Optional[float] = None
    llm_ms: Optional[float] = None
    tts_ms: Optional[float] = None
    rtt_ms: Optional[float] = None
    stt_confidence: Optional[float] = None
    speaking_rate_wpm: Optional[float] = None
    bot_word_count: int = 0
    interruptions: int = 0
    interrupted_bot_text: Optional[str] = None
    bot_speaking_ms_at_interruption: Optional[float] = None
    interruption_session_elapsed_ms: Optional[float] = None


@dataclass
class SessionResult:
    turns: list[TurnResult] = field(default_factory=list)
    session_final: dict = field(default_factory=dict)
    session_quality: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "user_text": t.user_text,
                    "bot_text": t.bot_text,
                    "stt_ms": t.stt_ms,
                    "llm_ms": t.llm_ms,
                    "tts_ms": t.tts_ms,
                    "rtt_ms": t.rtt_ms,
                    "stt_confidence": t.stt_confidence,
                    "speaking_rate_wpm": t.speaking_rate_wpm,
                    "bot_word_count": t.bot_word_count,
                    "interruptions": t.interruptions,
                    "interrupted_bot_text": t.interrupted_bot_text,
                    "bot_speaking_ms_at_interruption": t.bot_speaking_ms_at_interruption,
                    "interruption_session_elapsed_ms": t.interruption_session_elapsed_ms,
                }
                for t in self.turns
            ],
            "session_final": self.session_final,
            "session_quality": self.session_quality,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

CHUNK_BYTES = 320          # 160 samples × 2 bytes = 10 ms at 16 kHz
CHUNK_INTERVAL = 0.010     # 10 ms between chunks
SILENCE_DURATION = 0.600   # 600 ms of silence to trigger VAD end-of-speech
SILENCE_BYTES = bytes(int(16000 * SILENCE_DURATION) * 2)  # zeros

BAR_WIDTH = 24
TRANSCRIPT_WIDTH = 72  # max chars for transcript lines before truncation


def _trunc(text: str, width: int = TRANSCRIPT_WIDTH) -> str:
    return (text[:width - 1] + "…") if len(text) > width else text


class LiveDisplay:
    """3-line ANSI in-place display.

    Line 1: progress bar + phase
    Line 2: latest bot transcript
    Line 3: latest user (WAV) transcript
    """

    LINES = 3

    def __init__(self, cond_label: str, wav_duration_s: float):
        self._cond = cond_label
        self._dur  = wav_duration_s
        self._phase       = "connecting"
        self._pct         = 0.0
        self._stream_start: Optional[float] = None
        self.bot_text     = ""
        self.user_text    = ""
        self._drawn       = False

    def update(
        self,
        phase: Optional[str] = None,
        pct: Optional[float] = None,
        stream_start: Optional[float] = None,
    ):
        if phase        is not None: self._phase        = phase
        if pct          is not None: self._pct          = pct
        if stream_start is not None: self._stream_start = stream_start
        self._draw()

    def _draw(self):
        elapsed = (time.monotonic() - self._stream_start) if self._stream_start else 0.0
        filled  = int(BAR_WIDTH * self._pct)
        bar     = "█" * filled + "░" * (BAR_WIDTH - filled)
        line1 = (
            f"  [{bar}] {self._pct * 100:5.1f}%"
            f"  {elapsed:.1f}s / {self._dur:.1f}s"
            f"  filter={self._cond}  {self._phase}"
        )
        line2 = f"  Bot: {_trunc(self.bot_text)}"
        line3 = f"  You: {_trunc(self.user_text)}"

        out = f"\033[{self.LINES}A" if self._drawn else ""
        self._drawn = True
        for line in (line1, line2, line3):
            out += f"\033[2K\r{line}\n"
        sys.stderr.write(out)
        sys.stderr.flush()

    def finish(self, ok: bool):
        self._phase = "done ✓" if ok else "error ✗"
        self._pct   = 1.0
        self._draw()
        sys.stderr.write("\n")  # leave a blank line below the block
        sys.stderr.flush()


async def run_session(
    url: str,
    wav_path: Path,
    filter_name: str,
    quail_license_key: str,
    quail_model_id: str,
    greeting_timeout: float,
    response_timeout: float,
    quality_timeout: float,
) -> SessionResult:
    """Run one session: connect, send filter selection, wait for greeting,
    stream WAV, wait for bot response, collect quality scores, disconnect.

    Args:
        filter_name: One of "none" or "quail".

    Returns SessionResult with all collected data.
    """
    result = SessionResult()

    # Load WAV
    try:
        pcm = load_wav_pcm16(wav_path)
    except Exception as e:
        result.error = f"WAV load failed: {e}"
        return result

    pcm_bytes      = pcm.tobytes()
    n_chunks       = max(1, (len(pcm_bytes) + CHUNK_BYTES - 1) // CHUNK_BYTES)
    wav_duration_s = len(pcm_bytes) / (16000 * 2)
    cond_label     = filter_name

    disp = LiveDisplay(cond_label, wav_duration_s)
    disp.update(phase="waiting for greeting")

    # --- session state ---
    greeting_done = asyncio.Event()
    response_done = asyncio.Event()
    session_quality_done = asyncio.Event()
    wav_done_flag = False
    current_turn: Optional[TurnResult] = None
    # pending_metrics_turn holds the last turn that received a metrics_update.
    # It may differ from current_turn if background noise fired user_speaking:True
    # while the bot was still speaking (overwriting current_turn with an empty one).
    pending_metrics_turn: Optional[TurnResult] = None
    bot_is_speaking: bool = False

    def handle_message(msg: dict):
        nonlocal current_turn, pending_metrics_turn, bot_is_speaking

        mtype = msg.get("type")

        if mtype == "user_speaking":
            if msg.get("value") is True:
                # Don't create a new turn if the bot is currently speaking —
                # background noise / bleed-through commonly triggers VAD during
                # bot TTS playback, which would overwrite the turn that already
                # received a metrics_update and has valid timing data.
                if not bot_is_speaking:
                    current_turn = TurnResult(turn_id=len(result.turns) + 1)

        elif mtype == "transcription":
            text = msg.get("text", "")
            disp.user_text = text
            if current_turn:
                current_turn.user_text = text

        elif mtype == "bot_text":
            text = msg.get("text", "")
            disp.bot_text = text
            # Store in current_turn; also update pending_metrics_turn if it's
            # already set (they may be the same object, that's fine).
            if current_turn:
                current_turn.bot_text = text
            if pending_metrics_turn and pending_metrics_turn is not current_turn:
                pending_metrics_turn.bot_text = text

        elif mtype == "metrics_update":
            turn_data = msg.get("turn", {})
            # Always apply metrics to current_turn (the turn we expect to commit).
            target = current_turn
            if target:
                target.stt_ms            = turn_data.get("stt_ms")
                target.llm_ms            = turn_data.get("llm_ms")
                target.tts_ms            = turn_data.get("tts_ms")
                target.rtt_ms            = turn_data.get("rtt_ms")
                target.stt_confidence    = turn_data.get("stt_confidence")
                target.speaking_rate_wpm = turn_data.get("speaking_rate_wpm")
                target.bot_word_count                  = turn_data.get("bot_word_count", 0)
                target.interruptions                   = turn_data.get("interruptions", 0)
                target.interrupted_bot_text            = turn_data.get("interrupted_bot_text")
                target.bot_speaking_ms_at_interruption = turn_data.get("bot_speaking_ms_at_interruption")
                target.interruption_session_elapsed_ms = turn_data.get("interruption_session_elapsed_ms")
                pending_metrics_turn = target
            result.session_final = msg.get("session", {})

        elif mtype == "bot_speaking":
            speaking = msg.get("value", False)
            bot_is_speaking = speaking
            if not speaking:
                if not greeting_done.is_set():
                    greeting_done.set()
                else:
                    # Prefer the turn that actually received metrics data.
                    # If current_turn was overwritten by a noise VAD event,
                    # pending_metrics_turn still points to the real turn.
                    commit = pending_metrics_turn or current_turn
                    if commit:
                        result.turns.append(commit)
                    pending_metrics_turn = None
                    current_turn = None
                    if wav_done_flag:
                        response_done.set()

        elif mtype == "session_quality":
            result.session_quality = {
                "overall_score":  msg.get("overall_score"),
                "overall_issue":  msg.get("overall_issue", "none"),
                "summary":        msg.get("summary", ""),
                "flagged_turns":  msg.get("flagged_turns", []),
            }
            session_quality_done.set()

    try:
        async with websockets.connect(url, max_size=None) as ws:
            select_msg: dict = {"type": "filter_select", "filter": filter_name}
            if filter_name == "quail" and quail_license_key:
                select_msg["license_key"] = quail_license_key
            if filter_name == "quail" and quail_model_id:
                select_msg["model_id"] = quail_model_id
            await ws.send(json.dumps(select_msg))

            recv_exception: Optional[Exception] = None

            async def receiver():
                nonlocal recv_exception
                try:
                    async for message in ws:
                        if isinstance(message, str):
                            try:
                                handle_message(json.loads(message))
                            except Exception:
                                pass
                        # binary = TTS audio, ignore
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    recv_exception = e

            recv_task = asyncio.create_task(receiver())

            # --- wait for greeting (with live redraws so the display appears) ---
            greeting_deadline = time.monotonic() + greeting_timeout
            while not greeting_done.is_set():
                if time.monotonic() > greeting_deadline:
                    result.error = f"Greeting timeout after {greeting_timeout}s"
                    recv_task.cancel()
                    disp.finish(ok=False)
                    return result
                disp.update()
                await asyncio.sleep(0.3)

            # --- sender: stream WAV chunks at real-time pace ---
            stream_start = time.monotonic()
            disp.update(phase="streaming", pct=0.0, stream_start=stream_start)

            async def sender():
                nonlocal wav_done_flag
                start        = stream_start
                update_every = max(1, n_chunks // 100)  # redraw ~100× across the file
                for i in range(n_chunks):
                    chunk = pcm_bytes[i * CHUNK_BYTES: (i + 1) * CHUNK_BYTES]
                    if len(chunk) < CHUNK_BYTES:
                        chunk = chunk + bytes(CHUNK_BYTES - len(chunk))
                    try:
                        await ws.send(chunk)
                    except Exception:
                        return
                    # Clock-compensated sleep to maintain real-time pacing
                    next_send  = start + (i + 1) * CHUNK_INTERVAL
                    sleep_time = next_send - time.monotonic()
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    if i % update_every == 0 or i == n_chunks - 1:
                        disp.update(pct=(i + 1) / n_chunks)

                # Mark WAV content done BEFORE silence burst so that any
                # bot_speaking:false that arrives during or after the burst
                # is treated as a completed response turn.
                wav_done_flag = True

                # Silence burst → triggers Silero VAD end-of-speech
                disp.update(phase="silence burst")
                for i in range(0, len(SILENCE_BYTES), CHUNK_BYTES):
                    chunk = SILENCE_BYTES[i: i + CHUNK_BYTES]
                    if len(chunk) < CHUNK_BYTES:
                        chunk = chunk + bytes(CHUNK_BYTES - len(chunk))
                    try:
                        await ws.send(chunk)
                    except Exception:
                        return
                    await asyncio.sleep(CHUNK_INTERVAL)

                # If the bot already responded during the silence burst, set done.
                if result.turns and not response_done.is_set():
                    response_done.set()

                disp.update(phase="waiting for bot", pct=1.0)

            send_task = asyncio.create_task(sender())
            await send_task

            # --- wait for bot response (with live redraws so transcripts update) ---
            response_deadline = time.monotonic() + response_timeout
            while not response_done.is_set():
                if time.monotonic() > response_deadline:
                    result.error = f"Response timeout after {response_timeout}s"
                    recv_task.cancel()
                    disp.finish(ok=False)
                    break
                disp.update()
                await asyncio.sleep(0.3)

            # --- wait for end-of-session quality score ---
            if response_done.is_set():
                disp.update(phase="quality scoring")
                quality_deadline = time.monotonic() + quality_timeout
                while time.monotonic() < quality_deadline:
                    await asyncio.sleep(0.5)
                    if session_quality_done.is_set():
                        break

            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass

            if recv_exception:
                result.error = str(recv_exception)

    except Exception as e:
        result.error = f"WebSocket error: {e}"

    disp.finish(ok=not result.error)

    # Per-turn summary printed below the 3-line display block
    if result.turns:
        for t in result.turns:
            rtt  = f"{t.rtt_ms:.0f}ms"       if t.rtt_ms        is not None else "N/A"
            stt  = f"{t.stt_ms:.0f}ms"       if t.stt_ms        is not None else "N/A"
            conf = f"{t.stt_confidence:.2f}" if t.stt_confidence is not None else "N/A"
            print(f"       turn {t.turn_id}: RTT={rtt}  STT={stt}  conf={conf}", file=sys.stderr)
        if result.session_quality:
            sq = result.session_quality
            score = sq.get("overall_score", "?")
            issue = sq.get("overall_issue", "?")
            flagged = len(sq.get("flagged_turns", []))
            print(f"       quality: score={score}/5  issue={issue}  flagged={flagged} turns", file=sys.stderr)
    elif result.error:
        print(f"  ERROR: {result.error}", file=sys.stderr)
    else:
        print(f"  (no turns captured)", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Git branch helper
# ---------------------------------------------------------------------------

def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def _is_processed(entry: dict) -> bool:
    """Return True if an entry has valid (error-free) results for both conditions."""
    for cond in ("filter_none", "filter_quail"):
        sess = entry.get(cond)
        if not sess or sess.get("error"):
            return False
    return True


async def run_batch(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect WAV files
    wav_files: list[Path] = []
    for dirpath, _, filenames in os.walk(input_dir, followlinks=True):
        for fname in sorted(filenames):
            if not fname.lower().endswith(".wav"):
                continue
            if args.files and not fnmatch.fnmatch(fname, args.files):
                continue
            wav_files.append(Path(dirpath) / fname)
    wav_files.sort()

    if not wav_files:
        print("No WAV files found.", file=sys.stderr)
        sys.exit(1)

    n = len(wav_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "batch_results.json"

    # Load existing results so we can skip already-processed files
    existing_by_file: dict[str, dict] = {}
    if output_path.exists() and not args.force:
        try:
            existing_data = json.loads(output_path.read_text())
            for entry in existing_data.get("results", []):
                key = entry.get("file")
                if key:
                    existing_by_file[key] = entry
        except Exception as e:
            print(f"WARNING: could not load existing results: {e}", file=sys.stderr)

    results: list[dict] = []
    skipped = 0

    for idx, wav_path in enumerate(wav_files, 1):
        try:
            rel = wav_path.relative_to(input_dir)
        except ValueError:
            rel = wav_path.name
        file_key = str(rel)

        # Skip if already processed (unless --force)
        existing = existing_by_file.get(file_key)
        if existing and _is_processed(existing):
            print(f"\n[{idx}/{n}] {rel}  (skipping — already processed)", file=sys.stderr)
            results.append(existing)
            skipped += 1
            continue

        print(f"\n[{idx}/{n}] {rel}", file=sys.stderr)

        file_result = {"file": file_key, "filter_none": None, "filter_quail": None}

        for filter_name in ("none", "quail"):
            session = await run_session(
                url=args.url,
                wav_path=wav_path,
                filter_name=filter_name,
                quail_license_key=args.quail_license_key,
                quail_model_id=args.quail_model_id,
                greeting_timeout=args.greeting_timeout,
                response_timeout=args.response_timeout,
                quality_timeout=args.quality_timeout,
            )
            key = f"filter_{filter_name}"
            file_result[key] = session.to_dict()

            # Brief pause between conditions so server pipeline fully resets
            await asyncio.sleep(1.0)

        results.append(file_result)

        # Save incrementally after each file so partial results aren't lost
        output_data = {
            "metadata": {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "server_url": args.url,
                "n_files": len(results),
                "git_branch": _git_branch(),
                "input_dir": str(input_dir),
            },
            "results": results,
        }
        output_path.write_text(json.dumps(output_data, indent=2))

    n_new = len(wav_files) - skipped
    print(
        f"\nDone. {n_new} processed, {skipped} skipped. Results saved to {output_path}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-test WAV files through the Pipecat pipeline (no filter vs Quail).",
    )
    parser.add_argument("--url", default="ws://localhost:7860/ws", help="WebSocket server URL")
    parser.add_argument("--input-dir", default="input", help="Directory containing WAV files")
    parser.add_argument("--output-dir", default="results", help="Directory for JSON output")
    parser.add_argument("--greeting-timeout", type=float, default=30.0,
                        help="Seconds to wait for initial bot greeting")
    parser.add_argument("--response-timeout", type=float, default=60.0,
                        help="Seconds to wait for bot response after WAV finishes")
    parser.add_argument("--quality-timeout", type=float, default=15.0,
                        help="Seconds to collect quality scores after response")
    parser.add_argument("--quail-license-key", default="",
                        help="ai|coustics Quail license key (overrides AIC_LICENSE_KEY env var)")
    parser.add_argument("--quail-model-id", default="quail-vf-l-16khz",
                        help="Quail model ID (default: quail-vf-l-16khz)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrency level (keep at 1 for single-pipeline servers)")
    parser.add_argument("--files", default=None,
                        help="Glob pattern to filter WAV filenames (e.g. 'customer_*.wav')")
    parser.add_argument("--force", action="store_true",
                        help="Re-process all files, ignoring existing results")

    args = parser.parse_args()
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
