# Known Bugs

## [BUG-1] Bot re-greets on every turn when interrupted early

**Status**: Present in committed code. Fix was drafted but reverted to keep the
DeepFilter PR focused.

**Symptom**: The bot repeats its opening greeting ("Thank you for calling WestJet,
this is Maya. How can I assist you today?") on every conversation turn instead of
maintaining context. Quality-judge scores 1–2 / 5 with issue `repetitive`.

**Consistently reproducible with**: recordings that have background noise and
`VADParams(min_volume=0.0)` — any audio triggers VAD as "user speaking",
which interrupts the bot before Cartesia has returned word timestamps.

---

### Root cause

`CartesiaTTSService` is configured with `push_text_frames=False` (Cartesia
default). This means the pipeline's `LLMAssistantAggregator` does **not** see
regular `TextFrame`s from the LLM; instead it accumulates `TTSTextFrame`s
that CartesiaTTS pushes word-by-word, timed to Cartesia's word-timestamp
streaming API.

The assistant context is only updated when one of these happens:

1. The word-timed `LLMFullResponseEndFrame` arrives at `assistant_aggregator`
   (i.e., the audio has **finished playing**), or
2. An `InterruptionFrame` arrives and `assistant_aggregator._handle_interruptions()`
   calls `push_aggregation()` — but only if `_aggregation` is non-empty.

The race condition:

```
LLM generates response  →  TextFrames → BotTextNotifier → CartesiaTTS
                                                           │
                                          sends text to Cartesia API
                                                           │
                                    Cartesia API returns audio + word timestamps
                                                           │
                                    word-timed TTSTextFrame → assistant_aggregator._aggregation
```

With a fast LLM (Gemini Flash):

- The LLM's `LLMFullResponseEndFrame` arrives at `BotTextNotifier` and clears
  `_chunks` **before** Cartesia has returned any word timestamps.
- With `VADParams(min_volume=0.0)`, background noise triggers VAD almost
  immediately, generating an `InterruptionFrame`.
- The `InterruptionFrame` travels downstream in microseconds (pure async
  queue ops), reaching `assistant_aggregator` **before** any `TTSTextFrame`s
  from Cartesia's word-timestamp API.
- `assistant_aggregator._aggregation` is empty →
  `push_aggregation()` returns `""` → **no assistant message is added to
  the LLM context**.
- The next LLM call sees `[system, user: "..."]` with no assistant history →
  the LLM treats it as a fresh conversation and generates a new greeting.

This affects **every turn** in noisy recordings, not just the first.

---

### Relevant code paths

| File | Location | Role |
|---|---|---|
| `src/pipecat/services/cartesia/tts.py` | line 272 | `push_text_frames=False`, `pause_frame_processing=True` |
| `src/pipecat/services/tts_service.py` | `WordTTSService._words_task_handler` | pushes timed `TTSTextFrame`s and the word-timed `LLMFullResponseEndFrame` |
| `src/pipecat/processors/aggregators/llm_response_universal.py` | `LLMAssistantAggregator.push_aggregation()` line 921 | returns `""` if `_aggregation` is empty |
| `src/pipecat/processors/aggregators/llm_response_universal.py` | `_handle_interruptions()` line 953 | calls `push_aggregation()` on interrupt |
| `examples/wav-tester/bot.py` | `BotTextNotifier._start_interruption()` | **fix goes here** |

---

### Proposed fix

Add a two-stage text buffer to `BotTextNotifier` in `bot.py`.

`BotTextNotifier` sits **upstream** of `CartesiaTTSService` in the pipeline and
therefore receives the `InterruptionFrame` **before** `assistant_aggregator`
does. It also has the full LLM response text (from `TextFrame`s).

**Stage 1 — `_chunks`**: accumulates `TextFrame` text while the LLM is
streaming. Cleared when `LLMFullResponseEndFrame` arrives from the LLM.

**Stage 2 — `_pending_context_text`**: set to the full response text when
`LLMFullResponseEndFrame` clears `_chunks`. Held until the bot finishes
speaking normally (`BotStoppedSpeakingFrame` upstream) or is interrupted.

On `_start_interruption()`:

```python
async def _start_interruption(self):
    # Use _chunks if LLM is still streaming; fall back to _pending_context_text
    # if LLM finished but Cartesia hasn't returned word-timestamps yet.
    text = "".join(self._chunks).strip() if self._chunks else self._pending_context_text
    if text:
        self._context.add_message({"role": "assistant", "content": text})
    self._chunks = []
    self._first_token_seen = False
    self._pending_context_text = ""
    await super()._start_interruption()
```

On `BotStoppedSpeakingFrame` (upstream direction, normal completion):

```python
# word-timed mechanism already saved context correctly; clear the fallback
self._pending_context_text = ""
```

On `LLMFullResponseStartFrame` (downstream, safety net):

```python
# new response starting — clear any leftover pending text
self._pending_context_text = ""
```

**`BotTextNotifier.__init__`** needs `context: LLMContext` passed in:

```python
bot_text_notifier = BotTextNotifier(session=session, context=context)
```

**Edge case**: if some `TTSTextFrame`s arrived before the interruption,
`assistant_aggregator` will also call `push_aggregation()` with that partial
text, resulting in two assistant messages in context (full + partial). This is
a minor quality issue but still prevents re-greeting. A cleaner fix would
require a framework-level change to `LLMAssistantAggregator`.

---

### Is this a framework bug?

Partially. The `LLMAssistantAggregator` / `CartesiaTTSService` combination
silently loses context on early interruption — this affects any Pipecat bot
using CartesiaTTS with interruptions enabled, not just the wav-tester. A
proper framework fix would be to have `LLMAssistantAggregator` fall back to
the LLM text (via a `TextFrame` accumulator) when `_aggregation` is empty at
interruption time, instead of silently returning `""` from `push_aggregation()`.
