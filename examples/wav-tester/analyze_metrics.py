#!/usr/bin/env python3
"""Offline analysis script for WAV-Tester batch results.

Reads one or more batch_*.json files produced by batch_test.py and generates
a Markdown report comparing DeepFilter-off vs DeepFilter-on conditions.

Usage:
    python analyze_metrics.py results/batch_20260304_143022.json
    python analyze_metrics.py results/batch_*.json --csv --output results/
"""

import argparse
import csv
import difflib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import markdown as md_lib


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy/numpy required)
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _std(vals: list[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    m = _mean(vals)
    variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance)


def _median(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


def _percentile(vals: list[float], pct: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    idx = (len(s) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _welch_t_pvalue(a: list[float], b: list[float]) -> Optional[float]:
    """Two-sided Welch's t-test p-value (no external deps)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = _mean(a), _mean(b)
    sa2 = sum((x - ma) ** 2 for x in a) / (na - 1)
    sb2 = sum((x - mb) ** 2 for x in b) / (nb - 1)
    if sa2 + sb2 == 0:
        return 1.0
    t_stat = (ma - mb) / math.sqrt(sa2 / na + sb2 / nb)
    # Welch–Satterthwaite degrees of freedom
    dof = (sa2 / na + sb2 / nb) ** 2 / (
        (sa2 / na) ** 2 / (na - 1) + (sb2 / nb) ** 2 / (nb - 1)
    )
    # p-value via regularized incomplete beta function approximation
    # Using a simple Gaussian approximation for large dof
    x = abs(t_stat)
    if dof > 30:
        # Standard normal approximation
        p_one = 0.5 * math.erfc(x / math.sqrt(2))
        return 2 * p_one
    # For smaller dof, use a simple integration (not highly accurate but adequate)
    # Wilson-Hilferty approximation for t-distribution
    v = dof
    z = ((x / math.sqrt(v)) ** (2 / 3) * (1 - 2 / (9 * v)) - (1 - 2 / (9 * v))) / math.sqrt(
        2 / (9 * v) + (x / math.sqrt(v)) ** (4 / 3) * 2 / (9 * v)
    )
    try:
        p_one = 0.5 * math.erfc(z / math.sqrt(2))
    except Exception:
        p_one = 0.5
    return min(1.0, 2 * p_one)


def _fmt(v: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}{suffix}"


def _fmt_delta(off: Optional[float], on: Optional[float], decimals: int = 1) -> tuple[str, str]:
    """Return (absolute delta string, percent delta string)."""
    if off is None or on is None:
        return "N/A", "N/A"
    delta = on - off
    pct = (delta / off * 100) if off != 0 else 0.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{decimals}f}", f"{sign}{pct:.1f}%"


def _sig_stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(paths: list[Path]) -> list[dict]:
    """Load and merge results from one or more batch JSON files."""
    all_results = []
    for path in paths:
        data = json.loads(path.read_text())
        all_results.extend(data.get("results", []))
    return all_results


def extract_turns(
    results: list[dict], condition: str, min_turns: int
) -> tuple[list[dict], list[dict]]:
    """Extract all turn dicts and session_quality dicts for a condition.

    Returns (turns, session_qualities) where session_qualities is one dict per
    session (not per turn) containing overall_score, overall_issue, summary,
    and flagged_turns.
    """
    turns: list[dict] = []
    qualities: list[dict] = []
    for entry in results:
        sess = entry.get(condition) or {}
        if sess.get("error"):
            continue
        sess_turns = sess.get("turns", [])
        if len(sess_turns) < min_turns:
            continue
        turns.extend(sess_turns)
        sq = sess.get("session_quality")
        if sq:
            qualities.append(sq)
    return turns, qualities


def _collect(turns: list[dict], key: str) -> list[float]:
    return [v for t in turns if (v := t.get(key)) is not None]


# ---------------------------------------------------------------------------
# Conversation diff helpers
# ---------------------------------------------------------------------------

def _esc_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _user_text_html_for_side(my_text: str, other_text: str) -> str:
    """Render my_text with word-level highlights relative to other_text.

    Words only in my_text   → red mark (wdiff-del)
    Words changed vs other  → yellow mark (wdiff-chg)
    Words identical to other → plain
    """
    if not my_text:
        return "<em>—</em>"
    if my_text == other_text:
        return _esc_html(my_text)
    my_words  = my_text.split()
    oth_words = other_text.split()
    sm = difflib.SequenceMatcher(None, my_words, oth_words, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        chunk = " ".join(my_words[i1:i2])
        if not chunk:
            continue
        if tag == "equal":
            parts.append(_esc_html(chunk))
        elif tag == "replace":
            parts.append(f'<mark class="wdiff-chg">{_esc_html(chunk)}</mark>')
        elif tag == "delete":
            parts.append(f'<mark class="wdiff-del">{_esc_html(chunk)}</mark>')
        # "insert" means other has extra words — invisible in my view
    return " ".join(parts)


def _conversation_diff_html(off_turns: list[dict], on_turns: list[dict]) -> str:
    """Side-by-side full conversation view (user + bot) per file.

    Alignment is by user text (difflib). Each aligned pair is one table row.
    User STT differences are highlighted at word level.
    Bot text is shown muted when it differs (responses naturally vary).
    Interruption badges show what the bot was saying and for how long.
    """
    off_texts = [(t.get("user_text") or "").strip() for t in off_turns]
    on_texts  = [(t.get("user_text") or "").strip() for t in on_turns]
    matcher   = difflib.SequenceMatcher(None, off_texts, on_texts, autojunk=False)

    # Build aligned pairs: (off_turn | None, on_turn | None, opcode_tag)
    pairs: list[tuple[dict | None, dict | None, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append((off_turns[i1 + k], on_turns[j1 + k], "equal"))
        elif tag == "replace":
            off_chunk = list(range(i1, i2))
            on_chunk  = list(range(j1, j2))
            for k in range(max(len(off_chunk), len(on_chunk))):
                off_t = off_turns[off_chunk[k]] if k < len(off_chunk) else None
                on_t  = on_turns[on_chunk[k]]   if k < len(on_chunk)  else None
                pairs.append((off_t, on_t, "replace"))
        elif tag == "delete":
            for k in range(i1, i2):
                pairs.append((off_turns[k], None, "delete"))
        elif tag == "insert":
            for k in range(j1, j2):
                pairs.append((None, on_turns[k], "insert"))

    def conf_badge(conf: float | None) -> str:
        if conf is None:
            return ""
        cls = "conf-high" if conf >= 0.90 else ("conf-mid" if conf >= 0.80 else "conf-low")
        return f'<span class="cbadge {cls}">{conf:.3f}</span>'

    def metrics_line(t: dict) -> str:
        parts = []
        if (v := t.get("stt_ms")) is not None:
            parts.append(f"STT {v:.0f} ms")
        if (v := t.get("rtt_ms")) is not None:
            parts.append(f"RTT {v:.0f} ms")
        if not parts:
            return ""
        return f'<div class="cv-metrics">{" · ".join(parts)}</div>'

    def interruption_badge(t: dict) -> str:
        n = t.get("interruptions", 0)
        if n is None:
            n = 0
        if n == 0:
            return ""
        ibt = (t.get("interrupted_bot_text") or "").strip()
        ims = t.get("bot_speaking_ms_at_interruption")
        ses = t.get("interruption_session_elapsed_ms")
        details = []
        if ibt:
            truncated = ibt[:120] + ("…" if len(ibt) > 120 else "")
            details.append(
                f'<div class="intr-detail">'
                f'Bot was saying: <em>"{_esc_html(truncated)}"</em>'
                f'</div>'
            )
        timing_parts = []
        if ims is not None:
            timing_parts.append(f"after {ims:.0f} ms of bot speech")
        if ses is not None:
            timing_parts.append(f"T+{ses / 1000:.2f}s into session")
        if timing_parts:
            details.append(
                f'<div class="intr-detail intr-timing">'
                f'Interrupted {" · ".join(timing_parts)}'
                f'</div>'
            )
        label = f"⚡ Interrupted{f' ({n}×)' if n > 1 else ''}"
        return f'<div class="intr-badge">{label}{"".join(details)}</div>'

    def turn_cell(t: dict | None, other: dict | None, tag: str, is_off: bool) -> str:
        if t is None:
            return '<td class="cv-cell cv-empty"><span class="cv-none">—</span></td>'

        user_text  = (t.get("user_text") or "").strip()
        bot_text   = (t.get("bot_text")  or "").strip()
        conf       = t.get("stt_confidence")
        tid        = t.get("turn_id", "?")
        other_user = (other.get("user_text") or "").strip() if other else ""
        other_bot  = (other.get("bot_text")  or "").strip() if other else ""

        # User text: word-level diff when aligned with other side
        if tag == "equal" or tag == "replace":
            user_html = _user_text_html_for_side(user_text, other_user)
        elif tag == "delete":
            # Only exists in filter-off — mark entire text as removed
            user_html = f'<mark class="wdiff-del">{_esc_html(user_text)}</mark>' if user_text else "<em>—</em>"
        else:  # insert
            # Only exists in filter-on — mark entire text as added
            user_html = f'<mark class="wdiff-ins">{_esc_html(user_text)}</mark>' if user_text else "<em>—</em>"

        # Cell background class
        cell_cls = "cv-cell"
        if tag == "delete":
            cell_cls += " cv-del"
        elif tag == "insert":
            cell_cls += " cv-ins"

        # Bot text: muted styling when it differs (responses naturally vary)
        bot_differs = bot_text != other_bot and other is not None
        bot_cls  = "cv-bot cv-bot-muted" if bot_differs else "cv-bot"
        bot_html = _esc_html(bot_text) if bot_text else '<em class="cv-none">—</em>'

        return (
            f'<td class="{cell_cls}">'
            f'<div class="cv-turn-id">Turn {tid}</div>'
            + interruption_badge(t)
            + f'<div class="cv-user">'
              f'<span class="cv-speaker">👤</span>'
              f'<span class="cv-utext">{user_html}</span>'
              + conf_badge(conf)
              + f'</div>'
            + metrics_line(t)
            + f'<div class="{bot_cls}">'
              f'<span class="cv-speaker">🤖</span>'
              f'<span class="cv-btext">{bot_html}</span>'
              f'</div>'
            + '</td>'
        )

    rows_html = [
        f'<tr>{turn_cell(off_t, on_t, tag, True)}{turn_cell(on_t, off_t, tag, False)}</tr>'
        for off_t, on_t, tag in pairs
    ]

    return (
        '<table class="cv-table">'
        '<thead><tr>'
        '<th class="cv-th cv-th-off">Filter OFF</th>'
        '<th class="cv-th cv-th-on">Filter ON</th>'
        '</tr></thead>'
        '<tbody>' + "".join(rows_html) + "</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

METRICS = [
    ("stt_ms",             "STT Latency",       "ms",  1),
    ("llm_ms",             "LLM TTFT",          "ms",  1),
    ("tts_ms",             "TTS Latency",        "ms",  1),
    ("rtt_ms",             "RTT",               "ms",  1),
    ("stt_confidence",     "STT Confidence",    "",    3),
    ("speaking_rate_wpm",  "Speaking Rate",     "wpm", 1),
    ("bot_word_count",     "Bot Words/Turn",    "",    1),
    ("interruptions",      "Interruptions/Turn","",    2),
]

ISSUES = [
    "none", "context_loss", "repetitive", "wrong_entity",
    "unhelpful", "irrelevant", "escalation_failure", "too_long", "error",
]


def build_report(results: list[dict], min_turns: int) -> str:
    turns_off, qual_off = extract_turns(results, "filter_none",  min_turns)
    turns_on,  qual_on  = extract_turns(results, "filter_quail", min_turns)

    lines: list[str] = []

    def h1(t): lines.append(f"# {t}")
    def h2(t): lines.append(f"\n## {t}")
    def h3(t): lines.append(f"\n### {t}")
    def para(t): lines.append(f"\n{t}")
    def bullet(t): lines.append(f"- {t}")
    def rule(): lines.append("\n---")

    # -----------------------------------------------------------------------
    # Pre-compute all data so sections can be emitted in any order
    # -----------------------------------------------------------------------

    # STT confidence
    conf_off   = _collect(turns_off, "stt_confidence")
    conf_on    = _collect(turns_on,  "stt_confidence")
    m_conf_off = _mean(conf_off)
    m_conf_on  = _mean(conf_on)

    # RTT
    rtt_off   = _collect(turns_off, "rtt_ms")
    rtt_on    = _collect(turns_on,  "rtt_ms")
    m_rtt_off = _mean(rtt_off)
    m_rtt_on  = _mean(rtt_on)

    # Quality scores (one per session)
    scores_off  = [q.get("overall_score") for q in qual_off
                   if q.get("overall_score") is not None]
    scores_on   = [q.get("overall_score") for q in qual_on
                   if q.get("overall_score") is not None]
    m_score_off = _mean(scores_off)
    m_score_on  = _mean(scores_on)

    # Per-file stats (needed by Top Movers and Per-File Results)
    def _sess_stats(entry: dict, cond: str) -> dict:
        sess = entry.get(cond) or {}
        if sess.get("error"):
            return {"turns": 0, "rtt": None, "conf": None, "score": None,
                    "issue": None, "summary": None, "flagged": 0}
        t_list = sess.get("turns", [])
        rtt_vals  = [t["rtt_ms"]         for t in t_list if t.get("rtt_ms")         is not None]
        conf_vals = [t["stt_confidence"] for t in t_list if t.get("stt_confidence") is not None]
        sq = sess.get("session_quality") or {}
        raw_score = sq.get("overall_score")
        score = raw_score if raw_score is not None else None
        return {
            "turns":   len(t_list),
            "rtt":     _mean(rtt_vals),
            "conf":    _mean(conf_vals),
            "score":   score,
            "issue":   sq.get("overall_issue"),
            "summary": sq.get("summary"),
            "flagged": len(sq.get("flagged_turns", [])),
        }

    file_rows: list[dict] = [
        {"file": e.get("file", "?"),
         "off":  _sess_stats(e, "filter_none"),
         "on":   _sess_stats(e, "filter_quail"),
         "entry": e}
        for e in results
    ]

    # Confidence buckets
    def conf_bucket(vals: list[float]) -> dict:
        buckets = {"≥0.95": 0, "0.90–0.95": 0, "0.80–0.90": 0, "<0.80": 0}
        for v in vals:
            if   v >= 0.95: buckets["≥0.95"]     += 1
            elif v >= 0.90: buckets["0.90–0.95"]  += 1
            elif v >= 0.80: buckets["0.80–0.90"]  += 1
            else:           buckets["<0.80"]       += 1
        return buckets

    cb_off = conf_bucket(conf_off)
    cb_on  = conf_bucket(conf_on)
    n_conf_off = len(conf_off) or 1
    n_conf_on  = len(conf_on)  or 1

    # Session-level issue counts (one per session)
    def count_issues(quals: list[dict]) -> dict:
        c: dict[str, int] = {k: 0 for k in ISSUES}
        for q in quals:
            iss = q.get("overall_issue", "none")
            if iss in c:
                c[iss] += 1
            else:
                c["error"] = c.get("error", 0) + 1
        return c

    # Flagged-turn counts (aggregated from all sessions)
    def count_flagged_issues(quals: list[dict]) -> dict:
        c: dict[str, int] = {k: 0 for k in ISSUES}
        for q in quals:
            for ft in q.get("flagged_turns", []):
                iss = ft.get("issue", "none")
                if iss in c:
                    c[iss] += 1
                else:
                    c["error"] = c.get("error", 0) + 1
        return c

    ic_off = count_issues(qual_off)
    ic_on  = count_issues(qual_on)
    nq_off = len(qual_off) or 1
    nq_on  = len(qual_on)  or 1
    fi_off = count_flagged_issues(qual_off)
    fi_on  = count_flagged_issues(qual_on)

    # Sorted scorable file rows for Top Movers
    scorable = [r for r in file_rows
                if r["off"]["conf"] is not None and r["on"]["conf"] is not None]
    scorable.sort(key=lambda r: r["on"]["conf"] - r["off"]["conf"])

    # -----------------------------------------------------------------------
    # Report header
    # -----------------------------------------------------------------------
    h1("DeepFilter Impact Analysis")
    para(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — "
         f"{len(results)} files, {len(turns_off)} off-turns, {len(turns_on)} on-turns*")

    # -----------------------------------------------------------------------
    # 1. Executive Summary
    # -----------------------------------------------------------------------
    h2("1. Executive Summary")

    if m_conf_off is not None and m_conf_on is not None:
        d_conf = m_conf_on - m_conf_off
        direction = "improved" if d_conf > 0 else "degraded"
        bullet(f"STT confidence **{direction}** by {abs(d_conf):.3f} "
               f"({m_conf_off:.3f} → {m_conf_on:.3f}) with DeepFilter on.")

    if m_rtt_off is not None and m_rtt_on is not None:
        d_rtt = m_rtt_on - m_rtt_off
        overhead = "added" if d_rtt > 0 else "reduced"
        bullet(f"DeepFilter {overhead} **{abs(d_rtt):.0f} ms** of RTT overhead "
               f"({m_rtt_off:.0f} → {m_rtt_on:.0f} ms mean).")

    if m_score_off is not None and m_score_on is not None:
        d_score = m_score_on - m_score_off
        qdirection = "higher" if d_score > 0 else "lower"
        bullet(f"Conversation quality scores were **{qdirection}** with DeepFilter on "
               f"({m_score_off:.2f}/5 → {m_score_on:.2f}/5).")

    # -----------------------------------------------------------------------
    # 2. Latency Breakdown
    # -----------------------------------------------------------------------
    h2("2. Latency Breakdown")
    para("STT is directly affected by audio quality; LLM and TTS should be near-identical.")
    lines.append("")

    lines.append("| Stage | Off mean | Off p95 | On mean | On p95 | Δ mean |")
    lines.append("|-------|----------|---------|---------|--------|--------|")
    for key, label in [("stt_ms", "STT"), ("llm_ms", "LLM"), ("tts_ms", "TTS"), ("rtt_ms", "RTT")]:
        off_v = _collect(turns_off, key);  on_v = _collect(turns_on, key)
        m_off = _mean(off_v);  p95_off = _percentile(off_v, 95)
        m_on  = _mean(on_v);   p95_on  = _percentile(on_v,  95)
        d_abs, _ = _fmt_delta(m_off, m_on, 0)
        lines.append(f"| {label} | {_fmt(m_off, 0)} ms | {_fmt(p95_off, 0)} ms "
                     f"| {_fmt(m_on, 0)} ms | {_fmt(p95_on, 0)} ms | {d_abs} ms |")

    # -----------------------------------------------------------------------
    # 3. Top Movers (STT Confidence)
    # -----------------------------------------------------------------------
    h2("3. Top Movers (STT Confidence)")

    h3("Most Improved")
    improved = list(reversed(scorable[-5:])) if scorable else []
    if improved:
        lines.append("| File | Off conf | On conf | Δ conf |")
        lines.append("|------|----------|---------|--------|")
        for r in improved:
            d = r["on"]["conf"] - r["off"]["conf"]
            lines.append(f"| {r['file']} | {_fmt(r['off']['conf'], 3)} | {_fmt(r['on']['conf'], 3)} "
                         f"| {'+' if d >= 0 else ''}{d:.3f} |")
    else:
        para("*Not enough data.*")

    h3("Most Degraded")
    degraded = [r for r in scorable[:5] if r["on"]["conf"] - r["off"]["conf"] < 0]
    if degraded:
        lines.append("| File | Off conf | On conf | Δ conf |")
        lines.append("|------|----------|---------|--------|")
        for r in degraded:
            d = r["on"]["conf"] - r["off"]["conf"]
            lines.append(f"| {r['file']} | {_fmt(r['off']['conf'], 3)} | {_fmt(r['on']['conf'], 3)} "
                         f"| {'+' if d >= 0 else ''}{d:.3f} |")
    else:
        para("*No files degraded by DeepFilter.*")

    # -----------------------------------------------------------------------
    # 4. Full Conversation Comparison
    # -----------------------------------------------------------------------
    h2("4. Full Conversation Comparison")
    para(
        "Side-by-side view of the complete conversation (user + bot) for each file. "
        "Turns are diff-aligned by user speech. "
        "User STT output is compared word-by-word — differences are the primary quality signal. "
        "Bot responses are shown for context; they naturally vary between runs and are only lightly muted when different."
    )
    para(
        "*Legend: &nbsp;"
        "<mark style='background:#fff3b0;padding:1px 4px;border-radius:3px'>yellow word</mark> = changed between conditions &nbsp;·&nbsp; "
        "<mark style='background:#ffd0d0;padding:1px 4px;border-radius:3px'>red word</mark> = only in this side &nbsp;·&nbsp; "
        "<mark style='background:#d0f0d0;padding:1px 4px;border-radius:3px'>green word</mark> = only in this side &nbsp;·&nbsp; "
        "<span style='background:#fff0f0;padding:1px 4px;border-radius:3px'>red cell</span> = turn only in filter-off &nbsp;·&nbsp; "
        "<span style='background:#f0fff0;padding:1px 4px;border-radius:3px'>green cell</span> = turn only in filter-on &nbsp;·&nbsp; "
        "⚡ badge = user interrupted the bot mid-response*"
    )

    for row in file_rows:
        off_turns_list = (row["entry"].get("filter_none")  or {}).get("turns", [])
        on_turns_list  = (row["entry"].get("filter_quail") or {}).get("turns", [])
        if not off_turns_list and not on_turns_list:
            continue
        h3(row["file"])
        lines.append("")
        lines.append(_conversation_diff_html(off_turns_list, on_turns_list))
        lines.append("")

    # -----------------------------------------------------------------------
    # 5. Aggregate Metrics Table
    # -----------------------------------------------------------------------
    h2("5. Aggregate Metrics Table")
    para("*`*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` not significant*")
    lines.append("")

    lines.append("| Metric | Off Mean | On Mean | Δ Abs | Δ % | p-value |")
    lines.append("|--------|----------|---------|-------|-----|---------|")
    for key, label, unit, dec in METRICS:
        off_vals = _collect(turns_off, key);  on_vals = _collect(turns_on, key)
        m_off = _mean(off_vals);  m_on = _mean(on_vals)
        d_abs, d_pct = _fmt_delta(m_off, m_on, dec)
        p = _welch_t_pvalue(off_vals, on_vals)
        p_str = f"{p:.3f} {_sig_stars(p)}" if p is not None else "N/A"
        suffix = f" {unit}" if unit else ""
        lines.append(f"| {label} | {_fmt(m_off, dec)}{suffix} | {_fmt(m_on, dec)}{suffix} "
                     f"| {d_abs}{suffix} | {d_pct} | {p_str} |")

    # -----------------------------------------------------------------------
    # 6. STT Quality
    # -----------------------------------------------------------------------
    h2("6. STT Quality")

    lines.append("| Metric | Filter Off | Filter On | Delta |")
    lines.append("|--------|-----------|-----------|-------|")
    for key, label, dec in [
        ("stt_confidence",    "Confidence mean",        3),
        ("speaking_rate_wpm", "Speaking rate mean (wpm)", 1),
    ]:
        off_v = _collect(turns_off, key);  on_v = _collect(turns_on, key)
        m_off = _mean(off_v);  m_on = _mean(on_v)
        d_abs, _ = _fmt_delta(m_off, m_on, dec)
        lines.append(f"| {label} | {_fmt(m_off, dec)} | {_fmt(m_on, dec)} | {d_abs} |")

    h3("Confidence Distribution")
    lines.append("| Range | Off count | Off % | On count | On % |")
    lines.append("|-------|-----------|-------|----------|------|")
    for bucket in ["≥0.95", "0.90–0.95", "0.80–0.90", "<0.80"]:
        co = cb_off[bucket];  coo = cb_on[bucket]
        lines.append(f"| {bucket} | {co} | {co/n_conf_off*100:.1f}% | {coo} | {coo/n_conf_on*100:.1f}% |")

    # -----------------------------------------------------------------------
    # 7. Conversation Quality
    # -----------------------------------------------------------------------
    h2("7. Conversation Quality")
    para(f"End-of-session quality scores (1–5, holistic Ollama judge): "
         f"Off mean={_fmt(m_score_off, 2)} (n={len(scores_off)} sessions), "
         f"On mean={_fmt(m_score_on, 2)} (n={len(scores_on)} sessions)")

    h3("Session-Level Issue Distribution")
    para("One dominant issue per session (overall_issue field).")
    lines.append("| Issue | Off sessions | Off % | On sessions | On % |")
    lines.append("|-------|-------------|-------|------------|------|")
    for issue in ISSUES:
        co = ic_off.get(issue, 0);  coo = ic_on.get(issue, 0)
        lines.append(f"| {issue} | {co} | {co/nq_off*100:.1f}% | {coo} | {coo/nq_on*100:.1f}% |")

    h3("Flagged-Turn Issue Distribution")
    para("Individual turns flagged within sessions (may be more than one per session).")
    n_fi_off = sum(fi_off.values()) or 1
    n_fi_on  = sum(fi_on.values())  or 1
    lines.append("| Issue | Off flagged | Off % | On flagged | On % |")
    lines.append("|-------|------------|-------|-----------|------|")
    for issue in ISSUES:
        co = fi_off.get(issue, 0);  coo = fi_on.get(issue, 0)
        lines.append(f"| {issue} | {co} | {co/n_fi_off*100:.1f}% | {coo} | {coo/n_fi_on*100:.1f}% |")

    # Per-file quality summaries
    h3("Per-File Quality Notes")
    lines.append("| File | Off score | Off issue | On score | On issue | Summary |")
    lines.append("|------|-----------|-----------|----------|----------|---------|")
    for row in file_rows:
        off_s = row["off"];  on_s = row["on"]
        summary = (on_s.get("summary") or off_s.get("summary") or "").replace("|", "\\|")
        lines.append(
            f"| {row['file']} "
            f"| {_fmt(off_s['score'], 2)} | {off_s.get('issue') or 'N/A'} "
            f"| {_fmt(on_s['score'], 2)} | {on_s.get('issue') or 'N/A'} "
            f"| {summary[:120]} |"
        )

    # -----------------------------------------------------------------------
    # 8. Per-File Results
    # -----------------------------------------------------------------------
    h2("8. Per-File Results")

    lines.append("| File | Turns off | Turns on | RTT off | RTT on | Conf off | Conf on | Score off | Score on | Flagged off | Flagged on |")
    lines.append("|------|-----------|----------|---------|--------|----------|---------|-----------|----------|-------------|------------|")
    for row in file_rows:
        off_s = row["off"];  on_s = row["on"]
        lines.append(
            f"| {row['file']} "
            f"| {off_s['turns']} | {on_s['turns']} "
            f"| {_fmt(off_s['rtt'], 0)} ms | {_fmt(on_s['rtt'], 0)} ms "
            f"| {_fmt(off_s['conf'], 3)} | {_fmt(on_s['conf'], 3)} "
            f"| {_fmt(off_s['score'], 2)} | {_fmt(on_s['score'], 2)} "
            f"| {off_s.get('flagged', 'N/A')} | {on_s.get('flagged', 'N/A')} |"
        )

    # -----------------------------------------------------------------------
    # 9. Recommendation
    # -----------------------------------------------------------------------
    h2("9. Recommendation")

    if m_conf_off is not None and m_conf_on is not None and m_rtt_off is not None and m_rtt_on is not None:
        d_conf = m_conf_on - m_conf_off
        d_rtt  = m_rtt_on  - m_rtt_off
        if d_conf > 0.01 and d_rtt < 100:
            rec = (f"**Enable DeepFilter by default.** STT confidence improved by {d_conf:.3f} "
                   f"with only {d_rtt:.0f} ms RTT overhead — a favourable trade-off.")
        elif d_conf > 0.005 and d_rtt < 200:
            rec = (f"**Enable DeepFilter for noisy recordings.** Small confidence gain ({d_conf:.3f}) "
                   f"at {d_rtt:.0f} ms overhead — worthwhile only when audio is noisy.")
        elif d_conf < -0.005:
            rec = (f"**Keep DeepFilter off by default.** It decreased STT confidence by {abs(d_conf):.3f}. "
                   f"Investigate whether the model was loaded correctly.")
        else:
            rec = (f"**Results are inconclusive.** Confidence delta ({d_conf:+.3f}) is marginal; "
                   f"RTT overhead is {d_rtt:.0f} ms. Collect more data before deciding.")
        para(rec)
    else:
        para("*Insufficient data to make a recommendation.*")

    # -----------------------------------------------------------------------
    # 10. Interruption Analysis
    # -----------------------------------------------------------------------
    h2("10. Interruption Analysis")
    para(
        "An **interruption** is counted each time the VAD detects user audio while the bot "
        "is already playing back TTS audio. With DeepFilter **off**, background noise or "
        "acoustic bleed-through from the WAV can trigger spurious VAD events mid-response. "
        "With DeepFilter **on**, that noise is suppressed before reaching the VAD, so a lower "
        "interruption count indicates the filter is reducing false triggers."
    )

    intr_off = _collect(turns_off, "interruptions")
    intr_on  = _collect(turns_on,  "interruptions")

    total_intr_off = int(sum(intr_off))
    total_intr_on  = int(sum(intr_on))

    intr_turns_off = sum(1 for v in intr_off if v > 0)
    intr_turns_on  = sum(1 for v in intr_on  if v > 0)
    n_turns_off = len(intr_off) or 1
    n_turns_on  = len(intr_on)  or 1

    m_intr_off = _mean(intr_off)
    m_intr_on  = _mean(intr_on)
    d_intr_abs, d_intr_pct = _fmt_delta(m_intr_off, m_intr_on, 2)
    p_intr = _welch_t_pvalue(intr_off, intr_on)
    p_intr_str = f"{p_intr:.3f} {_sig_stars(p_intr)}" if p_intr is not None else "N/A"

    lines.append("")
    lines.append("| Metric | Filter Off | Filter On | Δ |")
    lines.append("|--------|-----------|-----------|---|")
    lines.append(f"| Total interruptions (all turns) | {total_intr_off} | {total_intr_on} | {total_intr_on - total_intr_off:+d} |")
    lines.append(f"| Mean interruptions/turn | {_fmt(m_intr_off, 2)} | {_fmt(m_intr_on, 2)} | {d_intr_abs} ({d_intr_pct}) |")
    lines.append(f"| Turns with ≥1 interruption | {intr_turns_off} ({intr_turns_off/n_turns_off*100:.1f}%) | {intr_turns_on} ({intr_turns_on/n_turns_on*100:.1f}%) | — |")
    lines.append(f"| p-value (Welch's t-test) | — | — | {p_intr_str} |")

    # Session-level totals from session_final (includes noise-VAD events not tied to a turn)
    sess_intr_off_list = [
        e.get("filter_none", {}).get("session_final", {}).get("total_interruptions")
        for e in results
        if (e.get("filter_none") or {}).get("session_final", {}).get("total_interruptions") is not None
    ]
    sess_intr_on_list = [
        e.get("filter_quail", {}).get("session_final", {}).get("total_interruptions")
        for e in results
        if (e.get("filter_quail") or {}).get("session_final", {}).get("total_interruptions") is not None
    ]
    if sess_intr_off_list or sess_intr_on_list:
        total_sess_off = int(sum(sess_intr_off_list)) if sess_intr_off_list else 0
        total_sess_on  = int(sum(sess_intr_on_list))  if sess_intr_on_list  else 0
        lines.append(f"| Total interruptions (session counters) | {total_sess_off} | {total_sess_on} | {total_sess_on - total_sess_off:+d} |")

    h3("Per-File Interruption Breakdown")
    lines.append("| File | Off total | On total | Δ | Off intr. turns | On intr. turns |")
    lines.append("|------|-----------|----------|---|-----------------|----------------|")
    for row in file_rows:
        off_turns_list = (row["entry"].get("filter_none")  or {}).get("turns", [])
        on_turns_list  = (row["entry"].get("filter_quail") or {}).get("turns", [])
        off_intr_vals = [t.get("interruptions", 0) or 0 for t in off_turns_list]
        on_intr_vals  = [t.get("interruptions", 0) or 0 for t in on_turns_list]
        off_total = int(sum(off_intr_vals))
        on_total  = int(sum(on_intr_vals))
        off_intr_t = sum(1 for v in off_intr_vals if v > 0)
        on_intr_t  = sum(1 for v in on_intr_vals  if v > 0)
        lines.append(
            f"| {row['file']} "
            f"| {off_total} | {on_total} | {on_total - off_total:+d} "
            f"| {off_intr_t}/{len(off_intr_vals) or '?'} turns "
            f"| {on_intr_t}/{len(on_intr_vals) or '?'} turns |"
        )

    # -----------------------------------------------------------------------
    # Appendix: Metric Definitions
    # -----------------------------------------------------------------------
    h2("Appendix: Metric Definitions")
    para("Glossary of every metric collected and reported above.")

    _METRIC_DEFS = [
        (
            "STT Latency",
            "Time from when the user stops speaking to when the speech-to-text engine returns a transcript.",
            "Recorded inside `WhisperSTTService`: timestamp taken at `_handle_user_stopped_speaking`, diff taken after `asyncio.to_thread(whisper.transcribe)` returns.",
            "Lower is better. Whisper on CPU is typically 1–2 s. Significant change between filter-off and filter-on would indicate the audio pre-processing is affecting Whisper's decoding time.",
        ),
        (
            "LLM TTFT",
            "Time To First Token — latency from when the transcript is sent to the LLM until the first response token arrives.",
            "Pipeline timer started when `LLMMessagesFrame` is pushed; stopped on the first `TextFrame` emitted by the LLM service.",
            "Lower is better. High values (> 2 s) indicate model load or network saturation. Should be nearly identical between filter conditions since LLM never sees the audio.",
        ),
        (
            "TTS Latency",
            "Time from when the LLM finishes generating text until the first audio chunk is synthesised and sent downstream.",
            "Pipeline timer from `TTSStartedFrame` to first `AudioRawFrame` out of the TTS service.",
            "Lower is better. Should be identical between filter conditions. `null` values mean the bot was interrupted before TTS completed.",
        ),
        (
            "RTT (Round-Trip Time)",
            "End-to-end latency for one conversational turn: from the moment the user's final word is detected until the bot's first audio byte is played back.",
            "Computed in `MetricsTracker` as `t_bot_audio_start − t_user_stopped`. Captures STT + LLM + TTS pipeline in sequence.",
            "Lower is better. The primary latency metric for perceived conversation responsiveness. DeepFilter should add at most a few ms here; large increases suggest a pipeline bottleneck.",
        ),
        (
            "STT Confidence",
            "How certain Whisper is about its transcription, expressed as a probability between 0 and 1.",
            "Derived from Whisper segment `avg_logprob` via `exp(avg_logprob)`, averaged across all accepted segments in a turn. Values near 1.0 indicate high confidence.",
            "Higher is better. The key signal for evaluating DeepFilter impact on audio quality. An increase with filter-on means DeepFilter improved the audio enough for Whisper to transcribe more reliably.",
        ),
        (
            "Speaking Rate (WPM)",
            "How fast the user spoke during a turn, in words per minute.",
            "Word count of the transcript divided by VAD-measured speech duration (`t_user_stopped − t_user_started`), converted to minutes.",
            "Informational. Helps distinguish genuine low-confidence transcripts (noisy audio) from short/fast utterances that are inherently harder to transcribe. `N/A` when VAD timing is unavailable.",
        ),
        (
            "Bot Words/Turn",
            "Number of words in the bot's response for a given turn.",
            "Simple whitespace split of the accumulated `bot_text` string at turn commit time.",
            "Informational. Very low values (0–1) usually indicate the turn was interrupted before the bot could respond fully. Consistent low values across a session may indicate pipeline issues.",
        ),
        (
            "Conversation Quality Score",
            "A holistic 1–5 rating of the bot's overall performance for the full session, assessed by an Ollama LLM judge after the pipeline finishes.",
            "After all turns complete, the full conversation transcript is sent to an Ollama model. It returns overall_score (1–5), overall_issue (dominant problem type), a summary sentence, and a list of flagged turns with per-turn issue notes.",
            "Higher is better. Scores ≥ 4 indicate a good session. Compare off vs on to see whether audio noise causes the bot to lose context or give wrong answers. Check flagged_turns for specific problem exchanges.",
        ),
        (
            "Issue Type",
            "Category of quality problem identified by the LLM judge for the session or for an individual flagged turn.",
            "LLM judge classifies sessions as one of: `none`, `context_loss`, `repetitive`, `wrong_entity`, `unhelpful`, `irrelevant`, `escalation_failure`, `too_long`, or `error`. Individual flagged turns use the same taxonomy.",
            "`none` is ideal. `context_loss` means the bot forgot earlier information (name, booking code, etc.). `wrong_entity` means it extracted wrong data. `repetitive` means it repeated itself. `escalation_failure` means it didn't escalate a frustrated customer.",
        ),
        (
            "Conf Δ (Transcript Comparison)",
            "Per-turn difference in STT confidence between filter-on and filter-off conditions.",
            "Computed as `conf_on − conf_off` for the same turn index (after diff-alignment). Positive = filter improved transcription; negative = filter degraded it.",
            "Look for consistent positive values on noisy sections of the WAV. Large negative values indicate DeepFilter over-processed the audio and hurt speech intelligibility.",
        ),
        (
            "p95 Latency",
            "The 95th-percentile value of a latency distribution — 95% of turns were faster than this.",
            "Computed from the sorted list of per-turn values for the given metric.",
            "More informative than the mean for latency. A low mean but high p95 indicates occasional very slow turns (e.g. long LLM responses). Compare p95 across conditions to detect tail-latency regressions.",
        ),
        (
            "p-value (statistical significance)",
            "Probability that the observed difference between filter-off and filter-on could have arisen by chance if there were no real effect.",
            "Two-sided Welch's t-test (unequal-variance) computed from per-turn values. Stars: `*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` not significant.",
            "Values below 0.05 indicate a statistically significant difference. With small sample sizes (< 20 files) most results will show `ns` — collect more data for reliable conclusions.",
        ),
        (
            "Interruptions",
            "Number of times the VAD detected user audio while the bot was actively playing back TTS audio during a given turn.",
            "Counted in MetricsTracker: each UserStartedSpeakingFrame that arrives while bot_speaking=True increments both TurnData.interruptions and SessionMetrics.total_interruptions.",
            "Lower is better. Non-zero values mean the VAD was triggered mid-response, which cuts off the bot and forces a re-prompt. With DeepFilter off, background noise in the WAV can cause spurious VAD triggers; DeepFilter on should suppress those and reduce interruption counts. A large drop in interruptions with filter-on is strong evidence the filter is removing noise that the VAD would otherwise react to.",
        ),
    ]
    def _esc_cell(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows_html = "".join(
        f"<tr><td><strong>{_esc_cell(name)}</strong></td>"
        f"<td>{_esc_cell(defn)}</td>"
        f"<td>{_esc_cell(measurement)}</td>"
        f"<td>{_esc_cell(guidance)}</td></tr>"
        for name, defn, measurement, guidance in _METRIC_DEFS
    )
    lines.append("")
    lines.append(
        '<table class="appendix-table">'
        "<thead><tr><th>Metric</th><th>Definition</th><th>How it's measured</th><th>What to look for</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )
    lines.append("")

    rule()
    para(f"*Report generated by analyze_metrics.py — {len(results)} files analysed.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(results: list[dict], output_path: Path) -> None:
    rows = []
    for entry in results:
        fname = entry.get("file", "")
        for cond_key, cond_label in [("filter_none", "off"), ("filter_quail", "on")]:
            sess = entry.get(cond_key) or {}
            if sess.get("error"):
                continue
            sq = sess.get("session_quality") or {}
            flagged_by_id = {ft.get("turn_id"): ft for ft in sq.get("flagged_turns", [])}
            for turn in sess.get("turns", []):
                tid = turn.get("turn_id")
                ft = flagged_by_id.get(tid, {})
                rows.append({
                    "file": fname,
                    "condition": cond_label,
                    "turn_id": tid,
                    "user_text": turn.get("user_text", ""),
                    "bot_text": turn.get("bot_text", ""),
                    "stt_ms": turn.get("stt_ms"),
                    "llm_ms": turn.get("llm_ms"),
                    "tts_ms": turn.get("tts_ms"),
                    "rtt_ms": turn.get("rtt_ms"),
                    "stt_confidence": turn.get("stt_confidence"),
                    "speaking_rate_wpm": turn.get("speaking_rate_wpm"),
                    "bot_word_count": turn.get("bot_word_count"),
                    "interruptions": turn.get("interruptions", 0),
                    "session_overall_score": sq.get("overall_score"),
                    "session_overall_issue": sq.get("overall_issue"),
                    "turn_flagged": bool(ft),
                    "turn_flag_issue": ft.get("issue"),
                    "turn_flag_note": ft.get("note"),
                })

    if not rows:
        print("No rows to write to CSV.", file=sys.stderr)
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepFilter Impact Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 14px; line-height: 1.6; color: #222; max-width: 1100px;
          margin: 40px auto; padding: 0 24px; }}
  h1 {{ font-size: 1.8em; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 1.3em; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 32px; }}
  h3 {{ font-size: 1.1em; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  th {{ background: #f0f0f0; font-weight: 600; text-align: left; }}
  th, td {{ border: 1px solid #d0d0d0; padding: 6px 10px; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  code {{ background: #f4f4f4; padding: 1px 5px; border-radius: 3px;
          font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 28px 0; }}
  ul {{ padding-left: 20px; }}
  em {{ color: #555; }}
  p {{ margin: 6px 0; }}
  /* ---- Conversation comparison table ---- */
  .cv-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; margin: 12px 0; }}
  .cv-th {{ background: #2c3e50; color: #fff; text-align: center; padding: 8px 12px;
             font-size: 0.9em; letter-spacing: 0.04em; border: 1px solid #1a252f; }}
  .cv-cell {{ width: 50%; padding: 10px 12px; border: 1px solid #ddd;
              vertical-align: top; word-break: break-word; }}
  .cv-del {{ background: #fff5f5; }}
  .cv-ins {{ background: #f5fff5; }}
  .cv-empty {{ background: #f7f7f7; text-align: center; color: #bbb; }}
  .cv-none  {{ color: #bbb; font-style: italic; font-size: 0.9em; }}
  .cv-turn-id {{ font-size: 0.72em; font-weight: 600; color: #999; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 6px; }}
  /* Interruption badge */
  .intr-badge {{ background: #fff8e1; border-left: 3px solid #f59e0b;
                 padding: 5px 8px; margin-bottom: 8px; border-radius: 0 4px 4px 0;
                 font-size: 0.82em; font-weight: 700; color: #92400e; }}
  .intr-detail {{ font-weight: 400; color: #555; margin-top: 3px; font-size: 0.95em; }}
  .intr-timing  {{ color: #888; font-size: 0.9em; }}
  /* User utterance row */
  .cv-user {{ display: flex; gap: 7px; align-items: baseline; margin-bottom: 3px; }}
  .cv-speaker {{ flex-shrink: 0; font-size: 1.05em; line-height: 1.4; }}
  .cv-utext {{ flex: 1; line-height: 1.5; }}
  /* Word-level diff marks */
  mark.wdiff-chg {{ background: #fff3b0; border-radius: 3px; padding: 0 2px; font-style: normal; }}
  mark.wdiff-del {{ background: #ffd0d0; border-radius: 3px; padding: 0 2px; font-style: normal; }}
  mark.wdiff-ins {{ background: #d0f0d0; border-radius: 3px; padding: 0 2px; font-style: normal; }}
  /* STT confidence badge */
  .cbadge {{ display: inline-block; font-size: 0.75em; font-weight: 700;
             padding: 1px 5px; border-radius: 3px; margin-left: 6px;
             vertical-align: middle; white-space: nowrap; }}
  .conf-high {{ background: #d1fae5; color: #065f46; }}
  .conf-mid  {{ background: #fef3c7; color: #92400e; }}
  .conf-low  {{ background: #fee2e2; color: #991b1b; }}
  /* Metrics line */
  .cv-metrics {{ font-size: 0.75em; color: #888; margin: 2px 0 6px 24px; }}
  /* Bot utterance row */
  .cv-bot {{ display: flex; gap: 7px; align-items: baseline; margin-top: 4px; }}
  .cv-btext {{ flex: 1; line-height: 1.5; color: #333; }}
  .cv-bot-muted .cv-btext {{ opacity: 0.55; font-style: italic; }}
  /* ---- Appendix table ---- */
  .appendix-table td {{ white-space: normal; word-break: break-word; vertical-align: top; }}
  .appendix-table td:nth-child(1) {{ width: 12%; font-weight: 600; }}
  .appendix-table td:nth-child(2) {{ width: 28%; }}
  .appendix-table td:nth-child(3) {{ width: 28%; }}
  .appendix-table td:nth-child(4) {{ width: 32%; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _save_html(report: str, path: Path) -> None:
    body = md_lib.markdown(report, extensions=["tables", "fenced_code"])
    path.write_text(_HTML_TEMPLATE.format(body=body), encoding="utf-8")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse DeepFilter impact from batch_test.py results.",
    )
    parser.add_argument("inputs", nargs="+", help="batch_*.json file(s)")
    parser.add_argument("--output", default="results/", help="Directory for report output")
    parser.add_argument("--csv", action="store_true", help="Also save per-turn CSV")
    parser.add_argument("--min-turns", type=int, default=1,
                        help="Skip sessions with fewer turns than this")

    args = parser.parse_args()

    # Expand glob patterns
    input_paths: list[Path] = []
    for pattern in args.inputs:
        matches = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        input_paths.extend(matches)

    if not input_paths:
        print("No matching input files found.", file=sys.stderr)
        sys.exit(1)

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        print(f"Files not found: {missing}", file=sys.stderr)
        sys.exit(1)

    results = load_results(input_paths)
    if not results:
        print("No results found in input files.", file=sys.stderr)
        sys.exit(1)

    report = build_report(results, min_turns=args.min_turns)

    # Print to stdout
    print(report)

    # Save HTML report
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis.html"
    _save_html(report, report_path)
    print(f"\nReport saved to {report_path}", file=sys.stderr)

    # Optional CSV
    if args.csv:
        csv_path = output_dir / "analysis.csv"
        write_csv(results, csv_path)


if __name__ == "__main__":
    main()
