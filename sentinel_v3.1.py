#!/usr/bin/env python3
"""
Sentinel — Agentic pattern-detection pipeline for SMB business data.

Architecture (4 stages; only Agent 2 and the quality checks are deterministic):
  Agent 0 (Schema)    — LLM, optional. Infers header row / date / metric columns.
  Agent 1 (Quality)   — LLM, optional. Data quality note, fed by the deterministic
                         checks in check_series_quality().
  Agent 2 (Detector)  — Deterministic. Rolling z-score, or weekly-seasonality-aware.
                         Not an LLM call on purpose: cheaper, testable, reproducible.
  Agent 3 (Narrator)  — LLM, optional. Turns flagged anomalies into a plain-English brief.

Usage:
    python sentinel.py sample_data.csv --date-column date --value-column sessions
    python sentinel.py sample_data.csv                        # auto-detect columns (needs ANTHROPIC_API_KEY)

Works WITHOUT an API key: Agent 2 (the core detection logic) runs standalone
and prints a raw statistical report — as long as you pass column names
explicitly. Set ANTHROPIC_API_KEY to unlock Agent 0 (auto-detect columns),
Agent 1 (data quality check), and Agent 3 (narrative brief).
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Agent 2: Detector (deterministic — no LLM, runs always)
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    date: str
    value: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    direction: str  # "spike" or "drop"


def normalize_date(raw: str, dayfirst: bool = False) -> str:
    """Accepts common export date formats, returns ISO YYYY-MM-DD.
    Handles: YYYY-MM-DD (already ISO), YYYYMMDD (GA4 raw export style),
    and slash dates. Slash dates like 03/04/2026 are ambiguous: by default
    they are read month-first (US); pass dayfirst=True for DD/MM/YYYY
    exports (e.g. most Indian and UK exports)."""
    raw = raw.strip()
    slash_formats = ("%d/%m/%Y", "%m/%d/%Y") if dayfirst else ("%m/%d/%Y", "%d/%m/%Y")
    for fmt in ("%Y-%m-%d", "%Y%m%d") + slash_formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: '{raw}'. Add a format to normalize_date() if this is a new export type.")


def find_header_row(path: str, expected_cols) -> int:
    """Some platform exports (GA4 in particular) put report title/date-range
    metadata lines before the real header row. Scan the first 20 lines and
    return the index of the first one that contains all expected column
    names — that's the real header."""
    with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig strips an Excel BOM
        for i, line in enumerate(f):
            if i > 20:
                break
            cells = [c.strip().strip('"') for c in line.split(",")]
            if all(any(exp.lower() == c.lower() for c in cells) for exp in expected_cols):
                return i
    return 0  # assume no junk header rows — first line is the real header


def load_series(path: str, date_col: str, value_col: str, dayfirst: bool = False):
    header_row = find_header_row(path, [date_col, value_col])

    dates, values = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for _ in range(header_row):
            next(f)  # skip metadata lines before the real header
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"'{path}' looks empty: no header row found.")
        # case-insensitive column matching, since exports vary ("Date" vs "date")
        field_map = {fn.lower(): fn for fn in reader.fieldnames}
        actual_date_col = field_map.get(date_col.lower())
        actual_value_col = field_map.get(value_col.lower())
        if not actual_date_col or not actual_value_col:
            raise ValueError(
                f"Columns not found. CSV has: {reader.fieldnames}. "
                f"Expected date_col='{date_col}', value_col='{value_col}'."
            )
        for row in reader:
            raw_date = row[actual_date_col]
            raw_value = row[actual_value_col]
            if not raw_date or not raw_value:
                continue  # skip blank trailing rows some exports include
            try:
                dates.append(normalize_date(raw_date, dayfirst=dayfirst))
                values.append(float(str(raw_value).replace(",", "")))  # some exports comma-format numbers
            except ValueError:
                continue  # skip rows that aren't real data (e.g. a trailing "Totals" row)
    return dates, values


def detect_anomalies(dates, values, window: int = 14, z_threshold: float = 2.0):
    """
    Rolling mean/stddev z-score detection.
    For each point (after the first `window` points), compare it to the
    trailing window's mean/std. Flag anything beyond z_threshold std devs.

    Note: this does NOT account for weekly seasonality — a normal Monday dip
    and a real anomaly look the same to this function. For data with a
    weekly pattern (most web/business metrics), prefer
    detect_anomalies_seasonal() instead. Kept here for backward
    compatibility and because it's the right choice for data with no
    weekly cycle (e.g. already-weekly-aggregated data).
    """
    anomalies = []
    for i in range(window, len(values)):
        trailing = values[i - window:i]
        mean = sum(trailing) / window
        variance = sum((x - mean) ** 2 for x in trailing) / window
        std = variance ** 0.5
        if std == 0:
            continue
        z = (values[i] - mean) / std
        if abs(z) >= z_threshold:
            anomalies.append(
                Anomaly(
                    date=dates[i],
                    value=values[i],
                    rolling_mean=round(mean, 2),
                    rolling_std=round(std, 2),
                    z_score=round(z, 2),
                    direction="spike" if z > 0 else "drop",
                )
            )
    return anomalies


def detect_anomalies_seasonal(dates, values, window: int = 28, z_threshold: float = 2.0):
    """
    Weekly-seasonality-aware anomaly detection.

    Problem this solves: a plain rolling z-score treats "it's always lower
    on Sundays" the same as a real anomaly, because it only looks at the
    recent trend, not the day-of-week pattern. For any metric with a weekly
    cycle (website traffic, orders, engagement — most business metrics),
    that produces false positives on exactly the days where nothing is wrong.

    How it works (classical multiplicative decomposition, not a library):
      1. For each point, look at the trailing `window` days.
      2. Estimate a seasonal factor per weekday: how much higher/lower that
         weekday tends to run vs. the window's overall average.
         e.g. factor 0.85 for Sunday means "Sundays run 15% below average."
      3. Compute this point's SEASONALLY-EXPECTED value: overall trailing
         mean × that weekday's factor.
      4. Compute the residual: actual value minus seasonally-expected value.
      5. Z-score the residual against the trailing window's residual std —
         NOT against raw values. This is what actually removes the weekly
         pattern from the comparison.

    Requires window >= 14 (two full weeks) to estimate weekday factors at
    all; window >= 21 is recommended for stability, since a 14-day window
    only gives 2 samples per weekday.
    """
    if window < 14:
        raise ValueError("Seasonal detection needs window >= 14 (two weeks) to estimate weekday factors.")

    anomalies = []
    for i in range(window, len(values)):
        trailing_vals = values[i - window:i]
        trailing_dates = dates[i - window:i]
        trailing_weekdays = [datetime.fromisoformat(d).weekday() for d in trailing_dates]

        overall_mean = sum(trailing_vals) / len(trailing_vals)
        if overall_mean == 0:
            continue

        # Seasonal factor per weekday (0=Monday ... 6=Sunday)
        weekday_sums = {wd: 0.0 for wd in range(7)}
        weekday_counts = {wd: 0 for wd in range(7)}
        for v, wd in zip(trailing_vals, trailing_weekdays):
            weekday_sums[wd] += v
            weekday_counts[wd] += 1

        seasonal_factor = {}
        n_groups_present = 0
        for wd in range(7):
            if weekday_counts[wd] > 0:
                weekday_avg = weekday_sums[wd] / weekday_counts[wd]
                seasonal_factor[wd] = weekday_avg / overall_mean
                n_groups_present += 1
            else:
                seasonal_factor[wd] = 1.0  # no data for this weekday yet — assume no seasonal effect

        # Deseasonalize the trailing window itself to get residual std.
        # IMPORTANT: each weekday's seasonal factor was estimated FROM these
        # same trailing points, so naive variance (divide by N) is biased
        # low — every point pulled its own weekday average toward itself,
        # shrinking its own residual. This is the same degrees-of-freedom
        # issue as one-way ANOVA: n_groups_present weekday means were fit
        # from this window, so divide by (N - n_groups_present), not N.
        # Skipping this correction was tested and produced ~2-3x too many
        # false positives on pure-noise seasonal data.
        residuals = [
            v - (overall_mean * seasonal_factor[wd])
            for v, wd in zip(trailing_vals, trailing_weekdays)
        ]
        dof = max(len(residuals) - n_groups_present, 1)
        resid_variance = sum(r ** 2 for r in residuals) / dof  # residuals are ~0-mean by construction
        resid_std = resid_variance ** 0.5
        if resid_std == 0:
            continue

        # This point's seasonally-expected value and residual
        cur_weekday = datetime.fromisoformat(dates[i]).weekday()
        expected = overall_mean * seasonal_factor[cur_weekday]
        residual = values[i] - expected
        z = residual / resid_std

        if abs(z) >= z_threshold:
            anomalies.append(
                Anomaly(
                    date=dates[i],
                    value=values[i],
                    rolling_mean=round(expected, 2),  # seasonally-adjusted expectation, not raw mean
                    rolling_std=round(resid_std, 2),
                    z_score=round(z, 2),
                    direction="spike" if z > 0 else "drop",
                )
            )
    return anomalies


def merge_consecutive_anomalies(anomalies):
    """Collapse consecutive flagged days into a single 'episode' — a decline
    trend should read as one event, not 15 separate alerts."""
    if not anomalies:
        return []
    episodes = [[anomalies[0]]]
    for a in anomalies[1:]:
        last_date = datetime.fromisoformat(episodes[-1][-1].date)
        cur_date = datetime.fromisoformat(a.date)
        if (cur_date - last_date).days <= 3 and a.direction == episodes[-1][-1].direction:
            episodes[-1].append(a)
        else:
            episodes.append([a])
    return episodes


# ---------------------------------------------------------------------------
# Deterministic data-quality checks (no LLM, always run)
#
# Agent 1 used to see only the first 5 rows, so it could not notice gaps,
# duplicates or unsorted dates anywhere else in the file. These checks run
# over the whole series in plain code, and their findings are passed to
# Agent 1 and printed in the report even when no API key is set.
# ---------------------------------------------------------------------------

def check_series_quality(dates, values) -> dict:
    """Whole-series checks on ISO dates. Returns a dict of findings."""
    unique_sorted = sorted(set(dates))
    counts = {}
    for d in dates:
        counts[d] = counts.get(d, 0) + 1
    duplicates = sorted(d for d, c in counts.items() if c > 1)

    missing_days, largest_gap = 0, 0
    for prev, cur in zip(unique_sorted, unique_sorted[1:]):
        gap = (datetime.fromisoformat(cur) - datetime.fromisoformat(prev)).days - 1
        if gap > 0:
            missing_days += gap
            largest_gap = max(largest_gap, gap)

    return {
        "n_points": len(dates),
        "first_date": unique_sorted[0] if unique_sorted else None,
        "last_date": unique_sorted[-1] if unique_sorted else None,
        "unsorted": list(dates) != sorted(dates),
        "duplicate_dates": duplicates,
        "missing_days": missing_days,
        "largest_gap_days": largest_gap,
    }


def describe_quality_issues(q: dict) -> list:
    """Human-readable lines for any problems found. Empty list = clean."""
    issues = []
    if q["unsorted"]:
        issues.append("Dates were not in chronological order (Sentinel sorted them before detection).")
    if q["duplicate_dates"]:
        shown = ", ".join(q["duplicate_dates"][:5])
        more = "" if len(q["duplicate_dates"]) <= 5 else f" and {len(q['duplicate_dates']) - 5} more"
        issues.append(f"{len(q['duplicate_dates'])} date(s) appear more than once: {shown}{more}.")
    if q["missing_days"]:
        issues.append(
            f"{q['missing_days']} calendar day(s) have no row (largest gap: {q['largest_gap_days']} day(s)). "
            "Detection windows count rows, not days, so gaps make results less reliable."
        )
    return issues


# ---------------------------------------------------------------------------
# Agent 0: Schema Detective (LLM — optional, requires ANTHROPIC_API_KEY)
#
# Solves a real, previously-open gap: using Sentinel on a new client's data
# meant a human manually inspecting the CSV and telling the tool which
# column is the date and which is the metric. That's still "custom parsing
# per client," just done by a person instead of code. This agent looks at
# the raw, UNPARSED file (arbitrary headers, arbitrary metadata rows) and
# infers that mapping itself.
#
# Scope, stated honestly: this handles a single flat CSV/table with one
# date column and one or more numeric metric columns — the same shape
# every export we've tested (GA4, Shopify-style, plain CSV) already has.
# It does NOT solve multi-sheet workbooks, unaggregated line-item
# transaction logs, or mixed-currency files in one column — those are a
# harder, still-open problem, not something this claims to fix.
# ---------------------------------------------------------------------------

def build_schema_inference_prompt(raw_sample: str) -> str:
    """Pure function: builds the prompt. Separated from the API call so the
    prompt itself — and the parsing of whatever comes back — can be unit
    tested without needing a live API key."""
    return (
        "You are looking at the first lines of a raw CSV export, exactly as "
        "a file on disk, including any metadata or title lines before the "
        "real header row. Identify: (1) which line number (0-indexed) is "
        "the actual column header row, (2) which column name is a date/time "
        "column, (3) which column name(s) contain a numeric metric suitable "
        "for trend analysis (skip ID columns, categorical text columns, and "
        "columns that are mostly empty). "
        "Respond with ONLY a JSON object, no other text, no markdown fences, "
        "in exactly this shape: "
        '{"header_row": <int>, "date_column": "<name or null>", '
        '"value_columns": ["<name>", ...], "notes": "<one short sentence>"}. '
        "If you cannot confidently identify a date column or any numeric "
        "metric column, set date_column to null and/or value_columns to an "
        "empty list — do not guess.\n\n"
        f"Raw file sample:\n{raw_sample}"
    )


def parse_schema_inference_response(response_text: str) -> dict:
    """Pure function: parses Agent 0's response into a usable dict, with
    honest fallback behavior for the ways a real LLM response can go wrong
    (wrapped in markdown fences, extra prose despite instructions, truly
    malformed JSON). Never silently invents a column mapping — an
    unparseable response returns date_column=None, which the caller must
    treat as "inference failed, ask the user," not as a guess."""
    text = response_text.strip()

    # Strip markdown code fences if the model added them despite instructions
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {
            "header_row": 0,
            "date_column": None,
            "value_columns": [],
            "notes": "Could not parse schema inference response — falling back to manual column specification.",
            "raw_response": response_text,
        }

    # Defensive defaults for a technically-valid-JSON-but-wrong-shape response
    return {
        "header_row": parsed.get("header_row", 0) if isinstance(parsed.get("header_row"), int) else 0,
        "date_column": parsed.get("date_column") if isinstance(parsed.get("date_column"), str) else None,
        "value_columns": parsed.get("value_columns") if isinstance(parsed.get("value_columns"), list) else [],
        "notes": parsed.get("notes", ""),
    }


def agent0_infer_schema(csv_path: str, api_key: str, sample_lines: int = 20) -> dict:
    """Reads the first `sample_lines` of the raw file and asks Claude to
    infer the schema. Returns the same dict shape as
    parse_schema_inference_response — callers check date_column is not
    None before trusting the result."""
    import itertools
    with open(csv_path, newline="") as f:
        sample = "".join(itertools.islice(f, sample_lines))
    prompt = build_schema_inference_prompt(sample)
    response = call_claude(prompt, api_key, max_tokens=300)
    return parse_schema_inference_response(response)


# ---------------------------------------------------------------------------
# Agent 1: Cleaner (LLM — optional, requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

def call_claude(prompt: str, api_key: str, max_tokens: int = 1000) -> str:
    import urllib.request

    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return "".join(block.get("text", "") for block in data.get("content", []))


def agent1_clean_summary(dates, values, api_key: str, quality: dict = None) -> str:
    """Ask Claude to sanity-check the data shape and flag obvious quality issues.
    It sees the first and last 5 points plus the deterministic findings from
    check_series_quality(), so it reasons about facts instead of guessing about
    gaps and duplicates it cannot see. (In a fuller version, this agent would
    take the RAW unparsed CSV text and return normalized JSON.)"""
    head = list(zip(dates[:5], values[:5]))
    tail = list(zip(dates[-5:], values[-5:]))
    facts = ""
    if quality is not None:
        found = describe_quality_issues(quality) or ["none"]
        facts = f" Automated checks over the full series found: {' '.join(found)}"
    prompt = (
        "You are a data quality checker. Here are the first and last 5 points of a "
        f"time series (date, value): first={head}, last={tail}. "
        f"Total points: {len(values)}.{facts} "
        "In 2-3 sentences, note anything that looks like a data quality issue "
        "(gaps, duplicate dates, obviously wrong scale) or say it looks clean. "
        "Do not contradict the automated checks."
    )
    return call_claude(prompt, api_key, max_tokens=200)


# ---------------------------------------------------------------------------
# Agent 3: Narrator (LLM — optional, requires ANTHROPIC_API_KEY)
#
# The brief is only shown if it passes a deterministic grounding check:
# every date and every number in it must exist in Agent 2's output. If the
# model's draft fails (after one retry), or the API call fails, Sentinel
# falls back to a templated brief built directly from the findings, so a
# made-up figure can never reach the report.
# ---------------------------------------------------------------------------

import re
from datetime import timedelta

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_PATTERN})\b", re.IGNORECASE)
_MONTH_DAY = re.compile(rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


@dataclass
class NarrativeResult:
    text: str          # the brief to show
    source: str        # "llm" (passed verification) or "fallback" (templated)
    attempts: int      # number of LLM attempts made
    violations: list   # problems found in the last rejected draft ([] if source == "llm")


def build_episode_facts(episodes) -> list:
    """The only facts Agent 3 is allowed to cite. All numbers are computed
    here, in code, so the model never has to do arithmetic."""
    facts = []
    for n, ep in enumerate(episodes, 1):
        avg_value = sum(a.value for a in ep) / len(ep)
        avg_expected = sum(a.rolling_mean for a in ep) / len(ep)
        pct = round((avg_value - avg_expected) / avg_expected * 100, 1) if avg_expected else None
        span = (datetime.fromisoformat(ep[-1].date) - datetime.fromisoformat(ep[0].date)).days + 1
        facts.append({
            "episode": n,
            "start": ep[0].date,
            "end": ep[-1].date,
            "direction": ep[0].direction,
            "days_flagged": len(ep),
            "span_days": span,
            "worst_z_score": max(abs(a.z_score) for a in ep),
            "value_start": ep[0].value,
            "value_end": ep[-1].value,
            "average_value": round(avg_value, 1),
            "average_expected": round(avg_expected, 1),
            "percent_vs_expected": pct,
        })
    return facts


def build_narration_prompt(metric_name: str, facts: list) -> str:
    return (
        f"You are writing a short weekly brief for a small business owner about "
        f"their '{metric_name}' metric. Below are statistically flagged anomaly "
        "episodes, already detected and measured. Do not re-derive anything. "
        "Rules: (1) use ONLY dates and numbers that appear in the facts, and write "
        "dates as YYYY-MM-DD; (2) do not calculate, estimate, or mention any other "
        "figures (no window sizes, no week counts, no totals you worked out); "
        "(3) frame possible business reasons as questions to investigate, never as "
        "confirmed causes; (4) the summary is plain English, under 100 words, no "
        "preamble. Respond with ONLY a JSON object, no markdown fences, in exactly "
        'this shape: {"summary": "<text>", "questions": ["<question>", ...]} '
        f"with 1 to 3 questions. Facts: {json.dumps(facts)}"
    )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_narrative_response(response_text: str):
    """Returns {"summary": str, "questions": [str, ...]} or None if the
    response is not usable. Never guesses."""
    try:
        parsed = json.loads(_strip_code_fences(response_text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    questions = parsed.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    return {
        "summary": summary.strip(),
        "questions": [q.strip() for q in questions if isinstance(q, str) and q.strip()],
    }


def compose_narrative_text(parsed: dict) -> str:
    text = parsed["summary"]
    if parsed["questions"]:
        text += "\n\n**Worth checking:**\n" + "\n".join(f"- {q}" for q in parsed["questions"])
    return text


def check_narrative_grounding(text: str, facts: list) -> list:
    """Returns a list of violations (empty list = every date and number in
    `text` exists in `facts`). Numbers match if they round to a fact at the
    precision written, so "23%" matches 23.4 and "1,023" matches 1023.4."""
    allowed_dates = set()
    for f in facts:
        d, last = datetime.fromisoformat(f["start"]), datetime.fromisoformat(f["end"])
        while d <= last:
            allowed_dates.add(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
    allowed_month_days = {(int(x[5:7]), int(x[8:10])) for x in allowed_dates}

    allowed_numbers = {float(len(facts))}
    for f in facts:
        for k, v in f.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                allowed_numbers.add(abs(float(v)))
    allowed_numbers |= {float(x[:4]) for x in allowed_dates}  # years may be mentioned

    violations = []

    for m in _ISO_DATE.findall(text):
        if m not in allowed_dates:
            violations.append(f"date {m} is not inside any detected episode")
    text = _ISO_DATE.sub(" ", text)

    for m in _DAY_MONTH.finditer(text):
        month, day = _MONTHS[m.group(2).lower()], int(m.group(1))
        if (month, day) not in allowed_month_days:
            violations.append(f"date '{m.group(0)}' is not inside any detected episode")
    text = _DAY_MONTH.sub(" ", text)
    for m in _MONTH_DAY.finditer(text):
        month, day = _MONTHS[m.group(1).lower()], int(m.group(2))
        if (month, day) not in allowed_month_days:
            violations.append(f"date '{m.group(0)}' is not inside any detected episode")
    text = _MONTH_DAY.sub(" ", text)

    for m in _NUMBER.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            value = abs(float(raw))
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(value - a) <= tol for a in allowed_numbers):
            violations.append(f"number '{m.group(0)}' does not match any detected figure")
    return violations


def render_fallback_brief(metric_name: str, facts: list) -> str:
    """Templated brief built only from the facts. Used when the model's draft
    fails verification or the API call fails. Passes its own grounding check."""
    lines = [f"{metric_name}: {len(facts)} unusual period(s) detected."]
    for f in facts:
        pct = f["percent_vs_expected"]
        if pct is None:
            rel = ""
        else:
            rel = f", {abs(pct)}% {'above' if pct > 0 else 'below'} expected"
        period = f["start"] if f["start"] == f["end"] else f"{f['start']} to {f['end']}"
        lines.append(
            f"- {f['direction'].capitalize()}: {period} ({f['days_flagged']} flagged day(s)). "
            f"Average {f['average_value']} vs about {f['average_expected']} expected{rel}; "
            f"worst z-score {f['worst_z_score']}."
        )
    lines.append("")
    lines.append(
        "**Worth checking:** a change in marketing channels or campaigns, a pricing or "
        "promotion change, a site or tracking problem, or a seasonal or calendar effect."
    )
    return "\n".join(lines)


def agent3_narrate_verified(episodes, metric_name: str, api_key: str,
                            max_retries: int = 1, llm=None) -> NarrativeResult:
    """Asks the model for a JSON brief, verifies it against the detected
    facts, retries once with the violations listed, then falls back to the
    templated brief. API failures also fall back instead of crashing."""
    llm = llm or call_claude
    facts = build_episode_facts(episodes)
    base_prompt = build_narration_prompt(metric_name, facts)
    violations, attempts = [], 0

    for _ in range(max_retries + 1):
        attempts += 1
        prompt = base_prompt
        if violations:
            prompt += (
                " Your previous answer was rejected for these problems: "
                + "; ".join(violations)
                + ". Rewrite it using only dates and numbers from the facts."
            )
        try:
            raw = llm(prompt, api_key, max_tokens=500)
        except Exception as e:  # network / API failure: degrade, don't crash
            violations = [f"API call failed: {type(e).__name__}: {e}"]
            break
        parsed = parse_narrative_response(raw)
        if parsed is None:
            violations = ["response was not valid JSON with a non-empty 'summary' string"]
            continue
        text = compose_narrative_text(parsed)
        violations = check_narrative_grounding(text, facts)
        if not violations:
            return NarrativeResult(text, "llm", attempts, [])

    return NarrativeResult(render_fallback_brief(metric_name, facts), "fallback", attempts, violations)


def agent3_narrate(episodes, metric_name: str, api_key: str) -> str:
    """Backward-compatible wrapper: returns just the verified brief text."""
    return agent3_narrate_verified(episodes, metric_name, api_key).text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_markdown_report(csv_path, value_col, quality_note, episodes, narrative,
                          quality_issues=None, narrative_note=None):
    lines = [
        f"# Sentinel Brief — {value_col}",
        f"*Source: {csv_path} · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    lines.append("## Data Quality")
    if quality_issues is not None:
        if quality_issues:
            lines.append("**Automated checks:**")
            lines.extend(f"- {issue}" for issue in quality_issues)
        else:
            lines.append("**Automated checks:** no gaps, duplicates or ordering problems found.")
        lines.append("")
    lines.append("**Agent 1 note:** " + (quality_note or "_Skipped — no ANTHROPIC_API_KEY set._"))
    lines.append("")

    lines.append(f"## Detected Anomalies (Agent 2) — {len(episodes)} episode(s)")
    if not episodes:
        lines.append("No anomalies above threshold.")
    else:
        lines.append("| Direction | Start | End | Days | Worst z-score |")
        lines.append("|---|---|---|---|---|")
        for ep in episodes:
            lines.append(
                f"| {ep[0].direction} | {ep[0].date} | {ep[-1].date} | "
                f"{len(ep)} | {max(abs(a.z_score) for a in ep)} |"
            )
    lines.append("")

    lines.append("## Narrative Brief (Agent 3)")
    lines.append(narrative or "_Skipped — no ANTHROPIC_API_KEY set._")
    if narrative and narrative_note:
        lines.append("")
        lines.append(f"_{narrative_note}_")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sentinel: agentic anomaly detection for business metrics")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--date-column", default=None,
                         help="Date column name. If omitted, requires ANTHROPIC_API_KEY "
                              "so Agent 0 can auto-detect it.")
    parser.add_argument("--value-column", default=None,
                         help="Value column name. If omitted, requires ANTHROPIC_API_KEY "
                              "so Agent 0 can auto-detect it.")
    parser.add_argument("--window", type=int, default=None,
                         help="Rolling window size (days). Default: 14 normally, 28 with --seasonal.")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument(
        "--dayfirst", action="store_true",
        help="Read ambiguous slash dates (03/04/2026) as DD/MM/YYYY instead of MM/DD/YYYY."
    )
    parser.add_argument(
        "--seasonal", action="store_true",
        help="Use weekly-seasonality-aware detection instead of plain rolling z-score. "
             "Recommended for data with a weekly cycle (traffic, orders, engagement). "
             "Requires --window >= 14 (defaults to 28 if not set explicitly)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write the markdown brief (default: <csv_basename>_brief.md)"
    )
    parser.add_argument(
        "--no-md", action="store_true",
        help="Skip writing the markdown file, print to stdout only"
    )
    args = parser.parse_args()
    window = args.window if args.window is not None else (28 if args.seasonal else 14)

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    date_col, value_col = args.date_column, args.value_column
    if date_col is None or value_col is None:
        if not api_key:
            print(
                "Error: --date-column and --value-column weren't given, and "
                "auto-detection (Agent 0) needs ANTHROPIC_API_KEY set. "
                "Either set the key, or pass both column names explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("--- Agent 0: Inferring schema from raw file ---")
        schema = agent0_infer_schema(args.csv_path, api_key)
        if schema["date_column"] is None or not schema["value_columns"]:
            print(
                f"Error: Agent 0 could not confidently infer the schema "
                f"({schema.get('notes', 'no reason given')}). "
                "Pass --date-column and --value-column explicitly instead.",
                file=sys.stderr,
            )
            sys.exit(1)
        date_col = schema["date_column"]
        value_col = schema["value_columns"][0]
        print(f"Detected: date_column='{date_col}', value_column='{value_col}'")
        if len(schema["value_columns"]) > 1:
            print(f"(Also found: {schema['value_columns'][1:]} — using the first. "
                  f"Pass --value-column explicitly to pick a different one.)")
        if schema.get("notes"):
            print(f"Note: {schema['notes']}")
        print()

    dates, values = load_series(args.csv_path, date_col, value_col, dayfirst=args.dayfirst)
    print(f"Loaded {len(values)} points from {args.csv_path}\n")

    quality = check_series_quality(dates, values)
    quality_issues = describe_quality_issues(quality)
    if quality["unsorted"]:
        dates, values = (list(t) for t in zip(*sorted(zip(dates, values), key=lambda p: p[0])))
    if quality_issues:
        print("--- Automated data checks ---")
        for issue in quality_issues:
            print(f"  ! {issue}")
        print()

    quality_note = None
    if api_key:
        print("--- Agent 1: Data quality check ---")
        quality_note = agent1_clean_summary(dates, values, api_key, quality=quality)
        print(quality_note)
        print()
    else:
        print("[Agent 1 skipped — set ANTHROPIC_API_KEY to enable data quality check]\n")

    try:
        if args.seasonal:
            print(f"--- Agent 2: Detecting (weekly-seasonality-aware, window={window}) ---")
            anomalies = detect_anomalies_seasonal(dates, values, window=window, z_threshold=args.z_threshold)
        else:
            anomalies = detect_anomalies(dates, values, window=window, z_threshold=args.z_threshold)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    episodes = merge_consecutive_anomalies(anomalies)

    print(f"--- Agent 2: Detected {len(episodes)} anomaly episode(s) ---")
    for ep in episodes:
        print(
            f"  {ep[0].direction.upper()}: {ep[0].date} -> {ep[-1].date} "
            f"({len(ep)} day(s), worst z-score {max(abs(a.z_score) for a in ep)})"
        )
    if not episodes:
        print("No anomalies above threshold.")
    print()

    narrative, narrative_note = None, None
    if episodes and api_key:
        print("--- Agent 3: Narrative brief ---")
        result = agent3_narrate_verified(episodes, value_col, api_key)
        narrative = result.text
        if result.source == "llm":
            narrative_note = "Every date and figure in this brief was checked against Agent 2's output."
        else:
            narrative_note = ("The model's draft could not be verified against the detected "
                              "episodes, so this is a templated brief built directly from them.")
            print(f"[Agent 3 draft rejected after {result.attempts} attempt(s): "
                  f"{'; '.join(result.violations)}. Using templated brief.]")
        print(narrative)
    elif not api_key:
        print("[Agent 3 skipped — set ANTHROPIC_API_KEY to enable the narrative brief]")

    if not args.no_md:
        report = build_markdown_report(args.csv_path, value_col, quality_note, episodes, narrative,
                                       quality_issues, narrative_note)
        out_path = args.output or f"{os.path.splitext(os.path.basename(args.csv_path))[0]}_brief.md"
        with open(out_path, "w") as f:
            f.write(report)
        print(f"\nMarkdown brief written to {out_path}")


if __name__ == "__main__":
    main()
