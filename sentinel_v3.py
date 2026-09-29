#!/usr/bin/env python3
"""
Sentinel — Agentic pattern-detection pipeline for SMB business data.

Architecture (3 agents, only 2 of which are LLM calls — see README for why):
  Agent 1 (Cleaner)   — LLM. Normalizes a messy CSV into consistent JSON.
  Agent 2 (Detector)  — Deterministic. Rolling mean/stddev anomaly detection.
                         Not an LLM call on purpose: cheaper, testable, reproducible.
  Agent 3 (Narrator)   — LLM. Turns flagged anomalies into a plain-English brief.

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


def normalize_date(raw: str) -> str:
    """Accepts common export date formats, returns ISO YYYY-MM-DD.
    Handles: YYYY-MM-DD (already ISO), YYYYMMDD (GA4 raw export style),
    MM/DD/YYYY (common US export style)."""
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%d/%m/%Y"):
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
    with open(path, newline="") as f:
        for i, line in enumerate(f):
            if i > 20:
                break
            cells = [c.strip().strip('"') for c in line.split(",")]
            if all(any(exp.lower() == c.lower() for c in cells) for exp in expected_cols):
                return i
    return 0  # assume no junk header rows — first line is the real header


def load_series(path: str, date_col: str, value_col: str):
    header_row = find_header_row(path, [date_col, value_col])

    dates, values = [], []
    with open(path, newline="") as f:
        for _ in range(header_row):
            next(f)  # skip metadata lines before the real header
        reader = csv.DictReader(f)
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
                dates.append(normalize_date(raw_date))
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


def agent1_clean_summary(dates, values, api_key: str) -> str:
    """Ask Claude to sanity-check the data shape and flag obvious quality issues.
    (In a fuller version, this agent would take the RAW unparsed CSV text and
    return normalized JSON. Kept simple here since load_series already parses.)"""
    sample = list(zip(dates[:5], values[:5]))
    prompt = (
        "You are a data quality checker. Here is a sample of a time series "
        f"(date, value): {sample}. Total points: {len(values)}. "
        "In 2-3 sentences, note anything that looks like a data quality issue "
        "(gaps, duplicate dates, obviously wrong scale) or say it looks clean."
    )
    return call_claude(prompt, api_key, max_tokens=200)


# ---------------------------------------------------------------------------
# Agent 3: Narrator (LLM — optional, requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

def agent3_narrate(episodes, metric_name: str, api_key: str) -> str:
    episode_summaries = []
    for ep in episodes:
        episode_summaries.append({
            "start": ep[0].date,
            "end": ep[-1].date,
            "direction": ep[0].direction,
            "days": len(ep),
            "worst_z_score": max(abs(a.z_score) for a in ep),
            "value_start": ep[0].value,
            "value_end": ep[-1].value,
        })
    prompt = (
        f"You are writing a short weekly brief for a small business owner about "
        f"their '{metric_name}' metric. Here are statistically flagged anomaly "
        f"episodes (already detected — do not re-derive, just interpret): "
        f"{json.dumps(episode_summaries)}. "
        "Write a plain-English brief: what changed, roughly when, and one or two "
        "plausible business reasons worth checking (e.g. seasonality, a channel "
        "change, a pricing change) — frame these as questions to investigate, "
        "not confirmed causes. Keep it under 150 words. No preamble."
    )
    return call_claude(prompt, api_key, max_tokens=400)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_markdown_report(csv_path, value_col, quality_note, episodes, narrative):
    lines = [
        f"# Sentinel Brief — {value_col}",
        f"*Source: {csv_path} · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    lines.append("## Data Quality (Agent 1)")
    lines.append(quality_note or "_Skipped — no ANTHROPIC_API_KEY set._")
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

    dates, values = load_series(args.csv_path, date_col, value_col)
    print(f"Loaded {len(values)} points from {args.csv_path}\n")

    quality_note = None
    if api_key:
        print("--- Agent 1: Data quality check ---")
        quality_note = agent1_clean_summary(dates, values, api_key)
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

    narrative = None
    if episodes and api_key:
        print("--- Agent 3: Narrative brief ---")
        narrative = agent3_narrate(episodes, args.value_column, api_key)
        print(narrative)
    elif not api_key:
        print("[Agent 3 skipped — set ANTHROPIC_API_KEY to enable the narrative brief]")

    if not args.no_md:
        report = build_markdown_report(args.csv_path, args.value_column, quality_note, episodes, narrative)
        out_path = args.output or f"{os.path.splitext(os.path.basename(args.csv_path))[0]}_brief.md"
        with open(out_path, "w") as f:
            f.write(report)
        print(f"\nMarkdown brief written to {out_path}")


if __name__ == "__main__":
    main()
