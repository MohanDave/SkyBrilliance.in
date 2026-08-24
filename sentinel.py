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

Works WITHOUT an API key: Agent 2 (the core detection logic) runs standalone
and prints a raw statistical report. Set ANTHROPIC_API_KEY to unlock Agents 1 and 3
for CSV normalization and the narrative brief.
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


def load_series(path: str, date_col: str, value_col: str):
    dates, values = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if date_col not in reader.fieldnames or value_col not in reader.fieldnames:
            raise ValueError(
                f"Columns not found. CSV has: {reader.fieldnames}. "
                f"Expected date_col='{date_col}', value_col='{value_col}'."
            )
        for row in reader:
            dates.append(row[date_col])
            values.append(float(row[value_col]))
    return dates, values


def detect_anomalies(dates, values, window: int = 14, z_threshold: float = 2.0):
    """
    Rolling mean/stddev z-score detection.
    For each point (after the first `window` points), compare it to the
    trailing window's mean/std. Flag anything beyond z_threshold std devs.
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
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--window", type=int, default=14, help="Rolling window size (days)")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument(
        "--output", default=None,
        help="Path to write the markdown brief (default: <csv_basename>_brief.md)"
    )
    parser.add_argument(
        "--no-md", action="store_true",
        help="Skip writing the markdown file, print to stdout only"
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    dates, values = load_series(args.csv_path, args.date_column, args.value_column)
    print(f"Loaded {len(values)} points from {args.csv_path}\n")

    quality_note = None
    if api_key:
        print("--- Agent 1: Data quality check ---")
        quality_note = agent1_clean_summary(dates, values, api_key)
        print(quality_note)
        print()
    else:
        print("[Agent 1 skipped — set ANTHROPIC_API_KEY to enable data quality check]\n")

    anomalies = detect_anomalies(dates, values, window=args.window, z_threshold=args.z_threshold)
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
