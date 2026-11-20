#!/usr/bin/env python3
"""
Evaluation harness for Sentinel's deterministic detectors (Agent 2).

Generates synthetic daily series with a weekly pattern and multiplicative
noise, injects anomalies on KNOWN dates, runs each detector configuration,
and measures how well it finds them.

Each synthetic series contains three injected events:
  spike  - one day, +40% by default
  drop   - one day, -40% by default
  shift  - a sustained level shift, -25% for 14 days by default

Metrics (per detector configuration, averaged over runs):
  recall_spike / recall_drop / recall_shift  - share of events flagged on at least one day
  shift_coverage                              - share of the shift's days that were flagged
  mean_delay_days                             - days from event start to first flag (detected events)
  false_alarms_per_100_clean_days             - flags outside every event window (plus a short
                                                grace period after each event ends)
  false_alarms_event_free                     - flags per 100 days on series with NO injected events;
                                                the detector's inherent false-alarm rate, free of any
                                                contamination from earlier anomalies

Usage:
    python evaluate.py                          # 30 runs, prints a Markdown table
    python evaluate.py --runs 100 --seed 7 --noise 0.08 --output eval_results.md
"""

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta

import sentinel

WEEKDAY_BASE = {0: 1000, 1: 980, 2: 960, 3: 990, 4: 900, 5: 500, 6: 450}
GRACE_DAYS = 3        # flags this many days after an event ends are not counted as false alarms
SHIFT_LENGTH = 14
WARMUP = 70           # events start after this index so every window is full
MIN_DAYS = WARMUP + 75 + SHIFT_LENGTH + 10   # smallest series that fits all events

DEFAULT_CONFIGS = [
    (mode, window, z)
    for mode, windows in (("plain", (14, 28)), ("seasonal", (28, 56)))
    for window in windows
    for z in (2.0, 2.5, 3.0)
]


@dataclass
class Event:
    kind: str    # "spike", "drop" or "shift"
    start: int   # index, inclusive
    end: int     # index, inclusive


def generate_series(seed: int, n_days: int = 200, noise: float = 0.06,
                    spike: float = 0.40, drop: float = 0.40, shift: float = 0.25,
                    start: str = "2026-01-05"):
    """Returns (dates, values, events). Reproducible for a given seed.
    2026-01-05 is a Monday."""
    if n_days < MIN_DAYS:
        raise ValueError(f"n_days must be at least {MIN_DAYS} to fit a warm-up period and all three events.")
    rng = random.Random(seed)
    first = date.fromisoformat(start)
    dates = [(first + timedelta(days=i)).isoformat() for i in range(n_days)]
    baseline = [WEEKDAY_BASE[(first + timedelta(days=i)).weekday()] for i in range(n_days)]
    values = [b * (1 + rng.gauss(0, noise)) for b in baseline]

    # one event per zone, so events never overlap and each has a full window behind it
    spike_i = rng.randint(WARMUP + 5, WARMUP + 30)
    drop_i = rng.randint(WARMUP + 40, WARMUP + 65)
    shift_i = rng.randint(WARMUP + 75, n_days - SHIFT_LENGTH - 5)
    events = [
        Event("spike", spike_i, spike_i),
        Event("drop", drop_i, drop_i),
        Event("shift", shift_i, shift_i + SHIFT_LENGTH - 1),
    ]
    for ev in events:
        factor = {"spike": 1 + spike, "drop": 1 - drop, "shift": 1 - shift}[ev.kind]
        for i in range(ev.start, ev.end + 1):
            values[i] = baseline[i] * factor * (1 + rng.gauss(0, noise))
    return dates, values, events


def generate_clean_series(seed: int, n_days: int = 200, noise: float = 0.06,
                          start: str = "2026-01-05"):
    """Same weekly pattern and noise as generate_series, with no events."""
    rng = random.Random(seed + 100_000)
    first = date.fromisoformat(start)
    dates = [(first + timedelta(days=i)).isoformat() for i in range(n_days)]
    values = [WEEKDAY_BASE[(first + timedelta(days=i)).weekday()] * (1 + rng.gauss(0, noise))
              for i in range(n_days)]
    return dates, values


def run_detector(dates, values, mode: str, window: int, z: float) -> set:
    """Returns the set of flagged day indices."""
    fn = sentinel.detect_anomalies_seasonal if mode == "seasonal" else sentinel.detect_anomalies
    index = {d: i for i, d in enumerate(dates)}
    return {index[a.date] for a in fn(dates, values, window=window, z_threshold=z)}


def score_run(events, flagged: set, n_days: int, window: int) -> dict:
    """Scores one detector run against the known events."""
    result = {"detected": {}, "delay": {}, "shift_coverage": None}
    for ev in events:
        hits = sorted(i for i in flagged if ev.start <= i <= ev.end)
        result["detected"][ev.kind] = bool(hits)
        if hits:
            result["delay"][ev.kind] = hits[0] - ev.start
        if ev.kind == "shift":
            result["shift_coverage"] = len(hits) / (ev.end - ev.start + 1)

    zones = [(ev.start, ev.end + GRACE_DAYS) for ev in events]
    in_zone = lambda i: any(lo <= i <= hi for lo, hi in zones)
    clean_days = sum(1 for i in range(window, n_days) if not in_zone(i))
    false_flags = sum(1 for i in flagged if not in_zone(i))
    result["false_flags"] = false_flags
    result["clean_days"] = clean_days
    result["false_alarms_per_100"] = false_flags / clean_days * 100 if clean_days else 0.0
    return result


def evaluate_config(runs, mode: str, window: int, z: float, clean_runs=None) -> dict:
    """`runs` is a list of (dates, values, events); `clean_runs` an optional list
    of (dates, values) with no events. Returns one summary row."""
    clean_rates = []
    for dates, values in (clean_runs or []):
        flagged = run_detector(dates, values, mode, window, z)
        evaluated = len(dates) - window
        clean_rates.append(len(flagged) / evaluated * 100 if evaluated > 0 else 0.0)
    scored = []
    for dates, values, events in runs:
        flagged = run_detector(dates, values, mode, window, z)
        scored.append(score_run(events, flagged, len(dates), window))

    n = len(scored)
    delays = [d for s in scored for d in s["delay"].values()]
    return {
        "mode": mode,
        "window": window,
        "z": z,
        "recall_spike": sum(s["detected"]["spike"] for s in scored) / n,
        "recall_drop": sum(s["detected"]["drop"] for s in scored) / n,
        "recall_shift": sum(s["detected"]["shift"] for s in scored) / n,
        "shift_coverage": sum(s["shift_coverage"] for s in scored) / n,
        "mean_delay_days": sum(delays) / len(delays) if delays else None,
        "false_alarms_per_100_clean_days": sum(s["false_alarms_per_100"] for s in scored) / n,
        "false_alarms_event_free": (sum(clean_rates) / len(clean_rates)) if clean_rates else None,
    }


def run_evaluation(n_runs: int = 30, seed: int = 1, noise: float = 0.06,
                   spike: float = 0.40, drop: float = 0.40, shift: float = 0.25,
                   configs=None) -> list:
    configs = configs or DEFAULT_CONFIGS
    runs = [generate_series(seed + k, noise=noise, spike=spike, drop=drop, shift=shift)
            for k in range(n_runs)]
    clean_runs = [generate_clean_series(seed + k, noise=noise) for k in range(n_runs)]
    return [evaluate_config(runs, mode, window, z, clean_runs) for mode, window, z in configs]


def format_markdown(rows, n_runs, seed, noise, spike, drop, shift) -> str:
    pct = lambda x: f"{x * 100:.0f}%"
    lines = [
        "# Sentinel detector evaluation",
        "",
        f"{n_runs} synthetic series (200 days each, seed {seed}), weekly pattern with "
        f"{noise * 100:.0f}% multiplicative noise. Each contains one spike (+{spike * 100:.0f}%), "
        f"one drop (-{drop * 100:.0f}%), and one {SHIFT_LENGTH}-day level shift (-{shift * 100:.0f}%).",
        "",
        "| Mode | Window | z | Spike found | Drop found | Shift found | Shift days flagged | Mean delay (days) | False alarms / 100 days, event-free data | False alarms / 100 clean days, data with events |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        delay = "n/a" if r["mean_delay_days"] is None else f"{r['mean_delay_days']:.1f}"
        ef = "n/a" if r["false_alarms_event_free"] is None else f"{r['false_alarms_event_free']:.2f}"
        lines.append(
            f"| {r['mode']} | {r['window']} | {r['z']} | {pct(r['recall_spike'])} | "
            f"{pct(r['recall_drop'])} | {pct(r['recall_shift'])} | {pct(r['shift_coverage'])} | "
            f"{delay} | {ef} | {r['false_alarms_per_100_clean_days']:.2f} |"
        )
    lines += [
        "",
        "How to read this: a detector is only useful if it finds the injected events (found columns) "
        "without crying wolf (false alarms). 'Shift days flagged' shows how much of a sustained change "
        f"is still being reported once the window has absorbed it. Flags up to {GRACE_DAYS} days after an "
        "event ends are not counted as false alarms. Synthetic data is a sanity check, not proof of "
        "performance on real client data.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Evaluate Sentinel's detectors on synthetic data with injected anomalies.")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--noise", type=float, default=0.06, help="multiplicative noise, 0.06 = 6%%")
    p.add_argument("--spike", type=float, default=0.40)
    p.add_argument("--drop", type=float, default=0.40)
    p.add_argument("--shift", type=float, default=0.25)
    p.add_argument("--output", default=None, help="write the Markdown table to this file")
    p.add_argument("--json", action="store_true", help="print raw JSON rows instead of Markdown")
    a = p.parse_args()

    rows = run_evaluation(a.runs, a.seed, a.noise, a.spike, a.drop, a.shift)
    if a.json:
        print(json.dumps(rows, indent=2))
        return
    md = format_markdown(rows, a.runs, a.seed, a.noise, a.spike, a.drop, a.shift)
    print(md)
    if a.output:
        with open(a.output, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
