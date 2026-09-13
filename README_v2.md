# Sentinel

Agentic anomaly detection for small-business metrics. Give it a CSV of a
metric over time (website sessions, engagement rate, orders — anything with
a date and a number), and it tells you when something changed and offers a
plain-English read on what to check.

Built by SkyBrilliance to encode, in code, the kind of trend-decline catch
we've previously done manually for clients.

## Architecture — 3 agents, only 2 are LLM calls

| Agent | Type | Job |
|---|---|---|
| 1. Cleaner | LLM | Sanity-checks the data for obvious quality issues |
| 2. Detector | **Deterministic** | Rolling mean/stddev z-score anomaly detection |
| 3. Narrator | LLM | Turns flagged anomalies into a plain-English brief |

**Why Agent 2 isn't an LLM call:** anomaly detection on numeric time series
is arithmetic. An LLM is slower, more expensive, and non-deterministic for a
job a rolling z-score does exactly and reproducibly. The LLM's job is
*interpretation* (agents 1 and 3), not *arithmetic* (agent 2). This is a
deliberate architecture choice, not a limitation.

## Quick start

```bash
# Works immediately, no API key needed — runs Agent 2 standalone,
# writes sample_data_brief.md
python sentinel.py sample_data.csv --value-column sessions

# Full pipeline (Agents 1 and 3 activate)
export ANTHROPIC_API_KEY=your_key_here
python sentinel.py sample_data.csv --value-column sessions
```

Output: prints progress to stdout AND writes a markdown brief
(`<csv_name>_brief.md` by default) with a summary table of detected
episodes and the narrative writeup.

Flags:
- `--date-column` (default: `date`)
- `--value-column` (required)
- `--window` — rolling window size in days (default: 14)
- `--z-threshold` — how many standard deviations count as anomalous (default: 2.0)
- `--output` — custom path for the markdown brief
- `--no-md` — skip writing the markdown file, stdout only

## Weekly-seasonality-aware detection (`--seasonal`)

The default detector (`detect_anomalies`) compares each point to a plain
rolling mean/std. That works, but it can't tell "this is just a normal
Sunday" apart from a real anomaly — any metric with a weekly cycle (traffic,
orders, engagement) will produce false positives on its quiet days.

`--seasonal` uses `detect_anomalies_seasonal` instead: it estimates a
per-weekday seasonal factor from the trailing window, compares each point to
its *seasonally-expected* value rather than the raw average, and z-scores
the leftover residual instead of the raw number.

```bash
python sentinel.py your_data.csv --value-column sessions --seasonal
```

Requires `--window >= 14` (two weeks, to see every weekday at least twice);
defaults to 28 (four weeks) when `--seasonal` is set and `--window` isn't
given explicitly. More history gives more stable weekday estimates.

**An honest note on how this was built:** the first version of this had a
real bug, caught by its own test suite — the seasonal factor for each
weekday was estimated from the same window used to measure the noise level,
so each point's value was pulling its own weekday's average toward itself.
That artificially shrank the measured noise and inflated z-scores (a
degrees-of-freedom bias, the same category of issue Bessel's correction
fixes for ordinary variance, but here with 7 estimated parameters instead of
1). Fixed by dividing by `(N - 7)` instead of `N` when computing residual
variance, matching standard one-way ANOVA convention. Left as a genuine
build note rather than describing this as clean from the start.

**Also honest:** at any fixed z-score threshold, some false positives are
statistically expected on purely clean data — about 4.55% of points at the
default `z_threshold=2.0` (two-tailed), regardless of whether seasonality is
handled correctly. Seasonal detection removes the *systematic* false
positives caused by ignoring weekly patterns; it doesn't and can't eliminate
the baseline statistical false-positive rate of any threshold-based method.
For lower noise, raise `--z-threshold` to 2.5–3.0.

The loader was written to survive common export quirks rather than assume a
clean CSV:
- Skips metadata/title lines before the real header row (GA4's UI export
  does this)
- Parses `YYYYMMDD`, `YYYY-MM-DD`, and `MM/DD/YYYY` dates
- Strips comma-formatted numbers (`"1,150"` → `1150`)
- Skips blank rows and trailing "Totals" rows instead of crashing

**This has been tested against synthetic files that mimic these patterns —
not against a real GA4/Shopify/IG export yet.** Before claiming "just drop
in your GA4 export" publicly, run it against an actual export and fix
whatever breaks. Real files always have one quirk a synthetic test won't.

## Running tests

The deterministic core (Agent 2, both detectors, and the CSV loader) has a
full unit test suite — 31 tests, stdlib `unittest`, no external dependencies:

```bash
python -m unittest test_sentinel.py -v
```

## Current limitations (honest, not marketing copy)

- The seasonal detector handles weekly cycles only (7-day period). Monthly
  or annual seasonality (e.g. a retail spike every December) isn't modeled.
- Seasonal detection needs real history to be reliable — `--window` below
  ~21-28 days gives noisy per-weekday estimates.

- Single metric, single CSV, single run — no scheduling, no multi-metric
  correlation yet.
- Loader has not been validated against a real platform export (see above).
- Z-score threshold is a blunt instrument on noisy/seasonal data. It hasn't
  been validated against real client data yet — tune `--window` and
  `--z-threshold` before trusting it on anything live.
- Agent 1 currently only reviews a small sample of the data, not the full
  raw file. A fuller version would feed it the raw unparsed CSV text and
  have it return normalized JSON.
- Agent 1 and Agent 3 (the LLM calls) aren't covered by the test suite —
  they require a live API key and network access, which unit tests
  shouldn't depend on. Verify those by running the pipeline directly with
  `ANTHROPIC_API_KEY` set.
- `sentinel_demo.html` (the browser version) has NOT been updated with
  `--seasonal` support yet — it still only ports the original plain
  detector. Treat that as a known gap, not an oversight to discover later.

## Roadmap (not yet built — do not claim these publicly)

- Multi-metric correlation ("sessions dropped, did ad spend also change?")
- Scheduled runs + email/Slack delivery of the brief
- Support for common export formats beyond generic CSV (GA4, Shopify, IG Insights)
