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

## Current limitations (honest, not marketing copy)

- Single metric, single CSV, single run — no scheduling, no multi-metric
  correlation yet.
- Z-score threshold is a blunt instrument on noisy/seasonal data. It hasn't
  been validated against real client data yet — tune `--window` and
  `--z-threshold` before trusting it on anything live.
- Agent 1 currently only reviews a small sample of the data, not the full
  raw file. A fuller version would feed it the raw unparsed CSV text and
  have it return normalized JSON.
- No tests yet (next commit should add unit tests for `detect_anomalies`
  and `merge_consecutive_anomalies` — the deterministic core is the part
  most worth testing rigorously).

## Roadmap (not yet built — do not claim these publicly)

- Multi-metric correlation ("sessions dropped, did ad spend also change?")
- Scheduled runs + email/Slack delivery of the brief
- Support for common export formats beyond generic CSV (GA4, Shopify, IG Insights)
