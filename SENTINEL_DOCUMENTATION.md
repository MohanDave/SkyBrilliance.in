# Sentinel: Documentation

**Owner:** SkyBrilliance
**Version:** v3 plus the changes in section 12 (suggested tag: v3.2)
**Status:** Working prototype. Not published, no pilot users yet.
**Verified against:** `sentinel.py`, `evaluate.py`, and their tests (88 tests passing).

---

## 1. What Sentinel is

Sentinel takes a business metric exported as CSV (website sessions, orders, and similar), finds unusual days, and explains them in a short plain-English brief for a business owner without an analyst.

The design rule is that **detection is deterministic and the LLMs only read and write, under checks**. The statistics are plain code, so results are reproducible and unit-testable. LLM calls are used where language is the job: understanding a messy file, commenting on data quality, and writing the brief. The brief is never shown unless every date and number in it can be traced to the detector's output.

## 2. Pipeline

| Stage | Role | Type | Needs API key |
|---|---|---|---|
| Agent 0: Schema | Infers the header row, date column, and metric column(s) from the raw file | LLM | Yes (only when column names aren't given) |
| Data checks | Gaps, duplicate dates, unsorted dates, over the whole series | Deterministic | No |
| Agent 1: Quality | Short data-quality note, given the data-check findings plus the first and last 5 points | LLM | Yes |
| Agent 2: Detector | Finds anomalous days (plain or weekly-seasonal z-score), then groups them into episodes | Deterministic | No |
| Agent 3: Narrator | Writes a brief from precomputed facts, returned as JSON, then verified against those facts | LLM + deterministic check | Yes |

Without an API key, Sentinel still runs the data checks and Agent 2 and writes a statistical report, as long as you pass `--date-column` and `--value-column`.

## 3. Files

| File | Purpose |
|---|---|
| `sentinel.py` | The pipeline and CLI. Standard library only. |
| `test_sentinel.py` | 74 unit tests for the loader, detectors, data checks, Agent 0 parsing, and the verified narrator |
| `evaluate.py` | Evaluation harness: injects known anomalies into synthetic data and scores the detectors |
| `test_evaluate.py` | 14 tests for the harness itself |
| `eval_results.md` | Latest harness output (30 runs, seed 1) |

Other project files (README, LICENSE, `.gitignore`, `requirements.txt`, `sample_data.csv`, and `sentinel_demo.html`) were not part of this review. See section 14.

## 4. Quick start

Requires Python 3.7 or later (as far as I can tell from the language features used). No third-party packages are imported.

```bash
# Statistical report only, no API key needed
python sentinel.py sample_data.csv --date-column date --value-column sessions --seasonal

# Full pipeline, letting Agent 0 find the columns
export ANTHROPIC_API_KEY=your-key
python sentinel.py sample_data.csv --seasonal

# Tests and evaluation
python -m unittest test_sentinel.py test_evaluate.py
python evaluate.py --runs 30 --seed 1 --output eval_results.md
```

A Markdown brief is written to `<csv_name>_brief.md` unless you pass `--output` or `--no-md`. Use `--seasonal` for any daily business metric with a weekly cycle (section 9 shows why).

## 5. CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `csv_path` | required | Path to the CSV |
| `--date-column` | none | Date column name. If missing (with `--value-column`), Agent 0 infers it and needs an API key. |
| `--value-column` | none | Metric column name. Same rule. If Agent 0 finds several metrics it uses the first. |
| `--window` | 14 (28 with `--seasonal`) | Trailing window, counted in rows |
| `--z-threshold` | 2.0 | Flag points at or beyond this many standard deviations |
| `--seasonal` | off | Use weekly-seasonality-aware detection |
| `--dayfirst` | off | Read ambiguous slash dates as DD/MM/YYYY |
| `--output` | `<csv_name>_brief.md` | Where to write the Markdown brief |
| `--no-md` | off | Print only, don't write the file |

Environment: `ANTHROPIC_API_KEY`.

## 6. Input data

- One flat CSV with one date column and one numeric metric column per run.
- **Header detection:** the loader scans the first 20 lines for the line containing the expected column names, so GA4-style exports with title and date-range lines above the header work.
- **Column names** are matched case-insensitively.
- **Dates:** `YYYY-MM-DD`, `YYYYMMDD` (GA4), and slash dates. Slash dates such as `03/04/2026` are read **month-first by default**. Pass `--dayfirst` for DD/MM exports, which is what most Indian exports use.
- **Numbers** with thousands separators (`"1,150"`) are accepted.
- **Skipped rows:** blank dates or values, and rows that don't parse (such as a trailing "Totals" row).
- **Encoding:** UTF-8, including the BOM that Excel's "CSV UTF-8" adds.
- An empty file raises a clear error.

## 7. Detection (Agent 2)

### Plain mode (default)

For each point after the first `window` points, compare it to the mean and standard deviation of the previous `window` points. Flag it if `|z| >= z_threshold`. Windows with zero variance are skipped. Standard deviation is the population form (divide by N).

Use this only for data with no weekly cycle, such as already-weekly aggregates.

### Seasonal mode (`--seasonal`)

For each point:

1. Take the trailing window (default 28, minimum 14, 21 or more recommended).
2. Estimate a factor per weekday: that weekday's average divided by the window's overall mean.
3. Expected value = overall mean × the factor for the point's weekday.
4. Residual = actual minus expected.
5. Z-score the residual against the window's residual standard deviation, using N minus the number of weekday groups as the divisor.

In seasonal mode, the reported `rolling_mean` is the seasonally-adjusted expectation, not the raw mean.

### Episodes

Flagged days are merged into one episode when they are at most 3 days apart and in the same direction (spike or drop). A sustained decline reads as one event instead of many alerts.

## 8. LLM agents and verification

- **Model:** `claude-sonnet-4-6`, hard-coded in `call_claude()`.
- **Transport:** the standard-library `urllib` call to the Messages API. No timeout or retry inside `call_claude()`.
- **Agent 0** asks for a JSON object (header row, date column, value columns, notes). `parse_schema_inference_response()` fails safe: a malformed or wrong-type response yields `date_column=None`, which the CLI treats as "ask the user", never as a guess. An API failure in Agent 0 or Agent 1 is not caught and ends the run.
- **Agent 1** receives the first and last 5 points plus the findings from the deterministic data checks, and is told not to contradict them.

### Agent 3: verified narration

1. **Facts are computed in code** (`build_episode_facts`): start and end dates, direction, days flagged, worst z-score, start and end values, average value, average expected value, and percent versus expected. The model does no arithmetic.
2. **The model returns JSON** (`summary` plus up to three `questions`) and is told to use only dates and numbers from the facts, write dates as `YYYY-MM-DD`, and frame causes as questions.
3. **A deterministic grounding check** (`check_narrative_grounding`) extracts every date and number from the draft. Dates must fall inside a detected episode (ISO dates and month-name dates such as "April 3" or "3rd April" are both checked). Numbers must match a fact at the precision written, so "23%" matches 23.4 and "1,023" matches 1023.4. Digits attached to words ("GA4", "Q2") are ignored.
4. **If the draft fails,** it is retried once with the problems listed. If it fails again, returns invalid JSON, or the API call fails, Sentinel uses a **templated brief** built only from the facts (which passes its own check). It never crashes on an Agent 3 API error.
5. **The report says which one you got:** "Every date and figure in this brief was checked against Agent 2's output", or a note that the draft could not be verified and a templated brief is shown.

What the check does not do: it verifies dates and numbers, not meaning. A draft can pass and still say something misleading in words, for example calling a drop "dramatic". The questions-not-causes rule is enforced only by the prompt.

## 9. Evaluation

`evaluate.py` generates 200-day series with a weekly pattern and 6% multiplicative noise, injects one spike (+40%), one drop (-40%), and one 14-day level shift (-25%) on known dates, and scores every detector configuration. It also runs each configuration on event-free series to measure the inherent false-alarm rate.

Latest run (30 series, seed 1):

| Mode | Window | z | Spike found | Drop found | Shift found | Shift days flagged | Mean delay (days) | False alarms / 100 days, event-free data | False alarms / 100 clean days, data with events |
|---|---|---|---|---|---|---|---|---|---|
| plain | 14 | 2.0 | 60% | 33% | 43% | 4% | 0.3 | 0.13 | 0.23 |
| plain | 14 | 2.5 | 33% | 0% | 0% | 0% | 0.0 | 0.00 | 0.02 |
| plain | 14 | 3.0 | 3% | 0% | 0% | 0% | 0.0 | 0.00 | 0.02 |
| plain | 28 | 2.0 | 57% | 33% | 60% | 5% | 0.6 | 0.04 | 0.05 |
| plain | 28 | 2.5 | 30% | 3% | 0% | 0% | 0.0 | 0.00 | 0.00 |
| plain | 28 | 3.0 | 3% | 0% | 0% | 0% | 0.0 | 0.00 | 0.00 |
| seasonal | 28 | 2.0 | 100% | 97% | 100% | 38% | 0.0 | 9.59 | 6.83 |
| seasonal | 28 | 2.5 | 100% | 97% | 100% | 25% | 0.1 | 4.63 | 3.13 |
| seasonal | 28 | 3.0 | 93% | 93% | 97% | 16% | 0.1 | 1.96 | 1.22 |
| seasonal | 56 | 2.0 | 100% | 93% | 100% | 50% | 0.1 | 7.59 | 3.59 |
| seasonal | 56 | 2.5 | 100% | 80% | 100% | 31% | 0.2 | 3.03 | 1.40 |
| seasonal | 56 | 3.0 | 93% | 73% | 100% | 19% | 0.3 | 1.06 | 0.59 |

**What the results say**

- **Plain mode is not usable on data with a weekly cycle.** The normal weekday-to-weekend swing is larger than the injected events, so it finds few of them. At z of 2.5 or above it finds no more than a third of the spikes and almost no drops or shifts. Use `--seasonal`.
- **Seasonal mode finds the events** (93% to 100% of spikes, 73% to 97% of drops, with near-zero delay) **but raises more false alarms than the z threshold suggests.** On event-free data at z = 3.0 with a 28-day window it flags about 2 days per 100, against about 0.3 expected for normal noise. At z = 2.0 it flags about 9.6 per 100 days, against about 4.5. A larger window lowers false alarms but misses more drops (73% at window 56, z = 3.0).
- **Sustained shifts are found but under-reported.** Seasonal mode catches the shift 97% to 100% of the time, but flags only 16% to 50% of the shift's days, because the window absorbs the new level.

Scratch experiments (not shipped): working in log space reduced false alarms only slightly, and correcting for the uncertainty in each weekday's estimate reduced them by roughly 20% to 40% without reaching the expected rate. The cause is not fully pinned down.

Synthetic data is a sanity check, not proof of performance on real client data.

## 10. Testing

```bash
python -m unittest test_sentinel.py test_evaluate.py -v
```

88 tests cover date normalization (including day-first), header detection, loading (including BOM and empty files), both detectors, episode merging, the data checks, report building, an end-to-end `main()` run with a mocked model, Agent 0's response parsing, the grounding check (including rejected invented numbers and dates), the retry and fallback paths of the narrator, and the evaluation harness's generator and scoring.

Not covered: real LLM responses. The grounding check is tested against scripted responses, so how often a live model passes it first time is unmeasured.

## 11. Known limitations

- **Seasonal false alarms run above the nominal rate** (section 9).
- **Sustained shifts are under-reported** (section 9).
- **Earlier anomalies contaminate later windows.** Flagged points stay in the trailing window.
- **Windows count rows, not days.** Gaps shift the window silently. The data checks warn, but detection doesn't adjust.
- **Duplicate dates are warned about, not removed.**
- **The grounding check verifies figures, not meaning** (section 8).
- **One metric per run.** Multiple metrics from Agent 0 use only the first.
- **Header detection splits on plain commas,** so a quoted comma in a header can confuse it.
- **Agents 0 and 1 have no API error handling.** Agent 3 degrades to the templated brief.
- **The model name is hard-coded.**

## 12. Changelog

**v3.1: fixes to the uploaded v3**

- **Bug fix:** auto-detected metric names now reach Agent 3 and the report. Before, the brief said "their 'None' metric".
- **Bug fix:** CSVs with an Excel UTF-8 BOM no longer fail with "Columns not found".
- **Bug fix:** empty files give a clear error instead of a `TypeError`.
- **New:** `--dayfirst` for DD/MM/YYYY slash dates.
- **New:** whole-series data checks (unsorted, duplicate, and missing dates), shown in the report even without an API key and passed to Agent 1, which now also sees the last 5 points.

**v3.2: verified narration and evaluation**

- **New:** Agent 3 returns structured JSON from precomputed facts, is verified by a deterministic grounding check, retries once, and falls back to a templated brief. The report states which one was used.
- **New:** `evaluate.py` and `test_evaluate.py`, with `eval_results.md`.
- **Tests:** 39 became 88.

## 13. Planned next

- **Reduce seasonal false alarms and under-reporting of shifts,** using the harness to prove each change (options: exclude flagged points from the baseline, correct for weekday-estimate uncertainty, median and MAD, a separate level-shift episode type).
- **Use calendar days instead of rows** by reindexing to a full daily calendar.
- **Retry and timeout handling** inside `call_claude()`, covering Agents 0 and 1.
- **Run the full pipeline against the live API** on real exports and record how often Agent 3 passes verification on the first try.
- **Publish on GitHub** with `eval_results.md` in the README, then recruit pilot users.

## 14. Open items

- Review `sentinel_demo.html`, the README, `requirements.txt`, and the licence. They were not provided, so this document makes no claims about them. Check in particular whether the demo's JavaScript port includes seasonal mode and the newer behaviour.
- Decide the default for ambiguous dates. Month-first is the current default, which is risky for Indian users.
- Decide whether `--seasonal` should become the default, given section 9.
