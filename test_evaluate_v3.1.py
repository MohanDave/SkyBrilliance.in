#!/usr/bin/env python3
"""Tests for the evaluation harness itself, so its numbers can be trusted.

Run: python -m unittest test_evaluate.py -v
"""

import unittest
from datetime import date

import evaluate as ev
from evaluate import Event


class TestGenerateSeries(unittest.TestCase):

    def test_same_seed_is_reproducible(self):
        self.assertEqual(ev.generate_series(5), ev.generate_series(5))

    def test_different_seeds_differ(self):
        self.assertNotEqual(ev.generate_series(5)[1], ev.generate_series(6)[1])

    def test_dates_are_daily_and_start_on_a_monday(self):
        dates, values, _ = ev.generate_series(1, n_days=ev.MIN_DAYS)
        self.assertEqual(len(dates), ev.MIN_DAYS)
        self.assertEqual(date.fromisoformat(dates[0]).weekday(), 0)
        self.assertEqual(date.fromisoformat(dates[1]).toordinal() - date.fromisoformat(dates[0]).toordinal(), 1)

    def test_events_fit_inside_series_and_do_not_overlap(self):
        for seed in range(50):
            dates, _, events = ev.generate_series(seed)
            self.assertEqual([e.kind for e in events], ["spike", "drop", "shift"])
            for e in events:
                self.assertGreaterEqual(e.start, ev.WARMUP)
                self.assertLess(e.end, len(dates))
            for a, b in zip(events, events[1:]):
                self.assertLess(a.end + ev.GRACE_DAYS, b.start)

    def test_shift_has_expected_length(self):
        _, _, events = ev.generate_series(3)
        shift = events[2]
        self.assertEqual(shift.end - shift.start + 1, ev.SHIFT_LENGTH)

    def test_injected_spike_is_actually_higher_than_baseline(self):
        dates, values, events = ev.generate_series(2, noise=0.0)
        spike = events[0]
        wd = date.fromisoformat(dates[spike.start]).weekday()
        self.assertAlmostEqual(values[spike.start], ev.WEEKDAY_BASE[wd] * 1.4, places=6)

    def test_too_short_series_raises_clear_error(self):
        with self.assertRaises(ValueError):
            ev.generate_series(1, n_days=30)

    def test_clean_series_has_no_events_and_is_reproducible(self):
        self.assertEqual(ev.generate_clean_series(4), ev.generate_clean_series(4))


class TestScoreRun(unittest.TestCase):

    def test_known_scoring_example(self):
        events = [Event("spike", 50, 50), Event("shift", 70, 83)]
        flagged = {50, 72, 73, 90}
        r = ev.score_run(events, flagged, n_days=100, window=20)
        self.assertTrue(r["detected"]["spike"])
        self.assertTrue(r["detected"]["shift"])
        self.assertEqual(r["delay"], {"spike": 0, "shift": 2})
        self.assertAlmostEqual(r["shift_coverage"], 2 / 14)
        # zones: 50..53 (4 days) and 70..86 (17 days); evaluated days 20..99 = 80
        self.assertEqual(r["clean_days"], 80 - 21)
        self.assertEqual(r["false_flags"], 1)              # only index 90
        self.assertAlmostEqual(r["false_alarms_per_100"], 1 / 59 * 100)

    def test_missed_event_is_reported_as_not_detected(self):
        r = ev.score_run([Event("drop", 60, 60)], set(), n_days=100, window=20)
        self.assertFalse(r["detected"]["drop"])
        self.assertEqual(r["delay"], {})
        self.assertEqual(r["false_flags"], 0)

    def test_flags_within_grace_period_are_not_false_alarms(self):
        r = ev.score_run([Event("spike", 60, 60)], {61, 63}, n_days=100, window=20)
        self.assertEqual(r["false_flags"], 0)
        r = ev.score_run([Event("spike", 60, 60)], {64}, n_days=100, window=20)
        self.assertEqual(r["false_flags"], 1)


class TestRunEvaluation(unittest.TestCase):

    def test_returns_a_row_per_config_with_valid_ranges(self):
        configs = [("plain", 14, 2.0), ("seasonal", 28, 3.0)]
        rows = ev.run_evaluation(n_runs=4, seed=1, configs=configs)
        self.assertEqual(len(rows), 2)
        for r in rows:
            for key in ("recall_spike", "recall_drop", "recall_shift", "shift_coverage"):
                self.assertTrue(0.0 <= r[key] <= 1.0, key)
            self.assertGreaterEqual(r["false_alarms_per_100_clean_days"], 0.0)
            self.assertGreaterEqual(r["false_alarms_event_free"], 0.0)

    def test_seasonal_detector_finds_large_obvious_events(self):
        # sanity check on the harness: big events, low noise, strict threshold
        rows = ev.run_evaluation(n_runs=10, seed=1, noise=0.03, spike=0.8, drop=0.8, shift=0.5,
                                 configs=[("seasonal", 28, 3.0)])
        self.assertGreaterEqual(rows[0]["recall_spike"], 0.9)
        self.assertGreaterEqual(rows[0]["recall_drop"], 0.9)

    def test_markdown_output_contains_table_and_all_configs(self):
        rows = ev.run_evaluation(n_runs=3, seed=1)
        md = ev.format_markdown(rows, 3, 1, 0.06, 0.4, 0.4, 0.25)
        self.assertIn("| Mode | Window | z |", md)
        self.assertEqual(md.count("| seasonal |"), 6)
        self.assertEqual(md.count("| plain |"), 6)


if __name__ == "__main__":
    unittest.main()
