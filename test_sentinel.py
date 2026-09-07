#!/usr/bin/env python3
"""
Unit tests for Sentinel's deterministic core (Agent 2 and the CSV loader).

Deliberately does NOT test agent1_clean_summary / agent3_narrate / call_claude —
those require a live API key and network access, which unit tests shouldn't
depend on. If you want to verify those, run sentinel.py directly with
ANTHROPIC_API_KEY set (see README).

Run: python -m unittest test_sentinel.py -v
"""

import csv
import os
import tempfile
import unittest

from sentinel import (
    Anomaly,
    normalize_date,
    find_header_row,
    load_series,
    detect_anomalies,
    merge_consecutive_anomalies,
)


class TestNormalizeDate(unittest.TestCase):

    def test_iso_format(self):
        self.assertEqual(normalize_date("2026-08-26"), "2026-08-26")

    def test_ga4_style_yyyymmdd(self):
        self.assertEqual(normalize_date("20260826"), "2026-08-26")

    def test_us_slash_format(self):
        self.assertEqual(normalize_date("08/26/2026"), "2026-08-26")

    def test_strips_whitespace(self):
        self.assertEqual(normalize_date("  2026-08-26  "), "2026-08-26")

    def test_unrecognized_format_raises(self):
        with self.assertRaises(ValueError):
            normalize_date("Totals")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            normalize_date("")


class TestFindHeaderRow(unittest.TestCase):

    def _write_temp_csv(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_clean_csv_header_is_row_zero(self):
        path = self._write_temp_csv("date,sessions\n2026-01-01,100\n")
        self.addCleanup(os.remove, path)
        self.assertEqual(find_header_row(path, ["date", "sessions"]), 0)

    def test_ga4_style_metadata_lines_skipped(self):
        content = (
            "# Report: Sessions by Date\n"
            "# Date range: Jun 1, 2026 - Aug 29, 2026\n"
            "\n"
            "Date,Sessions\n"
            "20260601,1205\n"
        )
        path = self._write_temp_csv(content)
        self.addCleanup(os.remove, path)
        self.assertEqual(find_header_row(path, ["Date", "Sessions"]), 3)

    def test_missing_columns_falls_back_to_zero(self):
        path = self._write_temp_csv("foo,bar\n1,2\n")
        self.addCleanup(os.remove, path)
        self.assertEqual(find_header_row(path, ["date", "sessions"]), 0)


class TestLoadSeries(unittest.TestCase):

    def _write_temp_csv(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_clean_csv(self):
        path = self._write_temp_csv("date,sessions\n2026-01-01,100\n2026-01-02,110\n")
        self.addCleanup(os.remove, path)
        dates, values = load_series(path, "date", "sessions")
        self.assertEqual(dates, ["2026-01-01", "2026-01-02"])
        self.assertEqual(values, [100.0, 110.0])

    def test_comma_formatted_numbers(self):
        path = self._write_temp_csv('date,sessions\n2026-01-01,"1,150"\n')
        self.addCleanup(os.remove, path)
        dates, values = load_series(path, "date", "sessions")
        self.assertEqual(values, [1150.0])

    def test_skips_blank_rows(self):
        path = self._write_temp_csv("date,sessions\n2026-01-01,100\n2026-01-02,\n")
        self.addCleanup(os.remove, path)
        dates, values = load_series(path, "date", "sessions")
        self.assertEqual(len(values), 1)

    def test_skips_trailing_totals_row(self):
        path = self._write_temp_csv("date,sessions\n2026-01-01,100\nTotals,100\n")
        self.addCleanup(os.remove, path)
        dates, values = load_series(path, "date", "sessions")
        self.assertEqual(len(values), 1)

    def test_case_insensitive_column_match(self):
        path = self._write_temp_csv("Date,Sessions\n2026-01-01,100\n")
        self.addCleanup(os.remove, path)
        dates, values = load_series(path, "date", "sessions")
        self.assertEqual(values, [100.0])

    def test_ga4_style_metadata_and_yyyymmdd(self):
        content = (
            "# Report: Sessions by Date\n\n"
            "Date,Sessions\n"
            "20260601,1205\n"
            "20260602,1198\n"
        )
        path = self._write_temp_csv(content)
        self.addCleanup(os.remove, path)
        dates, values = load_series(path, "Date", "Sessions")
        self.assertEqual(dates, ["2026-06-01", "2026-06-02"])
        self.assertEqual(values, [1205.0, 1198.0])

    def test_missing_column_raises(self):
        path = self._write_temp_csv("date,sessions\n2026-01-01,100\n")
        self.addCleanup(os.remove, path)
        with self.assertRaises(ValueError):
            load_series(path, "date", "pageviews")


class TestDetectAnomalies(unittest.TestCase):

    def test_no_anomalies_in_flat_series(self):
        values = [100.0] * 30
        dates = [f"2026-01-{i+1:02d}" for i in range(30)]
        anomalies = detect_anomalies(dates, values, window=14, z_threshold=2.0)
        self.assertEqual(anomalies, [])

    def test_detects_obvious_spike(self):
        # Realistic baseline needs SOME variance — a perfectly flat baseline
        # (std=0) causes the point right after it to be skipped by design
        # (see test_zero_variance_window_does_not_crash), so it would never
        # get flagged. Alternating values give the window a small, real std.
        baseline = [100.0, 102.0, 98.0, 101.0, 99.0] * 4  # 20 points, mild noise
        values = baseline + [500.0] + [100.0] * 5
        dates = [f"2026-01-{i+1:02d}" for i in range(len(values))]
        anomalies = detect_anomalies(dates, values, window=14, z_threshold=2.0)
        self.assertTrue(any(a.direction == "spike" for a in anomalies))

    def test_detects_obvious_drop(self):
        baseline = [100.0, 102.0, 98.0, 101.0, 99.0] * 4  # 20 points, mild noise
        values = baseline + [10.0] + [100.0] * 5
        dates = [f"2026-01-{i+1:02d}" for i in range(len(values))]
        anomalies = detect_anomalies(dates, values, window=14, z_threshold=2.0)
        self.assertTrue(any(a.direction == "drop" for a in anomalies))

    def test_zero_variance_window_does_not_crash(self):
        # a perfectly flat trailing window (std=0) must not raise ZeroDivisionError
        values = [100.0] * 25
        dates = [f"2026-01-{i+1:02d}" for i in range(25)]
        try:
            detect_anomalies(dates, values, window=14, z_threshold=2.0)
        except ZeroDivisionError:
            self.fail("detect_anomalies raised ZeroDivisionError on zero-variance window")

    def test_respects_window_param_short_series(self):
        # fewer points than the window means no anomalies can be evaluated
        values = [100.0, 200.0, 100.0]
        dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
        anomalies = detect_anomalies(dates, values, window=14, z_threshold=2.0)
        self.assertEqual(anomalies, [])


class TestMergeConsecutiveAnomalies(unittest.TestCase):

    def _anomaly(self, date, direction="drop"):
        return Anomaly(date=date, value=1.0, rolling_mean=1.0, rolling_std=1.0,
                        z_score=3.0, direction=direction)

    def test_empty_input(self):
        self.assertEqual(merge_consecutive_anomalies([]), [])

    def test_single_anomaly_is_one_episode(self):
        episodes = merge_consecutive_anomalies([self._anomaly("2026-01-01")])
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0]), 1)

    def test_consecutive_days_merge_into_one_episode(self):
        anomalies = [
            self._anomaly("2026-01-01"),
            self._anomaly("2026-01-02"),
            self._anomaly("2026-01-03"),
        ]
        episodes = merge_consecutive_anomalies(anomalies)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0]), 3)

    def test_gap_over_three_days_splits_episodes(self):
        anomalies = [
            self._anomaly("2026-01-01"),
            self._anomaly("2026-01-10"),  # 9-day gap
        ]
        episodes = merge_consecutive_anomalies(anomalies)
        self.assertEqual(len(episodes), 2)

    def test_direction_change_splits_episodes_even_if_adjacent(self):
        anomalies = [
            self._anomaly("2026-01-01", direction="drop"),
            self._anomaly("2026-01-02", direction="spike"),
        ]
        episodes = merge_consecutive_anomalies(anomalies)
        self.assertEqual(len(episodes), 2)


if __name__ == "__main__":
    unittest.main()
