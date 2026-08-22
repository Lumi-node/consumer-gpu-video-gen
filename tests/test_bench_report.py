"""Tests for the torch-free benchmark reporting logic."""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bench_report import (  # noqa: E402
    TOTAL_STAGE,
    RunResult,
    format_report,
    format_stage_table,
    format_summary_table,
    median,
    percentile,
    psnr_from_mse,
    results_from_json,
    results_to_json,
    speedup,
    stddev,
    vram_reduction_pct,
)


class TestStatistics(unittest.TestCase):
    def test_median_odd_and_even(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_median_single(self):
        self.assertEqual(median([7.5]), 7.5)

    def test_median_rejects_empty(self):
        with self.assertRaises(ValueError):
            median([])

    def test_median_is_robust_to_an_outlier(self):
        # The point of using median over mean for latency: one slow run from
        # thermal throttling must not move the headline number much.
        clean = [100, 101, 99, 100, 102]
        with_outlier = clean + [900]
        self.assertLess(abs(median(with_outlier) - median(clean)), 2.0)

    def test_percentile_endpoints(self):
        values = [10, 20, 30, 40]
        self.assertEqual(percentile(values, 0), 10)
        self.assertEqual(percentile(values, 100), 40)

    def test_percentile_interpolates(self):
        # rank = 0.5 * 3 = 1.5 -> midway between 20 and 30
        self.assertEqual(percentile([10, 20, 30, 40], 50), 25.0)

    def test_percentile_single_value(self):
        self.assertEqual(percentile([5], 90), 5)

    def test_percentile_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            percentile([1, 2], 101)

    def test_stddev_single_is_zero(self):
        self.assertEqual(stddev([42.0]), 0.0)

    def test_stddev_known_value(self):
        # Sample stddev of [2,4,4,4,5,5,7,9] is 2.138... (n-1 denominator)
        self.assertAlmostEqual(stddev([2, 4, 4, 4, 5, 5, 7, 9]), 2.13808993, places=6)


class TestQualityMetric(unittest.TestCase):
    def test_zero_mse_is_lossless(self):
        self.assertEqual(psnr_from_mse(0.0), float("inf"))

    def test_psnr_decreases_as_error_grows(self):
        self.assertGreater(psnr_from_mse(1e-5), psnr_from_mse(1e-3))

    def test_psnr_known_value(self):
        # mse = 0.01, data_range = 1.0 -> 10 * log10(100) = 20 dB
        self.assertAlmostEqual(psnr_from_mse(0.01), 20.0, places=9)

    def test_psnr_respects_data_range(self):
        self.assertAlmostEqual(psnr_from_mse(1.0, data_range=255.0), 48.1308036, places=5)

    def test_negative_mse_rejected(self):
        with self.assertRaises(ValueError):
            psnr_from_mse(-0.1)


def _run(label, total_ms, denoise_ms=None, alloc=0.0, reserved=0.0, psnr=None):
    stages = {}
    if denoise_ms is not None:
        stages["denoise"] = denoise_ms
    stages[TOTAL_STAGE] = total_ms
    return RunResult(
        label=label,
        stage_samples=stages,
        peak_allocated_gb=alloc,
        peak_reserved_gb=reserved,
        quality_psnr_db=psnr,
    )


class TestRunResult(unittest.TestCase):
    def test_missing_stage_returns_none(self):
        result = _run("bf16", [100.0])
        self.assertIsNone(result.median_ms("nonexistent"))
        self.assertIsNone(result.p90_ms("nonexistent"))

    def test_total_stage_sorts_last(self):
        result = RunResult(
            label="x",
            stage_samples={"encode": [1.0], TOTAL_STAGE: [10.0], "decode": [2.0]},
        )
        self.assertEqual(result.stages(), ["encode", "decode", TOTAL_STAGE])

    def test_unaccounted_time(self):
        result = RunResult(
            label="x",
            stage_samples={"encode": [10.0], "denoise": [50.0], TOTAL_STAGE: [75.0]},
        )
        self.assertAlmostEqual(result.unaccounted_ms(), 15.0)

    def test_unaccounted_is_none_without_total(self):
        result = RunResult(label="x", stage_samples={"encode": [10.0]})
        self.assertIsNone(result.unaccounted_ms())


class TestComparisons(unittest.TestCase):
    def test_speedup_faster_is_greater_than_one(self):
        baseline = _run("bf16", [200.0])
        candidate = _run("fp4", [100.0])
        self.assertAlmostEqual(speedup(baseline, candidate), 2.0)

    def test_speedup_slower_is_less_than_one(self):
        # This is the case that matters for weight-only INT4: it can be SLOWER
        # than bf16, and the harness must be able to say so.
        baseline = _run("bf16", [100.0])
        candidate = _run("int4", [125.0])
        self.assertAlmostEqual(speedup(baseline, candidate), 0.8)

    def test_speedup_none_when_stage_absent(self):
        baseline = _run("bf16", [100.0], denoise_ms=[80.0])
        candidate = _run("fp4", [50.0])
        self.assertIsNone(speedup(baseline, candidate, "denoise"))

    def test_vram_reduction(self):
        baseline = _run("bf16", [1.0], reserved=64.0)
        candidate = _run("int4", [1.0], reserved=16.0)
        self.assertAlmostEqual(vram_reduction_pct(baseline, candidate), 75.0)

    def test_vram_reduction_negative_when_worse(self):
        baseline = _run("bf16", [1.0], reserved=10.0)
        candidate = _run("bad", [1.0], reserved=12.0)
        self.assertAlmostEqual(vram_reduction_pct(baseline, candidate), -20.0)

    def test_vram_reduction_none_without_baseline_measurement(self):
        self.assertIsNone(vram_reduction_pct(_run("a", [1.0]), _run("b", [1.0])))


class TestFormatting(unittest.TestCase):
    def setUp(self):
        self.results = [
            _run("bf16", [200.0, 202.0], denoise_ms=[180.0], alloc=60.0, reserved=64.0),
            _run("int4", [250.0, 248.0], denoise_ms=[230.0], alloc=16.0, reserved=18.0, psnr=38.0),
            _run("nvfp4", [90.0, 92.0], denoise_ms=[75.0], alloc=15.0, reserved=17.0, psnr=34.5),
        ]

    def test_summary_lists_baseline_first(self):
        table = format_summary_table(self.results[::-1], "bf16")
        body = table.split("\n")
        first_row = [line for line in body if line.startswith("| **")][0]
        self.assertIn("bf16", first_row)
        self.assertIn("(baseline)", first_row)

    def test_summary_marks_reference_speedup(self):
        table = format_summary_table(self.results, "bf16")
        self.assertIn("1.00x (ref)", table)

    def test_summary_reports_slowdown_below_one(self):
        table = format_summary_table(self.results, "bf16")
        # 201 / 249 = 0.807...
        self.assertIn("0.81x", table)

    def test_summary_reports_speedup(self):
        table = format_summary_table(self.results, "bf16")
        # 201 / 91 = 2.20...
        self.assertIn("2.21x", table)

    def test_unknown_baseline_raises(self):
        with self.assertRaises(KeyError):
            format_summary_table(self.results, "does-not-exist")

    def test_stage_table_has_a_column_per_config(self):
        table = format_stage_table(self.results, "bf16")
        header = table.split("\n")[0]
        for label in ("bf16", "int4", "nvfp4"):
            self.assertIn(label, header)

    def test_stage_table_covers_union_of_stages(self):
        extra = _run("cached", [50.0])
        extra.stage_samples["vae_decode"] = [20.0]
        table = format_stage_table(self.results + [extra], "bf16")
        self.assertIn("vae_decode", table)
        self.assertIn("denoise", table)

    def test_empty_results_do_not_crash(self):
        self.assertIn("No results", format_summary_table([], "bf16"))
        self.assertIn("No results", format_stage_table([], "bf16"))

    def test_report_warns_on_visible_quality_loss(self):
        bad = _run("aggressive", [40.0], alloc=10.0, reserved=11.0, psnr=19.0)
        report = format_report(self.results + [bad], "bf16")
        self.assertIn("Warnings", report)
        self.assertIn("not free", report)

    def test_report_warns_on_unattributed_time(self):
        # denoise is only 20 of 200 ms, so 90% is unaccounted for.
        sparse = _run("sparse", [200.0], denoise_ms=[20.0])
        report = format_report([sparse], "sparse")
        self.assertIn("not attributed", report)

    def test_report_is_quiet_when_everything_is_healthy(self):
        clean = _run("bf16", [100.0], denoise_ms=[95.0], alloc=1.0, reserved=1.0)
        report = format_report([clean], "bf16")
        self.assertNotIn("Warnings", report)

    def test_lossless_quality_renders_as_lossless(self):
        exact = _run("identical", [100.0], psnr=float("inf"))
        table = format_summary_table([_run("bf16", [100.0]), exact], "bf16")
        self.assertIn("lossless", table)


class TestSerialization(unittest.TestCase):
    def test_round_trip_preserves_results(self):
        original = [
            _run("bf16", [200.0], denoise_ms=[180.0], alloc=60.0, reserved=64.0),
            _run("nvfp4", [90.0], denoise_ms=[75.0], alloc=15.0, reserved=17.0, psnr=34.5),
        ]
        restored = results_from_json(results_to_json(original))
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].label, "bf16")
        self.assertAlmostEqual(restored[1].quality_psnr_db, 34.5)
        self.assertAlmostEqual(restored[1].median_ms(), 90.0)

    def test_serialized_output_is_valid_json(self):
        payload = results_to_json([_run("bf16", [1.0])])
        self.assertIsInstance(json.loads(payload), list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
