"""Tests for the step-caching decision policy."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cache_policy import (  # noqa: E402
    CacheStats,
    StepCachePolicy,
    constant_threshold_schedule,
    linear_threshold_schedule,
)


def run_policy(policy, distances):
    """Drive a policy over a list of per-step distances; return the decisions."""
    policy.total_steps = len(distances)
    policy.reset()
    return [policy.should_compute(i, d) for i, d in enumerate(distances)]


class TestForcedSteps(unittest.TestCase):
    def test_warmup_steps_always_compute(self):
        policy = StepCachePolicy(threshold=1e9, warmup_steps=3, cooldown_steps=0)
        decisions = run_policy(policy, [0.0] * 10)
        self.assertEqual(decisions[:3], [True, True, True])
        self.assertEqual(policy.stats.forced_warmup, 3)

    def test_cooldown_steps_always_compute(self):
        policy = StepCachePolicy(threshold=1e9, warmup_steps=0, cooldown_steps=2)
        decisions = run_policy(policy, [0.0] * 10)
        self.assertEqual(decisions[-2:], [True, True])
        self.assertEqual(policy.stats.forced_cooldown, 2)

    def test_zero_distance_still_reuses_in_the_middle(self):
        # Cap disabled so this exercises warmup/cooldown alone; the cap has its
        # own tests in TestReuseCap.
        policy = StepCachePolicy(
            threshold=1e9, warmup_steps=1, cooldown_steps=1, max_consecutive_reuse=None
        )
        decisions = run_policy(policy, [0.0] * 6)
        self.assertEqual(decisions, [True, False, False, False, False, True])

    def test_no_cooldown_when_disabled(self):
        policy = StepCachePolicy(
            threshold=1e9, warmup_steps=0, cooldown_steps=0, max_consecutive_reuse=None
        )
        decisions = run_policy(policy, [0.0] * 5)
        self.assertEqual(decisions, [False] * 5)


class TestThresholdBehaviour(unittest.TestCase):
    def test_large_distance_forces_compute(self):
        policy = StepCachePolicy(threshold=0.1, warmup_steps=0, cooldown_steps=0)
        decisions = run_policy(policy, [0.5, 0.5, 0.5])
        self.assertEqual(decisions, [True, True, True])

    def test_accumulation_triggers_eventually(self):
        # Each step contributes 0.04; the 0.1 budget is crossed on the third.
        policy = StepCachePolicy(threshold=0.1, warmup_steps=0, cooldown_steps=0)
        decisions = run_policy(policy, [0.04] * 6)
        self.assertEqual(decisions, [False, False, True, False, False, True])

    def test_accumulator_resets_after_compute(self):
        policy = StepCachePolicy(threshold=0.1, warmup_steps=0, cooldown_steps=0)
        run_policy(policy, [0.04, 0.04, 0.04])
        self.assertAlmostEqual(policy.accumulated, 0.0)

    def test_zero_threshold_computes_every_step(self):
        policy = StepCachePolicy(threshold=0.0, warmup_steps=0, cooldown_steps=0)
        decisions = run_policy(policy, [0.0] * 5)
        self.assertEqual(decisions, [True] * 5)

    def test_negative_distance_rejected(self):
        policy = StepCachePolicy(total_steps=4, warmup_steps=0, cooldown_steps=0)
        policy.reset()
        with self.assertRaises(ValueError):
            policy.should_compute(0, -0.1)

    def test_negative_threshold_rejected(self):
        with self.assertRaises(ValueError):
            StepCachePolicy(threshold=-0.1)


class TestReuseCap(unittest.TestCase):
    def test_cap_bounds_consecutive_reuse(self):
        policy = StepCachePolicy(
            threshold=1e9, warmup_steps=0, cooldown_steps=0, max_consecutive_reuse=2
        )
        decisions = run_policy(policy, [0.0] * 9)
        # Two reuses then a forced compute, repeating.
        self.assertEqual(
            decisions, [False, False, True, False, False, True, False, False, True]
        )
        self.assertEqual(policy.stats.forced_by_reuse_cap, 3)

    def test_cap_of_zero_disables_reuse(self):
        policy = StepCachePolicy(
            threshold=1e9, warmup_steps=0, cooldown_steps=0, max_consecutive_reuse=0
        )
        self.assertEqual(run_policy(policy, [0.0] * 4), [True] * 4)

    def test_none_disables_the_cap(self):
        policy = StepCachePolicy(
            threshold=1e9, warmup_steps=0, cooldown_steps=0, max_consecutive_reuse=None
        )
        self.assertEqual(run_policy(policy, [0.0] * 20).count(False), 20)

    def test_negative_cap_rejected(self):
        with self.assertRaises(ValueError):
            StepCachePolicy(max_consecutive_reuse=-1)


class TestSchedules(unittest.TestCase):
    def test_constant_schedule_is_flat(self):
        schedule = constant_threshold_schedule(0.2)
        self.assertAlmostEqual(schedule(0, 10), 0.2)
        self.assertAlmostEqual(schedule(9, 10), 0.2)

    def test_linear_schedule_interpolates_endpoints(self):
        schedule = linear_threshold_schedule(0.3, 0.1)
        self.assertAlmostEqual(schedule(0, 11), 0.3)
        self.assertAlmostEqual(schedule(10, 11), 0.1)
        self.assertAlmostEqual(schedule(5, 11), 0.2)

    def test_linear_schedule_handles_single_step(self):
        self.assertAlmostEqual(linear_threshold_schedule(0.3, 0.1)(0, 1), 0.3)

    def test_timestep_aware_schedule_reuses_more_early_than_late(self):
        # The core claim of the timestep-aware idea: with a decreasing threshold,
        # reuse should concentrate in the first half of the schedule.
        policy = StepCachePolicy(
            threshold=linear_threshold_schedule(0.30, 0.02),
            warmup_steps=0,
            cooldown_steps=0,
            max_consecutive_reuse=None,
        )
        distances = [0.03] * 20
        decisions = run_policy(policy, distances)
        first_half_reuse = decisions[:10].count(False)
        second_half_reuse = decisions[10:].count(False)
        self.assertGreater(first_half_reuse, second_half_reuse)

    def test_sequence_schedule_indexes_by_step(self):
        policy = StepCachePolicy(
            threshold=[0.0, 1e9, 1e9], warmup_steps=0, cooldown_steps=0
        )
        decisions = run_policy(policy, [0.01, 0.01, 0.01])
        self.assertEqual(decisions, [True, False, False])

    def test_short_sequence_clamps_to_last_value(self):
        policy = StepCachePolicy(threshold=[1e9], warmup_steps=0, cooldown_steps=0)
        self.assertAlmostEqual(policy.threshold_at(99), 1e9)

    def test_empty_sequence_rejected(self):
        policy = StepCachePolicy(threshold=[], warmup_steps=0, cooldown_steps=0)
        with self.assertRaises(ValueError):
            policy.threshold_at(0)

    def test_negative_schedule_endpoints_rejected(self):
        with self.assertRaises(ValueError):
            linear_threshold_schedule(-0.1, 0.2)
        with self.assertRaises(ValueError):
            constant_threshold_schedule(-1.0)


class TestStats(unittest.TestCase):
    def test_counts_add_up(self):
        policy = StepCachePolicy(threshold=0.05, warmup_steps=1, cooldown_steps=1)
        run_policy(policy, [0.02] * 12)
        self.assertEqual(policy.stats.total_steps, 12)
        self.assertEqual(len(policy.stats.decisions), 12)

    def test_reuse_fraction(self):
        policy = StepCachePolicy(
            threshold=1e9, warmup_steps=0, cooldown_steps=0, max_consecutive_reuse=1
        )
        run_policy(policy, [0.0] * 10)
        self.assertAlmostEqual(policy.stats.reuse_fraction, 0.5)

    def test_theoretical_speedup(self):
        stats = CacheStats(computed=10, reused=10)
        self.assertAlmostEqual(stats.theoretical_speedup(), 2.0)

    def test_speedup_is_one_when_nothing_reused(self):
        self.assertAlmostEqual(CacheStats(computed=25, reused=0).theoretical_speedup(), 1.0)

    def test_empty_stats_are_safe(self):
        stats = CacheStats()
        self.assertEqual(stats.total_steps, 0)
        self.assertAlmostEqual(stats.reuse_fraction, 0.0)
        self.assertAlmostEqual(stats.theoretical_speedup(), 1.0)

    def test_summary_mentions_counts(self):
        policy = StepCachePolicy(threshold=1e9, warmup_steps=1, cooldown_steps=1)
        run_policy(policy, [0.0] * 8)
        self.assertIn("computed", policy.stats.summary())

    def test_reset_clears_previous_run(self):
        policy = StepCachePolicy(threshold=0.05, warmup_steps=0, cooldown_steps=0)
        run_policy(policy, [0.1] * 5)
        self.assertEqual(policy.stats.total_steps, 5)
        policy.reset()
        self.assertEqual(policy.stats.total_steps, 0)
        self.assertAlmostEqual(policy.accumulated, 0.0)
        self.assertEqual(policy.consecutive_reuse, 0)


class TestRealisticSchedule(unittest.TestCase):
    """A 25-step run with plausible distances should reuse a useful fraction."""

    def setUp(self):
        # Distances are large early (structure forming), small in the middle
        # (the regime caching targets), and rise again at the end (detail).
        self.distances = (
            [0.20, 0.15, 0.10]
            + [0.02] * 18
            + [0.08, 0.12, 0.18, 0.25]
        )[:25]

    def test_default_settings_reuse_but_do_not_run_away(self):
        policy = StepCachePolicy(threshold=0.15, warmup_steps=2, cooldown_steps=2)
        run_policy(policy, self.distances)
        self.assertGreater(policy.stats.reuse_fraction, 0.2)
        self.assertLess(policy.stats.reuse_fraction, 0.8)

    def test_higher_threshold_reuses_more(self):
        low = StepCachePolicy(threshold=0.05, warmup_steps=2, cooldown_steps=2)
        high = StepCachePolicy(threshold=0.40, warmup_steps=2, cooldown_steps=2)
        run_policy(low, self.distances)
        run_policy(high, self.distances)
        self.assertGreater(high.stats.reuse_fraction, low.stats.reuse_fraction)

    def test_first_and_last_steps_are_always_computed(self):
        policy = StepCachePolicy(threshold=0.15, warmup_steps=2, cooldown_steps=2)
        decisions = run_policy(policy, self.distances)
        self.assertTrue(decisions[0])
        self.assertTrue(decisions[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
