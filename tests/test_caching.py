"""
Tests for the step-caching transformer wrapper.

These require torch and are skipped automatically where it is not installed (the
policy logic they sit on top of is covered without torch in test_cache_policy.py).
They run on CPU -- no GPU needed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from utils.cache_policy import StepCachePolicy  # noqa: E402

if HAS_TORCH:
    from utils.caching import (  # noqa: E402
        StepCache,
        _rebuild_output,
        _split_output,
        cfg_note,
        relative_l1_distance,
    )


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestDistance(unittest.TestCase):
    def test_identical_tensors_have_zero_distance(self):
        a = torch.ones(4, 4)
        self.assertAlmostEqual(relative_l1_distance(a, a.clone()), 0.0)

    def test_distance_scales_with_change(self):
        previous = torch.ones(4, 4)
        small = previous + 0.01
        large = previous + 0.5
        self.assertLess(
            relative_l1_distance(small, previous), relative_l1_distance(large, previous)
        )

    def test_known_value(self):
        previous = torch.full((10,), 2.0)
        current = torch.full((10,), 3.0)
        # mean(|3-2|) / mean(|2|) = 1 / 2 = 0.5
        self.assertAlmostEqual(relative_l1_distance(current, previous), 0.5, places=6)

    def test_zero_previous_does_not_divide_by_zero(self):
        zeros = torch.zeros(4)
        self.assertEqual(relative_l1_distance(torch.ones(4), zeros), 0.0)

    def test_shape_change_is_infinite(self):
        self.assertEqual(
            relative_l1_distance(torch.ones(4), torch.ones(5)), float("inf")
        )


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestOutputPackaging(unittest.TestCase):
    def test_bare_tensor(self):
        t = torch.ones(2, 2)
        tensor, kind, cls = _split_output(t)
        self.assertIs(tensor, t)
        self.assertEqual(_rebuild_output(t, kind, cls).shape, t.shape)

    def test_tuple(self):
        t = torch.ones(2, 2)
        tensor, kind, cls = _split_output((t,))
        self.assertIs(tensor, t)
        rebuilt = _rebuild_output(t, kind, cls)
        self.assertIsInstance(rebuilt, tuple)
        self.assertIs(rebuilt[0], t)

    def test_sample_attribute_object(self):
        class Output:
            def __init__(self, sample):
                self.sample = sample

        t = torch.ones(2, 2)
        tensor, kind, cls = _split_output(Output(sample=t))
        self.assertIs(tensor, t)
        rebuilt = _rebuild_output(t, kind, cls)
        self.assertIs(rebuilt.sample, t)

    def test_positional_only_output_class(self):
        class PositionalOutput:
            def __init__(self, sample):
                self.sample = sample

            # Reject the keyword form to exercise the TypeError fallback.
            def __init_subclass__(cls):  # pragma: no cover
                pass

        t = torch.ones(2, 2)
        rebuilt = _rebuild_output(t, "sample", PositionalOutput)
        self.assertIs(rebuilt.sample, t)

    def test_unknown_output_type_raises(self):
        with self.assertRaises(TypeError):
            _split_output({"not": "a tensor"})


class CountingTransformer(torch.nn.Module if HAS_TORCH else object):
    """A stand-in transformer that records how many real forwards it ran."""

    def __init__(self, delta=0.1):
        super().__init__()
        self.calls = 0
        self.delta = delta

    def forward(self, hidden_states=None, **kwargs):
        self.calls += 1
        return hidden_states + self.delta


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestStepCache(unittest.TestCase):
    def _drive(self, cache, steps, drift=0.0):
        """Run `steps` fake denoising steps, returning the final latent."""
        latent = torch.zeros(2, 4)
        for _ in range(steps):
            latent = cache.transformer.forward(hidden_states=latent)
            latent = latent + drift
        return latent

    def test_install_and_uninstall_restore_forward(self):
        model = CountingTransformer()
        original = model.forward
        cache = StepCache(model, StepCachePolicy(total_steps=4), verbose=False)
        cache.install()
        self.assertIsNot(model.forward, original)
        cache.uninstall()
        self.assertIs(model.forward, original)

    def test_double_install_raises(self):
        model = CountingTransformer()
        cache = StepCache(model, StepCachePolicy(total_steps=4), verbose=False)
        cache.install()
        try:
            with self.assertRaises(RuntimeError):
                cache.install()
        finally:
            cache.uninstall()

    def test_context_manager_restores_on_exception(self):
        model = CountingTransformer()
        original = model.forward
        cache = StepCache(model, StepCachePolicy(total_steps=4), verbose=False)
        try:
            with cache:
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertIs(model.forward, original)

    def test_caching_reduces_real_forward_count(self):
        model = CountingTransformer()
        policy = StepCachePolicy(
            threshold=1e9, total_steps=10, warmup_steps=1, cooldown_steps=1,
            max_consecutive_reuse=None,
        )
        cache = StepCache(model, policy, verbose=False)
        with cache:
            self._drive(cache, 10)
        # Step 0 forced (no cache), step 9 is cooldown; the rest reuse.
        self.assertLess(model.calls, 10)
        self.assertGreater(policy.stats.reused, 0)

    def test_no_caching_when_threshold_is_zero(self):
        model = CountingTransformer()
        policy = StepCachePolicy(threshold=0.0, total_steps=6, warmup_steps=0, cooldown_steps=0)
        cache = StepCache(model, policy, verbose=False)
        with cache:
            self._drive(cache, 6)
        self.assertEqual(model.calls, 6)
        self.assertEqual(policy.stats.reused, 0)

    def test_reused_output_matches_residual_application(self):
        # With a constant-delta transformer the cached residual is exactly right,
        # so caching must be numerically lossless here.
        model = CountingTransformer(delta=0.25)
        policy = StepCachePolicy(
            threshold=1e9, total_steps=5, warmup_steps=1, cooldown_steps=0,
            max_consecutive_reuse=None,
        )
        cache = StepCache(model, policy, verbose=False)
        with cache:
            cached_result = self._drive(cache, 5)

        plain = torch.zeros(2, 4)
        for _ in range(5):
            plain = plain + 0.25
        self.assertTrue(torch.allclose(cached_result, plain, atol=1e-6))

    def test_first_step_always_computes(self):
        model = CountingTransformer()
        policy = StepCachePolicy(threshold=1e9, total_steps=3, warmup_steps=0, cooldown_steps=0)
        cache = StepCache(model, policy, verbose=False)
        with cache:
            self._drive(cache, 3)
        self.assertGreaterEqual(model.calls, 1)
        self.assertEqual(policy.stats.forced_no_cache, 1)

    def test_missing_hidden_states_falls_through(self):
        class NoTensor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, **kwargs):
                self.calls += 1
                return torch.ones(2, 2)

        model = NoTensor()
        cache = StepCache(model, StepCachePolicy(total_steps=3), verbose=False)
        with cache:
            for _ in range(3):
                model.forward(timestep=1)
        self.assertEqual(model.calls, 3)

    def test_positional_hidden_states_are_found(self):
        model = CountingTransformer()
        policy = StepCachePolicy(threshold=0.0, total_steps=3, warmup_steps=0, cooldown_steps=0)
        cache = StepCache(model, policy, verbose=False)
        with cache:
            latent = torch.zeros(2, 4)
            for _ in range(3):
                latent = model.forward(latent)
        self.assertEqual(model.calls, 3)

    def test_invalid_calls_per_step_rejected(self):
        with self.assertRaises(ValueError):
            StepCache(CountingTransformer(), StepCachePolicy(), calls_per_step=0)

    def test_uninstall_is_idempotent(self):
        cache = StepCache(CountingTransformer(), StepCachePolicy(), verbose=False)
        cache.uninstall()  # never installed
        cache.install()
        cache.uninstall()
        cache.uninstall()


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestCfgNote(unittest.TestCase):
    def test_enabled_note_mentions_doubled_batch(self):
        self.assertIn("doubled", cfg_note(3.5))

    def test_disabled_note_warns_about_distillation(self):
        note = cfg_note(1.0)
        self.assertIn("distilled", note)


if __name__ == "__main__":
    unittest.main(verbosity=2)
