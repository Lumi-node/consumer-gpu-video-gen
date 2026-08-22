"""Tests for precision backend selection and fallback chains."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.precision import (  # noqa: E402
    PRECISIONS,
    BackendAvailability,
    GpuCapability,
    Selection,
    estimate_weight_gb,
    expected_speed_note,
    select_backend,
    supports,
)

# Real devices this project targets.
RTX_5090 = GpuCapability(sm_major=12, sm_minor=0, name="RTX 5090", total_vram_gb=32.0)
RTX_4090 = GpuCapability(sm_major=8, sm_minor=9, name="RTX 4090", total_vram_gb=24.0)
RTX_3090 = GpuCapability(sm_major=8, sm_minor=6, name="RTX 3090", total_vram_gb=24.0)
T4 = GpuCapability(sm_major=7, sm_minor=5, name="Tesla T4", total_vram_gb=16.0)

ALL_BACKENDS = BackendAvailability(torchao_fp4=True, torchao_fp8=True, quanto=True)
NO_BACKENDS = BackendAvailability()
QUANTO_ONLY = BackendAvailability(quanto=True)


class TestCapability(unittest.TestCase):
    def test_sm_number(self):
        self.assertEqual(RTX_5090.sm, 120)
        self.assertEqual(RTX_4090.sm, 89)
        self.assertEqual(T4.sm, 75)

    def test_blackwell_detection(self):
        self.assertTrue(RTX_5090.is_blackwell)
        self.assertFalse(RTX_4090.is_blackwell)

    def test_str_includes_sm(self):
        self.assertIn("sm_120", str(RTX_5090))


class TestSupports(unittest.TestCase):
    def test_fp4_supported_on_blackwell(self):
        ok, reason = supports("nvfp4", RTX_5090, ALL_BACKENDS)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_fp4_rejected_on_ada(self):
        ok, reason = supports("nvfp4", RTX_4090, ALL_BACKENDS)
        self.assertFalse(ok)
        self.assertIn("sm_100", reason)
        self.assertIn("sm_89", reason)

    def test_fp8_supported_on_ada_but_not_ampere(self):
        self.assertTrue(supports("fp8", RTX_4090, ALL_BACKENDS)[0])
        self.assertFalse(supports("fp8", RTX_3090, ALL_BACKENDS)[0])

    def test_hardware_blocker_reported_before_library(self):
        # Ampere lacks fp8 hardware AND the library is missing; the message should
        # name the hardware, which is the blocker the user cannot pip-install away.
        ok, reason = supports("fp8", RTX_3090, NO_BACKENDS)
        self.assertFalse(ok)
        self.assertIn("compute capability", reason)

    def test_missing_library_reported_when_hardware_is_fine(self):
        ok, reason = supports("nvfp4", RTX_5090, NO_BACKENDS)
        self.assertFalse(ok)
        self.assertIn("torchao_fp4", reason)

    def test_unknown_precision_raises(self):
        with self.assertRaises(KeyError):
            supports("fp2", RTX_5090, ALL_BACKENDS)


class TestSelection(unittest.TestCase):
    def test_honors_supported_request(self):
        sel = select_backend("nvfp4", RTX_5090, ALL_BACKENDS)
        self.assertEqual(sel.precision, "nvfp4")
        self.assertFalse(sel.is_fallback)
        self.assertIn("using nvfp4", sel.explain())

    def test_falls_back_from_fp4_to_fp8_on_ada(self):
        sel = select_backend("nvfp4", RTX_4090, ALL_BACKENDS)
        self.assertEqual(sel.precision, "fp8")
        self.assertTrue(sel.is_fallback)
        self.assertIn("requested nvfp4", sel.explain())

    def test_falls_back_to_bf16_on_ampere(self):
        sel = select_backend("nvfp4", RTX_3090, ALL_BACKENDS)
        self.assertEqual(sel.precision, "bf16")

    def test_falls_back_when_library_missing(self):
        sel = select_backend("nvfp4", RTX_5090, NO_BACKENDS)
        self.assertEqual(sel.precision, "bf16")
        self.assertTrue(any("torchao_fp4" in r for r in sel.fallback_reasons))

    def test_auto_picks_fp4_on_5090(self):
        sel = select_backend("auto", RTX_5090, ALL_BACKENDS)
        self.assertEqual(sel.precision, "nvfp4")

    def test_auto_picks_fp8_on_4090(self):
        sel = select_backend("auto", RTX_4090, ALL_BACKENDS)
        self.assertEqual(sel.precision, "fp8")

    def test_auto_picks_bf16_on_3090(self):
        self.assertEqual(select_backend("auto", RTX_3090, ALL_BACKENDS).precision, "bf16")

    def test_auto_memory_goal_prefers_int4_over_bf16(self):
        # On Ada there is no fp4 hardware, so the memory chain should land on int4
        # rather than the fp8/bf16 the speed chain would choose.
        sel = select_backend("auto", RTX_4090, QUANTO_ONLY, goal="memory")
        self.assertEqual(sel.precision, "int4")

    def test_memory_goal_falls_back_within_memory_chain(self):
        sel = select_backend("nvfp4", RTX_4090, QUANTO_ONLY, goal="memory")
        self.assertEqual(sel.precision, "int4")

    def test_int4_still_available_on_turing(self):
        sel = select_backend("int4", T4, QUANTO_ONLY)
        self.assertEqual(sel.precision, "int4")

    def test_bf16_request_always_honored_on_modern_gpus(self):
        for gpu in (RTX_5090, RTX_4090, RTX_3090):
            self.assertEqual(select_backend("bf16", gpu, NO_BACKENDS).precision, "bf16")

    def test_invalid_goal_raises(self):
        with self.assertRaises(ValueError):
            select_backend("auto", RTX_5090, ALL_BACKENDS, goal="quality")

    def test_unknown_request_raises(self):
        with self.assertRaises(KeyError):
            select_backend("fp2", RTX_5090, ALL_BACKENDS)


class TestHonestyAboutSpeed(unittest.TestCase):
    """The table must not let weight-only modes masquerade as speed optimizations."""

    def test_int4_and_int8_do_not_claim_compute_acceleration(self):
        self.assertFalse(PRECISIONS["int4"].accelerates_compute)
        self.assertFalse(PRECISIONS["int8"].accelerates_compute)

    def test_fp4_and_fp8_do_claim_compute_acceleration(self):
        self.assertTrue(PRECISIONS["nvfp4"].accelerates_compute)
        self.assertTrue(PRECISIONS["fp8"].accelerates_compute)

    def test_int4_note_warns_about_latency(self):
        note = expected_speed_note("int4")
        self.assertIn("weight-only", note)
        self.assertIn("Benchmark", note)

    def test_fp4_note_promises_latency_win(self):
        self.assertIn("lower latency", expected_speed_note("nvfp4"))

    def test_int4_is_absent_from_the_speed_chain(self):
        from utils.precision import SPEED_CHAIN

        self.assertNotIn("int4", SPEED_CHAIN)
        self.assertNotIn("int8", SPEED_CHAIN)


class TestSizeEstimates(unittest.TestCase):
    def test_int4_quarters_weights(self):
        self.assertAlmostEqual(estimate_weight_gb(40.0, "int4"), 10.0)

    def test_fp8_halves_weights(self):
        self.assertAlmostEqual(estimate_weight_gb(40.0, "fp8"), 20.0)

    def test_bf16_is_unchanged(self):
        self.assertAlmostEqual(estimate_weight_gb(40.0, "bf16"), 40.0)

    def test_negative_size_rejected(self):
        with self.assertRaises(ValueError):
            estimate_weight_gb(-1.0, "int4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
