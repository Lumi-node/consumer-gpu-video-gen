"""Tests for benchmark config spec parsing."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bench_config import (  # noqa: E402
    ConfigSpec,
    parse_config_spec,
    parse_config_specs,
    pick_baseline,
)


class TestParseSpec(unittest.TestCase):
    def test_bare_precision(self):
        spec = parse_config_spec("bf16")
        self.assertEqual(spec.precision, "bf16")
        self.assertFalse(spec.cache)
        self.assertTrue(spec.use_cfg)

    def test_cache_modifier(self):
        spec = parse_config_spec("nvfp4+cache")
        self.assertEqual(spec.precision, "nvfp4")
        self.assertTrue(spec.cache)
        self.assertFalse(spec.timestep_aware)

    def test_timestep_aware_implies_cache(self):
        spec = parse_config_spec("int4+tscache")
        self.assertTrue(spec.cache)
        self.assertTrue(spec.timestep_aware)

    def test_nocfg_modifier(self):
        self.assertFalse(parse_config_spec("nvfp4+nocfg").use_cfg)

    def test_all_modifiers(self):
        spec = parse_config_spec("nvfp4+tscache+nocfg")
        self.assertEqual(spec.precision, "nvfp4")
        self.assertTrue(spec.cache)
        self.assertTrue(spec.timestep_aware)
        self.assertFalse(spec.use_cfg)

    def test_case_and_whitespace_insensitive(self):
        spec = parse_config_spec("  NVFP4 + Cache ")
        self.assertEqual(spec.precision, "nvfp4")
        self.assertTrue(spec.cache)

    def test_label_defaults_to_spec(self):
        self.assertEqual(parse_config_spec("nvfp4+cache").resolved_label(), "nvfp4+cache")

    def test_unknown_modifier_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_config_spec("nvfp4+turbo")
        self.assertIn("turbo", str(ctx.exception))

    def test_empty_spec_rejected(self):
        with self.assertRaises(ValueError):
            parse_config_spec("")
        with self.assertRaises(ValueError):
            parse_config_spec("   ")

    def test_describe_is_human_readable(self):
        text = parse_config_spec("nvfp4+tscache+nocfg").describe()
        self.assertIn("nvfp4", text)
        self.assertIn("timestep-aware", text)
        self.assertIn("CFG disabled", text)


class TestParseMany(unittest.TestCase):
    def test_parses_a_sweep(self):
        configs = parse_config_specs(["bf16", "int4", "nvfp4+cache"])
        self.assertEqual(len(configs), 3)
        self.assertEqual(configs[2].precision, "nvfp4")

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            parse_config_specs(["bf16", "bf16"])

    def test_same_precision_different_modifiers_is_allowed(self):
        configs = parse_config_specs(["nvfp4", "nvfp4+cache"])
        self.assertEqual(len(configs), 2)


class TestBaseline(unittest.TestCase):
    def test_prefers_plain_bf16(self):
        configs = parse_config_specs(["nvfp4", "bf16", "int4"])
        self.assertEqual(pick_baseline(configs), "bf16")

    def test_does_not_pick_modified_bf16(self):
        # bf16+cache is not a clean reference; fall through to the first config.
        configs = parse_config_specs(["nvfp4", "bf16+cache"])
        self.assertEqual(pick_baseline(configs), "nvfp4")

    def test_falls_back_to_first_config(self):
        configs = parse_config_specs(["int4", "nvfp4"])
        self.assertEqual(pick_baseline(configs), "int4")

    def test_explicit_request_honored(self):
        configs = parse_config_specs(["bf16", "int4"])
        self.assertEqual(pick_baseline(configs, "int4"), "int4")

    def test_unknown_request_rejected(self):
        configs = parse_config_specs(["bf16"])
        with self.assertRaises(ValueError):
            pick_baseline(configs, "nvfp4")

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            pick_baseline([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
