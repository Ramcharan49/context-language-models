# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only checks for the tool observation head/tail character limits."""

import unittest
from types import SimpleNamespace

from clm_harness.context_env.env import ContextEnv
from clm_harness.context_env.paths import Paths
from clm_harness.utils.budget import BudgetController


class ObservationTruncationTests(unittest.TestCase):
    def make_env(self, max_chars=10, obs_cfg=None):
        env = ContextEnv(
            paths=Paths(SimpleNamespace()),
            budget=BudgetController(0),
            observation_max_chars=max_chars,
            obs_cfg=obs_cfg,
        )
        self.addCleanup(env.host_mirror.unlink)
        return env

    def assert_truncated(self, result, head, tail, elided):
        # Compare the actual retained text as well as the reported lengths.
        _, sections = result.split("\n\n", 1)
        self.assertEqual(
            sections,
            f"---- HEAD ({len(head)} chars) ----\n{head}\n"
            f"---- {elided} chars elided ----\n"
            f"---- TAIL ({len(tail)} chars) ----\n{tail}",
        )

    def test_output_within_limit_is_unchanged(self):
        env = self.make_env()
        for output in ("", "short", "0123456789"):
            with self.subTest(output=output):
                self.assertEqual(env.truncate_observation(output), output)

    def test_default_sections_are_reduced_to_fit_limit(self):
        env = self.make_env()
        self.assert_truncated(
            env.truncate_observation("0123456789abcdefghij"), "01234", "fghij", 10
        )

    def test_custom_sections_and_hint(self):
        env = self.make_env(obs_cfg={
            "head_chars": 2, "tail_chars": 3, "too_long_hint": "Narrow the command."
        })
        result = env.truncate_observation("0123456789abcdefghij")
        self.assertTrue(result.startswith("Narrow the command.\n\n"))
        self.assert_truncated(result, "01", "hij", 15)

    def test_zero_tail_does_not_restore_full_output(self):
        env = self.make_env(obs_cfg={"head_chars": 4, "tail_chars": 0})
        self.assert_truncated(
            env.truncate_observation("0123456789abcdefghij"), "0123", "", 16
        )

    def test_zero_head_keeps_only_tail(self):
        env = self.make_env(obs_cfg={"head_chars": 0, "tail_chars": 4})
        self.assert_truncated(
            env.truncate_observation("0123456789abcdefghij"), "", "ghij", 16
        )

    def test_both_sections_can_be_empty(self):
        env = self.make_env(obs_cfg={"head_chars": 0, "tail_chars": 0})
        self.assert_truncated(
            env.truncate_observation("0123456789abcdefghij"), "", "", 20
        )

    def test_tiny_limit_does_not_restore_full_output(self):
        for max_chars in (0, 1):
            with self.subTest(max_chars=max_chars):
                env = self.make_env(max_chars=max_chars)
                self.assert_truncated(env.truncate_observation("0123456789"), "", "", 10)

    def test_formatted_result_preserves_exit_code_after_truncation(self):
        env = self.make_env(obs_cfg={"head_chars": 4, "tail_chars": 0})
        result = SimpleNamespace(stdout="0123456789abcdefghij\n", stderr="", return_code=2)
        text, exit_code = env.format_tool_result(result).rsplit("\n\n", 1)
        self.assertEqual(exit_code, "(exit_code=2)")
        self.assert_truncated(text, "0123", "", 16)


if __name__ == "__main__":
    unittest.main()
