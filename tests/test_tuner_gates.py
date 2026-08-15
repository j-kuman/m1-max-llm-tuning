from __future__ import annotations

import unittest

from tuner.gates import extract_round_stats
from tuner.promote import ScreenStats, classify


CFG = {
    "campaign": {"max_tokens": 512},
    "champion": {
        "mean_tps": 27.556,
        "rounds": 138,
        "tokens": 512,
    },
    "promotion": {
        "equal_round_min_tps_ratio": 0.995,
        "one_extra_round_min_tps_ratio": 1.005,
        "reject_round_delta_at_or_above": 2,
    },
}


class MissingRounds:
    pass


class EmptyRounds:
    accept_lens = []


class GoodRounds:
    accept_lens = [1, 2, 3, 4]


class GateTests(unittest.TestCase):
    def test_round_telemetry_missing_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_round_stats(MissingRounds(), 512)

    def test_zero_rounds_with_generated_tokens_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_round_stats(EmptyRounds(), 512)

    def test_normalized_efficiency(self) -> None:
        rounds, tpr = extract_round_stats(GoodRounds(), 12)
        self.assertEqual(rounds, 4)
        self.assertEqual(tpr, 3.0)

    def make_stats(self, **overrides) -> ScreenStats:
        data = dict(
            mean_tps=27.60,
            mean_rounds=138.0,
            mean_tokens=512.0,
            mean_tokens_per_round=512 / 138,
            count=1,
            min_tokens=512,
            max_tokens=512,
            min_rounds=138,
            max_rounds=138,
            distinct_hashes=1,
            text_sha256="abc",
        )
        data.update(overrides)
        return ScreenStats(**data)

    def test_hash_mismatch_rejects(self) -> None:
        decision, _ = classify(
            CFG,
            self.make_stats(text_sha256="bad"),
            reference_tokens=512,
            reference_hash="abc",
        )
        self.assertEqual(decision, "reject")

    def test_early_eos_cannot_look_like_round_improvement(self) -> None:
        decision, _ = classify(
            CFG,
            self.make_stats(
                mean_rounds=100.0,
                mean_tokens=300.0,
                mean_tokens_per_round=3.0,
                min_tokens=300,
                max_tokens=300,
                min_rounds=100,
                max_rounds=100,
            ),
            reference_tokens=512,
            reference_hash="abc",
        )
        self.assertEqual(decision, "reject")

    def test_nondeterministic_round_count_rejects(self) -> None:
        decision, _ = classify(
            CFG,
            self.make_stats(
                mean_rounds=137.5,
                min_rounds=137,
                max_rounds=138,
            ),
            reference_tokens=512,
            reference_hash="abc",
        )
        self.assertEqual(decision, "reject")

    def test_exact_same_round_candidate_can_advance(self) -> None:
        decision, _ = classify(
            CFG,
            self.make_stats(),
            reference_tokens=512,
            reference_hash="abc",
        )
        self.assertEqual(decision, "advance")


if __name__ == "__main__":
    unittest.main()
