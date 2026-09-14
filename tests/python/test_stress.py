import unittest

import belief_update


class StressBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = belief_update.update(
            prior=0.4, p_e_given_h=0.8, p_e_given_not_h=0.2
        )

    def test_relative_ranges_are_explicit_and_non_mutating(self) -> None:
        original_posterior = self.original.posterior
        ranges = belief_update.RelativeStressRanges(
            prior_percent=10.0,
            p_e_given_h_percent=10.0,
            p_e_given_not_h_percent=10.0,
        )

        stressed = belief_update.stress_update(
            result=self.original, ranges=ranges
        )

        self.assertEqual(
            stressed.range_mode, belief_update.StressRangeMode.relative_percent
        )
        self.assertAlmostEqual(stressed.resolved_bounds.prior.lower, 0.36)
        self.assertAlmostEqual(stressed.resolved_bounds.prior.upper, 0.44)
        self.assertEqual(
            stressed.threshold_crossing,
            belief_update.ThresholdCrossing.not_evaluated,
        )
        self.assertEqual(self.original.posterior, original_posterior)

    def test_absolute_ranges_and_threshold_match_cpp(self) -> None:
        ranges = belief_update.AbsoluteStressRanges(
            prior=belief_update.ProbabilityBounds(lower=0.3, upper=0.5),
            p_e_given_h=belief_update.ProbabilityBounds(
                lower=0.7, upper=0.9
            ),
            p_e_given_not_h=belief_update.ProbabilityBounds(
                lower=0.1, upper=0.3
            ),
        )

        stressed = belief_update.stress_update(
            result=self.original, ranges=ranges, threshold=0.8
        )

        self.assertEqual(
            stressed.range_mode, belief_update.StressRangeMode.absolute_bounds
        )
        self.assertAlmostEqual(stressed.worst_case_posterior, 0.5)
        self.assertEqual(
            stressed.threshold_crossing,
            belief_update.ThresholdCrossing.upward,
        )

    def test_invalid_range_is_rejected_without_clipping(self) -> None:
        ranges = belief_update.RelativeStressRanges(
            prior_percent=160.0,
            p_e_given_h_percent=10.0,
            p_e_given_not_h_percent=10.0,
        )

        with self.assertRaises(ValueError):
            belief_update.stress_update(result=self.original, ranges=ranges)


if __name__ == "__main__":
    unittest.main()
