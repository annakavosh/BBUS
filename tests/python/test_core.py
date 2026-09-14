import math
import unittest

import belief_update


class CoreBindingTest(unittest.TestCase):
    def test_hand_calculated_exact_update(self) -> None:
        result = belief_update.exact_binary_update(
            belief_update.BeliefState(0.4),
            belief_update.LikelihoodValues(0.8, 0.2),
        )

        self.assertAlmostEqual(result.likelihood_ratio, 4.0)
        self.assertAlmostEqual(result.prior_odds, 2.0 / 3.0)
        self.assertAlmostEqual(result.posterior_odds, 8.0 / 3.0)
        self.assertAlmostEqual(result.posterior, 8.0 / 11.0)
        self.assertAlmostEqual(result.belief_delta, 18.0 / 55.0)

    def test_hand_calculated_log_update(self) -> None:
        result = belief_update.log_odds_update(
            belief_update.BeliefState(0.2), math.log(2.0)
        )

        self.assertAlmostEqual(result.posterior_odds, 0.5)
        self.assertAlmostEqual(result.posterior, 1.0 / 3.0)

    def test_impossible_observation_fails_clearly(self) -> None:
        with self.assertRaises(ValueError):
            belief_update.exact_binary_update(
                belief_update.BeliefState(0.5),
                belief_update.LikelihoodValues(0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
