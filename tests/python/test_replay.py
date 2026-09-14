import unittest

import belief_update


def make_input(
    true_likelihood: float,
    false_likelihood: float,
    override_status: belief_update.OverrideStatus,
) -> belief_update.LikelihoodInput:
    return belief_update.LikelihoodInput(
        belief_update.LikelihoodValues(true_likelihood, false_likelihood),
        0.8,
        belief_update.Provenance(
            belief_update.SourceType.model,
            belief_update.SourceId("replay-model"),
            belief_update.SourceVersion("5"),
            belief_update.ProvenanceTimestamp(1_000),
            0.95,
            override_status,
        ),
    )


class ReplayTest(unittest.TestCase):
    def test_complete_result_replays_exactly(self) -> None:
        inputs = belief_update.UpdateInputs(
            belief_update.BeliefState(0.4),
            make_input(0.8, 0.2, belief_update.OverrideStatus.requested),
            make_input(0.6, 0.4, belief_update.OverrideStatus.not_requested),
            belief_update.ProvenanceTimestamp(1_100),
        )
        configuration = belief_update.UpdateConfig(
            belief_update.FreshnessRule(
                50, belief_update.ValidationDecision.warn
            )
        )

        original = belief_update.update(
            inputs, configuration, belief_update.UpdateAlgorithm.exact_binary
        )
        reproduced = belief_update.replay(original)

        self.assertEqual(original.posterior, reproduced.posterior)
        self.assertEqual(
            original.intermediate_calculations.posterior_odds,
            reproduced.intermediate_calculations.posterior_odds,
        )
        self.assertEqual(
            original.resolved_source,
            belief_update.ResolvedInputSource.explicit_override,
        )
        self.assertEqual(original.resolved_values.p_e_given_h, 0.8)
        self.assertEqual(original.provenance.source_id.value, "replay-model")
        self.assertEqual(
            original.configuration.freshness_rule.max_age_nanoseconds, 50
        )
        self.assertEqual(
            original.validation_warnings,
            (belief_update.ValidationCode.freshness_violation,),
        )
        self.assertEqual(original.library_version, belief_update.__version__)

        with self.assertRaises(AttributeError):
            original.posterior = 0.5


if __name__ == "__main__":
    unittest.main()
