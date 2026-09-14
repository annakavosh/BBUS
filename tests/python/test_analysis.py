import unittest

import belief_update


def make_input(
    p_e_given_h: float,
    p_e_given_not_h: float,
    source_type: belief_update.SourceType,
    source_id: str,
    override_status: belief_update.OverrideStatus,
) -> belief_update.LikelihoodInput:
    return belief_update.LikelihoodInput(
        likelihoods=belief_update.LikelihoodValues(
            evidence_given_hypothesis=p_e_given_h,
            evidence_given_not_hypothesis=p_e_given_not_h,
        ),
        evidence_weight=None,
        provenance=belief_update.Provenance(
            source_type=source_type,
            source_id=belief_update.SourceId(value=source_id),
            version=belief_update.SourceVersion(value="1"),
            timestamp=belief_update.ProvenanceTimestamp(
                unix_nanoseconds=1_000
            ),
            confidence=1.0,
            override_status=override_status,
        ),
    )


class AnalysisBindingTest(unittest.TestCase):
    def test_local_sensitivity_uses_cpp_result(self) -> None:
        result = belief_update.update(
            prior=0.4, p_e_given_h=0.8, p_e_given_not_h=0.2
        )

        sensitivity = belief_update.local_sensitivity(result=result)

        self.assertAlmostEqual(sensitivity.posterior, 8.0 / 11.0)
        self.assertAlmostEqual(
            sensitivity.d_posterior_d_prior, 100.0 / 121.0
        )

    def test_model_and_explicit_comparison_uses_cpp_updates(self) -> None:
        configuration = belief_update.UpdateConfig(freshness_rule=None)
        original = belief_update.update(
            inputs=belief_update.UpdateInputs(
                prior=belief_update.BeliefState(probability=0.4),
                explicit_override=make_input(
                    0.8,
                    0.3,
                    belief_update.SourceType.human,
                    "analyst-7",
                    belief_update.OverrideStatus.requested,
                ),
                likelihood_model_value=make_input(
                    0.8,
                    0.2,
                    belief_update.SourceType.model,
                    "model-a",
                    belief_update.OverrideStatus.not_requested,
                ),
                evaluation_time=belief_update.ProvenanceTimestamp(
                    unix_nanoseconds=1_000
                ),
            ),
            configuration=configuration,
            algorithm=belief_update.UpdateAlgorithm.exact_binary,
        )

        comparison = belief_update.compare_model_and_explicit(result=original)

        self.assertAlmostEqual(comparison.model_posterior, 8.0 / 11.0)
        self.assertAlmostEqual(comparison.explicit_posterior, 0.64)
        self.assertAlmostEqual(
            comparison.difference, 0.64 - (8.0 / 11.0)
        )

    def test_threshold_analysis_matches_cpp_conventions(self) -> None:
        result = belief_update.update(
            prior=0.4, p_e_given_h=0.8, p_e_given_not_h=0.3
        )

        threshold = belief_update.analyze_threshold(
            result=result, threshold=0.6
        )

        self.assertEqual(
            threshold.transition, belief_update.ThresholdTransition.upward
        )
        self.assertTrue(threshold.threshold_met)
        self.assertAlmostEqual(threshold.likelihood_ratio_at_threshold, 2.25)


if __name__ == "__main__":
    unittest.main()
