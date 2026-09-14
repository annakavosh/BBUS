import gc
import math
import unittest

import belief_update


def make_input(
    true_likelihood: float,
    false_likelihood: float,
    override_status: belief_update.OverrideStatus,
    timestamp: int = 1_000,
) -> belief_update.LikelihoodInput:
    return belief_update.LikelihoodInput(
        belief_update.LikelihoodValues(true_likelihood, false_likelihood),
        None,
        belief_update.Provenance(
            belief_update.SourceType.model,
            belief_update.SourceId("model-a"),
            belief_update.SourceVersion("3"),
            belief_update.ProvenanceTimestamp(timestamp),
            0.9,
            override_status,
        ),
    )


class ValidatorBindingTest(unittest.TestCase):
    def test_keyword_workflow_uses_one_configuration(self) -> None:
        configuration = belief_update.UpdateConfig(freshness_rule=None)
        evaluation_time = belief_update.ProvenanceTimestamp(
            unix_nanoseconds=1_000
        )
        likelihood_input = belief_update.LikelihoodInput(
            likelihoods=belief_update.LikelihoodValues(
                evidence_given_hypothesis=0.8,
                evidence_given_not_hypothesis=0.2,
            ),
            evidence_weight=None,
            provenance=belief_update.Provenance(
                source_type=belief_update.SourceType.human,
                source_id=belief_update.SourceId(value="analyst-7"),
                version=belief_update.SourceVersion(value="1"),
                timestamp=evaluation_time,
                confidence=0.9,
                override_status=belief_update.OverrideStatus.requested,
            ),
        )
        inputs = belief_update.UpdateInputs(
            prior=belief_update.BeliefState(probability=0.4),
            explicit_override=likelihood_input,
            likelihood_model_value=None,
            evaluation_time=evaluation_time,
        )

        validation = belief_update.Validator(
            configuration=configuration
        ).validate_input(likelihood_input, evaluation_time)
        self.assertTrue(validation.is_accepted)

        result = belief_update.update(
            inputs=inputs,
            configuration=configuration,
            algorithm=belief_update.UpdateAlgorithm.exact_binary,
        )
        self.assertAlmostEqual(result.posterior, 8.0 / 11.0)

    def test_scalar_failures_have_typed_codes(self) -> None:
        validator = belief_update.Validator(belief_update.UpdateConfig(None))

        self.assertEqual(
            validator.validate_probability(math.nan).code,
            belief_update.ValidationCode.probability_not_finite,
        )
        self.assertEqual(
            validator.validate_probability(math.inf).decision,
            belief_update.ValidationDecision.reject,
        )
        self.assertTrue(validator.validate_probability(math.inf).is_rejected)
        self.assertFalse(
            validator.validate_probability(math.inf).can_proceed
        )
        self.assertEqual(
            validator.validate_likelihood_ratio(0.0).code,
            belief_update.ValidationCode.likelihood_ratio_not_positive,
        )
        self.assertEqual(
            validator.validate_required_input(
                None, belief_update.ProvenanceTimestamp(1_000)
            ).code,
            belief_update.ValidationCode.missing_required_input,
        )

    def test_explicit_freshness_rule_can_warn(self) -> None:
        validator = belief_update.Validator(
            belief_update.UpdateConfig(
                belief_update.FreshnessRule(
                    50, belief_update.ValidationDecision.warn
                )
            )
        )
        result = validator.validate_input(
            make_input(0.8, 0.2, belief_update.OverrideStatus.not_requested),
            belief_update.ProvenanceTimestamp(1_100),
        )

        self.assertTrue(result.can_proceed)
        self.assertFalse(result.is_accepted)
        self.assertTrue(result.has_warning)
        self.assertEqual(
            result.code, belief_update.ValidationCode.freshness_violation
        )


class ResolverBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = belief_update.InputResolver(
            belief_update.Validator(belief_update.UpdateConfig(None))
        )
        self.evaluated = belief_update.ProvenanceTimestamp(1_100)

    def test_override_has_precedence_without_mutation(self) -> None:
        explicit_override = make_input(
            0.9, 0.1, belief_update.OverrideStatus.requested
        )
        model_value = make_input(
            0.7, 0.3, belief_update.OverrideStatus.not_requested
        )

        result = self.resolver.resolve(
            explicit_override, model_value, self.evaluated
        )

        self.assertEqual(
            result.resolved_input.source,
            belief_update.ResolvedInputSource.explicit_override,
        )
        self.assertEqual(
            result.resolved_input.input.likelihoods.evidence_given_hypothesis,
            0.9,
        )
        self.assertEqual(
            explicit_override.provenance.override_status,
            belief_update.OverrideStatus.requested,
        )

    def test_invalid_override_does_not_fall_back(self) -> None:
        invalid_override = make_input(
            0.8, 0.0, belief_update.OverrideStatus.requested
        )
        model_value = make_input(
            0.7, 0.3, belief_update.OverrideStatus.not_requested
        )

        result = self.resolver.resolve(
            invalid_override, model_value, self.evaluated
        )

        self.assertFalse(result.has_input)
        self.assertEqual(
            result.validation.code,
            belief_update.ValidationCode.likelihood_ratio_not_finite,
        )

    def test_missing_inputs_reject(self) -> None:
        result = self.resolver.resolve(None, None, self.evaluated)

        self.assertFalse(result.has_input)
        self.assertEqual(
            result.validation.code,
            belief_update.ValidationCode.missing_required_input,
        )

    def test_selected_temporary_remains_readable(self) -> None:
        result = self.resolver.resolve(
            make_input(0.6, 0.4, belief_update.OverrideStatus.requested),
            None,
            self.evaluated,
        )
        gc.collect()

        self.assertEqual(
            result.resolved_input.input.likelihoods.evidence_given_hypothesis,
            0.6,
        )


if __name__ == "__main__":
    unittest.main()
