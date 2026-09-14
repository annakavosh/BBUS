import math
import unittest

import belief_update


def make_model_output() -> belief_update.LikelihoodInput:
    return belief_update.LikelihoodInput(
        belief_update.LikelihoodValues(0.8, 0.2),
        None,
        belief_update.Provenance(
            belief_update.SourceType.model,
            belief_update.SourceId("python-test-model"),
            belief_update.SourceVersion("1"),
            belief_update.ProvenanceTimestamp(1_000),
            0.9,
            belief_update.OverrideStatus.not_requested,
        ),
    )


class PythonTestModel(belief_update.LikelihoodModel):
    def __init__(self) -> None:
        super().__init__()
        self._output = make_model_output()

    def infer(self) -> belief_update.LikelihoodInput:
        return self._output


class BindingEquivalenceTest(unittest.TestCase):
    def assert_matches_core(
        self,
        result: belief_update.UpdateResult,
        core: belief_update.BayesCalculation,
    ) -> None:
        values = result.intermediate_calculations
        self.assertEqual(values.prior, core.prior)
        self.assertEqual(values.likelihood_ratio, core.likelihood_ratio)
        self.assertEqual(values.prior_odds, core.prior_odds)
        self.assertEqual(values.posterior_odds, core.posterior_odds)
        self.assertEqual(values.posterior, core.posterior)
        self.assertEqual(values.belief_delta, core.belief_delta)

    def test_scalar_update_matches_cpp_core(self) -> None:
        core = belief_update.exact_binary_update(
            belief_update.BeliefState(0.4),
            belief_update.LikelihoodValues(0.8, 0.2),
        )
        result = belief_update.update(
            prior=0.4,
            p_e_given_h=0.8,
            p_e_given_not_h=0.2,
        )

        self.assert_matches_core(result, core)
        self.assertEqual(result.algorithm, belief_update.UpdateAlgorithm.exact_binary)
        self.assertEqual(
            result.resolved_source,
            belief_update.ResolvedInputSource.explicit_override,
        )
        self.assertEqual(
            result.provenance.source_type,
            belief_update.SourceType.direct_input,
        )
        self.assertEqual(result.provenance.source_id.value, "belief_update.direct")
        self.assertEqual(result.provenance.version.value, belief_update.__version__)
        self.assertEqual(result.provenance.timestamp.unix_nanoseconds, 0)
        self.assertEqual(result.provenance.confidence, 1.0)
        self.assertEqual(
            result.provenance.override_status,
            belief_update.OverrideStatus.applied,
        )
        self.assertIsNone(result.resolved_values.evidence_weight)
        self.assertEqual(result.validation_warnings, ())

    def test_model_update_matches_cpp_core_and_preserves_metadata(self) -> None:
        model = PythonTestModel()
        core = belief_update.exact_binary_update(
            belief_update.BeliefState(0.4),
            model.infer().likelihoods,
        )
        result = belief_update.update(prior=0.4, likelihood_model=model)

        self.assert_matches_core(result, core)
        self.assertEqual(
            result.resolved_source,
            belief_update.ResolvedInputSource.likelihood_model,
        )
        self.assertEqual(result.provenance.source_id.value, "python-test-model")
        self.assertEqual(result.provenance.confidence, 0.9)
        self.assertEqual(result.validation_warnings, ())

    def test_invalid_scalar_input_raises_clear_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "probability must be finite"):
            belief_update.update(
                prior=0.4,
                p_e_given_h=math.nan,
                p_e_given_not_h=0.2,
            )

        with self.assertRaisesRegex(
            ValueError, "likelihood ratio must be finite"
        ):
            belief_update.update(
                prior=0.4,
                p_e_given_h=0.8,
                p_e_given_not_h=0.0,
            )


if __name__ == "__main__":
    unittest.main()
