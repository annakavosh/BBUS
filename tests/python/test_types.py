import unittest

import belief_update


class TypeBindingTest(unittest.TestCase):
    def make_provenance(self) -> belief_update.Provenance:
        return belief_update.Provenance(
            belief_update.SourceType.model,
            belief_update.SourceId("risk-model"),
            belief_update.SourceVersion("2026.09"),
            belief_update.ProvenanceTimestamp(1_789_310_400_000_000_000),
            0.95,
            belief_update.OverrideStatus.not_requested,
        )

    def test_typed_provenance_round_trip(self) -> None:
        provenance = self.make_provenance()

        self.assertEqual(provenance.source_type, belief_update.SourceType.model)
        self.assertEqual(provenance.source_id.value, "risk-model")
        self.assertEqual(provenance.version.value, "2026.09")
        self.assertEqual(
            provenance.timestamp.unix_nanoseconds, 1_789_310_400_000_000_000
        )
        self.assertEqual(provenance.confidence, 0.95)

    def test_calculation_result_is_read_only(self) -> None:
        result = belief_update.exact_binary_update(
            belief_update.BeliefState(0.4),
            belief_update.LikelihoodValues(0.8, 0.2),
        )

        self.assertAlmostEqual(result.posterior, 8.0 / 11.0)
        with self.assertRaises(AttributeError):
            result.posterior = 0.5

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            belief_update.BeliefState(float("nan"))

    def test_likelihood_output_envelope(self) -> None:
        output = belief_update.LikelihoodInput(
            belief_update.LikelihoodValues(0.75, 0.25),
            0.6,
            self.make_provenance(),
        )

        self.assertEqual(output.p_e_given_h, 0.75)
        self.assertEqual(output.p_e_given_not_h, 0.25)
        self.assertEqual(output.evidence_weight, 0.6)
        self.assertEqual(output.confidence, 0.95)
        self.assertEqual(
            output.provenance.source_type, belief_update.SourceType.model
        )


if __name__ == "__main__":
    unittest.main()
