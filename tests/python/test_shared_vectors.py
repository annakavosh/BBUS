import csv
import math
from pathlib import Path
import sys
import unittest

import belief_update

TEST_DIRECTORY = Path(__file__).resolve().parent
if str(TEST_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TEST_DIRECTORY))

from reference_bayes import (
    ReferenceCalculation,
    exact_binary_reference,
    log_likelihood_ratio,
    log_odds_reference,
)


VECTOR_FILE = TEST_DIRECTORY.parent / "vectors" / "bayes_vectors.csv"
FIELDS = (
    "prior",
    "likelihood_ratio",
    "prior_odds",
    "posterior_odds",
    "posterior",
    "belief_delta",
)


def load_vectors() -> list[dict[str, str]]:
    with VECTOR_FILE.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def binding_exact(vector: dict[str, str]) -> belief_update.BayesCalculation:
    return belief_update.exact_binary_update(
        belief_update.BeliefState(float(vector["prior"])),
        belief_update.LikelihoodValues(
            float(vector["p_e_given_h"]),
            float(vector["p_e_given_not_h"]),
        ),
    )


def binding_log_odds(
    vector: dict[str, str],
) -> belief_update.BayesCalculation:
    prior = belief_update.BeliefState(float(vector["prior"]))
    likelihoods = belief_update.LikelihoodValues(
        float(vector["p_e_given_h"]),
        float(vector["p_e_given_not_h"]),
    )
    p_e_given_h = likelihoods.evidence_given_hypothesis
    p_e_given_not_h = likelihoods.evidence_given_not_hypothesis
    log_h = -math.inf if p_e_given_h == 0.0 else math.log(p_e_given_h)
    log_not_h = (
        -math.inf
        if p_e_given_not_h == 0.0
        else math.log(p_e_given_not_h)
    )
    return belief_update.log_odds_update(prior, log_h - log_not_h)


def reference_exact(vector: dict[str, str]) -> ReferenceCalculation:
    return exact_binary_reference(
        float(vector["prior"]),
        float(vector["p_e_given_h"]),
        float(vector["p_e_given_not_h"]),
    )


def reference_log_odds(vector: dict[str, str]) -> ReferenceCalculation:
    p_e_given_h = float(vector["p_e_given_h"])
    p_e_given_not_h = float(vector["p_e_given_not_h"])
    return log_odds_reference(
        float(vector["prior"]),
        log_likelihood_ratio(p_e_given_h, p_e_given_not_h),
    )


class SharedVectorTest(unittest.TestCase):
    def assert_value_close(
        self,
        actual: float,
        expected: float,
        absolute_tolerance: float,
        relative_tolerance: float,
        message: str,
    ) -> None:
        if math.isinf(expected):
            self.assertTrue(
                math.isinf(actual)
                and math.copysign(1.0, actual) == math.copysign(1.0, expected),
                message,
            )
            return
        self.assertTrue(
            math.isclose(
                actual,
                expected,
                abs_tol=absolute_tolerance,
                rel_tol=relative_tolerance,
            ),
            message,
        )

    def assert_matches_vector(
        self,
        calculation: object,
        vector: dict[str, str],
        implementation: str,
    ) -> None:
        expected = {
            "prior": float(vector["prior"]),
            "likelihood_ratio": float(vector["expected_likelihood_ratio"]),
            "prior_odds": float(vector["expected_prior_odds"]),
            "posterior_odds": float(vector["expected_posterior_odds"]),
            "posterior": float(vector["expected_posterior"]),
            "belief_delta": float(vector["expected_belief_delta"]),
        }
        absolute_tolerance = float(vector["absolute_tolerance"])
        relative_tolerance = float(vector["relative_tolerance"])
        for field in FIELDS:
            self.assert_value_close(
                float(getattr(calculation, field)),
                expected[field],
                absolute_tolerance,
                relative_tolerance,
                f"{vector['case']} {implementation} {field}",
            )

    def assert_equivalent(
        self,
        left: object,
        right: object,
        vector: dict[str, str],
        message: str,
    ) -> None:
        absolute_tolerance = float(vector["absolute_tolerance"])
        relative_tolerance = float(vector["relative_tolerance"])
        for field in FIELDS:
            self.assert_value_close(
                float(getattr(left, field)),
                float(getattr(right, field)),
                absolute_tolerance,
                relative_tolerance,
                f"{vector['case']} {message} {field}",
            )

    def test_valid_vectors_match_all_implementations(self) -> None:
        for vector in load_vectors():
            if vector["expectation"] != "valid":
                continue
            with self.subTest(case=vector["case"]):
                cpp_exact = binding_exact(vector)
                cpp_log = binding_log_odds(vector)
                python_exact = reference_exact(vector)
                python_log = reference_log_odds(vector)
                implementations = (
                    (cpp_exact, "C++ binding exact"),
                    (cpp_log, "C++ binding log odds"),
                    (python_exact, "pure Python exact"),
                    (python_log, "pure Python log odds"),
                )
                for calculation, name in implementations:
                    self.assert_matches_vector(calculation, vector, name)
                self.assert_equivalent(
                    cpp_exact, python_exact, vector, "exact equivalence"
                )
                self.assert_equivalent(
                    cpp_log, python_log, vector, "log-odds equivalence"
                )

    def test_invalid_vectors_are_rejected_everywhere(self) -> None:
        implementations = (
            binding_exact,
            binding_log_odds,
            reference_exact,
            reference_log_odds,
        )
        for vector in load_vectors():
            if vector["expectation"] != "invalid":
                continue
            for implementation in implementations:
                with self.subTest(
                    case=vector["case"], implementation=implementation.__name__
                ):
                    with self.assertRaises((ValueError, ArithmeticError)):
                        implementation(vector)


if __name__ == "__main__":
    unittest.main()
