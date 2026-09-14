#include <limits>
#include <optional>

#include "belief_update/validation/validator.hpp"
#include "test_support.hpp"

int main() {
  using namespace belief_update;

  const Validator validator{UpdateConfig{std::nullopt}};

  test_support::expect(
      validator
              .validate_required_input(nullptr, ProvenanceTimestamp{1'000})
              .code() == ValidationCode::missing_required_input,
      "missing required input is rejected by the validator");

  test_support::expect(validator.validate_probability(0.0).is_accepted(),
                       "zero probability is accepted");
  test_support::expect(validator.validate_probability(1.0).is_accepted(),
                       "unit probability is accepted");
  test_support::expect(validator.validate_probability(0.5).can_proceed(),
                       "accepted validation can proceed");
  test_support::expect(
      validator.validate_probability(-0.01).is_rejected(),
      "invalid probability reports rejection");
  test_support::expect(
      validator.validate_probability(-0.01).code() ==
          ValidationCode::probability_out_of_range,
      "negative probability is rejected");
  test_support::expect(
      validator.validate_probability(1.01).code() ==
          ValidationCode::probability_out_of_range,
      "probability above one is rejected");
  test_support::expect(
      validator
              .validate_probability(
                  std::numeric_limits<double>::quiet_NaN())
              .code() == ValidationCode::probability_not_finite,
      "NaN probability is rejected");
  test_support::expect(
      validator
              .validate_probability(std::numeric_limits<double>::infinity())
              .code() == ValidationCode::probability_not_finite,
      "infinite probability is rejected");

  test_support::expect(validator.validate_likelihood_ratio(2.0).is_accepted(),
                       "positive finite likelihood ratio is accepted");
  test_support::expect(
      validator.validate_likelihood_ratio(0.0).code() ==
          ValidationCode::likelihood_ratio_not_positive,
      "zero likelihood ratio is rejected");
  test_support::expect(
      validator.validate_likelihood_ratio(-1.0).code() ==
          ValidationCode::likelihood_ratio_not_positive,
      "negative likelihood ratio is rejected");
  test_support::expect(
      validator
              .validate_likelihood_ratio(
                  std::numeric_limits<double>::infinity())
              .code() == ValidationCode::likelihood_ratio_not_finite,
      "infinite likelihood ratio is rejected");

  test_support::expect(validator.validate_likelihoods(0.8, 0.2).is_accepted(),
                       "valid likelihood pair is accepted");
  test_support::expect(
      validator.validate_likelihoods(1.1, 0.2).code() ==
          ValidationCode::probability_out_of_range,
      "invalid likelihood probability is rejected before ratio evaluation");
  test_support::expect(
      validator.validate_likelihoods(0.8, 0.0).code() ==
          ValidationCode::likelihood_ratio_not_finite,
      "likelihood pair producing infinite ratio is rejected");

  return test_support::finish();
}
