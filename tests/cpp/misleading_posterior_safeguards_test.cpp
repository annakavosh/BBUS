#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "belief_update/core/bayes.hpp"
#include "belief_update/update/update.hpp"
#include "test_support.hpp"

namespace {

belief_update::LikelihoodInput make_input(
    double p_e_given_h, double p_e_given_not_h, double confidence,
    belief_update::OverrideStatus override_status, const char* source_id,
    std::int64_t timestamp = 1'000) {
  using namespace belief_update;
  return LikelihoodInput{
      LikelihoodValues{p_e_given_h, p_e_given_not_h}, std::nullopt,
      Provenance{SourceType::model,
                 SourceId{source_id},
                 SourceVersion{"1"},
                 ProvenanceTimestamp{timestamp},
                 confidence,
                 override_status}};
}

belief_update::UpdateResult update_model_only(
    belief_update::LikelihoodInput input,
    belief_update::UpdateConfig configuration,
    std::int64_t evaluation_time = 1'000) {
  using namespace belief_update;
  return update(UpdateInputs{BeliefState{0.4}, std::nullopt,
                             std::move(input),
                             ProvenanceTimestamp{evaluation_time}},
                std::move(configuration), UpdateAlgorithm::exact_binary);
}

}  // namespace

int main() {
  using namespace belief_update;

  static_assert(!std::is_default_constructible_v<Provenance>);
  static_assert(!std::is_default_constructible_v<LikelihoodInput>);

  const LikelihoodInput valid_model = make_input(
      0.8, 0.2, 1.0, OverrideStatus::not_requested, "valid-model");
  const UpdateResult recorded_model =
      update_model_only(valid_model, UpdateConfig{std::nullopt});
  test_support::expect(recorded_model.provenance().source_id().value() ==
                           "valid-model",
                       "numerically valid model assumptions are recorded");

  test_support::expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(update_model_only(
            make_input(0.8, 0.0, 1.0, OverrideStatus::not_requested,
                       "invalid-model"),
            UpdateConfig{std::nullopt}));
      },
      "non-finite model likelihood ratio is rejected");

  const LikelihoodInput stale_model = make_input(
      0.8, 0.2, 1.0, OverrideStatus::not_requested, "stale-model", 1'000);
  const UpdateResult stale_warning = update_model_only(
      stale_model,
      UpdateConfig{FreshnessRule{100, ValidationDecision::warn}}, 2'000);
  test_support::expect(
      stale_warning.validation_warnings().size() == 1 &&
          stale_warning.validation_warnings().front() ==
              ValidationCode::freshness_violation,
      "configured stale model violation warns");
  test_support::expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(update_model_only(
            stale_model,
            UpdateConfig{
                FreshnessRule{100, ValidationDecision::reject}},
            2'000));
      },
      "configured stale model violation rejects");

  const UpdateResult stale_recorded =
      update_model_only(stale_model, UpdateConfig{std::nullopt}, 2'000);
  test_support::expect(stale_recorded.validation_warnings().empty() &&
                           stale_recorded.provenance()
                                   .timestamp()
                                   .unix_nanoseconds() == 1'000,
                       "staleness is recorded without an implicit policy");

  const LikelihoodInput explicit_value = make_input(
      0.6, 0.4, 1.0, OverrideStatus::requested, "explicit-value");
  const UpdateResult override_result = update(
      UpdateInputs{BeliefState{0.4}, explicit_value, valid_model,
                   ProvenanceTimestamp{1'000}},
      UpdateConfig{std::nullopt}, UpdateAlgorithm::exact_binary);
  test_support::expect(
      override_result.resolved_source() ==
              ResolvedInputSource::explicit_override &&
          override_result.inputs().likelihood_model_value().has_value(),
      "valid override wins while the model candidate remains recorded");

  const LikelihoodInput invalid_override = make_input(
      0.6, 0.0, 1.0, OverrideStatus::requested, "invalid-override");
  test_support::expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(update(
            UpdateInputs{BeliefState{0.4}, invalid_override, valid_model,
                         ProvenanceTimestamp{1'000}},
            UpdateConfig{std::nullopt}, UpdateAlgorithm::exact_binary));
      },
      "invalid override rejects instead of falling back");

  const LikelihoodInput low_confidence = make_input(
      0.8, 0.2, 0.01, OverrideStatus::not_requested, "low-confidence");
  const UpdateResult low_confidence_result =
      update_model_only(low_confidence, UpdateConfig{std::nullopt});
  const BayesCalculation unweighted =
      exact_binary_update(BeliefState{0.4}, low_confidence.likelihoods());
  test_support::expect(low_confidence_result.provenance().confidence() == 0.01,
                       "low confidence is recorded");
  test_support::expect(low_confidence_result.posterior() ==
                           unweighted.posterior() &&
                           low_confidence_result.validation_warnings().empty(),
                       "confidence never silently changes the posterior");

  test_support::expect_throws<std::invalid_argument>(
      [] { static_cast<void>(SourceId{""}); },
      "missing provenance source id is rejected");

  const LikelihoodInput finite_extreme = make_input(
      1.0, 1e-300, 1.0, OverrideStatus::not_requested, "extreme-model");
  const UpdateResult extreme_result =
      update_model_only(finite_extreme, UpdateConfig{std::nullopt});
  test_support::expect(
      std::abs(extreme_result.intermediate_calculations().likelihood_ratio() /
                   1e300 -
               1.0) <= 1e-15,
      "finite extreme likelihood ratio is recorded unchanged");

  const LikelihoodInput repeated_source = make_input(
      0.8, 0.2, 1.0, OverrideStatus::not_requested, "repeated-source");
  const UpdateResult first =
      update_model_only(repeated_source, UpdateConfig{std::nullopt});
  const UpdateResult second = update(
      UpdateInputs{BeliefState{first.posterior()}, std::nullopt,
                   repeated_source, ProvenanceTimestamp{1'000}},
      UpdateConfig{std::nullopt}, UpdateAlgorithm::exact_binary);
  test_support::expect(
      first.provenance().source_id().value() ==
              second.provenance().source_id().value() &&
          second.validation_warnings().empty(),
      "repeated-source dependence is recorded without being guessed");

  return test_support::finish();
}
