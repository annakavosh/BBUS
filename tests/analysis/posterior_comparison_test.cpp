#include <optional>
#include <stdexcept>

#include "belief_update/analysis/posterior_comparison.hpp"
#include "belief_update/update/update.hpp"
#include "test_support.hpp"

namespace {

belief_update::LikelihoodInput make_input(
    double p_e_given_h, double p_e_given_not_h,
    belief_update::OverrideStatus override_status, const char* source_id) {
  using namespace belief_update;
  return LikelihoodInput{
      LikelihoodValues{p_e_given_h, p_e_given_not_h}, std::nullopt,
      Provenance{SourceType::model,
                 SourceId{source_id},
                 SourceVersion{"1"},
                 ProvenanceTimestamp{1'000},
                 1.0,
                 override_status}};
}

}  // namespace

int main() {
  using namespace belief_update;

  const LikelihoodInput explicit_value =
      make_input(0.8, 0.3, OverrideStatus::requested, "explicit-value");
  const LikelihoodInput model_value =
      make_input(0.8, 0.2, OverrideStatus::not_requested, "model-value");
  const UpdateResult original = update(
      UpdateInputs{BeliefState{0.4}, explicit_value, model_value,
                   ProvenanceTimestamp{1'000}},
      UpdateConfig{std::nullopt}, UpdateAlgorithm::exact_binary);
  const double original_posterior = original.posterior();

  const PosteriorComparisonResult comparison =
      compare_model_and_explicit(original);
  test_support::expect_near(comparison.model_posterior(), 8.0 / 11.0, 1e-15,
                            "model posterior is calculated independently");
  test_support::expect_near(comparison.explicit_posterior(), 0.64, 1e-15,
                            "explicit posterior is calculated independently");
  test_support::expect_near(comparison.difference(),
                            0.64 - (8.0 / 11.0), 1e-15,
                            "difference is explicit minus model");
  test_support::expect(
      comparison.model_result().resolved_source() ==
          ResolvedInputSource::likelihood_model,
      "model comparison result records model resolution");
  test_support::expect(
      comparison.explicit_result().resolved_source() ==
          ResolvedInputSource::explicit_override,
      "explicit comparison result records explicit resolution");
  test_support::expect(original.posterior() == original_posterior,
                       "comparison does not modify the original result");

  test_support::expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(compare_model_and_explicit(
            update(0.4, 0.8, 0.3)));
      },
      "comparison requires a model candidate");

  const UpdateResult model_only = update(
      UpdateInputs{BeliefState{0.4}, std::nullopt, model_value,
                   ProvenanceTimestamp{1'000}},
      UpdateConfig{std::nullopt}, UpdateAlgorithm::exact_binary);
  test_support::expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(compare_model_and_explicit(model_only)); },
      "comparison requires an explicit candidate");

  const LikelihoodInput invalid_unselected_model =
      make_input(0.8, 0.0, OverrideStatus::not_requested, "invalid-model");
  const UpdateResult valid_original = update(
      UpdateInputs{BeliefState{0.4}, explicit_value, invalid_unselected_model,
                   ProvenanceTimestamp{1'000}},
      UpdateConfig{std::nullopt}, UpdateAlgorithm::exact_binary);
  test_support::expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(compare_model_and_explicit(valid_original));
      },
      "comparison validates the previously unselected model candidate");

  return test_support::finish();
}
