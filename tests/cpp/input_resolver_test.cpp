#include <optional>
#include <type_traits>
#include <stdexcept>

#include "belief_update/resolver/input_resolver.hpp"
#include "test_support.hpp"

namespace {

belief_update::LikelihoodInput make_input(
    double given_hypothesis, double given_not_hypothesis,
    belief_update::OverrideStatus override_status,
    std::int64_t timestamp = 1'000) {
  using namespace belief_update;
  return LikelihoodInput{
      LikelihoodValues{given_hypothesis, given_not_hypothesis},
      std::nullopt,
      Provenance{SourceType::model,
                 SourceId{"model-a"},
                 SourceVersion{"3"},
                 ProvenanceTimestamp{timestamp},
                 0.9,
                 override_status}};
}

}  // namespace

int main() {
  using namespace belief_update;

  static_assert(!std::is_constructible_v<
                ResolvedInput, const LikelihoodInput&, ResolvedInputSource>);
  static_assert(!std::is_constructible_v<
                ResolutionResult, ValidationResult,
                std::optional<ResolvedInput>>);

  const LikelihoodInput explicit_override =
      make_input(0.9, 0.1, OverrideStatus::requested);
  const LikelihoodInput model_value =
      make_input(0.7, 0.3, OverrideStatus::not_requested);
  const InputResolver resolver{Validator{UpdateConfig{std::nullopt}}};
  const ProvenanceTimestamp evaluated{1'100};

  const ResolutionResult overridden =
      resolver.resolve(&explicit_override, &model_value, evaluated);
  test_support::expect(overridden.validation().is_accepted(),
                       "valid override resolution is accepted");
  test_support::expect(
      overridden.resolved_input().source() ==
          ResolvedInputSource::explicit_override,
      "explicit override has first precedence");
  test_support::expect(
      &overridden.resolved_input().input() == &explicit_override,
      "resolver refers to the exact override without modifying it");
  test_support::expect(
      explicit_override.likelihoods().evidence_given_hypothesis() == 0.9 &&
          explicit_override.provenance().override_status() ==
              OverrideStatus::requested,
      "resolution leaves override values and provenance unchanged");

  const ResolutionResult modeled =
      resolver.resolve(nullptr, &model_value, evaluated);
  test_support::expect(
      modeled.resolved_input().source() ==
          ResolvedInputSource::likelihood_model,
      "validated model value is used when override is absent");
  test_support::expect(&modeled.resolved_input().input() == &model_value,
                       "resolver does not copy or substitute model input");

  const ResolutionResult missing =
      resolver.resolve(nullptr, nullptr, evaluated);
  test_support::expect(
      !missing.has_input() &&
          missing.validation().code() ==
              ValidationCode::missing_required_input,
      "missing candidates produce a typed error");
  test_support::expect_throws<std::logic_error>(
      [&missing] { static_cast<void>(missing.resolved_input()); },
      "rejected resolution exposes no guessed value");

  const LikelihoodInput invalid_override =
      make_input(0.8, 0.0, OverrideStatus::requested);
  const ResolutionResult invalid =
      resolver.resolve(&invalid_override, &model_value, evaluated);
  test_support::expect(
      !invalid.has_input() &&
          invalid.validation().code() ==
              ValidationCode::likelihood_ratio_not_finite,
      "invalid explicit override rejects instead of falling back to model");

  const InputResolver warning_resolver{
      Validator{UpdateConfig{
          FreshnessRule{50, ValidationDecision::warn}}}};
  const ResolutionResult warned =
      warning_resolver.resolve(&explicit_override, &model_value, evaluated);
  test_support::expect(
      warned.validation().has_warning() && warned.has_input() &&
          warned.resolved_input().source() ==
              ResolvedInputSource::explicit_override &&
          warned.resolved_input()
                  .input()
                  .likelihoods()
                  .evidence_given_hypothesis() == 0.9 &&
          warned.resolved_input().input().provenance().confidence() == 0.9,
      "warning never down-weights likelihood or confidence");

  return test_support::finish();
}
