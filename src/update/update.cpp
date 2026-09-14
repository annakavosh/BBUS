#include "belief_update/update/update.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "belief_update/core/bayes.hpp"
#include "belief_update/model/likelihood_model.hpp"
#include "belief_update/resolver/input_resolver.hpp"
#include "belief_update/version.hpp"

namespace belief_update {
namespace {

[[nodiscard]] const char* rejection_message(ValidationCode code) noexcept {
  switch (code) {
    case ValidationCode::valid:
      return "update rejected with an invalid validation state";
    case ValidationCode::probability_not_finite:
      return "update rejected: probability must be finite";
    case ValidationCode::probability_out_of_range:
      return "update rejected: probability must be in [0, 1]";
    case ValidationCode::missing_required_input:
      return "update rejected: a likelihood input is required";
    case ValidationCode::empty_source_id:
      return "update rejected: provenance source id must not be empty";
    case ValidationCode::empty_source_version:
      return "update rejected: provenance source version must not be empty";
    case ValidationCode::malformed_source_type:
      return "update rejected: provenance source type is invalid";
    case ValidationCode::malformed_override_status:
      return "update rejected: provenance override status is invalid";
    case ValidationCode::likelihood_ratio_not_finite:
      return "update rejected: likelihood ratio must be finite";
    case ValidationCode::likelihood_ratio_not_positive:
      return "update rejected: likelihood ratio must be positive";
    case ValidationCode::freshness_violation:
      return "update rejected: provenance freshness limit was exceeded";
    case ValidationCode::freshness_timestamp_in_future:
      return "update rejected: provenance timestamp is in the future";
  }
  return "update rejected: unknown validation code";
}

[[nodiscard]] Provenance direct_input_provenance() {
  return Provenance{SourceType::direct_input,
                    SourceId{"belief_update.direct"},
                    SourceVersion{std::string{version}},
                    ProvenanceTimestamp{0},
                    1.0,
                    OverrideStatus::applied};
}

}  // namespace

UpdateResult update(UpdateInputs inputs, UpdateConfig configuration,
                    UpdateAlgorithm algorithm) {
  const Validator validator{configuration};
  const InputResolver resolver{validator};

  const LikelihoodInput* explicit_override =
      inputs.explicit_override().has_value()
          ? &*inputs.explicit_override()
          : nullptr;
  const LikelihoodInput* model_value =
      inputs.likelihood_model_value().has_value()
          ? &*inputs.likelihood_model_value()
          : nullptr;
  const ResolutionResult resolution = resolver.resolve(
      explicit_override, model_value, inputs.evaluation_time());

  if (!resolution.validation().can_proceed()) {
    throw std::invalid_argument(
        rejection_message(resolution.validation().code()));
  }

  const ResolvedInput& resolved = resolution.resolved_input();
  const LikelihoodValues& likelihoods = resolved.input().likelihoods();
  std::optional<ValidationCode> warning;
  if (resolution.validation().has_warning()) {
    warning = resolution.validation().code();
  }

  BayesCalculation calculation = [&] {
    switch (algorithm) {
      case UpdateAlgorithm::exact_binary:
        return exact_binary_update(inputs.prior(), likelihoods);
      case UpdateAlgorithm::log_odds: {
        const double log_likelihood_ratio =
            std::log(likelihoods.evidence_given_hypothesis()) -
            std::log(likelihoods.evidence_given_not_hypothesis());
        return log_odds_update(inputs.prior(), log_likelihood_ratio);
      }
    }
    throw std::invalid_argument("unknown update algorithm");
  }();

  return UpdateResult{std::move(inputs),
                      resolved.source(),
                      std::move(configuration),
                      algorithm,
                      std::move(calculation),
                      warning,
                      std::string{version}};
}

UpdateResult update(double prior, const LikelihoodModel& likelihood_model) {
  LikelihoodInput model_value = likelihood_model.infer();
  const ProvenanceTimestamp evaluation_time =
      model_value.provenance().timestamp();
  return update(
      UpdateInputs{BeliefState{prior}, std::nullopt,
                   std::optional<LikelihoodInput>{std::move(model_value)},
                   evaluation_time},
      UpdateConfig{std::nullopt}, UpdateAlgorithm::exact_binary);
}

UpdateResult update(double prior, double p_e_given_h,
                    double p_e_given_not_h) {
  LikelihoodInput direct_input{
      LikelihoodValues{p_e_given_h, p_e_given_not_h}, std::nullopt,
      direct_input_provenance()};
  return update(
      UpdateInputs{BeliefState{prior},
                   std::optional<LikelihoodInput>{std::move(direct_input)},
                   std::nullopt, ProvenanceTimestamp{0}},
      UpdateConfig{std::nullopt}, UpdateAlgorithm::exact_binary);
}

}  // namespace belief_update
