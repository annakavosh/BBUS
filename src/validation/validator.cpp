#include "belief_update/validation/validator.hpp"

#include <cmath>

namespace belief_update {
namespace {

[[nodiscard]] bool is_known(SourceType value) noexcept {
  switch (value) {
    case SourceType::direct_input:
    case SourceType::sensor:
    case SourceType::human:
    case SourceType::model:
    case SourceType::file:
    case SourceType::system:
      return true;
  }
  return false;
}

[[nodiscard]] bool is_known(OverrideStatus value) noexcept {
  switch (value) {
    case OverrideStatus::not_requested:
    case OverrideStatus::requested:
    case OverrideStatus::applied:
    case OverrideStatus::rejected:
      return true;
  }
  return false;
}

}  // namespace

ValidationResult Validator::validate_probability(double value) const noexcept {
  if (!std::isfinite(value)) {
    return ValidationResult::rejected(
        ValidationCode::probability_not_finite);
  }
  if (value < 0.0 || value > 1.0) {
    return ValidationResult::rejected(
        ValidationCode::probability_out_of_range);
  }
  return ValidationResult::accepted();
}

ValidationResult Validator::validate_likelihood_ratio(double value) const
    noexcept {
  if (!std::isfinite(value)) {
    return ValidationResult::rejected(
        ValidationCode::likelihood_ratio_not_finite);
  }
  if (value <= 0.0) {
    return ValidationResult::rejected(
        ValidationCode::likelihood_ratio_not_positive);
  }
  return ValidationResult::accepted();
}

ValidationResult Validator::validate_likelihoods(
    double evidence_given_hypothesis,
    double evidence_given_not_hypothesis) const noexcept {
  const ValidationResult hypothesis_result =
      validate_probability(evidence_given_hypothesis);
  if (!hypothesis_result.can_proceed()) {
    return hypothesis_result;
  }

  const ValidationResult not_hypothesis_result =
      validate_probability(evidence_given_not_hypothesis);
  if (!not_hypothesis_result.can_proceed()) {
    return not_hypothesis_result;
  }

  if (evidence_given_not_hypothesis == 0.0) {
    return ValidationResult::rejected(
        ValidationCode::likelihood_ratio_not_finite);
  }
  return validate_likelihood_ratio(evidence_given_hypothesis /
                                   evidence_given_not_hypothesis);
}

ValidationResult Validator::validate_provenance(
    const Provenance& provenance,
    ProvenanceTimestamp evaluation_time) const noexcept {
  if (!is_known(provenance.source_type())) {
    return ValidationResult::rejected(
        ValidationCode::malformed_source_type);
  }
  if (provenance.source_id().value().empty()) {
    return ValidationResult::rejected(ValidationCode::empty_source_id);
  }
  if (provenance.version().value().empty()) {
    return ValidationResult::rejected(ValidationCode::empty_source_version);
  }
  if (!is_known(provenance.override_status())) {
    return ValidationResult::rejected(
        ValidationCode::malformed_override_status);
  }

  const ValidationResult confidence =
      validate_probability(provenance.confidence());
  if (!confidence.can_proceed()) {
    return confidence;
  }

  if (!configuration_.freshness_rule().has_value()) {
    return ValidationResult::accepted();
  }

  const std::int64_t observed =
      provenance.timestamp().unix_nanoseconds();
  const std::int64_t evaluated = evaluation_time.unix_nanoseconds();
  if (observed > evaluated) {
    return freshness_result(ValidationCode::freshness_timestamp_in_future);
  }

  const std::uint64_t age = static_cast<std::uint64_t>(evaluated) -
                            static_cast<std::uint64_t>(observed);
  if (age > configuration_.freshness_rule()->max_age_nanoseconds()) {
    return freshness_result(ValidationCode::freshness_violation);
  }

  return ValidationResult::accepted();
}

ValidationResult Validator::validate(
    const LikelihoodInput& input,
    ProvenanceTimestamp evaluation_time) const noexcept {
  const LikelihoodValues& likelihoods = input.likelihoods();
  const ValidationResult likelihood_result = validate_likelihoods(
      likelihoods.evidence_given_hypothesis(),
      likelihoods.evidence_given_not_hypothesis());
  if (!likelihood_result.can_proceed()) {
    return likelihood_result;
  }

  if (input.evidence_weight().has_value()) {
    const ValidationResult weight_result =
        validate_probability(*input.evidence_weight());
    if (!weight_result.can_proceed()) {
      return weight_result;
    }
  }
  return validate_provenance(input.provenance(), evaluation_time);
}

ValidationResult Validator::validate_required_input(
    const LikelihoodInput* input,
    ProvenanceTimestamp evaluation_time) const noexcept {
  if (input == nullptr) {
    return ValidationResult::rejected(
        ValidationCode::missing_required_input);
  }
  return validate(*input, evaluation_time);
}

ValidationResult Validator::freshness_result(ValidationCode code) const
    noexcept {
  if (configuration_.freshness_rule()->violation_decision() ==
      ValidationDecision::warn) {
    return ValidationResult::warning(code);
  }
  return ValidationResult::rejected(code);
}

}  // namespace belief_update
