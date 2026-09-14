#pragma once

#include <cstdint>
#include <utility>

#include "belief_update/types/likelihood_input.hpp"
#include "belief_update/types/provenance.hpp"
#include "belief_update/types/update_config.hpp"
#include "belief_update/types/validation_result.hpp"

namespace belief_update {

class Validator final {
 public:
  explicit Validator(UpdateConfig configuration)
      : configuration_(std::move(configuration)) {}

  Validator(const Validator&) = default;
  Validator(Validator&&) noexcept = default;
  Validator& operator=(const Validator&) = delete;
  Validator& operator=(Validator&&) = delete;

  [[nodiscard]] ValidationResult validate_probability(double value) const
      noexcept;
  [[nodiscard]] ValidationResult validate_likelihood_ratio(double value) const
      noexcept;
  [[nodiscard]] ValidationResult validate_likelihoods(
      double evidence_given_hypothesis,
      double evidence_given_not_hypothesis) const noexcept;
  [[nodiscard]] ValidationResult validate_provenance(
      const Provenance& provenance,
      ProvenanceTimestamp evaluation_time) const noexcept;
  [[nodiscard]] ValidationResult validate(
      const LikelihoodInput& input,
      ProvenanceTimestamp evaluation_time) const noexcept;
  [[nodiscard]] ValidationResult validate_required_input(
      const LikelihoodInput* input,
      ProvenanceTimestamp evaluation_time) const noexcept;

 private:
  [[nodiscard]] ValidationResult freshness_result(
      ValidationCode code) const noexcept;

  UpdateConfig configuration_;
};

}  // namespace belief_update
