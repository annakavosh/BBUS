#pragma once

#include <utility>

#include "belief_update/types/likelihood_input.hpp"
#include "belief_update/types/provenance.hpp"
#include "belief_update/types/resolution_result.hpp"
#include "belief_update/validation/validator.hpp"

namespace belief_update {

class InputResolver final {
 public:
  explicit InputResolver(Validator validator)
      : validator_(std::move(validator)) {}

  InputResolver(const InputResolver&) = default;
  InputResolver(InputResolver&&) noexcept = default;
  InputResolver& operator=(const InputResolver&) = delete;
  InputResolver& operator=(InputResolver&&) = delete;

  // Candidate pointers are observed only for this call. A successful result
  // refers to the selected candidate, which must outlive the result.
  [[nodiscard]] ResolutionResult resolve(
      const LikelihoodInput* explicit_override,
      const LikelihoodInput* likelihood_model_value,
      ProvenanceTimestamp evaluation_time) const;

 private:
  Validator validator_;
};

}  // namespace belief_update
