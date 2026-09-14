#pragma once

#include <optional>
#include <utility>

#include "belief_update/types/belief_state.hpp"
#include "belief_update/types/likelihood_input.hpp"
#include "belief_update/types/provenance.hpp"

namespace belief_update {

class UpdateInputs final {
 public:
  explicit UpdateInputs(
      BeliefState prior,
      std::optional<LikelihoodInput> explicit_override,
      std::optional<LikelihoodInput> likelihood_model_value,
      ProvenanceTimestamp evaluation_time)
      : prior_(std::move(prior)),
        explicit_override_(std::move(explicit_override)),
        likelihood_model_value_(std::move(likelihood_model_value)),
        evaluation_time_(std::move(evaluation_time)) {}

  UpdateInputs(const UpdateInputs&) = default;
  UpdateInputs(UpdateInputs&&) noexcept = default;
  UpdateInputs& operator=(const UpdateInputs&) = delete;
  UpdateInputs& operator=(UpdateInputs&&) = delete;

  [[nodiscard]] const BeliefState& prior() const noexcept { return prior_; }
  [[nodiscard]] const std::optional<LikelihoodInput>& explicit_override()
      const noexcept {
    return explicit_override_;
  }
  [[nodiscard]] const std::optional<LikelihoodInput>& likelihood_model_value()
      const noexcept {
    return likelihood_model_value_;
  }
  [[nodiscard]] const ProvenanceTimestamp& evaluation_time() const noexcept {
    return evaluation_time_;
  }

 private:
  BeliefState prior_;
  std::optional<LikelihoodInput> explicit_override_;
  std::optional<LikelihoodInput> likelihood_model_value_;
  ProvenanceTimestamp evaluation_time_;
};

}  // namespace belief_update
