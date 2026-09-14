#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include "belief_update/types/likelihood_input.hpp"
#include "belief_update/types/validation_result.hpp"

namespace belief_update {

class InputResolver;

enum class ResolvedInputSource : std::uint8_t {
  explicit_override,
  likelihood_model,
};

class ResolvedInput final {
 public:
  ResolvedInput(const ResolvedInput&) = default;
  ResolvedInput(ResolvedInput&&) noexcept = default;
  ResolvedInput& operator=(const ResolvedInput&) = delete;
  ResolvedInput& operator=(ResolvedInput&&) = delete;

  [[nodiscard]] const LikelihoodInput& input() const noexcept { return *input_; }
  [[nodiscard]] ResolvedInputSource source() const noexcept { return source_; }

 private:
  friend class InputResolver;

  // This is a non-owning resolver output. The selected candidate must outlive
  // the ResolutionResult that contains this view.
  explicit ResolvedInput(const LikelihoodInput& input,
                         ResolvedInputSource source) noexcept
      : input_(&input), source_(source) {}

  const LikelihoodInput* input_;
  ResolvedInputSource source_;
};

class ResolutionResult final {
 public:
  ResolutionResult(const ResolutionResult&) = default;
  ResolutionResult(ResolutionResult&&) noexcept = default;
  ResolutionResult& operator=(const ResolutionResult&) = delete;
  ResolutionResult& operator=(ResolutionResult&&) = delete;

  [[nodiscard]] const ValidationResult& validation() const noexcept {
    return validation_;
  }
  [[nodiscard]] bool has_input() const noexcept {
    return resolved_input_.has_value();
  }
  [[nodiscard]] const ResolvedInput& resolved_input() const {
    if (!resolved_input_.has_value()) {
      throw std::logic_error("rejected resolution has no resolved input");
    }
    return *resolved_input_;
  }

 private:
  friend class InputResolver;

  explicit ResolutionResult(ValidationResult validation,
                            std::optional<ResolvedInput> resolved_input)
      : validation_(std::move(validation)),
        resolved_input_(std::move(resolved_input)) {
    if (validation_.can_proceed() != resolved_input_.has_value()) {
      throw std::invalid_argument(
          "accepted resolutions require an input; rejected resolutions forbid one");
    }
  }

  ValidationResult validation_;
  std::optional<ResolvedInput> resolved_input_;
};

}  // namespace belief_update
