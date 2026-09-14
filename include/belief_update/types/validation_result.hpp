#pragma once

#include <cstdint>

namespace belief_update {

class Validator;

enum class ValidationDecision : std::uint8_t {
  accept,
  warn,
  reject,
};

enum class ValidationCode : std::uint8_t {
  valid,
  probability_not_finite,
  probability_out_of_range,
  missing_required_input,
  empty_source_id,
  empty_source_version,
  malformed_source_type,
  malformed_override_status,
  likelihood_ratio_not_finite,
  likelihood_ratio_not_positive,
  freshness_violation,
  freshness_timestamp_in_future,
};

class ValidationResult final {
 public:
  ValidationResult(const ValidationResult&) = default;
  ValidationResult(ValidationResult&&) noexcept = default;
  ValidationResult& operator=(const ValidationResult&) = delete;
  ValidationResult& operator=(ValidationResult&&) = delete;

  [[nodiscard]] ValidationDecision decision() const noexcept {
    return decision_;
  }
  [[nodiscard]] ValidationCode code() const noexcept { return code_; }
  [[nodiscard]] bool can_proceed() const noexcept {
    return decision_ != ValidationDecision::reject;
  }
  [[nodiscard]] bool is_accepted() const noexcept {
    return decision_ == ValidationDecision::accept;
  }
  [[nodiscard]] bool has_warning() const noexcept {
    return decision_ == ValidationDecision::warn;
  }
  [[nodiscard]] bool is_rejected() const noexcept {
    return decision_ == ValidationDecision::reject;
  }

 private:
  friend class Validator;

  [[nodiscard]] static ValidationResult accepted() noexcept {
    return ValidationResult{ValidationDecision::accept, ValidationCode::valid};
  }

  [[nodiscard]] static ValidationResult warning(ValidationCode code) noexcept {
    return ValidationResult{ValidationDecision::warn, code};
  }

  [[nodiscard]] static ValidationResult rejected(ValidationCode code) noexcept {
    return ValidationResult{ValidationDecision::reject, code};
  }

  explicit ValidationResult(ValidationDecision decision,
                            ValidationCode code) noexcept
      : decision_(decision), code_(code) {}

  ValidationDecision decision_;
  ValidationCode code_;
};

}  // namespace belief_update
