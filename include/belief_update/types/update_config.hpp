#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include "belief_update/types/validation_result.hpp"

namespace belief_update {

class FreshnessRule final {
 public:
  explicit FreshnessRule(std::uint64_t max_age_nanoseconds,
                         ValidationDecision violation_decision)
      : max_age_nanoseconds_(max_age_nanoseconds),
        violation_decision_(violation_decision) {
    if (violation_decision_ == ValidationDecision::accept) {
      throw std::invalid_argument(
          "a freshness violation must warn or reject");
    }
  }

  FreshnessRule(const FreshnessRule&) = default;
  FreshnessRule(FreshnessRule&&) noexcept = default;
  FreshnessRule& operator=(const FreshnessRule&) = delete;
  FreshnessRule& operator=(FreshnessRule&&) = delete;

  [[nodiscard]] std::uint64_t max_age_nanoseconds() const noexcept {
    return max_age_nanoseconds_;
  }
  [[nodiscard]] ValidationDecision violation_decision() const noexcept {
    return violation_decision_;
  }

 private:
  std::uint64_t max_age_nanoseconds_;
  ValidationDecision violation_decision_;
};

class UpdateConfig final {
 public:
  explicit UpdateConfig(std::optional<FreshnessRule> freshness_rule)
      : freshness_rule_(std::move(freshness_rule)) {}

  UpdateConfig(const UpdateConfig&) = default;
  UpdateConfig(UpdateConfig&&) noexcept = default;
  UpdateConfig& operator=(const UpdateConfig&) = delete;
  UpdateConfig& operator=(UpdateConfig&&) = delete;

  [[nodiscard]] const std::optional<FreshnessRule>& freshness_rule()
      const noexcept {
    return freshness_rule_;
  }

 private:
  std::optional<FreshnessRule> freshness_rule_;
};

}  // namespace belief_update
