#pragma once

#include <cstdint>
#include <optional>

#include "belief_update/types/update_result.hpp"

namespace belief_update {

enum class ThresholdPosition : std::uint8_t {
  below,
  at,
  above,
};

enum class ThresholdTransition : std::uint8_t {
  none,
  upward,
  downward,
};

class ThresholdAnalysisResult final {
 public:
  ThresholdAnalysisResult(const ThresholdAnalysisResult&) = default;
  ThresholdAnalysisResult(ThresholdAnalysisResult&&) noexcept = default;
  ThresholdAnalysisResult& operator=(const ThresholdAnalysisResult&) = delete;
  ThresholdAnalysisResult& operator=(ThresholdAnalysisResult&&) = delete;

  [[nodiscard]] double threshold() const noexcept { return threshold_; }
  [[nodiscard]] double prior() const noexcept { return prior_; }
  [[nodiscard]] double posterior() const noexcept { return posterior_; }
  [[nodiscard]] double posterior_margin() const noexcept {
    return posterior_margin_;
  }
  [[nodiscard]] ThresholdPosition prior_position() const noexcept {
    return prior_position_;
  }
  [[nodiscard]] ThresholdPosition posterior_position() const noexcept {
    return posterior_position_;
  }
  [[nodiscard]] ThresholdTransition transition() const noexcept {
    return transition_;
  }
  [[nodiscard]] bool threshold_met() const noexcept {
    return posterior_ >= threshold_;
  }
  [[nodiscard]] double observed_likelihood_ratio() const noexcept {
    return observed_likelihood_ratio_;
  }
  [[nodiscard]] bool threshold_reachable() const noexcept {
    return threshold_reachable_;
  }
  // Empty when a certain prior makes the ratio non-unique or unreachable.
  [[nodiscard]] const std::optional<double>&
  likelihood_ratio_at_threshold() const noexcept {
    return likelihood_ratio_at_threshold_;
  }

 private:
  friend ThresholdAnalysisResult analyze_threshold(const UpdateResult&,
                                                    double);

  explicit ThresholdAnalysisResult(
      double threshold, double prior, double posterior,
      double posterior_margin, ThresholdPosition prior_position,
      ThresholdPosition posterior_position, ThresholdTransition transition,
      double observed_likelihood_ratio, bool threshold_reachable,
      std::optional<double> likelihood_ratio_at_threshold);

  double threshold_;
  double prior_;
  double posterior_;
  double posterior_margin_;
  ThresholdPosition prior_position_;
  ThresholdPosition posterior_position_;
  ThresholdTransition transition_;
  double observed_likelihood_ratio_;
  bool threshold_reachable_;
  std::optional<double> likelihood_ratio_at_threshold_;
};

[[nodiscard]] ThresholdAnalysisResult analyze_threshold(
    const UpdateResult& original, double threshold);

}  // namespace belief_update
