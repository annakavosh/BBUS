#include "belief_update/analysis/threshold_analysis.hpp"

#include <limits>
#include <optional>
#include <utility>

#include "belief_update/types/belief_state.hpp"

namespace belief_update {
namespace {

[[nodiscard]] ThresholdPosition position(double value,
                                         double threshold) noexcept {
  if (value < threshold) {
    return ThresholdPosition::below;
  }
  if (value > threshold) {
    return ThresholdPosition::above;
  }
  return ThresholdPosition::at;
}

[[nodiscard]] ThresholdTransition transition(
    ThresholdPosition prior, ThresholdPosition posterior) noexcept {
  if ((prior == ThresholdPosition::below &&
       posterior != ThresholdPosition::below) ||
      (prior == ThresholdPosition::at &&
       posterior == ThresholdPosition::above)) {
    return ThresholdTransition::upward;
  }
  if ((prior == ThresholdPosition::above &&
       posterior != ThresholdPosition::above) ||
      (prior == ThresholdPosition::at &&
       posterior == ThresholdPosition::below)) {
    return ThresholdTransition::downward;
  }
  return ThresholdTransition::none;
}

[[nodiscard]] std::optional<double> threshold_ratio(double prior,
                                                    double threshold) {
  if (prior == 0.0 || prior == 1.0) {
    return std::nullopt;
  }
  if (threshold == 0.0) {
    return 0.0;
  }
  if (threshold == 1.0) {
    return std::numeric_limits<double>::infinity();
  }

  const long double numerator =
      static_cast<long double>(threshold) * (1.0L - prior);
  const long double denominator =
      (1.0L - threshold) * static_cast<long double>(prior);
  const long double ratio = numerator / denominator;
  if (ratio > std::numeric_limits<double>::max()) {
    return std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(ratio);
}

}  // namespace

ThresholdAnalysisResult::ThresholdAnalysisResult(
    double threshold, double prior, double posterior,
    double posterior_margin, ThresholdPosition prior_position,
    ThresholdPosition posterior_position, ThresholdTransition transition_value,
    double observed_likelihood_ratio, bool threshold_reachable,
    std::optional<double> likelihood_ratio_at_threshold)
    : threshold_(threshold),
      prior_(prior),
      posterior_(posterior),
      posterior_margin_(posterior_margin),
      prior_position_(prior_position),
      posterior_position_(posterior_position),
      transition_(transition_value),
      observed_likelihood_ratio_(observed_likelihood_ratio),
      threshold_reachable_(threshold_reachable),
      likelihood_ratio_at_threshold_(likelihood_ratio_at_threshold) {}

ThresholdAnalysisResult analyze_threshold(const UpdateResult& original,
                                          double threshold) {
  const double checked_threshold = BeliefState{threshold}.probability();
  const double prior = original.inputs().prior().probability();
  const double posterior = original.posterior();
  const ThresholdPosition prior_position = position(prior, checked_threshold);
  const ThresholdPosition posterior_position =
      position(posterior, checked_threshold);
  const bool threshold_reachable =
      ((prior > 0.0 && prior < 1.0) && checked_threshold > 0.0 &&
       checked_threshold < 1.0) ||
      prior == checked_threshold;

  return ThresholdAnalysisResult{
      checked_threshold,
      prior,
      posterior,
      posterior - checked_threshold,
      prior_position,
      posterior_position,
      transition(prior_position, posterior_position),
      original.intermediate_calculations().likelihood_ratio(),
      threshold_reachable,
      threshold_ratio(prior, checked_threshold)};
}

}  // namespace belief_update
