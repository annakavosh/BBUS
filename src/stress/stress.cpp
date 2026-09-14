#include "belief_update/stress/stress.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

#include "belief_update/core/bayes.hpp"

namespace belief_update {
namespace {

[[nodiscard]] double checked_percentage(double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("stress percentage must be finite");
  }
  if (value < 0.0) {
    throw std::invalid_argument("stress percentage must be non-negative");
  }
  return value;
}

[[nodiscard]] ProbabilityBounds relative_bounds(
    double baseline, double percentage, const char* range_error) {
  const double fraction = percentage / 100.0;
  const double lower = baseline * (1.0 - fraction);
  const double upper = baseline * (1.0 + fraction);
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower < 0.0 ||
      upper > 1.0) {
    throw std::invalid_argument(range_error);
  }
  return ProbabilityBounds{lower, upper};
}

[[nodiscard]] ThresholdCrossing find_threshold_crossing(
    double original, double minimum, double maximum,
    const std::optional<double>& threshold) noexcept {
  if (!threshold.has_value()) {
    return ThresholdCrossing::not_evaluated;
  }

  const bool upward =
      (original < *threshold && maximum >= *threshold) ||
      (original == *threshold && maximum > *threshold);
  const bool downward =
      (original > *threshold && minimum <= *threshold) ||
      (original == *threshold && minimum < *threshold);
  if (upward && downward) {
    return ThresholdCrossing::both;
  }
  if (upward) {
    return ThresholdCrossing::upward;
  }
  if (downward) {
    return ThresholdCrossing::downward;
  }
  return ThresholdCrossing::none;
}

}  // namespace

ProbabilityBounds::ProbabilityBounds(double lower, double upper)
    : lower_(BeliefState{lower}.probability()),
      upper_(BeliefState{upper}.probability()) {
  if (lower_ > upper_) {
    throw std::invalid_argument(
        "probability bounds require lower <= upper");
  }
}

RelativeStressRanges::RelativeStressRanges(
    double prior_percent, double p_e_given_h_percent,
    double p_e_given_not_h_percent)
    : prior_percent_(checked_percentage(prior_percent)),
      p_e_given_h_percent_(checked_percentage(p_e_given_h_percent)),
      p_e_given_not_h_percent_(
          checked_percentage(p_e_given_not_h_percent)) {}

AbsoluteStressRanges::AbsoluteStressRanges(
    ProbabilityBounds prior, ProbabilityBounds p_e_given_h,
    ProbabilityBounds p_e_given_not_h)
    : prior_(std::move(prior)),
      p_e_given_h_(std::move(p_e_given_h)),
      p_e_given_not_h_(std::move(p_e_given_not_h)) {}

StressScenario::StressScenario(BeliefState prior,
                               LikelihoodValues likelihoods,
                               BayesCalculation calculation)
    : prior_(std::move(prior)),
      likelihoods_(std::move(likelihoods)),
      calculation_(std::move(calculation)) {}

StressResult::StressResult(
    StressRangeMode range_mode,
    std::optional<RelativeStressRanges> requested_relative_ranges,
    AbsoluteStressRanges resolved_bounds, double original_posterior,
    double original_belief_delta, StressScenario minimum_posterior_scenario,
    StressScenario maximum_posterior_scenario,
    StressScenario over_update_scenario,
    StressScenario under_update_scenario, double over_update_amount,
    double under_update_amount, std::optional<double> threshold,
    ThresholdCrossing threshold_crossing)
    : range_mode_(range_mode),
      requested_relative_ranges_(std::move(requested_relative_ranges)),
      resolved_bounds_(std::move(resolved_bounds)),
      original_posterior_(original_posterior),
      original_belief_delta_(original_belief_delta),
      minimum_posterior_scenario_(
          std::move(minimum_posterior_scenario)),
      maximum_posterior_scenario_(
          std::move(maximum_posterior_scenario)),
      over_update_scenario_(std::move(over_update_scenario)),
      under_update_scenario_(std::move(under_update_scenario)),
      over_update_amount_(over_update_amount),
      under_update_amount_(under_update_amount),
      threshold_(threshold),
      threshold_crossing_(threshold_crossing) {}

namespace detail {

class StressResultFactory final {
 public:
  [[nodiscard]] static StressScenario scenario(
      double prior, double p_e_given_h, double p_e_given_not_h) {
    BeliefState belief{prior};
    LikelihoodValues likelihoods{p_e_given_h, p_e_given_not_h};
    BayesCalculation calculation =
        exact_binary_update(belief, likelihoods);
    return StressScenario{std::move(belief), std::move(likelihoods),
                          std::move(calculation)};
  }

  [[nodiscard]] static StressResult result(
      StressRangeMode range_mode,
      std::optional<RelativeStressRanges> requested_relative_ranges,
      AbsoluteStressRanges resolved_bounds, double original_posterior,
      double original_belief_delta,
      const StressScenario& minimum_posterior_scenario,
      const StressScenario& maximum_posterior_scenario,
      const StressScenario& over_update_scenario,
      const StressScenario& under_update_scenario, double over_update_amount,
      double under_update_amount, std::optional<double> threshold,
      ThresholdCrossing threshold_crossing) {
    return StressResult{
        range_mode,
        std::move(requested_relative_ranges),
        std::move(resolved_bounds),
        original_posterior,
        original_belief_delta,
        minimum_posterior_scenario,
        maximum_posterior_scenario,
        over_update_scenario,
        under_update_scenario,
        over_update_amount,
        under_update_amount,
        threshold,
        threshold_crossing};
  }
};

}  // namespace detail

namespace {

[[nodiscard]] StressResult analyze(
    const UpdateResult& original, const AbsoluteStressRanges& bounds,
    StressRangeMode range_mode,
    std::optional<RelativeStressRanges> requested_relative_ranges,
    std::optional<double> threshold) {
  if (threshold.has_value()) {
    threshold = BeliefState{*threshold}.probability();
  }

  const std::array<double, 2> priors{bounds.prior().lower(),
                                      bounds.prior().upper()};
  const std::array<double, 2> likelihoods_h{
      bounds.p_e_given_h().lower(), bounds.p_e_given_h().upper()};
  const std::array<double, 2> likelihoods_not_h{
      bounds.p_e_given_not_h().lower(),
      bounds.p_e_given_not_h().upper()};
  std::array<std::optional<StressScenario>, 8> scenarios;
  std::size_t index = 0;
  for (const double prior : priors) {
    for (const double p_e_given_h : likelihoods_h) {
      for (const double p_e_given_not_h : likelihoods_not_h) {
        scenarios[index].emplace(detail::StressResultFactory::scenario(
            prior, p_e_given_h, p_e_given_not_h));
        ++index;
      }
    }
  }

  const StressScenario* minimum_posterior = &*scenarios.front();
  const StressScenario* maximum_posterior = &*scenarios.front();
  const StressScenario* largest_update = &*scenarios.front();
  const StressScenario* smallest_update = &*scenarios.front();
  for (const std::optional<StressScenario>& candidate : scenarios) {
    const StressScenario& scenario = *candidate;
    if (scenario.posterior() < minimum_posterior->posterior()) {
      minimum_posterior = &scenario;
    }
    if (scenario.posterior() > maximum_posterior->posterior()) {
      maximum_posterior = &scenario;
    }
    if (std::abs(scenario.belief_delta()) >
        std::abs(largest_update->belief_delta())) {
      largest_update = &scenario;
    }
    if (std::abs(scenario.belief_delta()) <
        std::abs(smallest_update->belief_delta())) {
      smallest_update = &scenario;
    }
  }

  const double original_posterior = original.posterior();
  const double original_update =
      std::abs(original.intermediate_calculations().belief_delta());
  const double over_update_amount =
      std::max(0.0,
               std::abs(largest_update->belief_delta()) - original_update);
  const double under_update_amount =
      std::max(0.0,
               original_update - std::abs(smallest_update->belief_delta()));
  const ThresholdCrossing threshold_crossing = find_threshold_crossing(
      original_posterior, minimum_posterior->posterior(),
      maximum_posterior->posterior(), threshold);

  return detail::StressResultFactory::result(
      range_mode, std::move(requested_relative_ranges), bounds,
      original_posterior,
      original.intermediate_calculations().belief_delta(),
      *minimum_posterior, *maximum_posterior, *largest_update,
      *smallest_update, over_update_amount, under_update_amount, threshold,
      threshold_crossing);
}

}  // namespace

StressResult stress_update(const UpdateResult& original,
                           const RelativeStressRanges& ranges,
                           std::optional<double> threshold) {
  const LikelihoodInput& values = original.resolved_values();
  const AbsoluteStressRanges resolved{
      relative_bounds(original.inputs().prior().probability(),
                      ranges.prior_percent(),
                      "relative prior range exceeds [0, 1]"),
      relative_bounds(values.p_e_given_h(), ranges.p_e_given_h_percent(),
                      "relative P(E|H) range exceeds [0, 1]"),
      relative_bounds(values.p_e_given_not_h(),
                      ranges.p_e_given_not_h_percent(),
                      "relative P(E|not H) range exceeds [0, 1]")};
  return analyze(original, resolved, StressRangeMode::relative_percent, ranges,
                 threshold);
}

StressResult stress_update(const UpdateResult& original,
                           const AbsoluteStressRanges& ranges,
                           std::optional<double> threshold) {
  return analyze(original, ranges, StressRangeMode::absolute_bounds,
                 std::nullopt, threshold);
}

}  // namespace belief_update
