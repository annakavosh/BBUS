#pragma once

#include <cstdint>
#include <optional>

#include "belief_update/types/bayes_calculation.hpp"
#include "belief_update/types/belief_state.hpp"
#include "belief_update/types/likelihood_values.hpp"
#include "belief_update/types/update_result.hpp"

namespace belief_update {

enum class StressRangeMode : std::uint8_t {
  relative_percent,
  absolute_bounds,
};

enum class ThresholdCrossing : std::uint8_t {
  not_evaluated,
  none,
  upward,
  downward,
  both,
};

class ProbabilityBounds final {
 public:
  explicit ProbabilityBounds(double lower, double upper);

  ProbabilityBounds(const ProbabilityBounds&) = default;
  ProbabilityBounds(ProbabilityBounds&&) noexcept = default;
  ProbabilityBounds& operator=(const ProbabilityBounds&) = delete;
  ProbabilityBounds& operator=(ProbabilityBounds&&) = delete;

  [[nodiscard]] double lower() const noexcept { return lower_; }
  [[nodiscard]] double upper() const noexcept { return upper_; }

 private:
  double lower_;
  double upper_;
};

class RelativeStressRanges final {
 public:
  // Percent values are relative percentages: 10.0 means baseline +/- 10%.
  explicit RelativeStressRanges(double prior_percent,
                                double p_e_given_h_percent,
                                double p_e_given_not_h_percent);

  RelativeStressRanges(const RelativeStressRanges&) = default;
  RelativeStressRanges(RelativeStressRanges&&) noexcept = default;
  RelativeStressRanges& operator=(const RelativeStressRanges&) = delete;
  RelativeStressRanges& operator=(RelativeStressRanges&&) = delete;

  [[nodiscard]] double prior_percent() const noexcept {
    return prior_percent_;
  }
  [[nodiscard]] double p_e_given_h_percent() const noexcept {
    return p_e_given_h_percent_;
  }
  [[nodiscard]] double p_e_given_not_h_percent() const noexcept {
    return p_e_given_not_h_percent_;
  }

 private:
  double prior_percent_;
  double p_e_given_h_percent_;
  double p_e_given_not_h_percent_;
};

class AbsoluteStressRanges final {
 public:
  explicit AbsoluteStressRanges(ProbabilityBounds prior,
                                ProbabilityBounds p_e_given_h,
                                ProbabilityBounds p_e_given_not_h);

  AbsoluteStressRanges(const AbsoluteStressRanges&) = default;
  AbsoluteStressRanges(AbsoluteStressRanges&&) noexcept = default;
  AbsoluteStressRanges& operator=(const AbsoluteStressRanges&) = delete;
  AbsoluteStressRanges& operator=(AbsoluteStressRanges&&) = delete;

  [[nodiscard]] const ProbabilityBounds& prior() const noexcept {
    return prior_;
  }
  [[nodiscard]] const ProbabilityBounds& p_e_given_h() const noexcept {
    return p_e_given_h_;
  }
  [[nodiscard]] const ProbabilityBounds& p_e_given_not_h() const noexcept {
    return p_e_given_not_h_;
  }

 private:
  ProbabilityBounds prior_;
  ProbabilityBounds p_e_given_h_;
  ProbabilityBounds p_e_given_not_h_;
};

namespace detail {
class StressResultFactory;
}

class StressScenario final {
 public:
  StressScenario(const StressScenario&) = default;
  StressScenario(StressScenario&&) noexcept = default;
  StressScenario& operator=(const StressScenario&) = delete;
  StressScenario& operator=(StressScenario&&) = delete;

  [[nodiscard]] const BeliefState& prior() const noexcept { return prior_; }
  [[nodiscard]] const LikelihoodValues& likelihoods() const noexcept {
    return likelihoods_;
  }
  [[nodiscard]] const BayesCalculation& calculation() const noexcept {
    return calculation_;
  }
  [[nodiscard]] double posterior() const noexcept {
    return calculation_.posterior();
  }
  [[nodiscard]] double belief_delta() const noexcept {
    return calculation_.belief_delta();
  }

 private:
  friend class detail::StressResultFactory;

  explicit StressScenario(BeliefState prior, LikelihoodValues likelihoods,
                          BayesCalculation calculation);

  BeliefState prior_;
  LikelihoodValues likelihoods_;
  BayesCalculation calculation_;
};

class StressResult final {
 public:
  StressResult(const StressResult&) = default;
  StressResult(StressResult&&) noexcept = default;
  StressResult& operator=(const StressResult&) = delete;
  StressResult& operator=(StressResult&&) = delete;

  [[nodiscard]] StressRangeMode range_mode() const noexcept {
    return range_mode_;
  }
  [[nodiscard]] const std::optional<RelativeStressRanges>&
  requested_relative_ranges() const noexcept {
    return requested_relative_ranges_;
  }
  [[nodiscard]] const AbsoluteStressRanges& resolved_bounds() const noexcept {
    return resolved_bounds_;
  }
  [[nodiscard]] double original_posterior() const noexcept {
    return original_posterior_;
  }
  [[nodiscard]] double original_belief_delta() const noexcept {
    return original_belief_delta_;
  }
  [[nodiscard]] const StressScenario& minimum_posterior_scenario()
      const noexcept {
    return minimum_posterior_scenario_;
  }
  [[nodiscard]] const StressScenario& maximum_posterior_scenario()
      const noexcept {
    return maximum_posterior_scenario_;
  }
  [[nodiscard]] const StressScenario& worst_case_scenario() const noexcept {
    return minimum_posterior_scenario_;
  }
  [[nodiscard]] double worst_case_posterior() const noexcept {
    return minimum_posterior_scenario_.posterior();
  }
  [[nodiscard]] const StressScenario& over_update_scenario() const noexcept {
    return over_update_scenario_;
  }
  [[nodiscard]] const StressScenario& under_update_scenario() const noexcept {
    return under_update_scenario_;
  }
  // Over/under update compare absolute belief-delta magnitude at the supplied
  // boundary cases with the magnitude of the recorded update.
  [[nodiscard]] bool over_update() const noexcept {
    return over_update_amount_ > 0.0;
  }
  [[nodiscard]] bool under_update() const noexcept {
    return under_update_amount_ > 0.0;
  }
  [[nodiscard]] double over_update_amount() const noexcept {
    return over_update_amount_;
  }
  [[nodiscard]] double under_update_amount() const noexcept {
    return under_update_amount_;
  }
  [[nodiscard]] const std::optional<double>& threshold() const noexcept {
    return threshold_;
  }
  [[nodiscard]] ThresholdCrossing threshold_crossing() const noexcept {
    return threshold_crossing_;
  }

 private:
  friend class detail::StressResultFactory;

  explicit StressResult(
      StressRangeMode range_mode,
      std::optional<RelativeStressRanges> requested_relative_ranges,
      AbsoluteStressRanges resolved_bounds, double original_posterior,
      double original_belief_delta, StressScenario minimum_posterior_scenario,
      StressScenario maximum_posterior_scenario,
      StressScenario over_update_scenario,
      StressScenario under_update_scenario, double over_update_amount,
      double under_update_amount, std::optional<double> threshold,
      ThresholdCrossing threshold_crossing);

  StressRangeMode range_mode_;
  std::optional<RelativeStressRanges> requested_relative_ranges_;
  AbsoluteStressRanges resolved_bounds_;
  double original_posterior_;
  double original_belief_delta_;
  StressScenario minimum_posterior_scenario_;
  StressScenario maximum_posterior_scenario_;
  StressScenario over_update_scenario_;
  StressScenario under_update_scenario_;
  double over_update_amount_;
  double under_update_amount_;
  std::optional<double> threshold_;
  ThresholdCrossing threshold_crossing_;
};

[[nodiscard]] StressResult stress_update(
    const UpdateResult& original, const RelativeStressRanges& ranges,
    std::optional<double> threshold = std::nullopt);

[[nodiscard]] StressResult stress_update(
    const UpdateResult& original, const AbsoluteStressRanges& ranges,
    std::optional<double> threshold = std::nullopt);

}  // namespace belief_update
