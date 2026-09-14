#pragma once

#include <cmath>
#include <stdexcept>

#include "belief_update/types/belief_state.hpp"

namespace belief_update {

class LikelihoodValues;

class BayesCalculation final {
 public:
  BayesCalculation(const BayesCalculation&) = default;
  BayesCalculation(BayesCalculation&&) noexcept = default;
  BayesCalculation& operator=(const BayesCalculation&) = delete;
  BayesCalculation& operator=(BayesCalculation&&) = delete;

  [[nodiscard]] double prior() const noexcept { return prior_; }
  [[nodiscard]] double likelihood_ratio() const noexcept {
    return likelihood_ratio_;
  }
  [[nodiscard]] double prior_odds() const noexcept { return prior_odds_; }
  [[nodiscard]] double posterior_odds() const noexcept {
    return posterior_odds_;
  }
  [[nodiscard]] double posterior() const noexcept { return posterior_; }
  [[nodiscard]] double belief_delta() const noexcept { return belief_delta_; }

 private:
  friend BayesCalculation exact_binary_update(
      const BeliefState&, const LikelihoodValues&);
  friend BayesCalculation log_odds_update(const BeliefState&, double);

  explicit BayesCalculation(BeliefState prior, double likelihood_ratio,
                            double prior_odds, double posterior_odds,
                            BeliefState posterior)
      : prior_(prior.probability()),
        likelihood_ratio_(checked_extended_nonnegative(likelihood_ratio)),
        prior_odds_(checked_extended_nonnegative(prior_odds)),
        posterior_odds_(checked_extended_nonnegative(posterior_odds)),
        posterior_(posterior.probability()),
        belief_delta_(posterior_ - prior_) {}

  [[nodiscard]] static double checked_extended_nonnegative(double value) {
    if (std::isnan(value) || value < 0.0) {
      throw std::invalid_argument(
          "likelihood ratios and odds must be non-negative");
    }
    return value;
  }

  double prior_;
  double likelihood_ratio_;
  double prior_odds_;
  double posterior_odds_;
  double posterior_;
  double belief_delta_;
};

}  // namespace belief_update
