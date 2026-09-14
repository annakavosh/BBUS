#pragma once

#include "belief_update/detail/probability.hpp"

namespace belief_update {

class LikelihoodValues final {
 public:
  explicit LikelihoodValues(double evidence_given_hypothesis,
                            double evidence_given_not_hypothesis)
      : evidence_given_hypothesis_(
            detail::checked_probability(evidence_given_hypothesis)),
        evidence_given_not_hypothesis_(
            detail::checked_probability(evidence_given_not_hypothesis)) {}

  LikelihoodValues(const LikelihoodValues&) = default;
  LikelihoodValues(LikelihoodValues&&) noexcept = default;
  LikelihoodValues& operator=(const LikelihoodValues&) = delete;
  LikelihoodValues& operator=(LikelihoodValues&&) = delete;

  [[nodiscard]] double evidence_given_hypothesis() const noexcept {
    return evidence_given_hypothesis_;
  }

  [[nodiscard]] double evidence_given_not_hypothesis() const noexcept {
    return evidence_given_not_hypothesis_;
  }

 private:
  double evidence_given_hypothesis_;
  double evidence_given_not_hypothesis_;
};

}  // namespace belief_update
