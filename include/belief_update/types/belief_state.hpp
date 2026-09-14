#pragma once

#include "belief_update/detail/probability.hpp"

namespace belief_update {

class BeliefState final {
 public:
  explicit BeliefState(double probability)
      : probability_(detail::checked_probability(probability)) {}

  BeliefState(const BeliefState&) = default;
  BeliefState(BeliefState&&) noexcept = default;
  BeliefState& operator=(const BeliefState&) = delete;
  BeliefState& operator=(BeliefState&&) = delete;

  [[nodiscard]] double probability() const noexcept { return probability_; }

 private:
  double probability_;
};

}  // namespace belief_update
