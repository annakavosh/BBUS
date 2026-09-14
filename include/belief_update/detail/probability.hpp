#pragma once

#include <cmath>
#include <stdexcept>

namespace belief_update::detail {

[[nodiscard]] inline double checked_probability(double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("probability must be finite");
  }
  if (value < 0.0 || value > 1.0) {
    throw std::invalid_argument("probability must be in [0, 1]");
  }
  return value;
}

}  // namespace belief_update::detail
