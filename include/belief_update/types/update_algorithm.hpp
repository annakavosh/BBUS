#pragma once

#include <cstdint>

namespace belief_update {

enum class UpdateAlgorithm : std::uint8_t {
  exact_binary,
  log_odds,
};

}  // namespace belief_update
