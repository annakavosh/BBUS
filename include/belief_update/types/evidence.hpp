#pragma once

#include <cstdint>
#include <utility>

#include "belief_update/types/provenance.hpp"

namespace belief_update {

enum class EvidenceValue : std::uint8_t {
  absent,
  present,
};

class Evidence final {
 public:
  explicit Evidence(EvidenceValue value, Provenance provenance)
      : value_(value), provenance_(std::move(provenance)) {}

  Evidence(const Evidence&) = default;
  Evidence(Evidence&&) noexcept = default;
  Evidence& operator=(const Evidence&) = delete;
  Evidence& operator=(Evidence&&) = delete;

  [[nodiscard]] EvidenceValue value() const noexcept { return value_; }
  [[nodiscard]] const Provenance& provenance() const noexcept {
    return provenance_;
  }

 private:
  EvidenceValue value_;
  Provenance provenance_;
};

}  // namespace belief_update
