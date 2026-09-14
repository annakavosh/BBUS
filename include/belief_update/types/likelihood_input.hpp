#pragma once

#include <optional>
#include <utility>

#include "belief_update/detail/probability.hpp"
#include "belief_update/types/likelihood_values.hpp"
#include "belief_update/types/provenance.hpp"

namespace belief_update {

class LikelihoodInput final {
 public:
  explicit LikelihoodInput(LikelihoodValues likelihoods,
                           std::optional<double> evidence_weight,
                           Provenance provenance)
      : likelihoods_(std::move(likelihoods)),
        evidence_weight_(checked_weight(evidence_weight)),
        provenance_(std::move(provenance)) {}

  LikelihoodInput(const LikelihoodInput&) = default;
  LikelihoodInput(LikelihoodInput&&) noexcept = default;
  LikelihoodInput& operator=(const LikelihoodInput&) = delete;
  LikelihoodInput& operator=(LikelihoodInput&&) = delete;

  [[nodiscard]] const LikelihoodValues& likelihoods() const noexcept {
    return likelihoods_;
  }
  [[nodiscard]] double p_e_given_h() const noexcept {
    return likelihoods_.evidence_given_hypothesis();
  }
  [[nodiscard]] double p_e_given_not_h() const noexcept {
    return likelihoods_.evidence_given_not_hypothesis();
  }
  [[nodiscard]] const std::optional<double>& evidence_weight() const noexcept {
    return evidence_weight_;
  }
  [[nodiscard]] double confidence() const noexcept {
    return provenance_.confidence();
  }
  [[nodiscard]] const Provenance& provenance() const noexcept {
    return provenance_;
  }

 private:
  [[nodiscard]] static std::optional<double> checked_weight(
      std::optional<double> value) {
    if (value.has_value()) {
      return detail::checked_probability(*value);
    }
    return std::nullopt;
  }

  LikelihoodValues likelihoods_;
  std::optional<double> evidence_weight_;
  Provenance provenance_;
};

}  // namespace belief_update
