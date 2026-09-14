#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "belief_update/types/bayes_calculation.hpp"
#include "belief_update/types/provenance.hpp"
#include "belief_update/types/resolution_result.hpp"
#include "belief_update/types/update_algorithm.hpp"
#include "belief_update/types/update_config.hpp"
#include "belief_update/types/update_inputs.hpp"
#include "belief_update/types/validation_result.hpp"

namespace belief_update {

class UpdateResult final {
 public:
  UpdateResult(const UpdateResult&) = default;
  UpdateResult(UpdateResult&&) noexcept = default;
  UpdateResult& operator=(const UpdateResult&) = delete;
  UpdateResult& operator=(UpdateResult&&) = delete;

  [[nodiscard]] const UpdateInputs& inputs() const noexcept { return inputs_; }
  [[nodiscard]] const LikelihoodInput& resolved_values() const noexcept {
    if (resolved_source_ == ResolvedInputSource::explicit_override) {
      return *inputs_.explicit_override();
    }
    return *inputs_.likelihood_model_value();
  }
  [[nodiscard]] const Provenance& provenance() const noexcept {
    return resolved_values().provenance();
  }
  [[nodiscard]] const UpdateConfig& configuration() const noexcept {
    return configuration_;
  }
  [[nodiscard]] UpdateAlgorithm algorithm() const noexcept {
    return algorithm_;
  }
  [[nodiscard]] ResolvedInputSource resolved_source() const noexcept {
    return resolved_source_;
  }
  [[nodiscard]] const BayesCalculation& intermediate_calculations()
      const noexcept {
    return intermediate_calculations_;
  }
  [[nodiscard]] double posterior() const noexcept {
    return intermediate_calculations_.posterior();
  }
  [[nodiscard]] std::span<const ValidationCode> validation_warnings()
      const noexcept {
    return {validation_warnings_.data(), validation_warning_count_};
  }
  [[nodiscard]] const std::string& library_version() const noexcept {
    return library_version_;
  }

 private:
  friend UpdateResult update(UpdateInputs, UpdateConfig, UpdateAlgorithm);

  explicit UpdateResult(UpdateInputs inputs, ResolvedInputSource resolved_source,
                        UpdateConfig configuration, UpdateAlgorithm algorithm,
                        BayesCalculation intermediate_calculations,
                        std::optional<ValidationCode> validation_warning,
                        std::string library_version)
      : inputs_(std::move(inputs)),
        resolved_source_(resolved_source),
        configuration_(std::move(configuration)),
        algorithm_(algorithm),
        intermediate_calculations_(std::move(intermediate_calculations)),
        validation_warnings_{
            validation_warning.value_or(ValidationCode::valid)},
        validation_warning_count_(validation_warning.has_value() ? 1U : 0U),
        library_version_(std::move(library_version)) {}

  UpdateInputs inputs_;
  ResolvedInputSource resolved_source_;
  UpdateConfig configuration_;
  UpdateAlgorithm algorithm_;
  BayesCalculation intermediate_calculations_;
  std::array<ValidationCode, 1> validation_warnings_;
  std::size_t validation_warning_count_;
  std::string library_version_;
};

}  // namespace belief_update
