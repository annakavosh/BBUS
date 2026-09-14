#include "belief_update/analysis/posterior_comparison.hpp"

#include <optional>
#include <stdexcept>
#include <utility>

#include "belief_update/types/update_inputs.hpp"
#include "belief_update/update/update.hpp"

namespace belief_update {

PosteriorComparisonResult::PosteriorComparisonResult(
    UpdateResult model_result, UpdateResult explicit_result)
    : model_result_(std::move(model_result)),
      explicit_result_(std::move(explicit_result)),
      difference_(explicit_result_.posterior() - model_result_.posterior()) {}

PosteriorComparisonResult compare_model_and_explicit(
    const UpdateResult& original) {
  const UpdateInputs& inputs = original.inputs();
  if (!inputs.likelihood_model_value().has_value()) {
    throw std::invalid_argument(
        "posterior comparison requires a likelihood-model value");
  }
  if (!inputs.explicit_override().has_value()) {
    throw std::invalid_argument(
        "posterior comparison requires an explicit value");
  }

  UpdateResult model_result = update(
      UpdateInputs{
          inputs.prior(), std::nullopt,
          std::optional<LikelihoodInput>{*inputs.likelihood_model_value()},
          inputs.evaluation_time()},
      original.configuration(), original.algorithm());
  UpdateResult explicit_result = update(
      UpdateInputs{
          inputs.prior(),
          std::optional<LikelihoodInput>{*inputs.explicit_override()},
          std::nullopt, inputs.evaluation_time()},
      original.configuration(), original.algorithm());

  return PosteriorComparisonResult{std::move(model_result),
                                   std::move(explicit_result)};
}

}  // namespace belief_update
