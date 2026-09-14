#include "belief_update/resolver/input_resolver.hpp"

#include <optional>

namespace belief_update {

ResolutionResult InputResolver::resolve(
    const LikelihoodInput* explicit_override,
    const LikelihoodInput* likelihood_model_value,
    ProvenanceTimestamp evaluation_time) const {
  if (explicit_override != nullptr) {
    const ValidationResult validation =
        validator_.validate_required_input(explicit_override, evaluation_time);
    if (!validation.can_proceed()) {
      return ResolutionResult{validation, std::nullopt};
    }
    return ResolutionResult{
        validation,
        ResolvedInput{*explicit_override,
                      ResolvedInputSource::explicit_override}};
  }

  if (likelihood_model_value != nullptr) {
    const ValidationResult validation =
        validator_.validate_required_input(likelihood_model_value,
                                           evaluation_time);
    if (!validation.can_proceed()) {
      return ResolutionResult{validation, std::nullopt};
    }
    return ResolutionResult{
        validation,
        ResolvedInput{*likelihood_model_value,
                      ResolvedInputSource::likelihood_model}};
  }

  return ResolutionResult{
      validator_.validate_required_input(nullptr, evaluation_time),
      std::nullopt};
}

}  // namespace belief_update
