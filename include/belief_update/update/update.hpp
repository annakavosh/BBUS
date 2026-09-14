#pragma once

#include "belief_update/types/update_algorithm.hpp"
#include "belief_update/types/update_config.hpp"
#include "belief_update/types/update_inputs.hpp"
#include "belief_update/types/update_result.hpp"

namespace belief_update {

class LikelihoodModel;

[[nodiscard]] UpdateResult update(UpdateInputs inputs,
                                  UpdateConfig configuration,
                                  UpdateAlgorithm algorithm);

// Convenience entry points use exact binary Bayes with no freshness rule.
// The scalar form records deterministic direct-input provenance.
[[nodiscard]] UpdateResult update(double prior,
                                  const LikelihoodModel& likelihood_model);

[[nodiscard]] UpdateResult update(double prior, double p_e_given_h,
                                  double p_e_given_not_h);

}  // namespace belief_update
