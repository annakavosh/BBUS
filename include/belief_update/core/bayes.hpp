#pragma once

#include "belief_update/types/bayes_calculation.hpp"
#include "belief_update/types/belief_state.hpp"
#include "belief_update/types/likelihood_values.hpp"

namespace belief_update {

// LikelihoodValues must describe the resolved observation: P(E|H) and P(E|not H).
// Throws std::domain_error when that observation has zero predictive mass.
[[nodiscard]] BayesCalculation exact_binary_update(
    const BeliefState& prior, const LikelihoodValues& likelihoods);

// log_likelihood_ratio is ln(P(E|H) / P(E|not H)) and may be infinite.
// Throws for NaN or an indeterminate infinite-ratio/certain-prior combination.
[[nodiscard]] BayesCalculation log_odds_update(
    const BeliefState& prior, double log_likelihood_ratio);

}  // namespace belief_update
