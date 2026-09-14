#include "belief_update/core/bayes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace belief_update {
namespace {

[[nodiscard]] double extended_ratio(double numerator, double denominator) {
  if (denominator == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return numerator / denominator;
}

[[nodiscard]] double logistic(double log_odds) noexcept {
  if (log_odds >= 0.0) {
    return 1.0 / (1.0 + std::exp(-log_odds));
  }

  const double exponential = std::exp(log_odds);
  return exponential / (1.0 + exponential);
}

}  // namespace

BayesCalculation exact_binary_update(const BeliefState& prior,
                                     const LikelihoodValues& likelihoods) {
  const double probability = prior.probability();
  const double given_hypothesis =
      likelihoods.evidence_given_hypothesis();
  const double given_not_hypothesis =
      likelihoods.evidence_given_not_hypothesis();
  const double scale = std::max(given_hypothesis, given_not_hypothesis);

  if (scale == 0.0) {
    throw std::domain_error(
        "observation has zero likelihood under both hypotheses");
  }

  const double hypothesis_mass =
      probability * (given_hypothesis / scale);
  const double not_hypothesis_mass =
      (1.0 - probability) * (given_not_hypothesis / scale);
  const double evidence_mass = hypothesis_mass + not_hypothesis_mass;

  if (evidence_mass == 0.0) {
    throw std::domain_error(
        "observation has zero probability under the prior");
  }

  const double likelihood_ratio =
      extended_ratio(given_hypothesis, given_not_hypothesis);
  const double prior_odds = extended_ratio(probability, 1.0 - probability);
  const double posterior_odds =
      extended_ratio(hypothesis_mass, not_hypothesis_mass);
  const BeliefState posterior{hypothesis_mass / evidence_mass};

  return BayesCalculation{prior, likelihood_ratio, prior_odds, posterior_odds,
                          posterior};
}

BayesCalculation log_odds_update(const BeliefState& prior,
                                 double log_likelihood_ratio) {
  if (std::isnan(log_likelihood_ratio)) {
    throw std::invalid_argument("log likelihood ratio must not be NaN");
  }

  const double probability = prior.probability();
  if ((probability == 0.0 && log_likelihood_ratio > 0.0 &&
       std::isinf(log_likelihood_ratio)) ||
      (probability == 1.0 && log_likelihood_ratio < 0.0 &&
       std::isinf(log_likelihood_ratio))) {
    throw std::domain_error(
        "infinite likelihood ratio conflicts with a certain prior");
  }

  const double likelihood_ratio = std::exp(log_likelihood_ratio);
  const double prior_odds = extended_ratio(probability, 1.0 - probability);

  if (probability == 0.0) {
    return BayesCalculation{prior, likelihood_ratio, prior_odds, 0.0,
                            BeliefState{0.0}};
  }
  if (probability == 1.0) {
    return BayesCalculation{prior,
                            likelihood_ratio,
                            prior_odds,
                            std::numeric_limits<double>::infinity(),
                            BeliefState{1.0}};
  }

  const double log_prior_odds =
      std::log(probability) - std::log1p(-probability);
  const double log_posterior_odds =
      log_prior_odds + log_likelihood_ratio;
  const double posterior_odds = std::exp(log_posterior_odds);
  const BeliefState posterior{logistic(log_posterior_odds)};

  return BayesCalculation{prior, likelihood_ratio, prior_odds, posterior_odds,
                          posterior};
}

}  // namespace belief_update
