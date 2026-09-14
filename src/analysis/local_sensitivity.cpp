#include "belief_update/analysis/local_sensitivity.hpp"

#include <algorithm>
#include <utility>

namespace belief_update {

LocalSensitivityResult::LocalSensitivityResult(
    BeliefState prior, LikelihoodValues likelihoods, double posterior,
    double d_posterior_d_prior, double d_posterior_d_p_e_given_h,
    double d_posterior_d_p_e_given_not_h)
    : prior_(std::move(prior)),
      likelihoods_(std::move(likelihoods)),
      posterior_(posterior),
      d_posterior_d_prior_(d_posterior_d_prior),
      d_posterior_d_p_e_given_h_(d_posterior_d_p_e_given_h),
      d_posterior_d_p_e_given_not_h_(d_posterior_d_p_e_given_not_h) {}

LocalSensitivityResult local_sensitivity(const UpdateResult& original) {
  const BeliefState prior = original.inputs().prior();
  const LikelihoodValues likelihoods = original.resolved_values().likelihoods();

  const double probability = prior.probability();
  const double p_e_given_h = likelihoods.evidence_given_hypothesis();
  const double p_e_given_not_h =
      likelihoods.evidence_given_not_hypothesis();
  const double scale = std::max(p_e_given_h, p_e_given_not_h);
  const double scaled_h = p_e_given_h / scale;
  const double scaled_not_h = p_e_given_not_h / scale;
  const double scaled_evidence =
      probability * scaled_h + (1.0 - probability) * scaled_not_h;
  const double squared_evidence = scaled_evidence * scaled_evidence;
  const double prior_complement_product =
      probability * (1.0 - probability);

  const double d_posterior_d_prior =
      (scaled_h * scaled_not_h) / squared_evidence;
  const double d_posterior_d_p_e_given_h =
      ((prior_complement_product * scaled_not_h) / squared_evidence) / scale;
  const double d_posterior_d_p_e_given_not_h =
      -((prior_complement_product * scaled_h) / squared_evidence) / scale;

  return LocalSensitivityResult{
      prior,
      likelihoods,
      original.posterior(),
      d_posterior_d_prior,
      d_posterior_d_p_e_given_h,
      d_posterior_d_p_e_given_not_h};
}

}  // namespace belief_update
