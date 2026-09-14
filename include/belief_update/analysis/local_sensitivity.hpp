#pragma once

#include "belief_update/types/belief_state.hpp"
#include "belief_update/types/likelihood_values.hpp"
#include "belief_update/types/update_result.hpp"

namespace belief_update {

class LocalSensitivityResult final {
 public:
  LocalSensitivityResult(const LocalSensitivityResult&) = default;
  LocalSensitivityResult(LocalSensitivityResult&&) noexcept = default;
  LocalSensitivityResult& operator=(const LocalSensitivityResult&) = delete;
  LocalSensitivityResult& operator=(LocalSensitivityResult&&) = delete;

  [[nodiscard]] const BeliefState& prior() const noexcept { return prior_; }
  [[nodiscard]] const LikelihoodValues& likelihoods() const noexcept {
    return likelihoods_;
  }
  [[nodiscard]] double posterior() const noexcept { return posterior_; }
  [[nodiscard]] double d_posterior_d_prior() const noexcept {
    return d_posterior_d_prior_;
  }
  [[nodiscard]] double d_posterior_d_p_e_given_h() const noexcept {
    return d_posterior_d_p_e_given_h_;
  }
  [[nodiscard]] double d_posterior_d_p_e_given_not_h() const noexcept {
    return d_posterior_d_p_e_given_not_h_;
  }

 private:
  friend LocalSensitivityResult local_sensitivity(const UpdateResult&);

  explicit LocalSensitivityResult(
      BeliefState prior, LikelihoodValues likelihoods, double posterior,
      double d_posterior_d_prior, double d_posterior_d_p_e_given_h,
      double d_posterior_d_p_e_given_not_h);

  BeliefState prior_;
  LikelihoodValues likelihoods_;
  double posterior_;
  double d_posterior_d_prior_;
  double d_posterior_d_p_e_given_h_;
  double d_posterior_d_p_e_given_not_h_;
};

[[nodiscard]] LocalSensitivityResult local_sensitivity(
    const UpdateResult& original);

}  // namespace belief_update
