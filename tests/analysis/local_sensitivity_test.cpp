#include <type_traits>

#include "belief_update/analysis/local_sensitivity.hpp"
#include "belief_update/update/update.hpp"
#include "test_support.hpp"

int main() {
  using namespace belief_update;

  static_assert(!std::is_default_constructible_v<LocalSensitivityResult>);
  static_assert(!std::is_copy_assignable_v<LocalSensitivityResult>);

  const UpdateResult original = update(0.4, 0.8, 0.2);
  const double original_posterior = original.posterior();
  const LocalSensitivityResult sensitivity = local_sensitivity(original);

  test_support::expect_near(sensitivity.posterior(), 8.0 / 11.0, 1e-15,
                            "sensitivity records the core posterior");
  test_support::expect_near(sensitivity.d_posterior_d_prior(), 100.0 / 121.0,
                            1e-15, "analytic prior derivative is correct");
  test_support::expect_near(sensitivity.d_posterior_d_p_e_given_h(),
                            30.0 / 121.0, 1e-15,
                            "analytic P(E|H) derivative is correct");
  test_support::expect_near(sensitivity.d_posterior_d_p_e_given_not_h(),
                            -120.0 / 121.0, 1e-15,
                            "analytic P(E|not H) derivative is correct");
  test_support::expect(original.posterior() == original_posterior,
                       "sensitivity does not modify the original result");

  const LocalSensitivityResult certain_prior =
      local_sensitivity(update(0.0, 0.8, 0.2));
  test_support::expect_near(certain_prior.d_posterior_d_prior(), 4.0, 1e-15,
                            "boundary prior derivative is finite");
  test_support::expect(certain_prior.d_posterior_d_p_e_given_h() == 0.0,
                       "likelihood derivative is zero at prior zero");
  test_support::expect(certain_prior.d_posterior_d_p_e_given_not_h() == 0.0,
                       "alternate likelihood derivative is zero at prior zero");

  return test_support::finish();
}
