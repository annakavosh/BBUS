#include <cmath>

#include "belief_update/core/bayes.hpp"
#include "test_support.hpp"

int main() {
  using namespace belief_update;

  // Hand calculation:
  // prior odds = 0.2 / 0.8 = 1/4
  // LR = 2
  // posterior odds = 1/2
  // posterior = (1/2) / (1 + 1/2) = 1/3
  const BayesCalculation result =
      log_odds_update(BeliefState{0.2}, std::log(2.0));

  test_support::expect_near(result.prior(), 0.2, 1e-15,
                            "log update reports the prior");
  test_support::expect_near(result.likelihood_ratio(), 2.0, 1e-15,
                            "log update reports the likelihood ratio");
  test_support::expect_near(result.prior_odds(), 0.25, 1e-15,
                            "log update computes prior odds");
  test_support::expect_near(result.posterior_odds(), 0.5, 1e-15,
                            "log update computes posterior odds");
  test_support::expect_near(result.posterior(), 1.0 / 3.0, 1e-15,
                            "log update computes the posterior");
  test_support::expect_near(result.belief_delta(), 2.0 / 15.0, 1e-15,
                            "log update computes the belief delta");

  const BayesCalculation exact =
      exact_binary_update(BeliefState{0.2}, LikelihoodValues{0.6, 0.3});
  test_support::expect_near(result.posterior(), exact.posterior(), 1e-15,
                            "exact and log updates agree");

  return test_support::finish();
}
