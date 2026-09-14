#include "belief_update/core/bayes.hpp"
#include "test_support.hpp"

int main() {
  using namespace belief_update;

  // Hand calculation:
  // LR = 0.8 / 0.2 = 4
  // prior odds = 0.4 / 0.6 = 2/3
  // posterior odds = 8/3
  // posterior = (8/3) / (1 + 8/3) = 8/11
  const BayesCalculation result =
      exact_binary_update(BeliefState{0.4}, LikelihoodValues{0.8, 0.2});

  test_support::expect_near(result.prior(), 0.4, 1e-15,
                            "exact update reports the prior");
  test_support::expect_near(result.likelihood_ratio(), 4.0, 1e-15,
                            "exact update computes the likelihood ratio");
  test_support::expect_near(result.prior_odds(), 2.0 / 3.0, 1e-15,
                            "exact update computes prior odds");
  test_support::expect_near(result.posterior_odds(), 8.0 / 3.0, 1e-15,
                            "exact update computes posterior odds");
  test_support::expect_near(result.posterior(), 8.0 / 11.0, 1e-15,
                            "exact update computes the posterior");
  test_support::expect_near(result.belief_delta(), 18.0 / 55.0, 1e-15,
                            "exact update computes the belief delta");

  const BayesCalculation neutral =
      exact_binary_update(BeliefState{0.73}, LikelihoodValues{0.05, 0.05});
  test_support::expect_near(neutral.likelihood_ratio(), 1.0, 0.0,
                            "equal likelihoods produce unit evidence");
  test_support::expect_near(neutral.posterior(), 0.73, 1e-15,
                            "unit evidence preserves the belief");
  test_support::expect_near(neutral.belief_delta(), 0.0, 1e-15,
                            "unit evidence has zero belief delta");

  return test_support::finish();
}
