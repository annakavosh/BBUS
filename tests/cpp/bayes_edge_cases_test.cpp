#include <cmath>
#include <limits>
#include <stdexcept>

#include "belief_update/core/bayes.hpp"
#include "test_support.hpp"

int main() {
  using namespace belief_update;

  const double smallest = std::numeric_limits<double>::denorm_min();
  const BayesCalculation tiny_equal =
      exact_binary_update(BeliefState{0.5},
                          LikelihoodValues{smallest, smallest});
  test_support::expect(tiny_equal.posterior() == 0.5,
                       "equal subnormal likelihoods do not underflow");

  const BayesCalculation decisive_true =
      exact_binary_update(BeliefState{smallest},
                          LikelihoodValues{smallest, 0.0});
  test_support::expect(decisive_true.posterior() == 1.0,
                       "exclusive subnormal evidence remains decisive");
  test_support::expect(std::isinf(decisive_true.likelihood_ratio()),
                       "zero alternative likelihood produces infinite LR");
  test_support::expect(std::isinf(decisive_true.posterior_odds()),
                       "certain posterior has infinite odds");

  const BayesCalculation decisive_false =
      exact_binary_update(BeliefState{0.5}, LikelihoodValues{0.0, 0.7});
  test_support::expect(decisive_false.posterior() == 0.0,
                       "zero hypothesis likelihood produces zero posterior");
  test_support::expect(decisive_false.likelihood_ratio() == 0.0,
                       "zero hypothesis likelihood produces zero LR");
  test_support::expect(decisive_false.belief_delta() == -0.5,
                       "belief delta preserves its sign");

  test_support::expect_throws<std::domain_error>(
      [] {
        static_cast<void>(exact_binary_update(BeliefState{0.5},
                                              LikelihoodValues{0.0, 0.0}));
      },
      "zero likelihood under both hypotheses is undefined");
  test_support::expect_throws<std::domain_error>(
      [] {
        static_cast<void>(exact_binary_update(BeliefState{0.0},
                                              LikelihoodValues{0.8, 0.0}));
      },
      "zero predictive evidence mass is undefined");
  test_support::expect_throws<std::domain_error>(
      [] {
        static_cast<void>(exact_binary_update(BeliefState{1.0},
                                              LikelihoodValues{0.0, 0.8}));
      },
      "zero predictive evidence mass is symmetric at a certain prior");

  const double infinity = std::numeric_limits<double>::infinity();
  const BayesCalculation log_positive =
      log_odds_update(BeliefState{0.5},
                      std::numeric_limits<double>::max());
  test_support::expect(log_positive.posterior() == 1.0,
                       "large positive log evidence is stable");
  test_support::expect(std::isinf(log_positive.posterior_odds()),
                       "large positive log odds saturate explicitly");

  const BayesCalculation log_negative =
      log_odds_update(BeliefState{0.5},
                      -std::numeric_limits<double>::max());
  test_support::expect(log_negative.posterior() == 0.0,
                       "large negative log evidence is stable");
  test_support::expect(log_negative.posterior_odds() == 0.0,
                       "large negative log odds underflow explicitly");

  const BayesCalculation certain_false =
      log_odds_update(BeliefState{0.0}, std::log(2.0));
  const BayesCalculation certain_true =
      log_odds_update(BeliefState{1.0}, -std::log(2.0));
  test_support::expect(certain_false.posterior() == 0.0,
                       "finite evidence preserves a zero prior");
  test_support::expect(certain_true.posterior() == 1.0,
                       "finite evidence preserves a certain prior");

  test_support::expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(log_odds_update(
            BeliefState{0.5}, std::numeric_limits<double>::quiet_NaN()));
      },
      "NaN log likelihood ratio is rejected");
  test_support::expect_throws<std::domain_error>(
      [infinity] {
        static_cast<void>(log_odds_update(BeliefState{0.0}, infinity));
      },
      "zero prior with infinite positive evidence is indeterminate");
  test_support::expect_throws<std::domain_error>(
      [infinity] {
        static_cast<void>(log_odds_update(BeliefState{1.0}, -infinity));
      },
      "certain prior with infinite negative evidence is indeterminate");

  return test_support::finish();
}
