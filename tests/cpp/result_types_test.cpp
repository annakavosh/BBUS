#include <optional>
#include <type_traits>

#include "belief_update/core/bayes.hpp"
#include "belief_update/types/update_config.hpp"
#include "test_support.hpp"

int main() {
  using namespace belief_update;

  static_assert(!std::is_default_constructible_v<BayesCalculation>);
  static_assert(!std::is_copy_assignable_v<BayesCalculation>);
  static_assert(!std::is_default_constructible_v<UpdateConfig>);

  const UpdateConfig config{
      FreshnessRule{1'000, ValidationDecision::reject}};
  test_support::expect(
      config.freshness_rule().has_value() &&
          config.freshness_rule()->max_age_nanoseconds() == 1'000 &&
          config.freshness_rule()->violation_decision() ==
              ValidationDecision::reject,
      "freshness configuration is explicit and immutable");
  test_support::expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(FreshnessRule{1'000, ValidationDecision::accept});
      },
      "freshness violations cannot be configured as accepted");

  const BayesCalculation result =
      exact_binary_update(BeliefState{0.4}, LikelihoodValues{0.8, 0.2});

  test_support::expect(result.prior() == 0.4, "result retains the prior");
  test_support::expect(result.likelihood_ratio() == 4.0,
                       "result retains the likelihood ratio");
  test_support::expect_near(result.prior_odds(), 2.0 / 3.0, 1e-15,
                            "result retains the prior odds");
  test_support::expect_near(result.posterior_odds(), 8.0 / 3.0, 1e-15,
                            "result retains the posterior odds");
  test_support::expect(result.posterior() == 8.0 / 11.0,
                       "result retains the posterior");
  test_support::expect_near(result.belief_delta(), 18.0 / 55.0, 1e-15,
                            "result derives the belief delta");

  return test_support::finish();
}
