#include <stdexcept>
#include <type_traits>

#include "belief_update/core/bayes.hpp"
#include "belief_update/stress/stress.hpp"
#include "belief_update/update/update.hpp"
#include "test_support.hpp"

namespace {

void expect_same_calculation(const belief_update::BayesCalculation& left,
                             const belief_update::BayesCalculation& right) {
  test_support::expect(left.prior() == right.prior(), "prior matches core");
  test_support::expect(left.likelihood_ratio() == right.likelihood_ratio(),
                       "likelihood ratio matches core");
  test_support::expect(left.prior_odds() == right.prior_odds(),
                       "prior odds match core");
  test_support::expect(left.posterior_odds() == right.posterior_odds(),
                       "posterior odds match core");
  test_support::expect(left.posterior() == right.posterior(),
                       "posterior matches core");
  test_support::expect(left.belief_delta() == right.belief_delta(),
                       "belief delta matches core");
}

}  // namespace

int main() {
  using namespace belief_update;

  static_assert(!std::is_default_constructible_v<ProbabilityBounds>);
  static_assert(!std::is_default_constructible_v<RelativeStressRanges>);
  static_assert(!std::is_default_constructible_v<AbsoluteStressRanges>);
  static_assert(!std::is_copy_assignable_v<StressResult>);

  const UpdateResult original = update(0.4, 0.8, 0.2);
  const double original_prior = original.inputs().prior().probability();
  const double original_posterior = original.posterior();
  const double original_ratio =
      original.intermediate_calculations().likelihood_ratio();
  const auto original_source = original.provenance().source_type();

  const AbsoluteStressRanges absolute_ranges{
      ProbabilityBounds{0.3, 0.5}, ProbabilityBounds{0.7, 0.9},
      ProbabilityBounds{0.1, 0.3}};
  const StressResult absolute =
      stress_update(original, absolute_ranges, 0.8);

  test_support::expect(
      absolute.range_mode() == StressRangeMode::absolute_bounds,
      "absolute range mode is recorded");
  test_support::expect(!absolute.requested_relative_ranges().has_value(),
                       "absolute analysis has no relative range record");
  test_support::expect(absolute.worst_case_posterior() == 0.5,
                       "minimum posterior is the worst case");
  test_support::expect_near(
      absolute.maximum_posterior_scenario().posterior(), 0.9, 1e-15,
      "maximum posterior is found within the bounds");
  test_support::expect(
      absolute.threshold_crossing() == ThresholdCrossing::upward,
      "upward threshold crossing is detected");
  test_support::expect(absolute.threshold().has_value() &&
                           *absolute.threshold() == 0.8,
                       "explicit threshold is recorded");
  test_support::expect(absolute.over_update(),
                       "larger update magnitude is detected");
  test_support::expect(absolute.under_update(),
                       "smaller update magnitude is detected");
  test_support::expect(absolute.over_update_amount() > 0.0,
                       "over-update amount is reported");
  test_support::expect(absolute.under_update_amount() > 0.0,
                       "under-update amount is reported");

  const StressScenario& minimum = absolute.minimum_posterior_scenario();
  const BayesCalculation expected_minimum = exact_binary_update(
      BeliefState{minimum.prior().probability()}, minimum.likelihoods());
  expect_same_calculation(minimum.calculation(), expected_minimum);

  test_support::expect(
      stress_update(original, absolute_ranges, 0.6).threshold_crossing() ==
          ThresholdCrossing::downward,
      "downward threshold crossing is detected");
  test_support::expect(
      stress_update(original, absolute_ranges, original.posterior())
              .threshold_crossing() == ThresholdCrossing::both,
      "both crossing directions are detected at the original posterior");
  test_support::expect(
      stress_update(original, absolute_ranges, 0.95).threshold_crossing() ==
          ThresholdCrossing::none,
      "an unreachable threshold has no crossing");

  test_support::expect(original.inputs().prior().probability() ==
                           original_prior,
                       "stress analysis does not modify the original prior");
  test_support::expect(original.posterior() == original_posterior,
                       "stress analysis does not modify the posterior");
  test_support::expect(
      original.intermediate_calculations().likelihood_ratio() ==
          original_ratio,
      "stress analysis does not modify intermediate calculations");
  test_support::expect(original.provenance().source_type() == original_source,
                       "stress analysis does not modify provenance");

  const RelativeStressRanges relative_ranges{10.0, 10.0, 10.0};
  const StressResult relative = stress_update(original, relative_ranges);
  test_support::expect(
      relative.range_mode() == StressRangeMode::relative_percent,
      "relative range mode is recorded");
  test_support::expect(
      relative.requested_relative_ranges().has_value() &&
          relative.requested_relative_ranges()->prior_percent() == 10.0,
      "user-supplied relative percentages are retained");
  test_support::expect_near(relative.resolved_bounds().prior().lower(), 0.36,
                            1e-15,
                            "relative prior lower bound is resolved");
  test_support::expect_near(relative.resolved_bounds().prior().upper(), 0.44,
                            1e-15,
                            "relative prior upper bound is resolved");
  test_support::expect(
      relative.threshold_crossing() == ThresholdCrossing::not_evaluated,
      "threshold analysis is omitted when no threshold is supplied");

  const StressScenario& relative_minimum =
      relative.minimum_posterior_scenario();
  const BayesCalculation expected_relative_minimum = exact_binary_update(
      BeliefState{relative.resolved_bounds().prior().lower()},
      LikelihoodValues{relative.resolved_bounds().p_e_given_h().lower(),
                       relative.resolved_bounds().p_e_given_not_h().upper()});
  expect_same_calculation(relative_minimum.calculation(),
                          expected_relative_minimum);

  test_support::expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(stress_update(
            original, RelativeStressRanges{160.0, 10.0, 10.0}));
      },
      "relative ranges outside [0, 1] are rejected rather than clipped");
  test_support::expect_throws<std::invalid_argument>(
      [] { static_cast<void>(ProbabilityBounds{0.7, 0.2}); },
      "reversed absolute bounds are rejected");
  test_support::expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(stress_update(original, absolute_ranges, 1.1)); },
      "invalid thresholds are rejected");

  const AbsoluteStressRanges impossible_ranges{
      ProbabilityBounds{0.0, 0.0}, ProbabilityBounds{0.8, 0.8},
      ProbabilityBounds{0.0, 0.0}};
  test_support::expect_throws<std::domain_error>(
      [&] { static_cast<void>(stress_update(original, impossible_ranges)); },
      "mathematically impossible stress corners fail through the core");

  return test_support::finish();
}
