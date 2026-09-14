#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "belief_update/analysis/threshold_analysis.hpp"
#include "belief_update/update/update.hpp"
#include "test_support.hpp"

int main() {
  using namespace belief_update;

  static_assert(!std::is_default_constructible_v<ThresholdAnalysisResult>);
  static_assert(!std::is_copy_assignable_v<ThresholdAnalysisResult>);

  const UpdateResult positive_update = update(0.4, 0.8, 0.3);
  const ThresholdAnalysisResult crossed =
      analyze_threshold(positive_update, 0.6);
  test_support::expect(crossed.prior_position() == ThresholdPosition::below,
                       "prior is below the threshold");
  test_support::expect(
      crossed.posterior_position() == ThresholdPosition::above,
      "posterior is above the threshold");
  test_support::expect(crossed.transition() == ThresholdTransition::upward,
                       "upward threshold transition is detected");
  test_support::expect(crossed.threshold_met(), "threshold is met");
  test_support::expect_near(crossed.posterior_margin(), 0.04, 1e-15,
                            "posterior margin is reported");
  test_support::expect_near(crossed.observed_likelihood_ratio(), 8.0 / 3.0,
                            1e-15, "observed likelihood ratio is retained");
  test_support::expect(
      crossed.threshold_reachable() &&
          crossed.likelihood_ratio_at_threshold().has_value(),
      "threshold is reachable from a non-certain prior");
  test_support::expect_near(*crossed.likelihood_ratio_at_threshold(), 2.25,
                            1e-15,
                            "analytic threshold likelihood ratio is correct");

  const ThresholdAnalysisResult not_crossed =
      analyze_threshold(positive_update, 0.7);
  test_support::expect(not_crossed.transition() == ThresholdTransition::none,
                       "uncrossed threshold is reported");
  test_support::expect(!not_crossed.threshold_met(),
                       "higher threshold is not met");
  test_support::expect_near(
      *not_crossed.likelihood_ratio_at_threshold(), 3.5, 1e-15,
      "higher threshold ratio is calculated analytically");

  const ThresholdAnalysisResult from_threshold =
      analyze_threshold(positive_update, 0.4);
  test_support::expect(from_threshold.prior_position() ==
                           ThresholdPosition::at &&
                           from_threshold.transition() ==
                               ThresholdTransition::upward,
                       "movement away from the threshold is classified");

  const ThresholdAnalysisResult downward =
      analyze_threshold(update(0.6, 0.2, 0.8), 0.4);
  test_support::expect(downward.transition() == ThresholdTransition::downward,
                       "downward threshold transition is detected");

  const ThresholdAnalysisResult unreachable =
      analyze_threshold(update(0.0, 0.8, 0.2), 0.5);
  test_support::expect(!unreachable.threshold_reachable(),
                       "certain prior cannot reach a different threshold");
  test_support::expect(
      !unreachable.likelihood_ratio_at_threshold().has_value(),
      "unreachable threshold has no required likelihood ratio");

  const ThresholdAnalysisResult certain_at_threshold =
      analyze_threshold(update(0.0, 0.8, 0.2), 0.0);
  test_support::expect(certain_at_threshold.threshold_reachable(),
                       "certain prior already at threshold is reachable");
  test_support::expect(
      !certain_at_threshold.likelihood_ratio_at_threshold().has_value(),
      "non-unique threshold ratio is not invented");

  const ThresholdAnalysisResult endpoint =
      analyze_threshold(positive_update, 1.0);
  test_support::expect(
      !endpoint.threshold_reachable() &&
          endpoint.likelihood_ratio_at_threshold().has_value() &&
          std::isinf(*endpoint.likelihood_ratio_at_threshold()),
      "endpoint threshold reports its unreachable limiting ratio");

  test_support::expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(analyze_threshold(positive_update, 1.1)); },
      "threshold above one is rejected");
  test_support::expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            analyze_threshold(positive_update, std::nan("")));
      },
      "NaN threshold is rejected");

  return test_support::finish();
}
