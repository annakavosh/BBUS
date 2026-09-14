#include <stdexcept>
#include <string_view>

#include "belief_update/core/bayes.hpp"
#include "belief_update/replay/replay.hpp"
#include "belief_update/update/update.hpp"
#include "belief_update/version.hpp"
#include "test_support.hpp"

namespace {

void expect_same_calculation(const belief_update::BayesCalculation& left,
                             const belief_update::BayesCalculation& right) {
  test_support::expect(left.prior() == right.prior(), "prior is equivalent");
  test_support::expect(left.likelihood_ratio() == right.likelihood_ratio(),
                       "likelihood ratio is equivalent");
  test_support::expect(left.prior_odds() == right.prior_odds(),
                       "prior odds are equivalent");
  test_support::expect(left.posterior_odds() == right.posterior_odds(),
                       "posterior odds are equivalent");
  test_support::expect(left.posterior() == right.posterior(),
                       "posterior is equivalent");
  test_support::expect(left.belief_delta() == right.belief_delta(),
                       "belief delta is equivalent");
}

}  // namespace

int main() {
  using namespace belief_update;

  const BayesCalculation core =
      exact_binary_update(BeliefState{0.4}, LikelihoodValues{0.8, 0.2});
  const UpdateResult convenient = update(0.4, 0.8, 0.2);

  expect_same_calculation(core, convenient.intermediate_calculations());
  test_support::expect(
      convenient.resolved_source() == ResolvedInputSource::explicit_override,
      "scalar input is resolved as an explicit override");
  test_support::expect(
      convenient.provenance().source_type() == SourceType::direct_input,
      "scalar input has typed direct-input provenance");
  test_support::expect(
      convenient.provenance().source_id().value() == "belief_update.direct",
      "scalar input source id is deterministic");
  test_support::expect(
      convenient.provenance().version().value() == version,
      "scalar input provenance records the library version");
  test_support::expect(
      convenient.provenance().timestamp().unix_nanoseconds() == 0,
      "scalar input uses the documented epoch timestamp");
  test_support::expect(convenient.provenance().confidence() == 1.0,
                       "scalar input is not automatically down-weighted");
  test_support::expect(
      convenient.provenance().override_status() == OverrideStatus::applied,
      "scalar input records its override status");
  test_support::expect(
      !convenient.configuration().freshness_rule().has_value(),
      "scalar convenience update has no freshness rule");
  test_support::expect(convenient.algorithm() == UpdateAlgorithm::exact_binary,
                       "scalar convenience update records its algorithm");
  test_support::expect(convenient.validation_warnings().empty(),
                       "valid scalar input has no warnings");

  const UpdateResult reproduced = replay(convenient);
  expect_same_calculation(convenient.intermediate_calculations(),
                          reproduced.intermediate_calculations());

  try {
    static_cast<void>(update(0.4, 0.8, 0.0));
    test_support::expect(false, "invalid likelihood ratio is rejected");
  } catch (const std::invalid_argument& exception) {
    test_support::expect(
        std::string_view{exception.what()} ==
            "update rejected: likelihood ratio must be finite",
        "invalid likelihood ratio has a clear exception");
  }

  return test_support::finish();
}
