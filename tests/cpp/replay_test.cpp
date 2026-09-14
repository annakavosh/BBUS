#include <optional>
#include <type_traits>

#include "belief_update/replay/replay.hpp"
#include "belief_update/update/update.hpp"
#include "belief_update/version.hpp"
#include "test_support.hpp"

namespace {

belief_update::LikelihoodInput make_input(
    double given_hypothesis, double given_not_hypothesis,
    belief_update::OverrideStatus override_status,
    std::int64_t timestamp = 1'000) {
  using namespace belief_update;
  return LikelihoodInput{
      LikelihoodValues{given_hypothesis, given_not_hypothesis}, 0.8,
      Provenance{SourceType::model,
                 SourceId{"replay-model"},
                 SourceVersion{"5"},
                 ProvenanceTimestamp{timestamp},
                 0.95,
                 override_status}};
}

void expect_same_calculation(const belief_update::BayesCalculation& left,
                             const belief_update::BayesCalculation& right) {
  test_support::expect(left.prior() == right.prior(),
                       "replay reproduces prior");
  test_support::expect(left.likelihood_ratio() == right.likelihood_ratio(),
                       "replay reproduces likelihood ratio");
  test_support::expect(left.prior_odds() == right.prior_odds(),
                       "replay reproduces prior odds");
  test_support::expect(left.posterior_odds() == right.posterior_odds(),
                       "replay reproduces posterior odds");
  test_support::expect(left.posterior() == right.posterior(),
                       "replay reproduces posterior");
  test_support::expect(left.belief_delta() == right.belief_delta(),
                       "replay reproduces belief delta");
}

}  // namespace

int main() {
  using namespace belief_update;

  static_assert(!std::is_default_constructible_v<UpdateResult>);
  static_assert(!std::is_copy_assignable_v<UpdateResult>);
  static_assert(!std::is_move_assignable_v<UpdateResult>);

  const LikelihoodInput explicit_override =
      make_input(0.8, 0.2, OverrideStatus::requested);
  const LikelihoodInput model_value =
      make_input(0.6, 0.4, OverrideStatus::not_requested);
  const UpdateInputs inputs{BeliefState{0.4}, explicit_override, model_value,
                            ProvenanceTimestamp{1'100}};
  const UpdateConfig configuration{
      FreshnessRule{50, ValidationDecision::warn}};

  const UpdateResult original =
      update(inputs, configuration, UpdateAlgorithm::exact_binary);

  test_support::expect(original.inputs().prior().probability() == 0.4,
                       "result owns the prior input");
  test_support::expect(
      original.inputs().explicit_override().has_value() &&
          original.inputs().likelihood_model_value().has_value(),
      "result owns every candidate input");
  test_support::expect(
      &original.resolved_values() == &*original.inputs().explicit_override(),
      "resolved values refer to the owned selected candidate");
  test_support::expect(original.resolved_source() ==
                           ResolvedInputSource::explicit_override,
                       "result records the resolver decision");
  test_support::expect(original.provenance().source_id().value() ==
                           "replay-model",
                       "result exposes selected provenance");
  test_support::expect(
      original.configuration().freshness_rule().has_value() &&
          original.configuration()
                  .freshness_rule()
                  ->max_age_nanoseconds() == 50,
      "result owns update configuration");
  test_support::expect(original.algorithm() == UpdateAlgorithm::exact_binary,
                       "result records the algorithm");
  test_support::expect(original.posterior() == 8.0 / 11.0,
                       "result exposes the posterior");
  test_support::expect(
      original.validation_warnings().size() == 1 &&
          original.validation_warnings().front() ==
              ValidationCode::freshness_violation,
      "result records validation warnings");
  test_support::expect(original.library_version() == version,
                       "result records the library version");

  const UpdateResult reproduced = replay(original);
  expect_same_calculation(original.intermediate_calculations(),
                          reproduced.intermediate_calculations());
  test_support::expect(reproduced.resolved_source() ==
                           original.resolved_source(),
                       "replay reproduces resolution");
  test_support::expect(reproduced.validation_warnings().size() ==
                           original.validation_warnings().size() &&
                           reproduced.validation_warnings().front() ==
                               original.validation_warnings().front(),
                       "replay reproduces validation warnings");
  test_support::expect(reproduced.library_version() ==
                           original.library_version(),
                       "replay reproduces library version");

  const UpdateInputs log_inputs{
      BeliefState{0.2}, std::nullopt,
      make_input(0.6, 0.3, OverrideStatus::not_requested),
      ProvenanceTimestamp{1'000}};
  const UpdateResult log_original =
      update(log_inputs, UpdateConfig{std::nullopt},
             UpdateAlgorithm::log_odds);
  const UpdateResult log_reproduced = replay(log_original);
  expect_same_calculation(log_original.intermediate_calculations(),
                          log_reproduced.intermediate_calculations());
  test_support::expect(log_original.validation_warnings().empty() &&
                           log_reproduced.validation_warnings().empty(),
                       "replay preserves absence of warnings");

  return test_support::finish();
}
