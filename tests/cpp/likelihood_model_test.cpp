#include <optional>
#include <stdexcept>
#include <utility>

#include "belief_update/core/bayes.hpp"
#include "belief_update/model/likelihood_model.hpp"
#include "belief_update/update/update.hpp"
#include "test_support.hpp"

namespace {

class ConstantLikelihoodModel final : public belief_update::LikelihoodModel {
 public:
  explicit ConstantLikelihoodModel(belief_update::LikelihoodInput output)
      : output_(std::move(output)) {}

  [[nodiscard]] belief_update::LikelihoodInput infer() const override {
    return output_;
  }

 private:
  belief_update::LikelihoodInput output_;
};

belief_update::Provenance make_provenance(
    belief_update::SourceType source_type, double confidence) {
  using namespace belief_update;
  return Provenance{source_type,
                    SourceId{"constant-test-model"},
                    SourceVersion{"1"},
                    ProvenanceTimestamp{1'000},
                    confidence,
                    OverrideStatus::not_requested};
}

}  // namespace

int main() {
  using namespace belief_update;

  const ConstantLikelihoodModel model{LikelihoodInput{
      LikelihoodValues{0.75, 0.25}, 0.6,
      make_provenance(SourceType::model, 0.9)}};

  const LikelihoodInput output = model.infer();
  test_support::expect(output.p_e_given_h() == 0.75,
                       "model returns P(E|H)");
  test_support::expect(output.p_e_given_not_h() == 0.25,
                       "model returns P(E|not H)");
  test_support::expect(output.evidence_weight().has_value() &&
                           *output.evidence_weight() == 0.6,
                       "model returns optional evidence weight");
  test_support::expect(output.confidence() == 0.9,
                       "model returns confidence");
  test_support::expect(output.provenance().source_type() == SourceType::model,
                       "model returns typed provenance");

  const BayesCalculation result =
      exact_binary_update(BeliefState{0.5}, output.likelihoods());
  test_support::expect_near(result.posterior(), 0.75, 1e-15,
                            "caller passes only likelihoods into the core");

  const UpdateResult model_update = update(0.5, model);
  test_support::expect(
      model_update.posterior() == result.posterior(),
      "model convenience update matches the C++ Bayesian core");
  test_support::expect(
      model_update.intermediate_calculations().likelihood_ratio() ==
          result.likelihood_ratio(),
      "model convenience update exposes matching intermediates");
  test_support::expect(
      model_update.resolved_source() == ResolvedInputSource::likelihood_model,
      "model convenience update records model resolution");
  test_support::expect(
      model_update.provenance().source_id().value() ==
          output.provenance().source_id().value(),
      "model convenience update preserves model provenance");

  test_support::expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(LikelihoodInput{
            LikelihoodValues{0.75, 0.25}, 1.1,
            make_provenance(SourceType::model, 0.9)});
      },
      "evidence weight outside [0, 1] is rejected");

  return test_support::finish();
}
