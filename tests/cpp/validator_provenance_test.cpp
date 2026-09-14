#include <optional>
#include <stdexcept>

#include "belief_update/validation/validator.hpp"
#include "test_support.hpp"

namespace {

belief_update::Provenance make_provenance(
    belief_update::ProvenanceTimestamp timestamp,
    belief_update::SourceType source_type = belief_update::SourceType::model,
    belief_update::OverrideStatus override_status =
        belief_update::OverrideStatus::not_requested) {
  using namespace belief_update;
  return Provenance{source_type, SourceId{"model-a"}, SourceVersion{"3"},
                    timestamp, 0.9, override_status};
}

}  // namespace

int main() {
  using namespace belief_update;

  const ProvenanceTimestamp observed{1'000};
  const ProvenanceTimestamp evaluated{2'000};
  const Provenance provenance = make_provenance(observed);

  const Validator no_freshness{UpdateConfig{std::nullopt}};
  test_support::expect(
      no_freshness.validate_provenance(provenance, evaluated).is_accepted(),
      "freshness is not inferred when no rule is configured");

  const Validator warning_validator{UpdateConfig{
      FreshnessRule{500, ValidationDecision::warn}}};
  const ValidationResult warning =
      warning_validator.validate_provenance(provenance, evaluated);
  test_support::expect(warning.decision() == ValidationDecision::warn,
                       "configured stale input can warn");
  test_support::expect(warning.code() == ValidationCode::freshness_violation,
                       "stale warning has a typed code");
  test_support::expect(warning.can_proceed() && !warning.is_accepted(),
                       "warning permits progress without claiming validity");

  const Validator rejecting_validator{UpdateConfig{
      FreshnessRule{500, ValidationDecision::reject}}};
  const ValidationResult rejection =
      rejecting_validator.validate_provenance(provenance, evaluated);
  test_support::expect(rejection.decision() == ValidationDecision::reject,
                       "configured stale input can reject");

  const Validator boundary_validator{UpdateConfig{
      FreshnessRule{1'000, ValidationDecision::reject}}};
  test_support::expect(
      boundary_validator.validate_provenance(provenance, evaluated).is_accepted(),
      "input exactly at the configured age is accepted");

  const ValidationResult future = warning_validator.validate_provenance(
      make_provenance(ProvenanceTimestamp{3'000}), evaluated);
  test_support::expect(
      future.code() == ValidationCode::freshness_timestamp_in_future &&
          future.has_warning(),
      "future timestamp follows the explicit freshness decision");

  const Provenance malformed_source = make_provenance(
      observed, static_cast<SourceType>(255), OverrideStatus::not_requested);
  test_support::expect(
      no_freshness.validate_provenance(malformed_source, evaluated).code() ==
          ValidationCode::malformed_source_type,
      "unknown source enum is malformed provenance");

  const Provenance malformed_override = make_provenance(
      observed, SourceType::model, static_cast<OverrideStatus>(255));
  test_support::expect(
      no_freshness.validate_provenance(malformed_override, evaluated).code() ==
          ValidationCode::malformed_override_status,
      "unknown override enum is malformed provenance");

  test_support::expect_throws<std::invalid_argument>(
      [] { static_cast<void>(FreshnessRule{10, ValidationDecision::accept}); },
      "freshness violation cannot be configured as accept");

  return test_support::finish();
}
