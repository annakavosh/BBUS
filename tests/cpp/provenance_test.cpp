#include <stdexcept>
#include <string>
#include <type_traits>

#include "belief_update/types/evidence.hpp"
#include "belief_update/types/provenance.hpp"
#include "test_support.hpp"

int main() {
  using namespace belief_update;

  static_assert(!std::is_default_constructible_v<Provenance>);
  static_assert(!std::is_copy_assignable_v<Provenance>);

  const ProvenanceTimestamp timestamp{1'725'000'000'000'000'000LL};
  const Provenance provenance{SourceType::model,
                              SourceId{"risk-model"},
                              SourceVersion{"2026.09"},
                              timestamp,
                              0.95,
                              OverrideStatus::not_requested};

  test_support::expect(provenance.source_type() == SourceType::model,
                       "source type is strongly typed");
  test_support::expect(provenance.source_id().value() == "risk-model",
                       "source id is retained");
  test_support::expect(provenance.version().value() == "2026.09",
                       "source version is retained");
  test_support::expect(provenance.timestamp().unix_nanoseconds() ==
                           timestamp.unix_nanoseconds(),
                       "caller-supplied timestamp is retained exactly");
  test_support::expect(provenance.confidence() == 0.95,
                       "confidence is retained");

  const Evidence evidence{EvidenceValue::present, provenance};
  test_support::expect(evidence.value() == EvidenceValue::present,
                       "evidence value is typed");
  test_support::expect(evidence.provenance().source_id().value() ==
                           "risk-model",
                       "evidence owns its provenance");

  test_support::expect_throws<std::invalid_argument>(
      [] { static_cast<void>(SourceId{std::string{}}); },
      "empty source ids are rejected");
  test_support::expect_throws<std::invalid_argument>(
      [] { static_cast<void>(SourceVersion{std::string{}}); },
      "empty source versions are rejected");
  test_support::expect_throws<std::invalid_argument>(
      [timestamp] {
        static_cast<void>(Provenance{SourceType::sensor,
                                     SourceId{"sensor-1"},
                                     SourceVersion{"1"},
                                     timestamp,
                                     1.1,
                                     OverrideStatus::rejected});
      },
      "invalid confidence is rejected");

  return test_support::finish();
}
