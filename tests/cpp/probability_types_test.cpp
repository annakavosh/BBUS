#include <limits>
#include <stdexcept>
#include <type_traits>

#include "belief_update/types/belief_state.hpp"
#include "belief_update/types/likelihood_values.hpp"
#include "test_support.hpp"

int main() {
  using belief_update::BeliefState;
  using belief_update::LikelihoodValues;

  static_assert(!std::is_default_constructible_v<BeliefState>);
  static_assert(!std::is_copy_assignable_v<BeliefState>);

  const BeliefState impossible{0.0};
  const BeliefState certain{1.0};
  test_support::expect(impossible.probability() == 0.0,
                       "zero is a valid belief probability");
  test_support::expect(certain.probability() == 1.0,
                       "one is a valid belief probability");

  const LikelihoodValues likelihoods{0.8, 0.25};
  test_support::expect(likelihoods.evidence_given_hypothesis() == 0.8,
                       "true-hypothesis likelihood is retained");
  test_support::expect(likelihoods.evidence_given_not_hypothesis() == 0.25,
                       "false-hypothesis likelihood is retained");

  test_support::expect_throws<std::invalid_argument>(
      [] { static_cast<void>(BeliefState{-0.01}); },
      "negative probabilities are rejected");
  test_support::expect_throws<std::invalid_argument>(
      [] { static_cast<void>(BeliefState{1.01}); },
      "probabilities above one are rejected");
  test_support::expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(
            BeliefState{std::numeric_limits<double>::quiet_NaN()});
      },
      "NaN probabilities are rejected");
  test_support::expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(LikelihoodValues{
            std::numeric_limits<double>::infinity(), 0.5});
      },
      "infinite likelihoods are rejected");

  return test_support::finish();
}
