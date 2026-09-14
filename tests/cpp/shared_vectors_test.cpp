#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "belief_update/core/bayes.hpp"
#include "test_support.hpp"

#ifndef BELIEF_UPDATE_VECTOR_FILE
#error "BELIEF_UPDATE_VECTOR_FILE must identify the shared vector CSV"
#endif

namespace {

struct TestVector final {
  std::string name;
  bool valid;
  double prior;
  double p_e_given_h;
  double p_e_given_not_h;
  double expected_likelihood_ratio;
  double expected_prior_odds;
  double expected_posterior_odds;
  double expected_posterior;
  double expected_belief_delta;
  double absolute_tolerance;
  double relative_tolerance;
};

[[nodiscard]] std::vector<std::string> split_csv_line(
    const std::string& line) {
  std::vector<std::string> fields;
  std::istringstream input{line};
  std::string field;
  while (std::getline(input, field, ',')) {
    fields.push_back(std::move(field));
  }
  return fields;
}

[[nodiscard]] std::vector<TestVector> load_vectors() {
  std::ifstream input{BELIEF_UPDATE_VECTOR_FILE};
  if (!input) {
    throw std::runtime_error("could not open shared Bayes vectors");
  }

  std::string line;
  static_cast<void>(std::getline(input, line));
  std::vector<TestVector> vectors;
  while (std::getline(input, line)) {
    const std::vector<std::string> fields = split_csv_line(line);
    if (fields.size() != 12U) {
      throw std::runtime_error("shared Bayes vector has wrong field count");
    }
    const bool valid = fields[1] == "valid";
    if (!valid && fields[1] != "invalid") {
      throw std::runtime_error("shared Bayes vector has unknown expectation");
    }
    vectors.push_back(TestVector{
        fields[0],
        valid,
        std::stod(fields[2]),
        std::stod(fields[3]),
        std::stod(fields[4]),
        valid ? std::stod(fields[5]) : 0.0,
        valid ? std::stod(fields[6]) : 0.0,
        valid ? std::stod(fields[7]) : 0.0,
        valid ? std::stod(fields[8]) : 0.0,
        valid ? std::stod(fields[9]) : 0.0,
        std::stod(fields[10]),
        std::stod(fields[11]),
    });
  }
  return vectors;
}

[[nodiscard]] bool close(double actual, double expected,
                         double absolute_tolerance,
                         double relative_tolerance) noexcept {
  if (std::isinf(expected)) {
    return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
  }
  if (!std::isfinite(actual)) {
    return false;
  }
  return std::abs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::abs(expected);
}

void expect_value(double actual, double expected, const TestVector& vector,
                  std::string_view field, std::string_view implementation) {
  const std::string description = vector.name + " " +
                                  std::string{implementation} + " " +
                                  std::string{field};
  test_support::expect(
      close(actual, expected, vector.absolute_tolerance,
            vector.relative_tolerance),
      description);
}

void expect_calculation(const belief_update::BayesCalculation& calculation,
                        const TestVector& vector,
                        std::string_view implementation) {
  expect_value(calculation.prior(), vector.prior, vector, "prior",
               implementation);
  expect_value(calculation.likelihood_ratio(),
               vector.expected_likelihood_ratio, vector, "likelihood ratio",
               implementation);
  expect_value(calculation.prior_odds(), vector.expected_prior_odds, vector,
               "prior odds", implementation);
  expect_value(calculation.posterior_odds(), vector.expected_posterior_odds,
               vector, "posterior odds", implementation);
  expect_value(calculation.posterior(), vector.expected_posterior, vector,
               "posterior", implementation);
  expect_value(calculation.belief_delta(), vector.expected_belief_delta, vector,
               "belief delta", implementation);
}

template <typename Function>
[[nodiscard]] bool rejects(Function&& function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

[[nodiscard]] belief_update::BayesCalculation exact_calculation(
    const TestVector& vector) {
  return belief_update::exact_binary_update(
      belief_update::BeliefState{vector.prior},
      belief_update::LikelihoodValues{vector.p_e_given_h,
                                      vector.p_e_given_not_h});
}

[[nodiscard]] belief_update::BayesCalculation log_calculation(
    const TestVector& vector) {
  const belief_update::BeliefState prior{vector.prior};
  const belief_update::LikelihoodValues likelihoods{
      vector.p_e_given_h, vector.p_e_given_not_h};
  const double log_h = likelihoods.evidence_given_hypothesis() == 0.0
                           ? -INFINITY
                           : std::log(
                                 likelihoods.evidence_given_hypothesis());
  const double log_not_h =
      likelihoods.evidence_given_not_hypothesis() == 0.0
          ? -INFINITY
          : std::log(likelihoods.evidence_given_not_hypothesis());
  return belief_update::log_odds_update(prior, log_h - log_not_h);
}

}  // namespace

int main() {
  const std::vector<TestVector> vectors = load_vectors();
  test_support::expect(!vectors.empty(), "shared vector file is not empty");

  for (const TestVector& vector : vectors) {
    if (vector.valid) {
      expect_calculation(exact_calculation(vector), vector, "C++ exact");
      expect_calculation(log_calculation(vector), vector, "C++ log odds");
      continue;
    }

    test_support::expect(
        rejects([&] { static_cast<void>(exact_calculation(vector)); }),
        vector.name + " C++ exact rejects invalid input");
    test_support::expect(
        rejects([&] { static_cast<void>(log_calculation(vector)); }),
        vector.name + " C++ log odds rejects invalid input");
  }

  return test_support::finish();
}
