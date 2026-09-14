#pragma once

#include <cmath>
#include <exception>
#include <iostream>
#include <string_view>

namespace test_support {

inline int failures = 0;

inline void expect(bool condition, std::string_view description) {
  if (!condition) {
    std::cerr << "FAILED: " << description << '\n';
    ++failures;
  }
}

inline void expect_near(double actual, double expected, double tolerance,
                        std::string_view description) {
  expect(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
         description);
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, std::string_view description) {
  try {
    function();
  } catch (const Exception&) {
    return;
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << description << " (unexpected exception: "
              << exception.what() << ")\n";
    ++failures;
    return;
  }

  std::cerr << "FAILED: " << description << " (no exception)\n";
  ++failures;
}

[[nodiscard]] inline int finish() noexcept { return failures == 0 ? 0 : 1; }

}  // namespace test_support
