#pragma once

#include "belief_update/types/update_result.hpp"

namespace belief_update {

class PosteriorComparisonResult final {
 public:
  PosteriorComparisonResult(const PosteriorComparisonResult&) = default;
  PosteriorComparisonResult(PosteriorComparisonResult&&) noexcept = default;
  PosteriorComparisonResult& operator=(const PosteriorComparisonResult&) =
      delete;
  PosteriorComparisonResult& operator=(PosteriorComparisonResult&&) = delete;

  [[nodiscard]] const UpdateResult& model_result() const noexcept {
    return model_result_;
  }
  [[nodiscard]] const UpdateResult& explicit_result() const noexcept {
    return explicit_result_;
  }
  [[nodiscard]] double model_posterior() const noexcept {
    return model_result_.posterior();
  }
  [[nodiscard]] double explicit_posterior() const noexcept {
    return explicit_result_.posterior();
  }
  // Difference is explicit posterior minus model posterior.
  [[nodiscard]] double difference() const noexcept { return difference_; }

 private:
  friend PosteriorComparisonResult compare_model_and_explicit(
      const UpdateResult&);

  explicit PosteriorComparisonResult(UpdateResult model_result,
                                     UpdateResult explicit_result);

  UpdateResult model_result_;
  UpdateResult explicit_result_;
  double difference_;
};

[[nodiscard]] PosteriorComparisonResult compare_model_and_explicit(
    const UpdateResult& original);

}  // namespace belief_update
