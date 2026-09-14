#pragma once

#include "belief_update/types/likelihood_input.hpp"

namespace belief_update {

class LikelihoodModel {
 public:
  virtual ~LikelihoodModel() = default;

  LikelihoodModel(const LikelihoodModel&) = delete;
  LikelihoodModel(LikelihoodModel&&) = delete;
  LikelihoodModel& operator=(const LikelihoodModel&) = delete;
  LikelihoodModel& operator=(LikelihoodModel&&) = delete;

  [[nodiscard]] virtual LikelihoodInput infer() const = 0;

 protected:
  LikelihoodModel() = default;
};

}  // namespace belief_update
