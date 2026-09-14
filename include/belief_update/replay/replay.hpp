#pragma once

#include "belief_update/types/update_result.hpp"

namespace belief_update {

[[nodiscard]] UpdateResult replay(const UpdateResult& recorded);

}  // namespace belief_update
