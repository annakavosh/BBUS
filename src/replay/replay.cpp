#include "belief_update/replay/replay.hpp"

#include <stdexcept>

#include "belief_update/update/update.hpp"
#include "belief_update/version.hpp"

namespace belief_update {

UpdateResult replay(const UpdateResult& recorded) {
  if (recorded.library_version() != version) {
    throw std::invalid_argument(
        "recorded result uses a different library version");
  }

  return update(recorded.inputs(), recorded.configuration(),
                recorded.algorithm());
}

}  // namespace belief_update
