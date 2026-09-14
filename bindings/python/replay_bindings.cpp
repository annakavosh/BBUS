#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "belief_update/replay/replay.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_replay(py::module_& module) {
  module.def("replay", &replay, py::arg("recorded"));
}

}  // namespace belief_update::python
