#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "belief_update/core/bayes.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_core(py::module_& module) {
  module.def(
      "exact_binary_update", &exact_binary_update, py::arg("prior"),
      py::arg("likelihoods"),
      "Low-level calculation for already-resolved likelihoods; it does not "
      "validate provenance or create an UpdateResult.");
  module.def(
      "log_odds_update", &log_odds_update, py::arg("prior"),
      py::arg("log_likelihood_ratio"),
      "Low-level log-odds calculation; it does not validate provenance or "
      "create an UpdateResult.");
}

}  // namespace belief_update::python
