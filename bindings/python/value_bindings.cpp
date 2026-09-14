#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "belief_update/types/belief_state.hpp"
#include "belief_update/types/likelihood_values.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_values(py::module_& module) {
  py::class_<BeliefState>(module, "BeliefState")
      .def(py::init<double>(), py::arg("probability"))
      .def_property_readonly("probability", &BeliefState::probability);

  py::class_<LikelihoodValues>(module, "LikelihoodValues")
      .def(py::init<double, double>(),
           py::arg("evidence_given_hypothesis"),
           py::arg("evidence_given_not_hypothesis"))
      .def_property_readonly("evidence_given_hypothesis",
                             &LikelihoodValues::evidence_given_hypothesis)
      .def_property_readonly(
          "evidence_given_not_hypothesis",
          &LikelihoodValues::evidence_given_not_hypothesis);

}

}  // namespace belief_update::python
