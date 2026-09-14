#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "belief_update/types/bayes_calculation.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_result(py::module_& module) {
  py::class_<BayesCalculation>(module, "BayesCalculation")
      .def_property_readonly("prior", &BayesCalculation::prior)
      .def_property_readonly("likelihood_ratio",
                             &BayesCalculation::likelihood_ratio)
      .def_property_readonly("prior_odds", &BayesCalculation::prior_odds)
      .def_property_readonly("posterior_odds",
                             &BayesCalculation::posterior_odds)
      .def_property_readonly("posterior", &BayesCalculation::posterior)
      .def_property_readonly("belief_delta",
                             &BayesCalculation::belief_delta);
}

}  // namespace belief_update::python
