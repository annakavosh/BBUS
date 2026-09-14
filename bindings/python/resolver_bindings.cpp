#include "bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "belief_update/resolver/input_resolver.hpp"
#include "belief_update/types/likelihood_input.hpp"
#include "belief_update/types/resolution_result.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_resolver(py::module_& module) {
  py::class_<LikelihoodInput>(module, "LikelihoodInput")
      .def(py::init<LikelihoodValues, std::optional<double>, Provenance>(),
           py::arg("likelihoods"), py::arg("evidence_weight"),
           py::arg("provenance"))
      .def_property_readonly("likelihoods", &LikelihoodInput::likelihoods)
      .def_property_readonly("p_e_given_h", &LikelihoodInput::p_e_given_h)
      .def_property_readonly("p_e_given_not_h",
                             &LikelihoodInput::p_e_given_not_h)
      .def_property_readonly("evidence_weight",
                             &LikelihoodInput::evidence_weight)
      .def_property_readonly("confidence", &LikelihoodInput::confidence)
      .def_property_readonly("provenance", &LikelihoodInput::provenance);

  py::enum_<ResolvedInputSource>(module, "ResolvedInputSource")
      .value("explicit_override", ResolvedInputSource::explicit_override)
      .value("likelihood_model", ResolvedInputSource::likelihood_model);

  py::class_<ResolvedInput>(module, "ResolvedInput")
      .def_property_readonly("input", &ResolvedInput::input)
      .def_property_readonly("source", &ResolvedInput::source);

  py::class_<ResolutionResult>(module, "ResolutionResult")
      .def_property_readonly("validation", &ResolutionResult::validation)
      .def_property_readonly("has_input", &ResolutionResult::has_input)
      .def_property_readonly("resolved_input",
                             &ResolutionResult::resolved_input);

  py::class_<InputResolver>(module, "InputResolver")
      .def(py::init<Validator>(), py::arg("validator"))
      .def("resolve", &InputResolver::resolve,
           py::arg("explicit_override"),
           py::arg("likelihood_model_value"), py::arg("evaluation_time"),
           py::keep_alive<0, 2>(), py::keep_alive<0, 3>());
}

}  // namespace belief_update::python
