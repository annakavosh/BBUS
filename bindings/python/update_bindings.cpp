#include "bindings.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "belief_update/model/likelihood_model.hpp"
#include "belief_update/types/update_algorithm.hpp"
#include "belief_update/types/update_inputs.hpp"
#include "belief_update/types/update_result.hpp"
#include "belief_update/update/update.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_update(py::module_& module) {
  py::enum_<UpdateAlgorithm>(module, "UpdateAlgorithm")
      .value("exact_binary", UpdateAlgorithm::exact_binary)
      .value("log_odds", UpdateAlgorithm::log_odds);

  py::class_<UpdateInputs>(module, "UpdateInputs")
      .def(py::init<BeliefState, std::optional<LikelihoodInput>,
                    std::optional<LikelihoodInput>, ProvenanceTimestamp>(),
           py::arg("prior"), py::arg("explicit_override"),
           py::arg("likelihood_model_value"), py::arg("evaluation_time"))
      .def_property_readonly("prior", &UpdateInputs::prior)
      .def_property_readonly("explicit_override",
                             &UpdateInputs::explicit_override)
      .def_property_readonly("likelihood_model_value",
                             &UpdateInputs::likelihood_model_value)
      .def_property_readonly("evaluation_time",
                             &UpdateInputs::evaluation_time);

  py::class_<UpdateResult>(module, "UpdateResult")
      .def_property_readonly("inputs", &UpdateResult::inputs)
      .def_property_readonly("resolved_values", &UpdateResult::resolved_values)
      .def_property_readonly("provenance", &UpdateResult::provenance)
      .def_property_readonly("configuration", &UpdateResult::configuration)
      .def_property_readonly("algorithm", &UpdateResult::algorithm)
      .def_property_readonly("resolved_source",
                             &UpdateResult::resolved_source)
      .def_property_readonly("intermediate_calculations",
                             &UpdateResult::intermediate_calculations)
      .def_property_readonly("posterior", &UpdateResult::posterior)
      .def_property_readonly(
          "validation_warnings",
          [](const UpdateResult& result) {
            py::tuple warnings(result.validation_warnings().size());
            std::size_t index = 0;
            for (const ValidationCode code : result.validation_warnings()) {
              warnings[index] = code;
              ++index;
            }
            return warnings;
          })
      .def_property_readonly("library_version", &UpdateResult::library_version);

  module.def(
      "update",
      static_cast<UpdateResult (*)(UpdateInputs, UpdateConfig,
                                   UpdateAlgorithm)>(&update),
      py::arg("inputs"), py::arg("configuration"), py::arg("algorithm"),
      "Validate and resolve the supplied inputs, run the selected algorithm, "
      "and return an immutable audit record.");
  module.def(
      "update",
      static_cast<UpdateResult (*)(double, const LikelihoodModel&)>(&update),
      py::arg("prior"), py::arg("likelihood_model"),
      "Run an exact-binary update from one model inference using the fixed "
      "convenience configuration.");
  module.def(
      "update", static_cast<UpdateResult (*)(double, double, double)>(&update),
      py::arg("prior"), py::arg("p_e_given_h"),
      py::arg("p_e_given_not_h"),
      "Run an exact-binary update using deterministic direct-input "
      "provenance and the fixed convenience configuration.");
}

}  // namespace belief_update::python
