#include "bindings.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "belief_update/types/validation_result.hpp"
#include "belief_update/validation/validator.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_validation(py::module_& module) {
  py::enum_<ValidationDecision>(module, "ValidationDecision")
      .value("accept", ValidationDecision::accept)
      .value("warn", ValidationDecision::warn)
      .value("reject", ValidationDecision::reject);

  py::enum_<ValidationCode>(module, "ValidationCode")
      .value("valid", ValidationCode::valid)
      .value("probability_not_finite",
             ValidationCode::probability_not_finite)
      .value("probability_out_of_range",
             ValidationCode::probability_out_of_range)
      .value("missing_required_input",
             ValidationCode::missing_required_input)
      .value("empty_source_id", ValidationCode::empty_source_id)
      .value("empty_source_version", ValidationCode::empty_source_version)
      .value("malformed_source_type", ValidationCode::malformed_source_type)
      .value("malformed_override_status",
             ValidationCode::malformed_override_status)
      .value("likelihood_ratio_not_finite",
             ValidationCode::likelihood_ratio_not_finite)
      .value("likelihood_ratio_not_positive",
             ValidationCode::likelihood_ratio_not_positive)
      .value("freshness_violation", ValidationCode::freshness_violation)
      .value("freshness_timestamp_in_future",
             ValidationCode::freshness_timestamp_in_future);

  py::class_<ValidationResult>(module, "ValidationResult")
      .def_property_readonly("decision", &ValidationResult::decision)
      .def_property_readonly("code", &ValidationResult::code)
      .def_property_readonly("can_proceed", &ValidationResult::can_proceed)
      .def_property_readonly("is_accepted", &ValidationResult::is_accepted)
      .def_property_readonly("has_warning", &ValidationResult::has_warning)
      .def_property_readonly("is_rejected", &ValidationResult::is_rejected);

  py::class_<FreshnessRule>(module, "FreshnessRule")
      .def(py::init<std::uint64_t, ValidationDecision>(),
           py::arg("max_age_nanoseconds"), py::arg("violation_decision"))
      .def_property_readonly("max_age_nanoseconds",
                             &FreshnessRule::max_age_nanoseconds)
      .def_property_readonly("violation_decision",
                             &FreshnessRule::violation_decision);

  py::class_<UpdateConfig>(module, "UpdateConfig")
      .def(py::init<std::optional<FreshnessRule>>(),
           py::arg("freshness_rule"))
      .def_property_readonly("freshness_rule",
                             &UpdateConfig::freshness_rule);

  py::class_<Validator>(module, "Validator")
      .def(py::init<UpdateConfig>(), py::arg("configuration"))
      .def("validate_probability", &Validator::validate_probability)
      .def("validate_likelihood_ratio",
           &Validator::validate_likelihood_ratio)
      .def("validate_likelihoods", &Validator::validate_likelihoods)
      .def("validate_provenance", &Validator::validate_provenance)
      .def("validate_input",
           py::overload_cast<const LikelihoodInput&, ProvenanceTimestamp>(
               &Validator::validate, py::const_))
      .def("validate_required_input", &Validator::validate_required_input);
}

}  // namespace belief_update::python
