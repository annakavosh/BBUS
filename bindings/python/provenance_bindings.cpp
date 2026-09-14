#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "belief_update/types/evidence.hpp"
#include "belief_update/types/provenance.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_provenance(py::module_& module) {
  py::enum_<SourceType>(module, "SourceType")
      .value("direct_input", SourceType::direct_input)
      .value("sensor", SourceType::sensor)
      .value("human", SourceType::human)
      .value("model", SourceType::model)
      .value("file", SourceType::file)
      .value("system", SourceType::system);

  py::enum_<OverrideStatus>(module, "OverrideStatus")
      .value("not_requested", OverrideStatus::not_requested)
      .value("requested", OverrideStatus::requested)
      .value("applied", OverrideStatus::applied)
      .value("rejected", OverrideStatus::rejected);

  py::class_<SourceId>(module, "SourceId")
      .def(py::init<std::string>(), py::arg("value"))
      .def_property_readonly("value", &SourceId::value);

  py::class_<SourceVersion>(module, "SourceVersion")
      .def(py::init<std::string>(), py::arg("value"))
      .def_property_readonly("value", &SourceVersion::value);

  py::class_<ProvenanceTimestamp>(module, "ProvenanceTimestamp")
      .def(py::init<std::int64_t>(), py::arg("unix_nanoseconds"))
      .def_property_readonly("unix_nanoseconds",
                             &ProvenanceTimestamp::unix_nanoseconds);

  py::class_<Provenance>(module, "Provenance")
      .def(py::init<SourceType, SourceId, SourceVersion, ProvenanceTimestamp,
                    double, OverrideStatus>(),
           py::arg("source_type"), py::arg("source_id"),
           py::arg("version"), py::arg("timestamp"), py::arg("confidence"),
           py::arg("override_status"))
      .def_property_readonly("source_type", &Provenance::source_type)
      .def_property_readonly("source_id", &Provenance::source_id)
      .def_property_readonly("version", &Provenance::version)
      .def_property_readonly("timestamp", &Provenance::timestamp)
      .def_property_readonly("confidence", &Provenance::confidence)
      .def_property_readonly("override_status", &Provenance::override_status);

  py::enum_<EvidenceValue>(module, "EvidenceValue")
      .value("absent", EvidenceValue::absent)
      .value("present", EvidenceValue::present);

  py::class_<Evidence>(module, "Evidence")
      .def(py::init<EvidenceValue, Provenance>(), py::arg("value"),
           py::arg("provenance"))
      .def_property_readonly("value", &Evidence::value)
      .def_property_readonly("provenance", &Evidence::provenance);
}

}  // namespace belief_update::python
