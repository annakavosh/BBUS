#include "bindings.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "belief_update/stress/stress.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_stress(py::module_& module) {
  py::enum_<StressRangeMode>(module, "StressRangeMode")
      .value("relative_percent", StressRangeMode::relative_percent)
      .value("absolute_bounds", StressRangeMode::absolute_bounds);

  py::enum_<ThresholdCrossing>(module, "ThresholdCrossing")
      .value("not_evaluated", ThresholdCrossing::not_evaluated)
      .value("none", ThresholdCrossing::none)
      .value("upward", ThresholdCrossing::upward)
      .value("downward", ThresholdCrossing::downward)
      .value("both", ThresholdCrossing::both);

  py::class_<ProbabilityBounds>(module, "ProbabilityBounds")
      .def(py::init<double, double>(), py::arg("lower"), py::arg("upper"))
      .def_property_readonly("lower", &ProbabilityBounds::lower)
      .def_property_readonly("upper", &ProbabilityBounds::upper);

  py::class_<RelativeStressRanges>(module, "RelativeStressRanges")
      .def(py::init<double, double, double>(), py::arg("prior_percent"),
           py::arg("p_e_given_h_percent"),
           py::arg("p_e_given_not_h_percent"))
      .def_property_readonly("prior_percent",
                             &RelativeStressRanges::prior_percent)
      .def_property_readonly("p_e_given_h_percent",
                             &RelativeStressRanges::p_e_given_h_percent)
      .def_property_readonly(
          "p_e_given_not_h_percent",
          &RelativeStressRanges::p_e_given_not_h_percent);

  py::class_<AbsoluteStressRanges>(module, "AbsoluteStressRanges")
      .def(py::init<ProbabilityBounds, ProbabilityBounds, ProbabilityBounds>(),
           py::arg("prior"), py::arg("p_e_given_h"),
           py::arg("p_e_given_not_h"))
      .def_property_readonly("prior", &AbsoluteStressRanges::prior)
      .def_property_readonly("p_e_given_h",
                             &AbsoluteStressRanges::p_e_given_h)
      .def_property_readonly("p_e_given_not_h",
                             &AbsoluteStressRanges::p_e_given_not_h);

  py::class_<StressScenario>(module, "StressScenario")
      .def_property_readonly("prior", &StressScenario::prior)
      .def_property_readonly("likelihoods", &StressScenario::likelihoods)
      .def_property_readonly("calculation", &StressScenario::calculation)
      .def_property_readonly("posterior", &StressScenario::posterior)
      .def_property_readonly("belief_delta", &StressScenario::belief_delta);

  py::class_<StressResult>(module, "StressResult")
      .def_property_readonly("range_mode", &StressResult::range_mode)
      .def_property_readonly("requested_relative_ranges",
                             &StressResult::requested_relative_ranges)
      .def_property_readonly("resolved_bounds",
                             &StressResult::resolved_bounds)
      .def_property_readonly("original_posterior",
                             &StressResult::original_posterior)
      .def_property_readonly("original_belief_delta",
                             &StressResult::original_belief_delta)
      .def_property_readonly("minimum_posterior_scenario",
                             &StressResult::minimum_posterior_scenario)
      .def_property_readonly("maximum_posterior_scenario",
                             &StressResult::maximum_posterior_scenario)
      .def_property_readonly("worst_case_scenario",
                             &StressResult::worst_case_scenario)
      .def_property_readonly("worst_case_posterior",
                             &StressResult::worst_case_posterior)
      .def_property_readonly("over_update_scenario",
                             &StressResult::over_update_scenario)
      .def_property_readonly("under_update_scenario",
                             &StressResult::under_update_scenario)
      .def_property_readonly("over_update", &StressResult::over_update)
      .def_property_readonly("under_update", &StressResult::under_update)
      .def_property_readonly("over_update_amount",
                             &StressResult::over_update_amount)
      .def_property_readonly("under_update_amount",
                             &StressResult::under_update_amount)
      .def_property_readonly("threshold", &StressResult::threshold)
      .def_property_readonly("threshold_crossing",
                             &StressResult::threshold_crossing);

  module.def(
      "stress_update",
      static_cast<StressResult (*)(const UpdateResult&,
                                   const RelativeStressRanges&,
                                   std::optional<double>)>(&stress_update),
      py::arg("result"), py::arg("ranges"),
      py::arg("threshold") = py::none());
  module.def(
      "stress_update",
      static_cast<StressResult (*)(const UpdateResult&,
                                   const AbsoluteStressRanges&,
                                   std::optional<double>)>(&stress_update),
      py::arg("result"), py::arg("ranges"),
      py::arg("threshold") = py::none());
}

}  // namespace belief_update::python
