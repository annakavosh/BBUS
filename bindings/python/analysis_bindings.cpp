#include "bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "belief_update/analysis/local_sensitivity.hpp"
#include "belief_update/analysis/posterior_comparison.hpp"
#include "belief_update/analysis/threshold_analysis.hpp"

namespace py = pybind11;

namespace belief_update::python {

void bind_analysis(py::module_& module) {
  py::class_<LocalSensitivityResult>(module, "LocalSensitivityResult")
      .def_property_readonly("prior", &LocalSensitivityResult::prior)
      .def_property_readonly("likelihoods",
                             &LocalSensitivityResult::likelihoods)
      .def_property_readonly("posterior", &LocalSensitivityResult::posterior)
      .def_property_readonly("d_posterior_d_prior",
                             &LocalSensitivityResult::d_posterior_d_prior)
      .def_property_readonly(
          "d_posterior_d_p_e_given_h",
          &LocalSensitivityResult::d_posterior_d_p_e_given_h)
      .def_property_readonly(
          "d_posterior_d_p_e_given_not_h",
          &LocalSensitivityResult::d_posterior_d_p_e_given_not_h);

  py::class_<PosteriorComparisonResult>(module,
                                        "PosteriorComparisonResult")
      .def_property_readonly("model_result",
                             &PosteriorComparisonResult::model_result)
      .def_property_readonly("explicit_result",
                             &PosteriorComparisonResult::explicit_result)
      .def_property_readonly("model_posterior",
                             &PosteriorComparisonResult::model_posterior)
      .def_property_readonly("explicit_posterior",
                             &PosteriorComparisonResult::explicit_posterior)
      .def_property_readonly("difference",
                             &PosteriorComparisonResult::difference);

  py::enum_<ThresholdPosition>(module, "ThresholdPosition")
      .value("below", ThresholdPosition::below)
      .value("at", ThresholdPosition::at)
      .value("above", ThresholdPosition::above);

  py::enum_<ThresholdTransition>(module, "ThresholdTransition")
      .value("none", ThresholdTransition::none)
      .value("upward", ThresholdTransition::upward)
      .value("downward", ThresholdTransition::downward);

  py::class_<ThresholdAnalysisResult>(module, "ThresholdAnalysisResult")
      .def_property_readonly("threshold",
                             &ThresholdAnalysisResult::threshold)
      .def_property_readonly("prior", &ThresholdAnalysisResult::prior)
      .def_property_readonly("posterior",
                             &ThresholdAnalysisResult::posterior)
      .def_property_readonly("posterior_margin",
                             &ThresholdAnalysisResult::posterior_margin)
      .def_property_readonly("prior_position",
                             &ThresholdAnalysisResult::prior_position)
      .def_property_readonly("posterior_position",
                             &ThresholdAnalysisResult::posterior_position)
      .def_property_readonly("transition",
                             &ThresholdAnalysisResult::transition)
      .def_property_readonly("threshold_met",
                             &ThresholdAnalysisResult::threshold_met)
      .def_property_readonly(
          "observed_likelihood_ratio",
          &ThresholdAnalysisResult::observed_likelihood_ratio)
      .def_property_readonly("threshold_reachable",
                             &ThresholdAnalysisResult::threshold_reachable)
      .def_property_readonly(
          "likelihood_ratio_at_threshold",
          &ThresholdAnalysisResult::likelihood_ratio_at_threshold);

  module.def("local_sensitivity", &local_sensitivity, py::arg("result"));
  module.def("compare_model_and_explicit", &compare_model_and_explicit,
             py::arg("result"));
  module.def("analyze_threshold", &analyze_threshold, py::arg("result"),
             py::arg("threshold"));
}

}  // namespace belief_update::python
