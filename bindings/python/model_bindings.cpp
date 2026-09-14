#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "belief_update/model/likelihood_model.hpp"

namespace py = pybind11;

namespace belief_update::python {
namespace {

class PythonLikelihoodModel final : public LikelihoodModel {
 public:
  PythonLikelihoodModel() = default;

  [[nodiscard]] LikelihoodInput infer() const override {
    PYBIND11_OVERRIDE_PURE(LikelihoodInput, LikelihoodModel, infer);
  }
};

}  // namespace

void bind_model(py::module_& module) {
  py::class_<LikelihoodModel, PythonLikelihoodModel>(module,
                                                     "LikelihoodModel")
      .def(py::init<>())
      .def("infer", &LikelihoodModel::infer);
}

}  // namespace belief_update::python
