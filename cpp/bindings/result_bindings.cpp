#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/results.hpp>
#include <hikoboshi/bindings/casters.hpp>
#include <hikoboshi/bindings/dict_builder.hpp>
#include <hikoboshi/bindings/tensor_type.hpp>

namespace hikoboshi::bindings {

PyObject* encode_result_to_python(const api::EncodeResult& result) {
  return PythonCaster<api::EncodeResult>::to_python(result);
}

PyObject* pairwise_result_to_python(const api::PairwiseResult& result) {
  return PythonCaster<api::PairwiseResult>::to_python(result);
}

PyObject* all_vs_all_result_to_python(const api::AllVsAllResult& result) {
  return PythonCaster<api::AllVsAllResult>::to_python(result);
}

int bind_result_types(PyObject* module) {
  if (PyModule_AddIntConstant(
          module, "ALIGNMENT_GAP_SENTINEL",
          universal::kAlignmentGapSentinel) < 0) {
    return -1;
  }
  return 0;
}

}  // namespace hikoboshi::bindings
