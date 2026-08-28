#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace hikoboshi::bindings {
int bind_all_vs_all(PyObject* module);
int bind_engine(PyObject* module);
int bind_exceptions(PyObject* module);
int bind_inverse_fold(PyObject* module);
int bind_numpy_conversion(PyObject* module);
int bind_result_types(PyObject* module);
int bind_score_alignment(PyObject* module);
int bind_tensor_type(PyObject* module);
}  // namespace hikoboshi::bindings

namespace {

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "Hikoboshi public API Python extension.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

#define PYBIND11_MODULE(name, variable)                                      \
  static int pybind11_init_##name(PyObject* variable);                       \
  PyMODINIT_FUNC PyInit_##name() {                                           \
    PyObject* module = PyModule_Create(&kModuleDef);                         \
    if (module == nullptr) {                                                 \
      return nullptr;                                                        \
    }                                                                        \
    if (pybind11_init_##name(module) < 0) {                                  \
      Py_DECREF(module);                                                     \
      return nullptr;                                                        \
    }                                                                        \
    return module;                                                           \
  }                                                                          \
  static int pybind11_init_##name(PyObject* variable)

PYBIND11_MODULE(_core, m) {
  if (PyModule_AddStringConstant(m, "__version__", "0.1.0") < 0) {
    return -1;
  }
  if (hikoboshi::bindings::bind_exceptions(m) < 0 ||
      hikoboshi::bindings::bind_result_types(m) < 0 ||
      hikoboshi::bindings::bind_tensor_type(m) < 0 ||
      hikoboshi::bindings::bind_numpy_conversion(m) < 0 ||
      hikoboshi::bindings::bind_engine(m) < 0 ||
      hikoboshi::bindings::bind_inverse_fold(m) < 0 ||
      hikoboshi::bindings::bind_all_vs_all(m) < 0 ||
      hikoboshi::bindings::bind_score_alignment(m) < 0) {
    return -1;
  }
  return 0;
}
