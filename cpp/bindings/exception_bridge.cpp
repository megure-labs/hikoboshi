#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/engine.hpp>

namespace hikoboshi::bindings {
namespace {

PyObject* g_hikoboshi_error = nullptr;
PyObject* g_invalid_argument_error = nullptr;
PyObject* g_failed_precondition_error = nullptr;
PyObject* g_unavailable_error = nullptr;
PyObject* g_unimplemented_error = nullptr;
PyObject* g_internal_error = nullptr;

const char* status_code_name(hikoboshi::universal::StatusCode code) noexcept {
  switch (code) {
    case hikoboshi::universal::StatusCode::Ok:
      return "ok";
    case hikoboshi::universal::StatusCode::InvalidArgument:
      return "invalid_argument";
    case hikoboshi::universal::StatusCode::FailedPrecondition:
      return "failed_precondition";
    case hikoboshi::universal::StatusCode::Unavailable:
      return "unavailable";
    case hikoboshi::universal::StatusCode::Unimplemented:
      return "unimplemented";
    case hikoboshi::universal::StatusCode::InternalError:
      return "internal_error";
  }
  return "internal_error";
}

PyObject* exception_type_for(hikoboshi::universal::StatusCode code) noexcept {
  switch (code) {
    case hikoboshi::universal::StatusCode::InvalidArgument:
      return g_invalid_argument_error;
    case hikoboshi::universal::StatusCode::FailedPrecondition:
      return g_failed_precondition_error;
    case hikoboshi::universal::StatusCode::Unavailable:
      return g_unavailable_error;
    case hikoboshi::universal::StatusCode::Unimplemented:
      return g_unimplemented_error;
    case hikoboshi::universal::StatusCode::InternalError:
      return g_internal_error;
    case hikoboshi::universal::StatusCode::Ok:
      return g_hikoboshi_error;
  }
  return g_internal_error;
}

int set_attr_string(PyObject* object, const char* name, const char* value) {
  PyObject* py_value = PyUnicode_FromString(value);
  if (py_value == nullptr) {
    return -1;
  }
  const int status = PyObject_SetAttrString(object, name, py_value);
  Py_DECREF(py_value);
  return status;
}

int add_exception(PyObject* module,
                  const char* attr_name,
                  const char* qualified_name,
                  PyObject* base,
                  PyObject** out) {
  PyObject* type = PyErr_NewException(qualified_name, base, nullptr);
  if (type == nullptr) {
    return -1;
  }
  *out = type;
  return PyModule_AddObject(module, attr_name, type);
}

}  // namespace

int raise_if_error(const hikoboshi::universal::Status& status) {
  if (status.code == hikoboshi::universal::StatusCode::Ok) {
    return 0;
  }

  PyObject* exception_type = exception_type_for(status.code);
  if (exception_type == nullptr) {
    exception_type = PyExc_RuntimeError;
  }
  const char* detail =
      (status.detail != nullptr && status.detail[0] != '\0')
          ? status.detail
          : status_code_name(status.code);

  PyObject* args = Py_BuildValue("(s)", detail);
  if (args == nullptr) {
    return -1;
  }
  PyObject* exception = PyObject_CallObject(exception_type, args);
  Py_DECREF(args);
  if (exception == nullptr) {
    return -1;
  }

  if (set_attr_string(exception, "code", status_code_name(status.code)) < 0 ||
      set_attr_string(exception, "status_code", status_code_name(status.code)) <
          0 ||
      set_attr_string(exception, "detail", detail) < 0) {
    Py_DECREF(exception);
    return -1;
  }

  PyErr_SetObject(exception_type, exception);
  Py_DECREF(exception);
  return -1;
}

int bind_exceptions(PyObject* module) {
  if (add_exception(module, "HikoboshiError", "hikoboshi._core.HikoboshiError",
                    PyExc_RuntimeError, &g_hikoboshi_error) < 0) {
    return -1;
  }
  if (add_exception(module, "InvalidArgumentError",
                    "hikoboshi._core.InvalidArgumentError", g_hikoboshi_error,
                    &g_invalid_argument_error) < 0) {
    return -1;
  }
  if (add_exception(module, "FailedPreconditionError",
                    "hikoboshi._core.FailedPreconditionError", g_hikoboshi_error,
                    &g_failed_precondition_error) < 0) {
    return -1;
  }
  if (add_exception(module, "UnavailableError",
                    "hikoboshi._core.UnavailableError", g_hikoboshi_error,
                    &g_unavailable_error) < 0) {
    return -1;
  }
  if (add_exception(module, "UnimplementedError",
                    "hikoboshi._core.UnimplementedError", g_hikoboshi_error,
                    &g_unimplemented_error) < 0) {
    return -1;
  }
  if (add_exception(module, "InternalError", "hikoboshi._core.InternalError",
                    g_hikoboshi_error, &g_internal_error) < 0) {
    return -1;
  }
  return 0;
}

}  // namespace hikoboshi::bindings
