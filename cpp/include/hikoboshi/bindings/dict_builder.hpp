#ifndef HIKOBOSHI_BINDINGS_DICT_BUILDER_HPP
#define HIKOBOSHI_BINDINGS_DICT_BUILDER_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstddef>
#include <string_view>

#include <hikoboshi/bindings/py_object_ref.hpp>

namespace hikoboshi::bindings {

class DictBuilder final {
 public:
  DictBuilder() : dict_(PyDict_New()) {}

  DictBuilder(const DictBuilder&) = delete;
  DictBuilder& operator=(const DictBuilder&) = delete;

  DictBuilder(DictBuilder&&) noexcept = default;
  DictBuilder& operator=(DictBuilder&&) noexcept = default;

  bool ok() const noexcept { return static_cast<bool>(dict_); }

  PyObject* get() const noexcept { return dict_.get(); }

  explicit operator bool() const noexcept { return ok(); }

  bool set_new_ref(const char* key, PyObject* value) {
    if (value == nullptr) {
      return false;
    }
    PyObjectRef owned(value);
    if (!dict_) {
      return false;
    }
    return PyDict_SetItemString(dict_.get(), key, owned.get()) == 0;
  }

  bool set_borrowed(const char* key, PyObject* value) {
    if (!dict_ || value == nullptr) {
      return false;
    }
    return PyDict_SetItemString(dict_.get(), key, value) == 0;
  }

  bool set_string_view(const char* key, std::string_view value) {
    return set_new_ref(key,
                       PyUnicode_FromStringAndSize(
                           value.empty() ? "" : value.data(),
                           static_cast<Py_ssize_t>(value.size())));
  }

  bool set_bool(const char* key, bool value) {
    return set_new_ref(key, PyBool_FromLong(value ? 1 : 0));
  }

  bool set_size(const char* key, std::size_t value) {
    return set_new_ref(key, PyLong_FromSize_t(value));
  }

  bool set_long(const char* key, long value) {
    return set_new_ref(key, PyLong_FromLong(value));
  }

  bool set_double(const char* key, double value) {
    return set_new_ref(key, PyFloat_FromDouble(value));
  }

  PyObject* release() noexcept { return dict_.release(); }

 private:
  PyObjectRef dict_;
};

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_DICT_BUILDER_HPP
