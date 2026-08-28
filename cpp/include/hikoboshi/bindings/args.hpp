#ifndef HIKOBOSHI_BINDINGS_ARGS_HPP
#define HIKOBOSHI_BINDINGS_ARGS_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/bindings/casters.hpp>

#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

namespace hikoboshi::bindings {

class TypedArgParser final {
 public:
  TypedArgParser(PyObject* args, PyObject* kwargs, const char* callable)
      : args_(args), kwargs_(kwargs), callable_(callable == nullptr ? "function" : callable) {
    if (args_ == nullptr) {
      positional_count_ = 0;
    } else if (!PyTuple_Check(args_)) {
      PyErr_Format(PyExc_TypeError, "%s() args must be a tuple", callable_);
      failed_ = true;
      return;
    } else {
      positional_count_ = PyTuple_GET_SIZE(args_);
    }

    if (kwargs_ != nullptr && !PyDict_Check(kwargs_)) {
      PyErr_Format(PyExc_TypeError, "%s() kwargs must be a dict", callable_);
      failed_ = true;
    }
  }

  TypedArgParser(const TypedArgParser&) = delete;
  TypedArgParser& operator=(const TypedArgParser&) = delete;

  bool ok() const noexcept { return !failed_; }

  template <typename T>
  bool required(const char* name, T& out) {
    PyObject* object = nullptr;
    bool present = false;
    if (!field(name, true, object, present)) {
      return false;
    }
    using Value = typename std::decay<T>::type;
    if (!PythonCaster<Value>::from_python(object, out)) {
      failed_ = true;
      return false;
    }
    return true;
  }

  template <typename T>
  bool optional(const char* name, T& out, const T& default_value) {
    out = default_value;
    PyObject* object = nullptr;
    bool present = false;
    if (!field(name, false, object, present)) {
      return false;
    }
    if (!present) {
      return true;
    }
    using Value = typename std::decay<T>::type;
    if (!PythonCaster<Value>::from_python(object, out)) {
      failed_ = true;
      return false;
    }
    return true;
  }

  bool required_object(const char* name, PyObject*& out) {
    PyObject* object = nullptr;
    bool present = false;
    if (!field(name, true, object, present)) {
      return false;
    }
    out = object;
    return true;
  }

  bool optional_object(const char* name, PyObject*& out, PyObject* default_value) {
    out = default_value;
    PyObject* object = nullptr;
    bool present = false;
    if (!field(name, false, object, present)) {
      return false;
    }
    if (present) {
      out = object;
    }
    return true;
  }

  bool finish() {
    if (failed_) {
      return false;
    }
    if (position_ < positional_count_) {
      PyErr_Format(PyExc_TypeError,
                   "%s() takes at most %zd positional argument%s (%zd given)",
                   callable_, position_, position_ == 1 ? "" : "s",
                   positional_count_);
      failed_ = true;
      return false;
    }
    if (kwargs_ == nullptr) {
      return true;
    }

    PyObject* key = nullptr;
    PyObject* value = nullptr;
    Py_ssize_t cursor = 0;
    while (PyDict_Next(kwargs_, &cursor, &key, &value) != 0) {
      if (!PyUnicode_Check(key)) {
        PyErr_Format(PyExc_TypeError, "%s() keyword names must be strings",
                     callable_);
        failed_ = true;
        return false;
      }
      Py_ssize_t size = 0;
      const char* name = PyUnicode_AsUTF8AndSize(key, &size);
      if (name == nullptr) {
        failed_ = true;
        return false;
      }
      if (!is_known(name, static_cast<std::size_t>(size))) {
        PyErr_Format(PyExc_TypeError,
                     "%s() got an unexpected keyword argument '%U'", callable_,
                     key);
        failed_ = true;
        return false;
      }
    }
    return true;
  }

 private:
  PyObject* keyword_item(const char* name) const {
    if (kwargs_ == nullptr) {
      return nullptr;
    }
    return PyDict_GetItemString(kwargs_, name);
  }

  bool field(const char* name, bool required, PyObject*& object, bool& present) {
    object = nullptr;
    present = false;
    if (failed_) {
      return false;
    }
    known_.push_back(name);

    PyObject* positional = nullptr;
    if (position_ < positional_count_) {
      positional = PyTuple_GET_ITEM(args_, position_);
    }
    PyObject* keyword = keyword_item(name);
    if (positional != nullptr && keyword != nullptr) {
      PyErr_Format(PyExc_TypeError,
                   "%s() got multiple values for argument '%s'", callable_,
                   name);
      failed_ = true;
      return false;
    }

    ++position_;
    if (positional != nullptr) {
      object = positional;
      present = true;
      return true;
    }
    if (keyword != nullptr) {
      object = keyword;
      present = true;
      return true;
    }
    if (required) {
      PyErr_Format(PyExc_TypeError, "%s() missing required argument '%s'",
                   callable_, name);
      failed_ = true;
      return false;
    }
    return true;
  }

  bool is_known(const char* name, std::size_t size) const noexcept {
    for (const char* known : known_) {
      if (std::strlen(known) == size &&
          std::strncmp(known, name, size) == 0) {
        return true;
      }
    }
    return false;
  }

  PyObject* args_ = nullptr;
  PyObject* kwargs_ = nullptr;
  const char* callable_ = "function";
  Py_ssize_t positional_count_ = 0;
  Py_ssize_t position_ = 0;
  bool failed_ = false;
  std::vector<const char*> known_;
};

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_ARGS_HPP
