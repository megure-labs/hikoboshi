#ifndef HIKOBOSHI_BINDINGS_LIST_BUILDER_HPP
#define HIKOBOSHI_BINDINGS_LIST_BUILDER_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstddef>
#include <limits>

#include <hikoboshi/bindings/py_object_ref.hpp>

namespace hikoboshi::bindings {

class ListBuilder final {
 public:
  explicit ListBuilder(std::size_t size) {
    if (size >
        static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max())) {
      PyErr_SetString(PyExc_OverflowError,
                      "list builder size is too large for Python");
      return;
    }
    list_.reset(PyList_New(static_cast<Py_ssize_t>(size)));
    size_ = list_ ? size : 0;
  }

  ListBuilder(const ListBuilder&) = delete;
  ListBuilder& operator=(const ListBuilder&) = delete;

  ListBuilder(ListBuilder&&) noexcept = default;
  ListBuilder& operator=(ListBuilder&&) noexcept = default;

  bool ok() const noexcept { return static_cast<bool>(list_); }

  PyObject* get() const noexcept { return list_.get(); }

  std::size_t size() const noexcept { return size_; }

  explicit operator bool() const noexcept { return ok(); }

  bool set_new_ref(std::size_t index, PyObject* value) {
    if (value == nullptr) {
      return false;
    }
    if (!list_) {
      Py_DECREF(value);
      return false;
    }
    if (index >= size_) {
      Py_DECREF(value);
      PyErr_SetString(PyExc_IndexError, "list builder index out of range");
      return false;
    }
    return PyList_SetItem(list_.get(), static_cast<Py_ssize_t>(index), value) ==
           0;
  }

  bool append_new_ref(PyObject* value) {
    if (value == nullptr) {
      return false;
    }
    PyObjectRef owned(value);
    if (!list_) {
      return false;
    }
    return PyList_Append(list_.get(), owned.get()) == 0;
  }

  PyObject* release() noexcept {
    size_ = 0;
    return list_.release();
  }

 private:
  PyObjectRef list_;
  std::size_t size_ = 0;
};

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_LIST_BUILDER_HPP
