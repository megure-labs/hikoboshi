#ifndef HIKOBOSHI_BINDINGS_PY_OBJECT_REF_HPP
#define HIKOBOSHI_BINDINGS_PY_OBJECT_REF_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <utility>

namespace hikoboshi::bindings {

class PyObjectRef final {
 public:
  PyObjectRef() noexcept = default;

  explicit PyObjectRef(PyObject* owned) noexcept : object_(owned) {}

  PyObjectRef(const PyObjectRef&) = delete;
  PyObjectRef& operator=(const PyObjectRef&) = delete;

  PyObjectRef(PyObjectRef&& other) noexcept : object_(other.release()) {}

  PyObjectRef& operator=(PyObjectRef&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  ~PyObjectRef() { Py_XDECREF(object_); }

  static PyObjectRef adopt(PyObject* owned) noexcept {
    return PyObjectRef(owned);
  }

  static PyObjectRef borrow(PyObject* borrowed) noexcept {
    Py_XINCREF(borrowed);
    return PyObjectRef(borrowed);
  }

  PyObject* get() const noexcept { return object_; }

  explicit operator bool() const noexcept { return object_ != nullptr; }

  PyObject* release() noexcept {
    PyObject* object = object_;
    object_ = nullptr;
    return object;
  }

  void reset(PyObject* owned = nullptr) noexcept {
    PyObject* old = std::exchange(object_, owned);
    Py_XDECREF(old);
  }

 private:
  PyObject* object_ = nullptr;
};

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_PY_OBJECT_REF_HPP
