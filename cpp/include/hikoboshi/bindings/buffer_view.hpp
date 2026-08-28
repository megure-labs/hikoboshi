#ifndef HIKOBOSHI_BINDINGS_BUFFER_VIEW_HPP
#define HIKOBOSHI_BINDINGS_BUFFER_VIEW_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hikoboshi::bindings {

enum class BufferContiguity {
  Any,
  C,
  Fortran,
};

namespace detail {

inline const char* buffer_label(const char* label) noexcept {
  return label == nullptr ? "buffer" : label;
}

inline char normalized_format_code(const char* format) noexcept {
  if (format == nullptr || format[0] == '\0') {
    return '\0';
  }
  if ((format[0] == '<' || format[0] == '>' || format[0] == '=' ||
       format[0] == '@' || format[0] == '!') &&
      format[1] != '\0') {
    return format[1];
  }
  return format[0];
}

template <typename T>
bool buffer_format_matches(const Py_buffer& view) noexcept {
  using Value = typename std::remove_cv<T>::type;
  if (view.itemsize != static_cast<Py_ssize_t>(sizeof(Value))) {
    return false;
  }

  const char code = normalized_format_code(view.format);
  if (code == '\0') {
    return true;
  }

  if constexpr (std::is_same<Value, float>::value) {
    return code == 'f';
  } else if constexpr (std::is_same<Value, double>::value) {
    return code == 'd';
  } else if constexpr (std::is_same<Value, bool>::value) {
    return code == '?';
  } else if constexpr (std::is_same<Value, char>::value) {
    return code == 'c' || code == 'b' || code == 'B';
  } else if constexpr (std::is_integral<Value>::value) {
    if constexpr (std::is_signed<Value>::value) {
      return code == 'b' || code == 'h' || code == 'i' || code == 'l' ||
             code == 'q' || code == 'n';
    } else {
      return code == 'B' || code == 'H' || code == 'I' || code == 'L' ||
             code == 'Q' || code == 'N';
    }
  } else {
    return false;
  }
}

inline bool buffer_is_contiguous(const Py_buffer& view,
                                 BufferContiguity contiguity) noexcept {
  switch (contiguity) {
    case BufferContiguity::Any:
      return true;
    case BufferContiguity::C:
      return PyBuffer_IsContiguous(const_cast<Py_buffer*>(&view), 'C') != 0;
    case BufferContiguity::Fortran:
      return PyBuffer_IsContiguous(const_cast<Py_buffer*>(&view), 'F') != 0;
  }
  return false;
}

}  // namespace detail

template <typename T, int NDim>
class BufferView final {
 public:
  static_assert(NDim >= 0, "BufferView dimension count must be non-negative");

  BufferView() = default;

  BufferView(const BufferView&) = delete;
  BufferView& operator=(const BufferView&) = delete;

  BufferView(BufferView&& other) noexcept
      : view_(other.view_), active_(other.active_) {
    other.view_ = {};
    other.active_ = false;
  }

  BufferView& operator=(BufferView&& other) noexcept {
    if (this != &other) {
      release();
      view_ = other.view_;
      active_ = other.active_;
      other.view_ = {};
      other.active_ = false;
    }
    return *this;
  }

  ~BufferView() { release(); }

  bool acquire(PyObject* object,
               const char* label = nullptr,
               BufferContiguity contiguity = BufferContiguity::Any) {
    release();
    const char* name = detail::buffer_label(label);
    if (object == nullptr) {
      PyErr_Format(PyExc_TypeError, "%s must not be null", name);
      return false;
    }

    Py_buffer view{};
    if (PyObject_GetBuffer(object, &view,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
      return false;
    }
    view_ = view;
    active_ = true;

    if (view_.ndim != NDim) {
      PyErr_Format(PyExc_ValueError, "%s must be a %dD buffer", name, NDim);
      release();
      return false;
    }
    if (view_.shape == nullptr) {
      PyErr_Format(PyExc_ValueError, "%s must expose shape metadata", name);
      release();
      return false;
    }
    for (int index = 0; index < NDim; ++index) {
      if (view_.shape[index] < 0) {
        PyErr_Format(PyExc_ValueError, "%s shape must not be negative", name);
        release();
        return false;
      }
    }
    if (!detail::buffer_format_matches<T>(view_)) {
      PyErr_Format(PyExc_TypeError, "%s has an incompatible dtype", name);
      release();
      return false;
    }
    if (!std::is_const<T>::value && view_.readonly != 0) {
      PyErr_Format(PyExc_TypeError, "%s must be writable", name);
      release();
      return false;
    }
    if (!detail::buffer_is_contiguous(view_, contiguity)) {
      PyErr_Format(PyExc_ValueError, "%s must be contiguous", name);
      release();
      return false;
    }
    return true;
  }

  void release() noexcept {
    if (active_) {
      PyBuffer_Release(&view_);
      view_ = {};
      active_ = false;
    }
  }

  bool active() const noexcept { return active_; }

  const Py_buffer& py_buffer() const noexcept { return view_; }

  T* data() const noexcept { return static_cast<T*>(view_.buf); }

  Py_ssize_t extent(int dimension) const noexcept {
    return view_.shape == nullptr ? 0 : view_.shape[dimension];
  }

  Py_ssize_t stride(int dimension) const noexcept {
    return view_.strides == nullptr ? 0 : view_.strides[dimension];
  }

  std::size_t element_count() const noexcept {
    if (!active_ || view_.itemsize <= 0) {
      return 0;
    }
    return static_cast<std::size_t>(view_.len / view_.itemsize);
  }

 private:
  Py_buffer view_{};
  bool active_ = false;
};

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_BUFFER_VIEW_HPP
