#ifndef HIKOBOSHI_BINDINGS_SPAN_CONVERSION_HPP
#define HIKOBOSHI_BINDINGS_SPAN_CONVERSION_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstddef>
#include <limits>
#include <string_view>
#include <type_traits>

#include <hikoboshi/bindings/list_builder.hpp>

namespace hikoboshi::bindings {

namespace detail {

template <typename T>
PyObject* scalar_to_pyobject(const T& value) {
  using Value = typename std::decay<T>::type;
  if constexpr (std::is_same<Value, bool>::value) {
    return PyBool_FromLong(value ? 1 : 0);
  } else if constexpr (std::is_same<Value, char>::value) {
    return PyUnicode_FromStringAndSize(&value, 1);
  } else if constexpr (std::is_floating_point<Value>::value) {
    return PyFloat_FromDouble(static_cast<double>(value));
  } else if constexpr (std::is_integral<Value>::value &&
                       std::is_signed<Value>::value) {
    return PyLong_FromLongLong(static_cast<long long>(value));
  } else if constexpr (std::is_integral<Value>::value &&
                       std::is_unsigned<Value>::value) {
    return PyLong_FromUnsignedLongLong(
        static_cast<unsigned long long>(value));
  } else if constexpr (std::is_same<Value, std::string_view>::value) {
    return PyUnicode_FromStringAndSize(value.empty() ? "" : value.data(),
                                       static_cast<Py_ssize_t>(value.size()));
  } else {
    static_assert(std::is_arithmetic<Value>::value,
                  "span_to_pylist needs a converter for this value type");
  }
}

}  // namespace detail

template <typename SpanLike, typename Converter>
PyObject* span_to_pylist(const SpanLike& span, Converter converter) {
  if (span.size > static_cast<std::size_t>(
                      std::numeric_limits<Py_ssize_t>::max())) {
    PyErr_SetString(PyExc_OverflowError, "span is too large for a Python list");
    return nullptr;
  }

  ListBuilder list(span.size);
  if (!list) {
    return nullptr;
  }
  for (std::size_t index = 0; index < span.size; ++index) {
    PyObject* value = converter(span.data[index]);
    if (!list.set_new_ref(index, value)) {
      return nullptr;
    }
  }
  return list.release();
}

template <typename SpanLike>
PyObject* span_to_pylist(const SpanLike& span) {
  return span_to_pylist(span, [](const auto& value) {
    return detail::scalar_to_pyobject(value);
  });
}

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_SPAN_CONVERSION_HPP
