#ifndef HIKOBOSHI_BINDINGS_TENSOR_TYPE_HPP
#define HIKOBOSHI_BINDINGS_TENSOR_TYPE_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/bindings/dlpack/dlpack_compat.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hikoboshi::bindings {

enum class TensorDType : std::uint8_t {
  Float32,
  Float64,
  Int32,
  UInt8,
};

struct TensorDTypeInfo {
  TensorDType dtype;
  const char* name;
  const char* buffer_format;
  std::size_t itemsize;
  std::uint8_t dlpack_code;
  std::uint8_t dlpack_bits;
  std::uint16_t dlpack_lanes;
};

struct TensorPayload {
  TensorDType dtype = TensorDType::Float32;
  std::vector<std::uint8_t> storage;
  std::vector<Py_ssize_t> shape;
  std::vector<Py_ssize_t> strides;
  std::vector<std::int64_t> dlpack_shape;
  std::vector<std::int64_t> dlpack_strides;
};

struct TensorObject {
  PyObject_HEAD
  TensorPayload payload;
};

const TensorDTypeInfo& tensor_dtype_info(TensorDType dtype) noexcept;
bool tensor_dtype_from_name(const char* name, TensorDType& out) noexcept;
bool tensor_dtype_from_buffer(const Py_buffer& view, TensorDType& out) noexcept;
std::size_t tensor_element_count(const TensorPayload& payload) noexcept;
PyObject* tensor_payload_to_list(const TensorPayload& payload);
PyObject* tensor_from_float32_matrix(const std::vector<float>& values,
                                     std::size_t row_count,
                                     std::size_t column_count);

PyTypeObject* tensor_type() noexcept;
int bind_tensor_type(PyObject* module);

PyObject* tensor_dlpack(TensorObject* self, PyObject* args, PyObject* kwargs);
PyObject* tensor_dlpack_device(TensorObject* self, PyObject* Py_UNUSED(ignored));

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_TENSOR_TYPE_HPP
