#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/bindings/args.hpp>
#include <hikoboshi/bindings/list_builder.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>
#include <hikoboshi/bindings/tensor_type.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

namespace hikoboshi::bindings {
namespace {

constexpr TensorDTypeInfo kDTypes[] = {
    {TensorDType::Float32, "float32", "f", sizeof(float),
     dlpack::kDLFloat, 32, 1},
    {TensorDType::Float64, "float64", "d", sizeof(double),
     dlpack::kDLFloat, 64, 1},
    {TensorDType::Int32, "int32", "i", sizeof(std::int32_t),
     dlpack::kDLInt, 32, 1},
    {TensorDType::UInt8, "uint8", "B", sizeof(std::uint8_t),
     dlpack::kDLUInt, 8, 1},
};

bool is_sequence_like(PyObject* object) noexcept {
  return object != nullptr && PySequence_Check(object) != 0 &&
         !PyUnicode_Check(object) && !PyBytes_Check(object) &&
         !PyByteArray_Check(object) && !PyMemoryView_Check(object);
}

char normalized_format_code(const char* format) noexcept {
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

bool parse_dtype(PyObject* dtype_object, TensorDType& dtype) {
  if (dtype_object == nullptr || dtype_object == Py_None) {
    return true;
  }
  if (!PyUnicode_Check(dtype_object)) {
    PyErr_SetString(PyExc_TypeError, "dtype must be a string or None");
    return false;
  }
  Py_ssize_t size = 0;
  const char* text = PyUnicode_AsUTF8AndSize(dtype_object, &size);
  if (text == nullptr) {
    return false;
  }
  const std::string_view name{text, static_cast<std::size_t>(size)};
  TensorDType parsed = TensorDType::Float32;
  if (!tensor_dtype_from_name(name.data(), parsed) ||
      name.size() != std::strlen(tensor_dtype_info(parsed).name)) {
    PyErr_Format(PyExc_TypeError,
                 "unsupported Tensor dtype '%U'; expected float32, float64, "
                 "int32, or uint8",
                 dtype_object);
    return false;
  }
  dtype = parsed;
  return true;
}

bool parse_shape(PyObject* shape_object, std::vector<Py_ssize_t>& shape) {
  shape.clear();
  if (shape_object == nullptr || shape_object == Py_None) {
    return true;
  }

  PyObjectRef sequence(PySequence_Fast(shape_object,
                                       "shape must be a sequence of integers"));
  if (!sequence) {
    return false;
  }
  const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.get());
  shape.reserve(static_cast<std::size_t>(size));
  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t index = 0; index < size; ++index) {
    if (!PyLong_Check(items[index]) || PyBool_Check(items[index])) {
      PyErr_SetString(PyExc_TypeError, "shape entries must be integers");
      return false;
    }
    const Py_ssize_t dim = PyLong_AsSsize_t(items[index]);
    if (dim == -1 && PyErr_Occurred()) {
      return false;
    }
    if (dim < 0) {
      PyErr_SetString(PyExc_ValueError, "shape entries must be non-negative");
      return false;
    }
    shape.push_back(dim);
  }
  return true;
}

bool checked_element_count(const std::vector<Py_ssize_t>& shape,
                           std::size_t& count) {
  count = 1;
  for (const Py_ssize_t dim : shape) {
    if (dim < 0) {
      PyErr_SetString(PyExc_ValueError, "shape entries must be non-negative");
      return false;
    }
    const std::size_t extent = static_cast<std::size_t>(dim);
    if (extent != 0 &&
        count > std::numeric_limits<std::size_t>::max() / extent) {
      PyErr_SetString(PyExc_OverflowError, "tensor shape is too large");
      return false;
    }
    count *= extent;
  }
  return true;
}

bool fill_contiguous_metadata(TensorPayload& payload) {
  const TensorDTypeInfo& info = tensor_dtype_info(payload.dtype);
  std::size_t element_count = 0;
  if (!checked_element_count(payload.shape, element_count)) {
    return false;
  }
  if (element_count != 0 &&
      info.itemsize > std::numeric_limits<std::size_t>::max() / element_count) {
    PyErr_SetString(PyExc_OverflowError, "tensor byte length is too large");
    return false;
  }

  const std::size_t rank = payload.shape.size();
  payload.strides.assign(rank, 0);
  payload.dlpack_shape.assign(rank, 0);
  payload.dlpack_strides.assign(rank, 0);

  std::size_t byte_stride = info.itemsize;
  std::int64_t element_stride = 1;
  for (std::size_t reverse = rank; reverse > 0; --reverse) {
    const std::size_t index = reverse - 1;
    if (byte_stride >
        static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max())) {
      PyErr_SetString(PyExc_OverflowError, "tensor byte strides are too large");
      return false;
    }
    payload.strides[index] = static_cast<Py_ssize_t>(byte_stride);
    payload.dlpack_strides[index] = element_stride;
    payload.dlpack_shape[index] = static_cast<std::int64_t>(payload.shape[index]);

    const std::size_t extent = static_cast<std::size_t>(payload.shape[index]);
    if (extent != 0 &&
        byte_stride > std::numeric_limits<std::size_t>::max() / extent) {
      PyErr_SetString(PyExc_OverflowError, "tensor byte strides are too large");
      return false;
    }
    if (extent != 0 &&
        element_stride >
            std::numeric_limits<std::int64_t>::max() /
                static_cast<std::int64_t>(extent)) {
      PyErr_SetString(PyExc_OverflowError,
                      "tensor element strides are too large");
      return false;
    }
    byte_stride *= extent;
    element_stride *= static_cast<std::int64_t>(extent);
  }

  return true;
}

template <typename T>
void append_value(std::vector<std::uint8_t>& storage, T value) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
  storage.insert(storage.end(), bytes, bytes + sizeof(T));
}

bool append_scalar(PyObject* object,
                   TensorDType dtype,
                   std::vector<std::uint8_t>& storage) {
  switch (dtype) {
    case TensorDType::Float32: {
      const double value = PyFloat_AsDouble(object);
      if (value == -1.0 && PyErr_Occurred()) {
        return false;
      }
      append_value(storage, static_cast<float>(value));
      return true;
    }
    case TensorDType::Float64: {
      const double value = PyFloat_AsDouble(object);
      if (value == -1.0 && PyErr_Occurred()) {
        return false;
      }
      append_value(storage, value);
      return true;
    }
    case TensorDType::Int32: {
      if (!PyLong_Check(object) || PyBool_Check(object)) {
        PyErr_SetString(PyExc_TypeError,
                        "int32 Tensor values must be integers");
        return false;
      }
      const long value = PyLong_AsLong(object);
      if (value == -1 && PyErr_Occurred()) {
        return false;
      }
      if (value < std::numeric_limits<std::int32_t>::min() ||
          value > std::numeric_limits<std::int32_t>::max()) {
        PyErr_SetString(PyExc_OverflowError,
                        "int32 Tensor value is outside the supported range");
        return false;
      }
      append_value(storage, static_cast<std::int32_t>(value));
      return true;
    }
    case TensorDType::UInt8: {
      if (!PyLong_Check(object) || PyBool_Check(object)) {
        PyErr_SetString(PyExc_TypeError,
                        "uint8 Tensor values must be integers");
        return false;
      }
      const unsigned long value = PyLong_AsUnsignedLong(object);
      if (value == static_cast<unsigned long>(-1) && PyErr_Occurred()) {
        return false;
      }
      if (value > std::numeric_limits<std::uint8_t>::max()) {
        PyErr_SetString(PyExc_OverflowError,
                        "uint8 Tensor value is outside the supported range");
        return false;
      }
      append_value(storage, static_cast<std::uint8_t>(value));
      return true;
    }
  }
  PyErr_SetString(PyExc_TypeError, "unsupported Tensor dtype");
  return false;
}

bool infer_shape(PyObject* object,
                 std::vector<Py_ssize_t>& shape,
                 std::size_t depth = 0) {
  if (!is_sequence_like(object)) {
    return true;
  }

  PyObjectRef sequence(PySequence_Fast(object, "Tensor data must be a sequence"));
  if (!sequence) {
    return false;
  }
  const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.get());
  if (shape.size() == depth) {
    shape.push_back(size);
  } else if (shape[depth] != size) {
    PyErr_SetString(PyExc_ValueError, "Tensor data must be rectangular");
    return false;
  }
  if (size == 0) {
    return true;
  }

  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  std::vector<Py_ssize_t> expected_child_shape;
  for (Py_ssize_t index = 0; index < size; ++index) {
    std::vector<Py_ssize_t> candidate = shape;
    if (!infer_shape(items[index], candidate, depth + 1)) {
      return false;
    }
    if (index == 0) {
      expected_child_shape = std::move(candidate);
    } else if (candidate != expected_child_shape) {
      PyErr_SetString(PyExc_ValueError, "Tensor data must be rectangular");
      return false;
    }
  }
  shape = std::move(expected_child_shape);
  return true;
}

bool flatten_data(PyObject* object,
                  const std::vector<Py_ssize_t>& shape,
                  std::size_t depth,
                  TensorDType dtype,
                  std::vector<std::uint8_t>& storage) {
  if (depth == shape.size()) {
    if (is_sequence_like(object)) {
      PyErr_SetString(PyExc_ValueError,
                      "Tensor data nesting is deeper than its shape");
      return false;
    }
    return append_scalar(object, dtype, storage);
  }

  PyObjectRef sequence(PySequence_Fast(object, "Tensor data must be a sequence"));
  if (!sequence) {
    return false;
  }
  const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.get());
  if (size != shape[depth]) {
    PyErr_SetString(PyExc_ValueError, "Tensor data does not match its shape");
    return false;
  }
  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t index = 0; index < size; ++index) {
    if (!flatten_data(items[index], shape, depth + 1, dtype, storage)) {
      return false;
    }
  }
  return true;
}

bool copy_from_buffer(PyObject* object,
                      PyObject* shape_object,
                      PyObject* dtype_object,
                      TensorPayload& payload) {
  Py_buffer view{};
  if (PyObject_GetBuffer(object, &view,
                         PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
    PyErr_Clear();
    return false;
  }

  bool ok = true;
  TensorDType buffer_dtype = TensorDType::Float32;
  if (!tensor_dtype_from_buffer(view, buffer_dtype)) {
    PyErr_SetString(PyExc_TypeError,
                    "buffer dtype is not supported by hikoboshi.Tensor");
    ok = false;
  }
  TensorDType dtype = buffer_dtype;
  if (ok && dtype_object != nullptr && dtype_object != Py_None) {
    if (!parse_dtype(dtype_object, dtype)) {
      ok = false;
    } else if (dtype != buffer_dtype) {
      PyErr_SetString(PyExc_TypeError,
                      "buffer dtype must match the requested Tensor dtype");
      ok = false;
    }
  }

  std::vector<Py_ssize_t> shape;
  if (ok && shape_object != nullptr && shape_object != Py_None) {
    ok = parse_shape(shape_object, shape);
  } else if (ok) {
    if (view.ndim < 0) {
      PyErr_SetString(PyExc_ValueError, "buffer ndim must not be negative");
      ok = false;
    } else if (view.ndim == 0) {
      shape.clear();
    } else if (view.shape == nullptr) {
      PyErr_SetString(PyExc_ValueError, "buffer must expose shape metadata");
      ok = false;
    } else {
      shape.assign(view.shape, view.shape + view.ndim);
    }
  }

  std::size_t count = 0;
  if (ok && !checked_element_count(shape, count)) {
    ok = false;
  }
  if (ok && view.itemsize <= 0) {
    PyErr_SetString(PyExc_ValueError, "buffer itemsize must be positive");
    ok = false;
  }
  if (ok && view.len % view.itemsize != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "buffer length must be a multiple of its itemsize");
    ok = false;
  }
  if (ok &&
      count !=
          static_cast<std::size_t>(view.len / std::max<Py_ssize_t>(view.itemsize, 1))) {
    PyErr_SetString(PyExc_ValueError,
                    "buffer length does not match the requested Tensor shape");
    ok = false;
  }
  if (ok && !PyBuffer_IsContiguous(&view, 'C')) {
    PyErr_SetString(PyExc_ValueError,
                    "hikoboshi.Tensor requires C-contiguous buffer input");
    ok = false;
  }

  if (ok) {
    payload.dtype = dtype;
    payload.shape = std::move(shape);
    if (view.len > 0) {
      payload.storage.assign(static_cast<const std::uint8_t*>(view.buf),
                             static_cast<const std::uint8_t*>(view.buf) +
                                 view.len);
    } else {
      payload.storage.clear();
    }
    ok = fill_contiguous_metadata(payload);
  }

  PyBuffer_Release(&view);
  return ok;
}

bool copy_from_sequence(PyObject* object,
                        PyObject* shape_object,
                        TensorDType dtype,
                        TensorPayload& payload) {
  std::vector<Py_ssize_t> inferred_shape;
  if (!infer_shape(object, inferred_shape)) {
    return false;
  }

  std::vector<Py_ssize_t> shape;
  if (shape_object != nullptr && shape_object != Py_None) {
    if (!parse_shape(shape_object, shape)) {
      return false;
    }
  } else {
    shape = std::move(inferred_shape);
  }

  std::vector<std::uint8_t> storage;
  if (shape.empty() && is_sequence_like(object)) {
    PyErr_SetString(PyExc_ValueError,
                    "scalar Tensor data must not be a sequence");
    return false;
  }

  if (shape_object != nullptr && shape_object != Py_None) {
    if (!flatten_data(object, inferred_shape, 0, dtype, storage)) {
      return false;
    }
    std::size_t requested_count = 0;
    if (!checked_element_count(shape, requested_count)) {
      return false;
    }
    const std::size_t parsed_count =
        storage.size() / tensor_dtype_info(dtype).itemsize;
    if (parsed_count != requested_count) {
      PyErr_SetString(PyExc_ValueError,
                      "Tensor data length does not match the requested shape");
      return false;
    }
  } else if (!flatten_data(object, shape, 0, dtype, storage)) {
    return false;
  }

  payload.dtype = dtype;
  payload.shape = std::move(shape);
  payload.storage = std::move(storage);
  return fill_contiguous_metadata(payload);
}

PyObject* scalar_at(const TensorPayload& payload, std::size_t element_offset) {
  const TensorDTypeInfo& info = tensor_dtype_info(payload.dtype);
  const std::uint8_t* ptr =
      payload.storage.data() + element_offset * info.itemsize;
  switch (payload.dtype) {
    case TensorDType::Float32: {
      float value = 0.0F;
      std::memcpy(&value, ptr, sizeof(value));
      return PyFloat_FromDouble(value);
    }
    case TensorDType::Float64: {
      double value = 0.0;
      std::memcpy(&value, ptr, sizeof(value));
      return PyFloat_FromDouble(value);
    }
    case TensorDType::Int32: {
      std::int32_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return PyLong_FromLong(value);
    }
    case TensorDType::UInt8: {
      std::uint8_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return PyLong_FromUnsignedLong(value);
    }
  }
  PyErr_SetString(PyExc_TypeError, "unsupported Tensor dtype");
  return nullptr;
}

PyObject* list_from_offset(const TensorPayload& payload,
                           std::size_t depth,
                           std::size_t element_offset) {
  if (depth == payload.shape.size()) {
    return scalar_at(payload, element_offset);
  }
  ListBuilder list(static_cast<std::size_t>(payload.shape[depth]));
  if (!list) {
    return nullptr;
  }
  const std::size_t stride =
      static_cast<std::size_t>(payload.dlpack_strides[depth]);
  for (Py_ssize_t index = 0; index < payload.shape[depth]; ++index) {
    PyObject* item =
        list_from_offset(payload, depth + 1,
                         element_offset + static_cast<std::size_t>(index) *
                                              stride);
    if (item == nullptr) {
      return nullptr;
    }
    if (!list.set_new_ref(static_cast<std::size_t>(index), item)) {
      return nullptr;
    }
  }
  return list.release();
}

PyObject* shape_tuple(const TensorPayload& payload) {
  PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(payload.shape.size()));
  if (tuple == nullptr) {
    return nullptr;
  }
  for (std::size_t index = 0; index < payload.shape.size(); ++index) {
    PyObject* item = PyLong_FromSsize_t(payload.shape[index]);
    if (item == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(index), item);
  }
  return tuple;
}

PyObject* tensor_new(PyTypeObject* type, PyObject*, PyObject*) {
  auto* self = reinterpret_cast<TensorObject*>(type->tp_alloc(type, 0));
  if (self == nullptr) {
    return nullptr;
  }
  try {
    new (&self->payload) TensorPayload();
  } catch (...) {
    type->tp_free(reinterpret_cast<PyObject*>(self));
    PyErr_NoMemory();
    return nullptr;
  }
  return reinterpret_cast<PyObject*>(self);
}

int tensor_init(TensorObject* self, PyObject* args, PyObject* kwargs) {
  PyObject* data = nullptr;
  PyObject* shape_object = Py_None;
  PyObject* dtype_object = Py_None;
  TypedArgParser parser(args, kwargs, "Tensor");
  if (!parser.required_object("data", data) ||
      !parser.optional_object("shape", shape_object, Py_None) ||
      !parser.optional_object("dtype", dtype_object, Py_None) ||
      !parser.finish()) {
    return -1;
  }

  TensorPayload payload;
  TensorDType dtype = TensorDType::Float32;
  if (!parse_dtype(dtype_object, dtype)) {
    return -1;
  }

  if (!copy_from_buffer(data, shape_object, dtype_object, payload)) {
    if (PyErr_Occurred() && !is_sequence_like(data)) {
      return -1;
    }
    PyErr_Clear();
    if (!copy_from_sequence(data, shape_object, dtype, payload)) {
      return -1;
    }
  }

  self->payload = std::move(payload);
  return 0;
}

void tensor_dealloc(TensorObject* self) {
  self->payload.~TensorPayload();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int tensor_getbuffer(PyObject* object, Py_buffer* view, int flags) {
  if (view == nullptr) {
    PyErr_SetString(PyExc_BufferError, "missing Py_buffer");
    return -1;
  }
  auto* self = reinterpret_cast<TensorObject*>(object);
  const TensorDTypeInfo& info = tensor_dtype_info(self->payload.dtype);
  view->buf = self->payload.storage.empty() ? nullptr : self->payload.storage.data();
  view->obj = object;
  Py_INCREF(object);
  view->len = static_cast<Py_ssize_t>(self->payload.storage.size());
  view->readonly = 0;
  view->itemsize = static_cast<Py_ssize_t>(info.itemsize);
  view->format =
      (flags & PyBUF_FORMAT) != 0 ? const_cast<char*>(info.buffer_format) : nullptr;
  view->ndim = static_cast<int>(self->payload.shape.size());
  view->shape = self->payload.shape.empty() ? nullptr : self->payload.shape.data();
  view->strides =
      (flags & PyBUF_STRIDES) != 0 && !self->payload.strides.empty()
          ? self->payload.strides.data()
          : nullptr;
  view->suboffsets = nullptr;
  view->internal = nullptr;
  return 0;
}

void tensor_releasebuffer(PyObject*, Py_buffer*) {}

Py_ssize_t tensor_length(TensorObject* self) {
  if (self->payload.shape.empty()) {
    PyErr_SetString(PyExc_TypeError, "scalar Tensor has no len()");
    return -1;
  }
  return self->payload.shape[0];
}

PyObject* tensor_item(TensorObject* self, Py_ssize_t index) {
  if (self->payload.shape.empty()) {
    PyErr_SetString(PyExc_IndexError, "scalar Tensor is not subscriptable");
    return nullptr;
  }
  const Py_ssize_t size = self->payload.shape[0];
  if (index < 0) {
    index += size;
  }
  if (index < 0 || index >= size) {
    PyErr_SetString(PyExc_IndexError, "Tensor index out of range");
    return nullptr;
  }
  const std::size_t offset =
      static_cast<std::size_t>(index) *
      static_cast<std::size_t>(self->payload.dlpack_strides[0]);
  return list_from_offset(self->payload, 1, offset);
}

PyObject* tensor_tolist(TensorObject* self, PyObject*) {
  return tensor_payload_to_list(self->payload);
}

PyObject* tensor_tolist_method(PyObject* self, PyObject*) {
  return tensor_tolist(reinterpret_cast<TensorObject*>(self), nullptr);
}

PyObject* tensor_dlpack_method(PyObject* self, PyObject* args, PyObject* kwargs) {
  return tensor_dlpack(reinterpret_cast<TensorObject*>(self), args, kwargs);
}

PyObject* tensor_dlpack_device_method(PyObject* self, PyObject* ignored) {
  return tensor_dlpack_device(reinterpret_cast<TensorObject*>(self), ignored);
}

PyObject* tensor_shape_get(TensorObject* self, void*) {
  return shape_tuple(self->payload);
}

PyObject* tensor_dtype_get(TensorObject* self, void*) {
  return PyUnicode_FromString(tensor_dtype_info(self->payload.dtype).name);
}

PyObject* tensor_device_get(TensorObject*, void*) {
  return PyUnicode_FromString("cpu");
}

PyObject* tensor_repr(TensorObject* self) {
  PyObjectRef shape(shape_tuple(self->payload));
  if (!shape) {
    return nullptr;
  }
  PyObjectRef shape_repr(PyObject_Repr(shape.get()));
  if (!shape_repr) {
    return nullptr;
  }
  return PyUnicode_FromFormat("Tensor(shape=%U, dtype='%s', device='cpu')",
                              shape_repr.get(),
                              tensor_dtype_info(self->payload.dtype).name);
}

PyMethodDef kTensorMethods[] = {
    {"tolist", tensor_tolist_method, METH_NOARGS,
     "Return the Tensor contents as nested Python lists."},
    {"__dlpack__", reinterpret_cast<PyCFunction>(tensor_dlpack_method),
     METH_VARARGS | METH_KEYWORDS,
     "Export this CPU Tensor as a DLPack v0.8-compatible capsule."},
    {"__dlpack_device__", tensor_dlpack_device_method,
     METH_NOARGS, "Return the DLPack device tuple for this Tensor."},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef kTensorGetSet[] = {
    {"shape", reinterpret_cast<getter>(tensor_shape_get), nullptr,
     const_cast<char*>("Tensor shape tuple."), nullptr},
    {"dtype", reinterpret_cast<getter>(tensor_dtype_get), nullptr,
     const_cast<char*>("Tensor dtype name."), nullptr},
    {"device", reinterpret_cast<getter>(tensor_device_get), nullptr,
     const_cast<char*>("Tensor device name."), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyType_Slot kTensorSlots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(tensor_dealloc)},
    {Py_tp_repr, reinterpret_cast<void*>(tensor_repr)},
    {Py_sq_length, reinterpret_cast<void*>(tensor_length)},
    {Py_sq_item, reinterpret_cast<void*>(tensor_item)},
    {Py_bf_getbuffer, reinterpret_cast<void*>(tensor_getbuffer)},
    {Py_bf_releasebuffer, reinterpret_cast<void*>(tensor_releasebuffer)},
    {Py_tp_methods, kTensorMethods},
    {Py_tp_getset, kTensorGetSet},
    {Py_tp_init, reinterpret_cast<void*>(tensor_init)},
    {Py_tp_new, reinterpret_cast<void*>(tensor_new)},
    {0, nullptr},
};

PyType_Spec kTensorSpec = {
    "hikoboshi.Tensor",
    sizeof(TensorObject),
    0,
    Py_TPFLAGS_DEFAULT,
    kTensorSlots,
};

PyTypeObject* kTensorType = nullptr;

int publish_parent_alias(PyObject* type) {
  PyObject* modules = PyImport_GetModuleDict();
  if (modules == nullptr) {
    return 0;
  }
  PyObject* parent = PyDict_GetItemString(modules, "hikoboshi");
  if (parent == nullptr) {
    return 0;
  }
  return PyObject_SetAttrString(parent, "Tensor", type);
}

}  // namespace

const TensorDTypeInfo& tensor_dtype_info(TensorDType dtype) noexcept {
  for (const TensorDTypeInfo& info : kDTypes) {
    if (info.dtype == dtype) {
      return info;
    }
  }
  return kDTypes[0];
}

bool tensor_dtype_from_name(const char* name, TensorDType& out) noexcept {
  if (name == nullptr) {
    return false;
  }
  for (const TensorDTypeInfo& info : kDTypes) {
    if (std::strcmp(name, info.name) == 0) {
      out = info.dtype;
      return true;
    }
  }
  return false;
}

bool tensor_dtype_from_buffer(const Py_buffer& view, TensorDType& out) noexcept {
  const char code = normalized_format_code(view.format);
  if (view.itemsize == static_cast<Py_ssize_t>(sizeof(float)) &&
      (code == 'f' || code == '\0')) {
    out = TensorDType::Float32;
    return true;
  }
  if (view.itemsize == static_cast<Py_ssize_t>(sizeof(double)) &&
      code == 'd') {
    out = TensorDType::Float64;
    return true;
  }
  if (view.itemsize == static_cast<Py_ssize_t>(sizeof(std::int32_t)) &&
      code == 'i') {
    out = TensorDType::Int32;
    return true;
  }
  if (view.itemsize == static_cast<Py_ssize_t>(sizeof(std::uint8_t)) &&
      (code == 'B' || code == 'b' || code == 'c')) {
    out = TensorDType::UInt8;
    return true;
  }
  return false;
}

std::size_t tensor_element_count(const TensorPayload& payload) noexcept {
  std::size_t count = 1;
  for (const Py_ssize_t dim : payload.shape) {
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

PyObject* tensor_payload_to_list(const TensorPayload& payload) {
  return list_from_offset(payload, 0, 0);
}

PyObject* tensor_from_float32_matrix(const std::vector<float>& values,
                                     std::size_t row_count,
                                     std::size_t column_count) {
  if (kTensorType == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "hikoboshi.Tensor type is not registered");
    return nullptr;
  }
  if (row_count >
          static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max()) ||
      column_count >
          static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max())) {
    PyErr_SetString(PyExc_OverflowError,
                    "encoded embedding shape is too large for Tensor");
    return nullptr;
  }
  if (row_count != 0 &&
      column_count > std::numeric_limits<std::size_t>::max() / row_count) {
    PyErr_SetString(PyExc_OverflowError,
                    "encoded embedding shape is too large for Tensor");
    return nullptr;
  }
  const std::size_t expected = row_count * column_count;
  if (values.size() < expected) {
    PyErr_SetString(PyExc_ValueError, "encoded embedding payload is incomplete");
    return nullptr;
  }
  if (expected != 0 &&
      sizeof(float) > std::numeric_limits<std::size_t>::max() / expected) {
    PyErr_SetString(PyExc_OverflowError,
                    "encoded embedding storage is too large for Tensor");
    return nullptr;
  }

  PyObjectRef object(tensor_new(kTensorType, nullptr, nullptr));
  if (!object) {
    return nullptr;
  }

  TensorPayload payload;
  payload.dtype = TensorDType::Float32;
  payload.shape = {static_cast<Py_ssize_t>(row_count),
                   static_cast<Py_ssize_t>(column_count)};
  const std::size_t byte_count = expected * sizeof(float);
  payload.storage.resize(byte_count);
  if (byte_count != 0) {
    std::memcpy(payload.storage.data(), values.data(), byte_count);
  }
  if (!fill_contiguous_metadata(payload)) {
    return nullptr;
  }

  auto* tensor = reinterpret_cast<TensorObject*>(object.get());
  tensor->payload = std::move(payload);
  return object.release();
}

PyTypeObject* tensor_type() noexcept {
  return kTensorType;
}

int bind_tensor_type(PyObject* module) {
  if (module == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "missing extension module");
    return -1;
  }
  if (kTensorType == nullptr) {
    PyObject* type = PyType_FromSpec(&kTensorSpec);
    if (type == nullptr) {
      return -1;
    }
    kTensorType = reinterpret_cast<PyTypeObject*>(type);
  }

  Py_INCREF(reinterpret_cast<PyObject*>(kTensorType));
  if (PyModule_AddObject(module, "Tensor",
                         reinterpret_cast<PyObject*>(kTensorType)) < 0) {
    Py_DECREF(reinterpret_cast<PyObject*>(kTensorType));
    return -1;
  }
  if (publish_parent_alias(reinterpret_cast<PyObject*>(kTensorType)) < 0) {
    return -1;
  }
  return 0;
}

}  // namespace hikoboshi::bindings
