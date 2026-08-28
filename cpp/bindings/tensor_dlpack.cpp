#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/bindings/args.hpp>
#include <hikoboshi/bindings/tensor_type.hpp>

#include <cstdint>
#include <limits>
#include <new>

namespace hikoboshi::bindings {
namespace {

void managed_tensor_deleter(dlpack::DLManagedTensor* self) {
  if (self == nullptr) {
    return;
  }
  PyObject* owner = reinterpret_cast<PyObject*>(self->manager_ctx);
  delete self;
  if (owner == nullptr || !Py_IsInitialized()) {
    return;
  }
  PyGILState_STATE gil_state = PyGILState_Ensure();
  Py_DECREF(owner);
  PyGILState_Release(gil_state);
}

void capsule_destructor(PyObject* capsule) {
  if (dlpack::kCapsuleDestructorSkipsUsedDltensor &&
      PyCapsule_IsValid(capsule, dlpack::kDLPackUsedCapsuleName)) {
    return;
  }
  auto* managed = static_cast<dlpack::DLManagedTensor*>(
      PyCapsule_GetPointer(capsule, dlpack::kDLPackCapsuleName));
  if (managed == nullptr) {
    PyErr_WriteUnraisable(capsule);
    return;
  }
  if (managed->deleter != nullptr) {
    managed->deleter(managed);
  }
}

bool tuple_matches_cpu_device(PyObject* object) {
  if (!PyTuple_Check(object) || PyTuple_GET_SIZE(object) != 2) {
    return false;
  }
  PyObject* device_type = PyTuple_GET_ITEM(object, 0);
  PyObject* device_id = PyTuple_GET_ITEM(object, 1);
  if (!PyLong_Check(device_type) || !PyLong_Check(device_id)) {
    return false;
  }
  const long type_value = PyLong_AsLong(device_type);
  if (type_value == -1 && PyErr_Occurred()) {
    return false;
  }
  const long id_value = PyLong_AsLong(device_id);
  if (id_value == -1 && PyErr_Occurred()) {
    return false;
  }
  return type_value == dlpack::kDLCPU && id_value == 0;
}

}  // namespace

PyObject* tensor_dlpack(TensorObject* self, PyObject* args, PyObject* kwargs) {
  PyObject* stream = Py_None;
  PyObject* max_version = Py_None;
  PyObject* dl_device = Py_None;
  PyObject* copy = Py_None;
  TypedArgParser parser(args, kwargs, "__dlpack__");
  if (!parser.optional_object("stream", stream, Py_None) ||
      !parser.optional_object("max_version", max_version, Py_None) ||
      !parser.optional_object("dl_device", dl_device, Py_None) ||
      !parser.optional_object("copy", copy, Py_None) || !parser.finish()) {
    return nullptr;
  }
  if (stream != Py_None) {
    PyErr_SetString(PyExc_BufferError,
                    "CPU hikoboshi.Tensor does not use DLPack streams");
    return nullptr;
  }
  if (dl_device != Py_None && !tuple_matches_cpu_device(dl_device)) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_BufferError,
                      "hikoboshi.Tensor can only export to CPU DLPack devices");
    }
    return nullptr;
  }
  if (copy == Py_True) {
    PyErr_SetString(PyExc_BufferError,
                    "hikoboshi.Tensor DLPack export is zero-copy only");
    return nullptr;
  }
  (void)max_version;

  if (self->payload.shape.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    PyErr_SetString(PyExc_OverflowError, "Tensor rank is too large for DLPack");
    return nullptr;
  }

  dlpack::DLManagedTensor* managed = nullptr;
  try {
    managed = new dlpack::DLManagedTensor{};
  } catch (const std::bad_alloc&) {
    PyErr_NoMemory();
    return nullptr;
  }

  const TensorDTypeInfo& dtype = tensor_dtype_info(self->payload.dtype);
  managed->dl_tensor.data = self->payload.storage.empty()
                                ? nullptr
                                : self->payload.storage.data();
  managed->dl_tensor.device = {dlpack::kDLCPU, 0};
  managed->dl_tensor.ndim = static_cast<std::int32_t>(self->payload.shape.size());
  managed->dl_tensor.dtype = {dtype.dlpack_code, dtype.dlpack_bits,
                              dtype.dlpack_lanes};
  managed->dl_tensor.shape = self->payload.dlpack_shape.empty()
                                 ? nullptr
                                 : self->payload.dlpack_shape.data();
  managed->dl_tensor.strides = self->payload.dlpack_strides.empty()
                                   ? nullptr
                                   : self->payload.dlpack_strides.data();
  managed->dl_tensor.byte_offset = 0;
  managed->manager_ctx = reinterpret_cast<void*>(self);
  managed->deleter = managed_tensor_deleter;
  Py_INCREF(reinterpret_cast<PyObject*>(self));

  PyObject* capsule =
      PyCapsule_New(managed, dlpack::kDLPackCapsuleName, capsule_destructor);
  if (capsule == nullptr) {
    managed->deleter(managed);
    return nullptr;
  }
  return capsule;
}

PyObject* tensor_dlpack_device(TensorObject*, PyObject*) {
  return Py_BuildValue("(ii)", static_cast<int>(dlpack::kDLCPU), 0);
}

}  // namespace hikoboshi::bindings
