#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/bindings/buffer_view.hpp>
#include <hikoboshi/bindings/dict_builder.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>

namespace hikoboshi::bindings {
namespace {

PyObject* shape_tuple(const Py_buffer& view) {
  PyObjectRef shape(PyTuple_New(view.ndim));
  if (!shape) {
    return nullptr;
  }
  for (int index = 0; index < view.ndim; ++index) {
    PyObject* item = PyLong_FromSsize_t(view.shape[index]);
    if (item == nullptr) {
      return nullptr;
    }
    PyTuple_SET_ITEM(shape.get(), index, item);
  }
  return shape.release();
}

PyObject* py_buffer_info(PyObject*, PyObject* object) {
  Py_buffer view{};
  if (PyObject_GetBuffer(object, &view,
                         PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
    return nullptr;
  }
  const bool is_float32 =
      detail::buffer_format_matches<float>(view);
  const bool is_c_contiguous = PyBuffer_IsContiguous(&view, 'C') != 0;
  PyObjectRef shape(shape_tuple(view));
  DictBuilder dict;
  if (!shape || !dict) {
    PyBuffer_Release(&view);
    return nullptr;
  }
  const char* format = view.format == nullptr ? "" : view.format;
  if (!dict.set_long("ndim", view.ndim) ||
      !dict.set_borrowed("shape", shape.get()) ||
      !dict.set_new_ref("itemsize", PyLong_FromSsize_t(view.itemsize)) ||
      !dict.set_new_ref("format", PyUnicode_FromString(format)) ||
      !dict.set_bool("c_contiguous", is_c_contiguous) ||
      !dict.set_bool("float32", is_float32)) {
    PyBuffer_Release(&view);
    return nullptr;
  }
  PyBuffer_Release(&view);
  return dict.release();
}

PyObject* py_accepts_float32_2d(PyObject*, PyObject* object) {
  BufferView<const float, 2> view;
  if (!view.acquire(object, "buffer", BufferContiguity::C)) {
    PyErr_Clear();
    Py_RETURN_FALSE;
  }
  Py_RETURN_TRUE;
}

PyMethodDef kMethods[] = {
    {"_buffer_info", py_buffer_info, METH_O,
     "Return buffer metadata used by Hikoboshi NumPy conversion tests."},
    {"_accepts_float32_2d", py_accepts_float32_2d, METH_O,
     "Return whether the object is a C-contiguous float32 2D buffer."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

int bind_numpy_conversion(PyObject* module) {
  for (PyMethodDef* method = kMethods; method->ml_name != nullptr; ++method) {
    PyObjectRef function(PyCFunction_NewEx(method, nullptr, nullptr));
    if (!function) {
      return -1;
    }
    if (PyModule_AddObject(module, method->ml_name, function.release()) < 0) {
      return -1;
    }
  }
  return 0;
}

}  // namespace hikoboshi::bindings
